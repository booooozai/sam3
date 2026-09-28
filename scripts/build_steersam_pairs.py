"""Build a validated, read-only training index for six SteerSAM datasets.

Run from the repository root. The source images and annotations are only read;
the output SQLite database and its audit JSON are new derived artifacts.
"""

import argparse
import hashlib
import json
import math
import pickle
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.train.data.pair_contract import SOURCES


def normalize_text(text):
    return " ".join(text.split()).casefold()


def file_identity(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def mask_rle(segmentation, height, width):
    if not segmentation:
        raise ValueError("missing_mask")
    try:
        if isinstance(segmentation, list):
            polygons = [p for p in segmentation if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]
            if not polygons or len(polygons) != len(segmentation):
                raise ValueError("invalid_polygon")
            rle = mask_util.merge(mask_util.frPyObjects(polygons, height, width))
        elif isinstance(segmentation, dict):
            if list(segmentation.get("size", [])) != [height, width]:
                raise ValueError("mask_size_mismatch")
            if isinstance(segmentation.get("counts"), list):
                rle = mask_util.frPyObjects(segmentation, height, width)
            else:
                rle = segmentation
        else:
            raise ValueError("invalid_mask_type")
        decoded = mask_util.decode(rle)
    except ValueError as exc:
        if str(exc) in ("invalid_polygon", "mask_size_mismatch", "invalid_mask_type"):
            raise
        raise ValueError("mask_decode_error") from exc
    except (TypeError, IndexError, OverflowError) as exc:
        raise ValueError("mask_decode_error") from exc
    if decoded.shape != (height, width) or not np.any(decoded):
        raise ValueError("empty_or_wrong_size_mask")
    return {
        "size": [height, width],
        "counts": rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"],
    }, float(np.count_nonzero(decoded))


def phrasecut_rle(polygons, height, width):
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("missing_mask")
    flattened = []
    for polygon in polygons:
        if not polygon:
            raise ValueError("invalid_polygon")
        if isinstance(polygon[0], (int, float)):
            coords = polygon
        else:
            coords = [coord for point in polygon for coord in point]
        if len(coords) < 6 or len(coords) % 2 or not all(math.isfinite(float(x)) for x in coords):
            raise ValueError("invalid_polygon")
        flattened.append(coords)
    return mask_rle(flattened, height, width)


def validate_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("invalid_box")
    try:
        x, y, w, h = map(float, box)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_box") from exc
    # PhraseCut/VG occasionally stores a region box slightly outside the
    # rasterized image because of its coordinate conversion. Clip overhangs
    # of at most 5% per axis; reject larger errors or boxes with no overlap.
    outside_fraction = max(-x / width, -y / height, (x + w - width) / width,
                           (y + h - height) / height, 0.0)
    if (
        not all(math.isfinite(v) for v in (x, y, w, h))
        or w <= 0 or h <= 0 or outside_fraction > 0.05
    ):
        raise ValueError("invalid_box")
    if outside_fraction == 0.0:
        return [x, y, w, h]
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    x, y = max(0.0, x), max(0.0, y)
    w, h = x1 - x, y1 - y
    if w <= 0 or h <= 0:
        raise ValueError("invalid_box")
    return [x, y, w, h]


def image_key_for_coco(image_id):
    return f"coco:{int(image_id)}"


def coco_image_path(data_root, image_id):
    name = f"{int(image_id):012d}.jpg"
    for split in ("train2017", "val2017"):
        path = f"coco/{split}/{name}"
        if (data_root / path).is_file():
            return path
    return f"coco/train2017/{name}"  # rejected later as missing_image


def coco_candidates(annotation_file, source, split, data_root):
    with annotation_file.open() as handle:
        document = json.load(handle)
    images = {int(item["id"]): item for item in document["images"]}
    categories = {
        int(item["id"]): item["name"].replace("_", " ") if source == "lvis" else item["name"]
        for item in document["categories"]
    }
    grouped = defaultdict(list)
    for annotation in document["annotations"]:
        grouped[(int(annotation["image_id"]), int(annotation["category_id"]))].append(annotation)
    for (image_id, category_id), targets in sorted(grouped.items()):
        non_crowd = [item for item in targets if not item.get("iscrowd", 0)]
        image = images[image_id]
        if source == "lvis":
            image_path = coco_image_path(data_root, image_id)
            exhaustive = category_id not in image.get("not_exhaustive_category_ids", [])
        else:
            image_path = f"coco/{split}2017/{image['file_name']}"
            exhaustive = True
        yield {
            "source": source, "split": split, "source_record_id": f"{image_id}:{category_id}",
            "image_key": image_key_for_coco(image_id), "image_path": image_path,
            "original_image_id": image_id, "original_category_id": category_id,
            "text": categories[category_id], "width": image["width"], "height": image["height"],
            "is_exhaustive": exhaustive, "is_pixel_exhaustive": exhaustive,
            "excluded_crowd_targets": len(targets) - len(non_crowd),
            "raw_targets": [(item["id"], item["bbox"], item.get("segmentation")) for item in non_crowd],
            "target_format": "coco",
        }


def refcoco_candidates(ref_file, instances_file, source, data_root):
    with ref_file.open("rb") as handle:
        refs = pickle.load(handle)  # local canonical REFER annotations
    with instances_file.open() as handle:
        document = json.load(handle)
    images = {int(item["id"]): item for item in document["images"]}
    annotations = {int(item["id"]): item for item in document["annotations"]}
    for ref in refs:
        split = ref["split"]
        if split not in ("train", "val", "test", "testA", "testB"):
            continue
        image_id, ann_id = int(ref["image_id"]), int(ref["ann_id"])
        image, annotation = images[image_id], annotations[ann_id]
        for sentence_index, sentence in enumerate(ref.get("sentences", [])):
            yield {
                "source": source, "split": split,
                "source_record_id": f"{ref['ref_id']}:{sentence.get('sent_id', sentence_index)}",
                "image_key": image_key_for_coco(image_id),
                "image_path": coco_image_path(data_root, image_id),
                "original_image_id": image_id,
                "original_category_id": int(ref.get("category_id", -1)),
                "text": sentence.get("sent", sentence.get("raw", "")),
                "width": image["width"], "height": image["height"],
                "is_exhaustive": True, "is_pixel_exhaustive": True,
                "excluded_crowd_targets": int(bool(annotation.get("iscrowd", 0))),
                "raw_targets": [] if annotation.get("iscrowd", 0) else [
                    (ann_id, annotation["bbox"], annotation.get("segmentation"))
                ],
                "target_format": "coco",
            }


def phrasecut_candidates(refer_file, metadata):
    split = refer_file.stem.removeprefix("refer_")
    with refer_file.open() as handle:
        tasks = json.load(handle)
    for task in tasks:
        image_id = int(task["image_id"])
        image = metadata[image_id]
        if image.get("split") != split:
            yield {"source": "phrasecut", "split": split, "invalid_reason": "split_mismatch"}
            continue
        ann_ids = task.get("ann_ids", [])
        boxes = task.get("instance_boxes", [])
        polygons = task.get("Polygons", [])
        # PhraseCut's ann_ids identify source VG annotations, not mask regions.
        # One annotation can produce several instance boxes/polygons (and vice
        # versa), so only the region-aligned arrays must have equal lengths.
        if not boxes or len(boxes) != len(polygons):
            yield {"source": "phrasecut", "split": split, "invalid_reason": "target_list_mismatch"}
            continue
        image_key = (
            image_key_for_coco(image["coco_id"])
            if image.get("coco_id") is not None else f"vg:{image_id}"
        )
        yield {
            "source": "phrasecut", "split": split, "source_record_id": str(task["task_id"]),
            "image_key": image_key, "image_path": f"PhraseCutDataset/images/{image_id}.jpg",
            "original_image_id": image_id, "original_category_id": -1,
            "source_annotation_ids": ann_ids,
            "text": task.get("phrase", ""), "width": image["width"], "height": image["height"],
            "is_exhaustive": True, "is_pixel_exhaustive": True,
            "raw_targets": [
                (f"{task['task_id']}:{i}", box, polygon)
                for i, (box, polygon) in enumerate(zip(boxes, polygons))
            ],
            "target_format": "phrasecut",
        }


class PairBuilder:
    def __init__(self, data_root, conn, tokenizer, max_targets=200, max_source_pairs=None):
        self.data_root = data_root
        self.conn = conn
        self.tokenizer = tokenizer
        self.max_targets = max_targets
        self.max_source_pairs = max_source_pairs
        self.image_cache = {}
        self.report = defaultdict(lambda: {"raw": 0, "accepted": 0, "excluded_crowd_targets": 0,
                                           "clipped_box_targets_in_valid_candidates": 0,
                                           "reasons": Counter(), "token_lengths": Counter()})
        self.retained_tied_conflicts = 0
        self._processed = Counter()

    def _image_size(self, rel_path):
        if rel_path not in self.image_cache:
            path = self.data_root / rel_path
            try:
                with Image.open(path) as image:
                    size = image.size
                    image.load()  # Verify pixel decoding, not only the file header.
                self.image_cache[rel_path] = size
            except (OSError, ValueError, Image.DecompressionBombError):
                self.image_cache[rel_path] = None
        return self.image_cache[rel_path]

    def add(self, candidate):
        source, split = candidate["source"], candidate["split"]
        if source not in SOURCES:
            raise ValueError(f"Unknown source {source!r}")
        key = (source, split)
        if self.max_source_pairs is not None and self._processed[key] >= self.max_source_pairs:
            return
        self._processed[key] += 1
        stats = self.report[key]
        stats["raw"] += 1
        stats["excluded_crowd_targets"] += candidate.get("excluded_crowd_targets", 0)
        try:
            if "invalid_reason" in candidate:
                raise ValueError(candidate["invalid_reason"])
            text = candidate["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty_text")
            token_count = len(self.tokenizer.encode(text)) + 2
            stats["token_lengths"][token_count] += 1
            if token_count > 32:
                raise ValueError("text_over_32_bpe")
            width, height = int(candidate["width"]), int(candidate["height"])
            if width <= 0 or height <= 0:
                raise ValueError("invalid_image_size")
            if self._image_size(candidate["image_path"]) != (width, height):
                raise ValueError("missing_corrupt_or_size_mismatch_image")
            raw_targets = candidate["raw_targets"]
            if not raw_targets:
                raise ValueError("crowd_only" if candidate.get("excluded_crowd_targets") else "no_targets")
            if len(raw_targets) > self.max_targets:
                raise ValueError("too_many_targets")
            targets = []
            clipped_box_targets = 0
            for ann_id, raw_box, raw_segmentation in raw_targets:
                box = validate_box(raw_box, width, height)
                was_clipped = list(map(float, raw_box)) != box
                clipped_box_targets += was_clipped
                if candidate["target_format"] == "phrasecut":
                    rle, area = phrasecut_rle(raw_segmentation, height, width)
                else:
                    rle, area = mask_rle(raw_segmentation, height, width)
                target = {"original_annotation_id": ann_id, "bbox": box,
                          "area": area, "segmentation": rle}
                if was_clipped:
                    target["original_bbox"] = raw_box
                targets.append(target)
            stats["clipped_box_targets_in_valid_candidates"] += clipped_box_targets
            signature_items = sorted(
                (tuple(item["bbox"]),
                 tuple(item["segmentation"]["size"]), item["segmentation"]["counts"])
                for item in targets
            )
            signature = hashlib.sha256(
                json.dumps(signature_items, separators=(",", ":")).encode()
            ).hexdigest()
            pair_id = f"{source}:{split}:{candidate['source_record_id']}"
            payload = {
                "pair_id": pair_id, "source": source, "split": split,
                "source_dataset": source, "original_split": split,
                "source_record_id": candidate["source_record_id"],
                "image_key": candidate["image_key"], "image_path": candidate["image_path"],
                "original_image_id": candidate["original_image_id"],
                "original_category_id": candidate["original_category_id"],
                "source_annotation_ids": candidate.get("source_annotation_ids"),
                "text": text, "token_count": token_count,
                "width": width, "height": height,
                "is_exhaustive": candidate["is_exhaustive"],
                "is_pixel_exhaustive": candidate["is_pixel_exhaustive"],
                "targets": targets,
            }
            if self._insert(payload, normalize_text(text), signature):
                stats["accepted"] += 1
        except ValueError as exc:
            stats["reasons"][str(exc)] += 1

    def _insert(self, payload, normalized_text, signature):
        source, split = payload["source"], payload["split"]
        existing = self.conn.execute(
            "SELECT id, source, target_count, signature FROM pairs "
            "WHERE split=? AND image_key=? AND normalized_text=?",
            (split, payload["image_key"], normalized_text),
        ).fetchall()
        target_count = len(payload["targets"])
        if any(previous_signature == signature for _, _, _, previous_signature in existing):
            self.report[(source, split)]["reasons"]["exact_duplicate"] += 1
            return False
        if any(previous_count > target_count for _, _, previous_count, _ in existing):
            self.report[(source, split)]["reasons"]["fewer_targets_conflict"] += 1
            return False
        for row_id, previous_source, previous_count, previous_signature in existing:
            if previous_count < target_count:
                self.conn.execute("DELETE FROM pairs WHERE id=?", (row_id,))
                self.report[(previous_source, split)]["reasons"]["replaced_by_more_targets"] += 1
            else:
                self.retained_tied_conflicts += 1
        self.conn.execute(
            "INSERT INTO pairs(pair_id,source,split,image_key,normalized_text,target_count,signature,payload) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (payload["pair_id"], source, split, payload["image_key"], normalized_text,
             target_count, signature, json.dumps(payload, separators=(",", ":"))),
        )
        return True


def create_schema(conn):
    conn.executescript("""
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO metadata VALUES ('schema_version', '1');
        INSERT INTO metadata VALUES ('build_state', 'building');
        CREATE TABLE pairs (
            id INTEGER PRIMARY KEY,
            pair_id TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            split TEXT NOT NULL,
            image_key TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            target_count INTEGER NOT NULL,
            signature TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX pairs_lookup ON pairs(split,image_key,normalized_text);
        CREATE INDEX pairs_source_split ON pairs(source,split);
        CREATE INDEX pairs_exact_cross_split
            ON pairs(image_key,normalized_text,signature,split);
    """)


def finalize_index(conn, max_source_pairs=None, max_targets=200):
    """Mark a finished build; partial indexes require explicit runtime opt-in."""
    values = {
        "build_state": "complete" if max_source_pairs is None else "partial",
        "build_parameters": json.dumps({
            "max_source_pairs": max_source_pairs, "max_targets": max_targets,
        }),
    }
    conn.executemany("INSERT OR REPLACE INTO metadata VALUES (?,?)", values.items())


def remove_cross_split_duplicates(conn):
    """Keep held-out copies of exact image/text/target training records."""
    predicate = """
        split='train' AND EXISTS (
            SELECT 1 FROM pairs AS held
            WHERE held.split!='train' AND held.image_key=pairs.image_key
              AND held.normalized_text=pairs.normalized_text
              AND held.signature=pairs.signature)
    """
    count = conn.execute(f"SELECT COUNT(*) FROM pairs WHERE {predicate}").fetchone()[0]
    conn.execute(f"DELETE FROM pairs WHERE {predicate}")
    return count


def source_candidates(data_root, input_files):
    for split in ("train", "val"):
        yield from coco_candidates(input_files[f"coco_{split}"], "coco", split, data_root)
    for split in ("train", "val"):
        yield from coco_candidates(input_files[f"lvis_{split}"], "lvis", split, data_root)
    for source in ("refcoco", "refcoco_plus", "refcocog"):
        yield from refcoco_candidates(input_files[f"{source}_refs"], input_files[f"{source}_instances"], source, data_root)
    with input_files["phrasecut_metadata"].open() as handle:
        metadata = {int(item["image_id"]): item for item in json.load(handle)}
    for split in ("train", "val", "test"):
        yield from phrasecut_candidates(input_files[f"phrasecut_{split}"], metadata)


def input_file_map(data_root):
    coco = data_root / "coco"
    phrasecut = data_root / "PhraseCutDataset"
    return {
        "coco_train": coco / "annotations/instances_train2017.json",
        "coco_val": coco / "annotations/instances_val2017.json",
        "lvis_train": data_root / "lvis/annotations/lvis_v1_train.json",
        "lvis_val": data_root / "lvis/annotations/lvis_v1_val.json",
        "refcoco_refs": coco / "refcoco/refs(unc).p",
        "refcoco_instances": coco / "refcoco/instances.json",
        "refcoco_plus_refs": coco / "refcoco+/refs(unc).p",
        "refcoco_plus_instances": coco / "refcoco+/instances.json",
        "refcocog_refs": coco / "refcocog/refs(umd).p",
        "refcocog_instances": coco / "refcocog/instances.json",
        "phrasecut_metadata": phrasecut / "image_data_split.json",
        "phrasecut_train": phrasecut / "refer_train.json",
        "phrasecut_val": phrasecut / "refer_val.json",
        "phrasecut_test": phrasecut / "refer_test.json",
    }


def build(args):
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = input_file_map(data_root)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source annotations: {missing}")
    tokenizer = SimpleTokenizer(str(Path(args.bpe_path).resolve()))
    with tempfile.NamedTemporaryFile(prefix="steersam_pairs_", suffix=".sqlite", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with sqlite3.connect(temporary) as conn:
            create_schema(conn)
            builder = PairBuilder(data_root, conn, tokenizer, args.max_targets, args.max_source_pairs)
            for index, candidate in enumerate(source_candidates(data_root, files), 1):
                builder.add(candidate)
                if index % 10000 == 0:
                    conn.commit()
                    print(f"Processed {index:,} raw pairs", flush=True)
            # A held-out exact record wins over a training duplicate. Different
            # descriptions of the same image remain in their native splits.
            cross_split_duplicates = remove_cross_split_duplicates(conn)
            final_counts = {f"{source}/{split}": count for source, split, count in conn.execute(
                "SELECT source,split,COUNT(*) FROM pairs GROUP BY source,split")}
            finalize_index(conn, args.max_source_pairs, args.max_targets)
            conn.commit()
        report = {
            "schema_version": 1,
            "data_root": str(data_root),
            "output": str(output),
            "partial_build_max_source_pairs": args.max_source_pairs,
            "max_targets": args.max_targets,
            "input_files": {name: file_identity(path) for name, path in files.items()},
            "cross_split_exact_training_duplicates_removed": cross_split_duplicates,
            "equal_count_different_annotation_pairs_retained": builder.retained_tied_conflicts,
            "final_counts": final_counts,
            "sources": {
                f"{source}/{split}": {
                    "raw": stat["raw"], "accepted_before_cross_split_filter": stat["accepted"],
                    "excluded_crowd_targets": stat["excluded_crowd_targets"],
                    "clipped_box_targets_in_valid_candidates": stat["clipped_box_targets_in_valid_candidates"],
                    "reasons": dict(stat["reasons"]),
                    "token_lengths": dict(sorted(stat["token_lengths"].items())),
                }
                for (source, split), stat in builder.report.items()
            },
        }
        temporary.replace(output)
        report_path = output.with_suffix(".audit.json")
        with report_path.open("w") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"Wrote {output} and {report_path}")
        print("Final pair counts:", final_counts)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data/steersam_pairs/pairs.sqlite")
    parser.add_argument("--bpe-path", default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--max-targets", type=int, default=200)
    parser.add_argument("--max-source-pairs", type=int, default=None,
                        help="CPU smoke only: process at most N raw candidates per source and split")
    args = parser.parse_args()
    if args.max_targets < 1 or (args.max_source_pairs is not None and args.max_source_pairs < 1):
        parser.error("--max-targets and --max-source-pairs must be positive")
    build(args)


if __name__ == "__main__":
    main()
