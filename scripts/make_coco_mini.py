#!/usr/bin/env python
"""Build a proportionally sampled mini COCO dataset.

The primary interface is a downsampling factor.  For example, ``--scale 50``
turns COCO train2017 (118,287 images) into roughly 2,366 images and tries to
keep its instance count close to 860,001 / 50, while covering every source
category that can be covered by the selected images.

The script currently supports annotation files whose annotations have a
``category_id``: ``instances`` and ``person_keypoints``.  COCO captions are
intentionally rejected because they require a separate sampling and
verification path.

Example
-------
    python scripts/make_coco_mini.py \
        --coco-root data/coco \
        --output-dir data/coco_mini \
        --scale 50 \
        --seed 42

The output keeps the COCO layout::

    <output-dir>/
      train2017/                    (symlinked or copied images)
      val2017/
      annotations/
        instances_train2017.json
        instances_val2017.json
        coco_mini_stats.json
      coco_mini_manifest.json       (files maintained by this script)

The number of images is the hard target.  The number of annotations is a
soft target: the sampler uses it while selecting images and reports the actual
deviation.  Exact annotation equality is deliberately not required because
images contain different numbers of instances and categories.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


IMAGE_SPLITS = (
    "train2017",
    "val2017",
    "test2017",
    "train2014",
    "val2014",
    "test2014",
)
SUPPORTED_ANN_TYPES = ("instances", "person_keypoints")
STRATEGY_CHOICES = ("proportional", "random")
STRATEGY_ALIASES = {"rare-category-first": "proportional", "stratified": "proportional"}
MANIFEST_NAME = "coco_mini_manifest.json"
COUNT_CANDIDATE_SAMPLE = 1024
CATEGORY_BALANCE_WEIGHT = 0.75

# These values are safely above the IDs used by COCO.  They namespace IDs by
# split when the source splits reuse an image or annotation ID.
IMAGE_ID_OFFSET = 10_000_000
ANN_ID_OFFSET = 100_000_000


# --------------------------------------------------------------------------- #
# IO and argument helpers
# --------------------------------------------------------------------------- #
def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def dump_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    log(f"  wrote {path}")


def round_positive(value: float) -> int:
    """Round a positive value using ordinary half-up rounding, with min 1."""
    return max(1, int(math.floor(value + 0.5)))


def round_nonnegative(value: float) -> int:
    """Round a non-negative value using ordinary half-up rounding."""
    return max(0, int(math.floor(value + 0.5)))


def parse_splits(spec: str, parser: argparse.ArgumentParser) -> List[str]:
    splits = [part.strip() for part in spec.split(",") if part.strip()]
    if not splits:
        parser.error("--splits must name at least one split")
    duplicates = sorted({s for s in splits if splits.count(s) > 1})
    if duplicates:
        parser.error(f"duplicate split(s) in --splits: {', '.join(duplicates)}")
    unknown = [s for s in splits if s not in IMAGE_SPLITS]
    if unknown:
        parser.error(f"unknown split(s): {', '.join(unknown)} (valid: {', '.join(IMAGE_SPLITS)})")
    return splits


def parse_num_images(spec: str, splits: Sequence[str]) -> Dict[str, int]:
    """Parse the legacy per-split image-count override."""
    spec = spec.strip()
    if "=" not in spec:
        try:
            value = int(spec)
        except ValueError:
            raise ValueError(f"--num-images: expected an integer, got '{spec}'") from None
        if value <= 0:
            raise ValueError("--num-images must be positive")
        return {split: value for split in splits}

    values: Dict[str, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        split, separator, raw_value = chunk.partition("=")
        if not separator or split.strip() not in splits:
            raise ValueError(f"invalid --num-images entry '{chunk}'")
        split = split.strip()
        if split in values:
            raise ValueError(f"duplicate entry for split '{split}' in --num-images")
        try:
            value = int(raw_value.strip())
        except ValueError:
            raise ValueError(
                f"--num-images: count for '{split}' must be an integer"
            ) from None
        if value <= 0:
            raise ValueError(f"--num-images: count for '{split}' must be positive")
        values[split] = value

    missing = [split for split in splits if split not in values]
    if missing:
        raise ValueError(f"--num-images is missing entries for: {', '.join(missing)}")
    return values


def validate_annotation_types(spec: str, parser: argparse.ArgumentParser) -> List[str]:
    ann_types = [part.strip() for part in spec.split(",") if part.strip()]
    if not ann_types:
        parser.error("--annotation-types must name at least one type")
    duplicates = sorted({t for t in ann_types if ann_types.count(t) > 1})
    if duplicates:
        parser.error(f"duplicate annotation type(s): {', '.join(duplicates)}")
    unsupported = [t for t in ann_types if t not in SUPPORTED_ANN_TYPES]
    if unsupported:
        parser.error(
            f"unsupported annotation type(s): {', '.join(unsupported)}; "
            f"supported: {', '.join(SUPPORTED_ANN_TYPES)}. "
            "COCO captions do not have category_id."
        )
    return ann_types


# --------------------------------------------------------------------------- #
# COCO indexing and sampling
# --------------------------------------------------------------------------- #
def index_annotations(
    coco: Dict[str, Any],
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Counter]]:
    """Return annotations and per-category counts grouped by image ID."""
    anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    cats_by_image: Dict[int, Counter] = defaultdict(Counter)
    for ann in coco.get("annotations", []):
        if "image_id" not in ann or "category_id" not in ann:
            raise ValueError("annotation is missing image_id or category_id")
        anns_by_image[ann["image_id"]].append(ann)
        cats_by_image[ann["image_id"]][ann["category_id"]] += 1
    return anns_by_image, cats_by_image


def category_to_images(
    image_ids: Iterable[int], cats_by_image: Dict[int, Counter]
) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = defaultdict(list)
    for image_id in image_ids:
        for category_id in cats_by_image[image_id]:
            result[category_id].append(image_id)
    return result


def proportional_category_targets(
    source_category_counts: Dict[int, int], target_annotations: int
) -> Dict[int, int]:
    """Allocate an integer annotation target across categories proportionally.

    Largest-remainder allocation keeps the per-category targets summing to the
    requested total instead of accumulating independent rounding errors.  If
    the requested total is smaller than the number of non-empty categories,
    every category still receives a target of one because category coverage is
    more important than an impossible exact total in that extreme case.
    """
    positive = [category for category, count in source_category_counts.items() if count > 0]
    if not positive or target_annotations <= 0:
        return {category: 0 for category in source_category_counts}
    if target_annotations < len(positive):
        return {
            category: 1 if source_category_counts.get(category, 0) > 0 else 0
            for category in source_category_counts
        }

    source_total = sum(source_category_counts[category] for category in positive)
    raw = {
        category: source_category_counts[category] * target_annotations / source_total
        for category in positive
    }
    targets = {category: max(1, int(math.floor(raw[category]))) for category in positive}
    remaining = target_annotations - sum(targets.values())

    if remaining > 0:
        order = sorted(
            positive,
            key=lambda category: (raw[category] - math.floor(raw[category]), category),
            reverse=True,
        )
        for index in range(remaining):
            targets[order[index % len(order)]] += 1
    elif remaining < 0:
        order = sorted(
            (category for category in positive if targets[category] > 1),
            key=lambda category: (raw[category] - math.floor(raw[category]), category),
        )
        index = 0
        while remaining < 0 and order:
            category = order[index % len(order)]
            if targets[category] > 1:
                targets[category] -= 1
                remaining += 1
            index += 1

    return {
        category: targets.get(category, 0) for category in source_category_counts
    }


def choose_coverage_image(
    candidates: Sequence[int],
    missing_categories: Set[int],
    image_categories: Dict[int, Set[int]],
    cats_by_image: Dict[int, Counter],
    image_ann_counts: Dict[int, int],
    target_ann_per_image: float,
    target_category_counts: Dict[int, int],
    rng: random.Random,
) -> int:
    """Choose a coverage image while respecting category-count targets."""
    scored: List[Tuple[int, float, float, float, float, int]] = []
    for image_id in candidates:
        new_categories = len(image_categories[image_id] & missing_categories)
        useful_category_count = sum(
            min(cats_by_image[image_id][category], max(1, target_category_counts.get(category, 0)))
            for category in missing_categories
        )
        category_overshoot = sum(
            max(0, cats_by_image[image_id][category] - target_category_counts.get(category, 0))
            for category in missing_categories
        )
        distance = abs(image_ann_counts[image_id] - target_ann_per_image)
        # Random tie-breaker preserves reproducibility without making the
        # category-coverage decision arbitrary.
        scored.append(
            (
                new_categories,
                float(useful_category_count),
                -float(category_overshoot),
                -distance,
                rng.random(),
                image_id,
            )
        )
    return max(scored)[-1]


def choose_count_image(
    candidates: Sequence[int],
    current_annotations: int,
    current_category_counts: Counter,
    target_annotations: int,
    target_category_counts: Dict[int, int],
    slots_after_choice: int,
    cats_by_image: Dict[int, Counter],
    image_ann_counts: Dict[int, int],
    rng: random.Random,
) -> int:
    """Choose an image that improves class proportions and total count."""
    remaining_after = max(0, target_annotations - current_annotations)
    ideal_count = remaining_after / max(1, slots_after_choice + 1)
    scored: List[Tuple[float, float, float, float, float, int]] = []
    for image_id in candidates:
        image_counts = cats_by_image[image_id]
        category_overshoot = 0.0
        category_error_delta = 0.0
        for category, added in image_counts.items():
            target = target_category_counts.get(category, 0)
            current = current_category_counts.get(category, 0)
            projected = current + added
            category_overshoot += max(0, projected - target) / max(1, target)
            category_error_delta += (
                abs(projected - target) - abs(current - target)
            ) / max(1, target)
        category_improvement = -category_error_delta
        total_fit = -abs(image_ann_counts[image_id] - ideal_count) / max(1, ideal_count)
        combined_score = total_fit + CATEGORY_BALANCE_WEIGHT * category_improvement
        scored.append(
            (
                combined_score,
                category_improvement,
                -category_overshoot,
                total_fit,
                rng.random(),
                image_id,
            )
        )
    return max(scored)[-1]


def sample_image_ids(
    image_ids: Sequence[int],
    cats_by_image: Dict[int, Counter],
    ann_counts: Dict[int, int],
    target_images: int,
    target_annotations: int,
    source_categories: Set[int],
    target_category_counts: Dict[int, int],
    strategy: str,
    rng: random.Random,
) -> Tuple[List[int], Set[int]]:
    """Select images and return ``(selected_ids, uncovered_categories)``.

    ``proportional`` first covers categories and then chooses images whose
    annotation counts move the result toward the proportional target.  It is
    intentionally heuristic: exact annotation equality is neither required
    nor always possible for a fixed number of images.
    """
    pool = list(image_ids)
    if not pool or target_images <= 0:
        return [], set(source_categories)

    target_images = min(target_images, len(pool))
    image_categories = {image_id: set(cats_by_image[image_id]) for image_id in pool}
    selected: List[int] = []
    selected_set: Set[int] = set()

    if strategy == "random":
        selected = rng.sample(pool, target_images)
        covered = set().union(*(image_categories[i] for i in selected)) if selected else set()
        return selected, source_categories - covered

    target_ann_per_image = target_annotations / max(1, target_images)
    category_images = category_to_images(pool, cats_by_image)
    missing = set(source_categories)
    current_category_counts: Counter = Counter()
    current_annotations = 0

    # Cover rare categories first.  Choosing an image that covers several
    # missing categories keeps the coverage phase small and leaves most of the
    # image budget for matching the desired instance count.
    while missing and len(selected) < target_images:
        available_categories = [
            category
            for category in missing
            if any(image_id not in selected_set for image_id in category_images.get(category, []))
        ]
        if not available_categories:
            break
        category = min(
            available_categories,
            key=lambda value: (len(category_images.get(value, [])), value),
        )
        candidates = [
            image_id
            for image_id in category_images.get(category, [])
            if image_id not in selected_set
        ]
        if not candidates:
            break
        image_id = choose_coverage_image(
            candidates,
            missing,
            image_categories,
            cats_by_image,
            ann_counts,
            target_ann_per_image,
            target_category_counts,
            rng,
        )
        selected.append(image_id)
        selected_set.add(image_id)
        current_category_counts.update(cats_by_image[image_id])
        current_annotations += ann_counts[image_id]
        missing -= image_categories[image_id]

    # Fill the remaining image budget based on annotation-count distance.
    while len(selected) < target_images:
        remaining_count = len(pool) - len(selected_set)
        if remaining_count <= COUNT_CANDIDATE_SAMPLE:
            candidates = [image_id for image_id in pool if image_id not in selected_set]
        else:
            # Scanning all 118k COCO images for every one of 2.3k selected
            # images is unnecessarily expensive.  A fresh candidate sample is
            # enough for this heuristic and keeps the work roughly linear in
            # the requested mini-set size.
            candidates = [
                image_id
                for image_id in rng.sample(pool, COUNT_CANDIDATE_SAMPLE)
                if image_id not in selected_set
            ]
        if not candidates:
            break
        image_id = choose_count_image(
            candidates,
            current_annotations,
            current_category_counts,
            target_annotations,
            target_category_counts,
            target_images - len(selected) - 1,
            cats_by_image,
            ann_counts,
            rng,
        )
        selected.append(image_id)
        selected_set.add(image_id)
        current_category_counts.update(cats_by_image[image_id])
        current_annotations += ann_counts[image_id]

    covered = set().union(*(image_categories[i] for i in selected)) if selected else set()
    return selected, source_categories - covered


# --------------------------------------------------------------------------- #
# Subset building and image materialization
# --------------------------------------------------------------------------- #
def build_subset(
    coco: Dict[str, Any],
    split: str,
    keep_ids: Sequence[int],
    split_index: int,
    ann_type: str,
    id_policy: str,
    prune_categories: bool,
) -> Tuple[Dict[str, Any], Dict[str, Dict[int, int]]]:
    keep = set(keep_ids)
    anns_by_image, _ = index_annotations(coco)
    images = [img for img in coco.get("images", []) if img["id"] in keep]
    images.sort(key=lambda image: image["id"])

    if id_policy == "remap":
        image_offset = (split_index + 1) * IMAGE_ID_OFFSET
        image_map = {img["id"]: image_offset + img["id"] for img in images}
        ann_offset = (split_index + 1) * ANN_ID_OFFSET
        next_ann_id = ann_offset + 1
    else:
        image_map = {img["id"]: img["id"] for img in images}
        next_ann_id = 0

    annotations: List[Dict[str, Any]] = []
    ann_map: Dict[int, int] = {}
    for image_id in sorted(keep):
        for ann in anns_by_image.get(image_id, []):
            if id_policy == "remap":
                new_ann_id = next_ann_id
                next_ann_id += 1
            else:
                new_ann_id = ann["id"]
            ann_map[ann["id"]] = new_ann_id
            new_ann = dict(ann)
            new_ann["id"] = new_ann_id
            new_ann["image_id"] = image_map[image_id]
            annotations.append(new_ann)

    new_images = []
    for image in images:
        new_image = dict(image)
        new_image["id"] = image_map[image["id"]]
        new_image["file_name"] = os.path.basename(image["file_name"])
        new_images.append(new_image)

    used_categories = {ann["category_id"] for ann in annotations}
    categories = [
        category
        for category in coco.get("categories", [])
        if not prune_categories or category["id"] in used_categories
    ]

    subset = {
        "info": {
            "description": f"coco_mini subset of {split}",
            "version": coco.get("info", {}).get("version", "1.0"),
            "year": coco.get("info", {}).get("year"),
            "contributor": "make_coco_mini.py",
            "source_split": split,
            "source_annotation_type": ann_type,
            "id_policy": id_policy,
        },
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": annotations,
        "categories": categories,
    }
    return subset, {"images": image_map, "annotations": ann_map}


def materialize_images(
    coco_root: str,
    split: str,
    images: Sequence[Dict[str, Any]],
    out_dir: str,
    link_mode: str,
) -> Tuple[int, List[str], List[str]]:
    """Create output images and return count, missing sources, managed paths."""
    src_dir = os.path.join(coco_root, split)
    dst_dir = os.path.join(out_dir, split)
    os.makedirs(dst_dir, exist_ok=True)
    missing: List[str] = []
    written: List[str] = []
    count = 0

    for image in images:
        filename = os.path.basename(image["file_name"])
        source = os.path.join(src_dir, filename)
        destination = os.path.join(dst_dir, filename)
        if not os.path.exists(source):
            missing.append(source)
            continue
        if os.path.lexists(destination):
            os.remove(destination)
        if link_mode == "symlink":
            os.symlink(os.path.abspath(source), destination)
        else:
            shutil.copy2(source, destination)
        count += 1
        written.append(os.path.join(split, filename))
    return count, missing, written


# --------------------------------------------------------------------------- #
# Statistics and verification
# --------------------------------------------------------------------------- #
def split_stats(
    subset: Dict[str, Any],
    target_images: int,
    target_annotations: int,
    source_categories: Set[int],
    source_category_counts: Dict[int, int],
    target_category_counts: Dict[int, int],
    source_image_count: int,
    source_annotation_count: int,
) -> Dict[str, Any]:
    category_names = {c["id"]: c.get("name", str(c["id"])) for c in subset["categories"]}
    per_category = Counter(ann["category_id"] for ann in subset["annotations"])
    areas = [ann.get("area", 0.0) for ann in subset["annotations"]]
    actual_images = len(subset["images"])
    actual_annotations = len(subset["annotations"])
    present = set(per_category)
    category_stats = {}
    relative_errors = []
    for category_id in sorted(source_categories):
        source_count = source_category_counts.get(category_id, 0)
        target_count = target_category_counts.get(category_id, 0)
        actual_count = per_category.get(category_id, 0)
        error = actual_count - target_count
        relative_error = abs(error) / max(1, target_count)
        relative_errors.append(relative_error)
        category_stats[category_names.get(category_id, str(category_id))] = {
            "category_id": category_id,
            "source_annotations": source_count,
            "target_annotations": target_count,
            "sample_annotations": actual_count,
            "error": error,
            "source_fraction": round(source_count / max(1, source_annotation_count), 6),
            "sample_fraction": round(actual_count / max(1, actual_annotations), 6),
        }
    return {
        "target_images": target_images,
        "num_images": actual_images,
        "target_annotations": target_annotations,
        "num_annotations": actual_annotations,
        "image_count_error": actual_images - target_images,
        "annotation_count_error": actual_annotations - target_annotations,
        "source_images": source_image_count,
        "source_annotations": source_annotation_count,
        "actual_image_scale": round(source_image_count / actual_images, 3) if actual_images else None,
        "actual_annotation_scale": round(source_annotation_count / actual_annotations, 3)
        if actual_annotations
        else None,
        "num_categories_total": len(subset["categories"]),
        "num_categories_present": len(present),
        "required_categories": len(source_categories),
        "missing_categories": sorted(source_categories - present),
        "category_distribution_metrics": {
            "mean_abs_relative_error": round(sum(relative_errors) / len(relative_errors), 6)
            if relative_errors
            else 0.0,
            "max_abs_relative_error": round(max(relative_errors), 6) if relative_errors else 0.0,
        },
        "category_stats": category_stats,
        "annotations_per_image": round(actual_annotations / actual_images, 2) if actual_images else 0.0,
        "num_crowd": sum(1 for ann in subset["annotations"] if ann.get("iscrowd")),
        "area_mean": round(sum(areas) / len(areas), 1) if areas else 0.0,
        "area_min": round(min(areas), 1) if areas else 0.0,
        "area_max": round(max(areas), 1) if areas else 0.0,
        "category_distribution": {
            category_names.get(category_id, str(category_id)): count
            for category_id, count in sorted(per_category.items(), key=lambda item: -item[1])
        },
    }


def verify_subset(subset: Dict[str, Any], image_dir: str) -> List[str]:
    """Run structural checks and return human-readable errors."""
    errors: List[str] = []
    image_ids = [image["id"] for image in subset["images"]]
    if len(image_ids) != len(set(image_ids)):
        errors.append("duplicate image ids")
    image_id_set = set(image_ids)

    ann_ids = [ann["id"] for ann in subset["annotations"]]
    if len(ann_ids) != len(set(ann_ids)):
        errors.append("duplicate annotation ids")

    category_ids = {category["id"] for category in subset["categories"]}
    sizes = {image["id"]: (image.get("width"), image.get("height")) for image in subset["images"]}

    for ann in subset["annotations"]:
        if ann["image_id"] not in image_id_set:
            errors.append(f"annotation {ann['id']} references unknown image {ann['image_id']}")
            continue
        if ann["category_id"] not in category_ids:
            errors.append(f"annotation {ann['id']} references unknown category {ann['category_id']}")
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"annotation {ann['id']} has invalid bbox")
        else:
            x, y, width, height = bbox
            if width <= 0 or height <= 0:
                errors.append(f"annotation {ann['id']} has non-positive bbox size")
            image_width, image_height = sizes[ann["image_id"]]
            if image_width and image_height and (
                x < -1 or y < -1 or x + width > image_width + 1 or y + height > image_height + 1
            ):
                errors.append(f"annotation {ann['id']} bbox outside image bounds")
        if ann.get("area", 0) <= 0:
            errors.append(f"annotation {ann['id']} has non-positive area")
        segmentation = ann.get("segmentation")
        if ann.get("iscrowd") == 0 and isinstance(segmentation, list):
            for polygon in segmentation:
                if len(polygon) < 6 or len(polygon) % 2 != 0:
                    errors.append(f"annotation {ann['id']} has malformed polygon")
                    break

    for image in subset["images"]:
        filename = os.path.basename(image["file_name"])
        if not os.path.exists(os.path.join(image_dir, filename)):
            errors.append(f"missing image file {os.path.join(image_dir, filename)}")
    return errors


def verify_with_pycocotools(ann_file: str) -> Optional[str]:
    try:
        from pycocotools.coco import COCO  # type: ignore
    except Exception:
        return None
    try:
        coco = COCO(ann_file)
        n_annotations = sum(len(value) for value in coco.imgToAnns.values())
        return f"pycocotools OK: {len(coco.imgs)} images / {n_annotations} annotations indexed"
    except Exception as exc:  # pragma: no cover - optional dependency
        return f"pycocotools FAILED: {exc}"


# --------------------------------------------------------------------------- #
# Manifest handling
# --------------------------------------------------------------------------- #
def read_manifest(out_dir: str) -> Optional[List[str]]:
    path = os.path.join(out_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    try:
        payload = load_json(path)
        managed = payload.get("managed_files", [])
        if not isinstance(managed, list) or not all(isinstance(item, str) for item in managed):
            raise ValueError("managed_files must be a list of strings")
        return managed
    except Exception as exc:
        log(f"WARNING: {path} cannot be parsed ({exc}); stale-file cleanup skipped")
        return []


def clean_managed_files(out_dir: str, managed: Sequence[str]) -> int:
    removed = 0
    root = os.path.abspath(out_dir)
    for relative in managed:
        path = os.path.abspath(os.path.join(root, relative))
        if os.path.commonpath((root, path)) != root:
            log(f"WARNING: ignoring manifest path outside output directory: {relative}")
            continue
        if os.path.lexists(path):
            os.remove(path)
            removed += 1
        directory = os.path.dirname(path)
        while directory != root:
            try:
                os.rmdir(directory)
            except OSError:
                break
            directory = os.path.dirname(directory)
    return removed


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-root", default="data/coco", help="source COCO root")
    parser.add_argument("--output-dir", default="data/coco_mini", help="destination mini-dataset root")
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="downsampling factor; default 50 means approximately 1/50 of images and annotations",
    )
    parser.add_argument(
        "--num-images",
        default=None,
        help="legacy image-count override: integer or 'train2017=2366,val2017=100'; cannot combine with --scale",
    )
    parser.add_argument(
        "--splits",
        default="train2017,val2017",
        help=f"comma-separated splits (valid: {', '.join(IMAGE_SPLITS)})",
    )
    parser.add_argument(
        "--annotation-types",
        default="instances",
        help="comma-separated annotation types: instances, person_keypoints",
    )
    parser.add_argument(
        "--strategy",
        default="proportional",
        help="proportional: cover categories and approach annotation target; random: uniform image sample",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="how images are placed in the output directory",
    )
    parser.add_argument(
        "--id-policy",
        choices=("auto", "keep", "remap"),
        default="auto",
        help="keep original IDs, remap by split, or auto-detect cross-split collisions",
    )
    parser.add_argument(
        "--prune-categories",
        action="store_true",
        help="drop category metadata absent from the sample; default keeps all source categories",
    )
    parser.add_argument(
        "--min-annotations",
        type=int,
        default=1,
        help="candidate images must have at least this many annotations (default: 1)",
    )
    parser.add_argument("--dry-run", action="store_true", help="show the plan without writing files")
    args = parser.parse_args(argv)

    if args.scale is not None and args.scale <= 0:
        parser.error("--scale must be positive")
    if args.num_images is not None and args.scale is not None:
        parser.error("--num-images cannot be combined with --scale")
    if args.scale is None:
        args.scale = 50.0
    if args.min_annotations < 0:
        parser.error("--min-annotations must be >= 0")

    splits = parse_splits(args.splits, parser)
    ann_types = validate_annotation_types(args.annotation_types, parser)
    strategy = args.strategy.strip().lower()
    if strategy in STRATEGY_ALIASES:
        log(f"NOTE: --strategy '{strategy}' is treated as 'proportional'")
        strategy = STRATEGY_ALIASES[strategy]
    if strategy not in STRATEGY_CHOICES:
        parser.error(f"unknown --strategy '{args.strategy}' (valid: {', '.join(STRATEGY_CHOICES)})")

    image_overrides: Optional[Dict[str, int]] = None
    if args.num_images is not None:
        try:
            image_overrides = parse_num_images(args.num_images, splits)
        except ValueError as exc:
            parser.error(str(exc))

    coco_root = os.path.abspath(args.coco_root)
    out_dir = os.path.abspath(args.output_dir)
    annotations_root = os.path.join(coco_root, "annotations")
    if not os.path.isdir(annotations_root):
        log(f"ERROR: {annotations_root} not found")
        return 1
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        parser.error(f"--output-dir exists but is not a directory: {out_dir}")

    output_existed = os.path.exists(out_dir)
    output_nonempty = output_existed and bool(os.listdir(out_dir))
    previous_manifest = read_manifest(out_dir)

    global_stats: Dict[str, Any] = {
        "source": coco_root,
        "output": out_dir,
        "scale_requested": args.scale,
        "image_overrides": image_overrides,
        "seed": args.seed,
        "strategy": strategy,
        "id_policy_requested": args.id_policy,
        "link_mode": args.link_mode,
        "min_annotations": args.min_annotations,
        "splits": {},
        "cross_split_checks": {},
        "verification": {},
    }

    sampled: List[Dict[str, Any]] = []
    errors: List[str] = []
    all_image_ids: Dict[int, str] = {}
    all_file_names: Dict[str, str] = {}
    id_conflicts: List[str] = []
    file_conflicts: List[str] = []

    # Pass 1 samples every split before writing anything.  This is required for
    # id-policy=auto to see collisions across all requested splits.
    for split_index, split in enumerate(splits):
        rng = random.Random(args.seed + split_index)
        for ann_type in ann_types:
            source_json = os.path.join(annotations_root, f"{ann_type}_{split}.json")
            if not os.path.exists(source_json):
                errors.append(f"missing annotation file: {source_json}")
                continue

            log(f"[{split}/{ann_type}] loading and sampling ...")
            try:
                coco = load_json(source_json)
                anns_by_image, cats_by_image = index_annotations(coco)
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                errors.append(f"{source_json}: cannot index annotations ({exc})")
                continue

            source_image_count = len(coco.get("images", []))
            source_annotation_count = len(coco.get("annotations", []))
            source_categories = {category["id"] for category in coco.get("categories", [])}
            source_category_counts = Counter(
                ann["category_id"] for ann in coco.get("annotations", [])
            )
            if not source_categories:
                errors.append(f"{source_json}: no categories found")
                continue

            image_entries = {image["id"]: image for image in coco.get("images", [])}
            candidates = [
                image_id
                for image_id, image in image_entries.items()
                if os.path.exists(
                    os.path.join(coco_root, split, os.path.basename(image["file_name"]))
                )
                and sum(cats_by_image[image_id].values()) >= args.min_annotations
            ]
            if not candidates:
                errors.append(f"{source_json}: no usable candidate images")
                continue

            if image_overrides is not None:
                target_images = image_overrides[split]
                target_annotations = round_nonnegative(
                    source_annotation_count * target_images / max(1, source_image_count)
                )
            else:
                target_images = round_positive(source_image_count / args.scale)
                target_annotations = round_nonnegative(source_annotation_count / args.scale)
            target_images = min(target_images, len(candidates))
            target_category_counts = proportional_category_targets(
                dict(source_category_counts), target_annotations
            )

            ann_counts = {image_id: len(anns_by_image.get(image_id, [])) for image_id in candidates}
            picked, uncovered = sample_image_ids(
                candidates,
                cats_by_image,
                ann_counts,
                target_images,
                target_annotations,
                source_categories,
                target_category_counts,
                strategy,
                rng,
            )
            if uncovered:
                log(
                    f"[{split}/{ann_type}] WARNING: {len(uncovered)} source categories "
                    "are not represented by the selected images"
                )
            sampled_category_counts = Counter()
            for image_id in picked:
                sampled_category_counts.update(cats_by_image[image_id])

            picked_set = set(picked)
            for image_id in picked:
                if image_id in all_image_ids and all_image_ids[image_id] != split:
                    id_conflicts.append(f"{image_id}: {all_image_ids[image_id]} <-> {split}")
                all_image_ids[image_id] = split
                filename = os.path.basename(image_entries[image_id]["file_name"])
                if filename in all_file_names and all_file_names[filename] != split:
                    file_conflicts.append(f"{filename}: {all_file_names[filename]} <-> {split}")
                all_file_names[filename] = split

            sampled.append(
                {
                    "split": split,
                    "ann_type": ann_type,
                    "split_index": split_index,
                    "source_json": source_json,
                    "picked": picked,
                    "requested_images": target_images,
                    "target_annotations": target_annotations,
                    "source_images": source_image_count,
                    "source_annotations": source_annotation_count,
                    "source_category_counts": dict(source_category_counts),
                    "target_category_counts": target_category_counts,
                    "sampled_category_counts": dict(sampled_category_counts),
                    "source_categories": source_categories,
                    "uncovered_categories": uncovered,
                    "sampled_annotations": sum(ann_counts[image_id] for image_id in picked_set),
                }
            )

    if errors:
        for error in errors:
            log(f"ERROR: {error}")
        return 2
    if not sampled:
        log("ERROR: no samples were produced")
        return 2

    if args.id_policy == "auto":
        policy = "remap" if id_conflicts else "keep"
        log(
            f"id-policy auto: {'remapping IDs' if id_conflicts else 'keeping original IDs'}"
        )
    else:
        policy = args.id_policy
    global_stats["id_policy_effective"] = policy
    global_stats["cross_split_checks"] = {
        "image_id_conflicts": id_conflicts[:20],
        "num_image_id_conflicts": len(id_conflicts),
        "file_name_conflicts": file_conflicts[:20],
        "num_file_name_conflicts": len(file_conflicts),
    }
    if id_conflicts and policy == "keep":
        log(f"WARNING: {len(id_conflicts)} cross-split image ID collision(s)")
    if file_conflicts:
        log(f"WARNING: {len(file_conflicts)} cross-split file-name collision(s)")

    if args.dry_run:
        log("\n=== dry-run plan (nothing will be written) ===")
        log(f"scale={args.scale} strategy={strategy} id_policy={policy}")
        for record in sampled:
            actual_scale = record["source_images"] / max(1, len(record["picked"]))
            category_errors = [
                abs(
                    record["sampled_category_counts"].get(category_id, 0)
                    - record["target_category_counts"].get(category_id, 0)
                )
                / max(1, record["target_category_counts"].get(category_id, 0))
                for category_id in record["source_categories"]
            ]
            log(
                f"  {record['split']}/{record['ann_type']}: "
                f"images {len(record['picked'])}/{record['requested_images']} "
                f"(source {record['source_images']}, scale {actual_scale:.2f}); "
                f"annotations {record['sampled_annotations']}/{record['target_annotations']}; "
                f"categories present {len(record['source_categories'] - record['uncovered_categories'])}/"
                f"{len(record['source_categories'])}; "
                f"mean/max class relative error "
                f"{sum(category_errors) / max(1, len(category_errors)):.4f}/"
                f"{max(category_errors) if category_errors else 0.0:.4f}"
            )
        if previous_manifest:
            log(f"a real run would clean {len(previous_manifest)} managed file(s)")
        return 0

    # Pass 2: clean only files recorded by the previous run, then build output.
    if previous_manifest:
        removed = clean_managed_files(out_dir, previous_manifest)
        if removed:
            log(f"cleaned {removed} stale file(s) from the previous manifest")
    elif output_nonempty:
        log(
            f"WARNING: {out_dir} is non-empty and has no {MANIFEST_NAME}; "
            "unmanaged files will be left untouched"
        )

    annotations_out = os.path.join(out_dir, "annotations")
    os.makedirs(annotations_out, exist_ok=True)
    written: Set[str] = set()
    ok = True

    for record in sampled:
        split = record["split"]
        ann_type = record["ann_type"]
        coco = load_json(record["source_json"])
        subset, mapping = build_subset(
            coco,
            split,
            record["picked"],
            record["split_index"],
            ann_type,
            policy,
            args.prune_categories,
        )

        output_json = os.path.join(annotations_out, f"{ann_type}_{split}.json")
        dump_json(output_json, subset)
        written.add(os.path.relpath(output_json, out_dir))

        linked_count, missing, linked_paths = materialize_images(
            coco_root, split, subset["images"], out_dir, args.link_mode
        )
        written.update(linked_paths)
        if missing:
            ok = False
            log(f"[{split}/{ann_type}] ERROR: {len(missing)} source image(s) missing")

        if policy == "remap":
            mapping_path = os.path.join(annotations_out, f"id_mapping_{ann_type}_{split}.json")
            dump_json(mapping_path, mapping)
            written.add(os.path.relpath(mapping_path, out_dir))

        stats = split_stats(
            subset,
            record["requested_images"],
            record["target_annotations"],
            record["source_categories"],
            record["source_category_counts"],
            record["target_category_counts"],
            record["source_images"],
            record["source_annotations"],
        )
        stats["images_linked"] = linked_count
        stats["id_policy"] = policy
        global_stats["splits"][f"{split}/{ann_type}"] = stats

        errors_in_subset = verify_subset(subset, os.path.join(out_dir, split))
        coco_check = verify_with_pycocotools(output_json)
        global_stats["verification"][f"{split}/{ann_type}"] = {
            "errors": errors_in_subset[:20],
            "num_errors": len(errors_in_subset),
            "pycocotools": coco_check,
        }
        if stats["missing_categories"] and not args.prune_categories:
            ok = False
            log(
                f"[{split}/{ann_type}] ERROR: missing {len(stats['missing_categories'])} "
                "category/categories in selected annotations"
            )
        if errors_in_subset:
            ok = False
            log(f"[{split}/{ann_type}] ERROR: {len(errors_in_subset)} verification error(s)")
        else:
            log(
                f"[{split}/{ann_type}] OK: images={stats['num_images']} "
                f"annotations={stats['num_annotations']} "
                f"categories={stats['num_categories_present']}/{stats['required_categories']} "
                f"annotation_error={stats['annotation_count_error']}"
            )

    stats_path = os.path.join(annotations_out, "coco_mini_stats.json")
    dump_json(stats_path, global_stats)
    written.add(os.path.relpath(stats_path, out_dir))
    dump_json(
        os.path.join(out_dir, MANIFEST_NAME),
        {"managed_files": sorted(written)},
    )

    log("\n=== summary ===")
    for key, stats in global_stats["splits"].items():
        log(
            f"{key}: images={stats['num_images']}/{stats['target_images']} "
            f"annotations={stats['num_annotations']}/{stats['target_annotations']} "
            f"categories={stats['num_categories_present']}/{stats['required_categories']}"
        )
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
