#!/usr/bin/env python
"""Sample a mini COCO-style dataset (e.g. ``coco_mini``) from the full COCO dataset.

The script keeps the original COCO format so that any COCO loader (pycocotools,
detectron2, SAM3 dataloader, ...) can consume the output directly.

Example
-------
    # 10 train images + 5 val images, images symlinked, seed fixed
    python scripts/make_coco_mini.py \
        --coco-root data/coco \
        --output-dir data/coco_mini \
        --num-images train2017=10,val2017=5 \
        --seed 42

    # 100 images per split, stratified over categories, real file copy
    python scripts/make_coco_mini.py \
        --num-images 100 --strategy stratified --link-mode copy

Output layout
-------------
    <output-dir>/
      train2017/                    (symlinked or copied images)
      val2017/
      annotations/
        instances_train2017.json
        instances_val2017.json
        id_mapping.json             (only when ids are remapped)
        coco_mini_stats.json        (sampling stats + verification report)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_SPLITS = ("train2017", "val2017", "test2017", "train2014", "val2014", "test2014")

# Offsets used when remapping ids so that ids stay globally unique across splits.
IMAGE_ID_OFFSET = 10_000_000
ANN_ID_OFFSET = 100_000_000


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def dump_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    log(f"  wrote {path}")


def parse_num_images(spec: str, splits: Sequence[str]) -> Dict[str, int]:
    """Parse ``--num-images``.

    Accepts either a single integer (applied to every split) or a comma
    separated ``split=count`` mapping, e.g. ``train2017=100,val2017=20``.
    """
    spec = spec.strip()
    if "=" not in spec:
        n = int(spec)
        if n <= 0:
            raise ValueError("--num-images must be positive")
        return {s: n for s in splits}

    mapping: Dict[str, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        split, _, value = chunk.partition("=")
        split, value = split.strip(), value.strip()
        if split not in splits:
            raise ValueError(f"unknown split '{split}' (available: {', '.join(splits)})")
        mapping[split] = int(value)
    missing = [s for s in splits if s not in mapping]
    if missing:
        raise ValueError(f"--num-images is missing entries for: {', '.join(missing)}")
    return mapping


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def index_annotations(coco: Dict[str, Any]) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Counter]]:
    """Return ``image_id -> annotations`` and ``image_id -> {category_id: count}``."""
    anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    cats_by_image: Dict[int, Counter] = defaultdict(Counter)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)
        cats_by_image[ann["image_id"]][ann["category_id"]] += 1
    return anns_by_image, cats_by_image


def sample_image_ids(
    image_ids: Sequence[int],
    cats_by_image: Dict[int, Counter],
    num: int,
    strategy: str,
    rng: random.Random,
) -> List[int]:
    """Pick ``num`` image ids.

    ``random``: uniform sample without replacement.
    ``stratified``: round-robin over categories (shuffled, fewest-images-first)
    so that rare categories are covered as early as possible; falls back to
    uniform sampling if the requested size exceeds what round-robin can reach.
    """
    pool = list(image_ids)
    if num >= len(pool):
        return rng.sample(pool, len(pool))

    if strategy == "random":
        return rng.sample(pool, num)

    if strategy == "stratified":
        cat_to_imgs: Dict[int, List[int]] = defaultdict(list)
        for img_id in pool:
            for cat_id in cats_by_image[img_id]:
                cat_to_imgs[cat_id].append(img_id)
        for img_ids in cat_to_imgs.values():
            rng.shuffle(img_ids)

        # Shuffle categories but visit rare ones first: they are the easiest to miss.
        cat_order = sorted(cat_to_imgs, key=lambda c: len(cat_to_imgs[c]))
        rng.shuffle(cat_order)
        cat_order.sort(key=lambda c: len(cat_to_imgs[c]))

        cursors = {c: 0 for c in cat_order}
        selected: List[int] = []
        seen: set[int] = set()
        progress = True
        while len(selected) < num and progress:
            progress = False
            for cat in cat_order:
                if len(selected) >= num:
                    break
                ptr = cursors[cat]
                imgs = cat_to_imgs[cat]
                while ptr < len(imgs):
                    img_id = imgs[ptr]
                    ptr += 1
                    if img_id not in seen:
                        seen.add(img_id)
                        selected.append(img_id)
                        progress = True
                        break
                cursors[cat] = ptr

        if len(selected) < num:  # extremely small split: pad uniformly
            leftover = [i for i in pool if i not in seen]
            selected.extend(rng.sample(leftover, num - len(selected)))
        return selected

    raise ValueError(f"unknown strategy '{strategy}'")


# --------------------------------------------------------------------------- #
# Subset building
# --------------------------------------------------------------------------- #
def build_subset(
    coco: Dict[str, Any],
    split: str,
    keep_ids: Sequence[int],
    split_index: int,
    id_policy: str,
    prune_categories: bool,
) -> Tuple[Dict[str, Any], Dict[str, Dict[int, int]]]:
    """Build the mini annotation dict for one split plus the id mapping."""
    keep = set(keep_ids)
    anns_by_image, _ = index_annotations(coco)

    images = [img for img in coco.get("images", []) if img["id"] in keep]
    images.sort(key=lambda i: i["id"])

    image_map: Dict[int, int] = {}
    if id_policy == "remap":
        offset = (split_index + 1) * IMAGE_ID_OFFSET
        image_map = {img["id"]: offset + img["id"] for img in images}
    else:
        image_map = {img["id"]: img["id"] for img in images}

    annotations: List[Dict[str, Any]] = []
    ann_map: Dict[int, int] = {}
    if id_policy == "remap":
        ann_offset = (split_index + 1) * ANN_ID_OFFSET
        next_ann_id = ann_offset + 1
        for img_id in sorted(keep):
            for ann in anns_by_image.get(img_id, []):
                new_id = next_ann_id
                next_ann_id += 1
                ann_map[ann["id"]] = new_id
                new_ann = dict(ann)
                new_ann["id"] = new_id
                new_ann["image_id"] = image_map[img_id]
                annotations.append(new_ann)
    else:
        for img_id in sorted(keep):
            for ann in anns_by_image.get(img_id, []):
                ann_map[ann["id"]] = ann["id"]
                annotations.append(dict(ann))

    new_images = []
    for img in images:
        new_img = dict(img)
        new_img["id"] = image_map[img["id"]]
        new_images.append(new_img)

    used_cats = {ann["category_id"] for ann in annotations}
    categories = [
        cat for cat in coco.get("categories", []) if (not prune_categories) or cat["id"] in used_cats
    ]

    subset: Dict[str, Any] = {
        # NOTE: no timestamp here on purpose -- it keeps the produced JSON byte-identical
        # across runs with the same seed (reproducibility check via md5).
        "info": {
            "description": f"coco_mini subset of {split}",
            "version": coco.get("info", {}).get("version", "1.0"),
            "year": coco.get("info", {}).get("year"),
            "contributor": "make_coco_mini.py",
            "source_split": split,
            "id_policy": id_policy,
        },
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": annotations,
        "categories": categories,
    }
    return subset, {"images": image_map, "annotations": ann_map}


def materialize_images(
    coco_root: str, split: str, images: Sequence[Dict[str, Any]], out_dir: str, link_mode: str
) -> Tuple[int, List[str]]:
    """Create ``out_dir/<split>/<file_name>`` as symlink or copy. Returns (n, missing)."""
    src_dir = os.path.join(coco_root, split)
    dst_dir = os.path.join(out_dir, split)
    os.makedirs(dst_dir, exist_ok=True)
    missing: List[str] = []
    count = 0
    for img in images:
        # file_name in COCO 2017 is just "<id>.jpg"; older dumps may contain "train2017/xxx.jpg".
        rel = os.path.basename(img["file_name"])
        src = os.path.join(src_dir, rel)
        dst = os.path.join(dst_dir, rel)
        if not os.path.exists(src):
            missing.append(src)
            continue
        if os.path.lexists(dst):
            os.remove(dst)
        if link_mode == "symlink":
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
        count += 1
    return count, missing


# --------------------------------------------------------------------------- #
# Statistics & verification
# --------------------------------------------------------------------------- #
def split_stats(subset: Dict[str, Any]) -> Dict[str, Any]:
    cat_name = {c["id"]: c.get("name", str(c["id"])) for c in subset["categories"]}
    per_cat = Counter(ann["category_id"] for ann in subset["annotations"])
    areas = [ann.get("area", 0.0) for ann in subset["annotations"]]
    n_img = len(subset["images"])
    n_ann = len(subset["annotations"])
    return {
        "num_images": n_img,
        "num_annotations": n_ann,
        "num_categories_total": len(subset["categories"]),
        "num_categories_present": len(per_cat),
        "annotations_per_image": round(n_ann / n_img, 2) if n_img else 0.0,
        "num_crowd": sum(1 for a in subset["annotations"] if a.get("iscrowd")),
        "area_mean": round(sum(areas) / len(areas), 1) if areas else 0.0,
        "area_min": round(min(areas), 1) if areas else 0.0,
        "area_max": round(max(areas), 1) if areas else 0.0,
        "category_distribution": {
            cat_name.get(cid, str(cid)): cnt for cid, cnt in sorted(per_cat.items(), key=lambda kv: -kv[1])
        },
    }


def verify_subset(subset: Dict[str, Any], img_dir: str) -> List[str]:
    """Structural self-check; returns a list of human readable errors (empty == OK)."""
    errors: List[str] = []
    img_ids = [i["id"] for i in subset["images"]]
    if len(img_ids) != len(set(img_ids)):
        errors.append("duplicate image ids")
    img_id_set = set(img_ids)

    ann_ids = [a["id"] for a in subset["annotations"]]
    if len(ann_ids) != len(set(ann_ids)):
        errors.append("duplicate annotation ids")

    cat_ids = {c["id"] for c in subset["categories"]}
    sizes = {i["id"]: (i.get("width"), i.get("height")) for i in subset["images"]}
    files = {os.path.basename(i["file_name"]) for i in subset["images"]}

    for ann in subset["annotations"]:
        if ann["image_id"] not in img_id_set:
            errors.append(f"annotation {ann['id']} references unknown image {ann['image_id']}")
            continue
        if ann["category_id"] not in cat_ids:
            errors.append(f"annotation {ann['id']} references unknown category {ann['category_id']}")
        bbox = ann.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            errors.append(f"annotation {ann['id']} has invalid bbox")
        else:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                errors.append(f"annotation {ann['id']} has non-positive bbox size")
            iw, ih = sizes[ann["image_id"]]
            if iw and ih and (x < -1 or y < -1 or x + w > iw + 1 or y + h > ih + 1):
                errors.append(f"annotation {ann['id']} bbox outside image bounds")
        if ann.get("area", 0) <= 0:
            errors.append(f"annotation {ann['id']} has non-positive area")
        seg = ann.get("segmentation")
        if ann.get("iscrowd") == 0 and isinstance(seg, list):
            for poly in seg:
                if len(poly) < 6 or len(poly) % 2 != 0:
                    errors.append(f"annotation {ann['id']} has malformed polygon")
                    break

    for img in subset["images"]:
        rel = os.path.basename(img["file_name"])
        if rel not in files:
            errors.append(f"image {img['id']} file_name mismatch")
        if not os.path.exists(os.path.join(img_dir, rel)):
            errors.append(f"missing image file {os.path.join(img_dir, rel)}")

    return errors


def verify_with_pycocotools(ann_file: str) -> Optional[str]:
    try:
        from pycocotools.coco import COCO  # type: ignore
    except Exception:
        return None
    try:
        coco = COCO(ann_file)
        n_ann = sum(len(v) for v in coco.imgToAnns.values())
        n_img = len(coco.imgs)
        return f"pycocotools OK: {n_img} images / {n_ann} annotations indexed"
    except Exception as exc:  # pragma: no cover - depends on optional dep
        return f"pycocotools FAILED: {exc}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco-root", default="data/coco", help="source COCO root (images + annotations/)")
    p.add_argument("--output-dir", default="data/coco_mini", help="destination root of the mini dataset")
    p.add_argument(
        "--splits",
        default="train2017,val2017",
        help="comma separated splits to sample (default: train2017,val2017)",
    )
    p.add_argument(
        "--num-images",
        default="10",
        help="images per split: an int (all splits) or 'train2017=100,val2017=20'",
    )
    p.add_argument(
        "--annotation-types",
        default="instances",
        help="comma separated annotation types: instances, captions, person_keypoints (default: instances)",
    )
    p.add_argument(
        "--strategy",
        choices=("random", "stratified"),
        default="random",
        help="random: uniform sample; stratified: round-robin over categories (default: random)",
    )
    p.add_argument("--seed", type=int, default=42, help="random seed, guarantees reproducibility")
    p.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="how images are placed in the output dir (default: symlink)",
    )
    p.add_argument(
        "--id-policy",
        choices=("auto", "keep", "remap"),
        default="auto",
        help="auto: keep ids unless a cross-split collision is detected (default)",
    )
    p.add_argument(
        "--prune-categories",
        action="store_true",
        help="drop categories absent from the sample instead of keeping the full 80-class list",
    )
    p.add_argument(
        "--min-annotations",
        type=int,
        default=1,
        help="only sample images with at least this many annotations (default: 1)",
    )
    p.add_argument("--dry-run", action="store_true", help="print the plan without writing anything")
    args = p.parse_args(argv)

    coco_root = os.path.abspath(args.coco_root)
    out_dir = os.path.abspath(args.output_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    ann_types = [t.strip() for t in args.annotation_types.split(",") if t.strip()]
    num_map = parse_num_images(args.num_images, splits)

    if not os.path.isdir(os.path.join(coco_root, "annotations")):
        log(f"ERROR: {coco_root}/annotations not found")
        return 1

    ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    global_stats: Dict[str, Any] = {
        "source": coco_root,
        "output": out_dir,
        "seed": args.seed,
        "strategy": args.strategy,
        "id_policy_requested": args.id_policy,
        "link_mode": args.link_mode,
        "num_images_request": num_map,
        "splits": {},
        "cross_split_checks": {},
        "verification": {},
    }
    all_file_names: Dict[str, str] = {}  # basename -> split
    all_image_ids: Dict[int, str] = {}  # (original) image id -> split
    id_conflicts: List[str] = []
    file_conflicts: List[str] = []
    ok = True

    for split_index, split in enumerate(splits):
        num = num_map[split]
        # Per-split RNG stream: same seed never yields the same sequence for two splits.
        rng = random.Random(args.seed + split_index)
        log(f"[{split}] loading annotations ...")

        for ann_type in ann_types:
            src_json = os.path.join(coco_root, "annotations", f"{ann_type}_{split}.json")
            if not os.path.exists(src_json):
                log(f"[{split}] WARNING: {src_json} not found, skipped")
                continue
            coco = load_json(src_json)

            img_dir = os.path.join(coco_root, split)
            candidates = [
                img["id"]
                for img in coco.get("images", [])
                if os.path.exists(os.path.join(img_dir, os.path.basename(img["file_name"])))
            ]
            _, cats_by_image = index_annotations(coco)
            if args.min_annotations > 0:
                candidates = [i for i in candidates if sum(cats_by_image[i].values()) >= args.min_annotations]
            if not candidates:
                log(f"[{split}] WARNING: no candidate images, skipped {ann_type}")
                continue

            n = min(num, len(candidates))
            if n < num:
                log(f"[{split}] WARNING: requested {num} but only {len(candidates)} usable images")
            picked = sample_image_ids(candidates, cats_by_image, n, args.strategy, rng)

            # ---- cross-split leakage / id collision bookkeeping -------------
            for src_img in coco["images"]:
                if src_img["id"] in set(picked):
                    rel = os.path.basename(src_img["file_name"])
                    if rel in all_file_names and all_file_names[rel] != split:
                        file_conflicts.append(f"{rel}: {all_file_names[rel]} <-> {split}")
                    all_file_names[rel] = split
            for img_id in picked:
                if img_id in all_image_ids and all_image_ids[img_id] != split:
                    id_conflicts.append(f"{img_id}: {all_image_ids[img_id]} <-> {split}")
                all_image_ids[img_id] = split

            policy = args.id_policy
            if policy == "auto":
                policy = "keep"
            subset, mapping = build_subset(coco, split, picked, split_index, policy, args.prune_categories)

            if args.dry_run:
                log(f"[{split}] dry-run: would write {len(subset['images'])} images "
                    f"/ {len(subset['annotations'])} annotations ({ann_type})")
                continue

            out_json = os.path.join(ann_dir, f"{ann_type}_{split}.json")
            dump_json(out_json, subset)

            n_linked, missing = materialize_images(coco_root, split, subset["images"], out_dir, args.link_mode)
            if missing:
                log(f"[{split}] WARNING: {len(missing)} source images missing, e.g. {missing[0]}")

            if policy == "remap":
                dump_json(os.path.join(ann_dir, f"id_mapping_{split}.json"), mapping)

            stats = split_stats(subset)
            stats["requested"] = num
            stats["images_linked"] = n_linked
            stats["id_policy"] = policy
            global_stats["splits"][f"{split}/{ann_type}"] = stats

            errs = verify_subset(subset, os.path.join(out_dir, split))
            coco_check = verify_with_pycocotools(out_json)
            global_stats["verification"][f"{split}/{ann_type}"] = {
                "errors": errs[:20],
                "num_errors": len(errs),
                "pycocotools": coco_check,
            }
            if errs:
                ok = False
                log(f"[{split}] {ann_type}: {len(errs)} verification error(s)")
            else:
                log(f"[{split}] {ann_type}: OK ({stats['num_images']} images, "
                    f"{stats['num_annotations']} annotations, {stats['num_categories_present']} categories)")

    global_stats["cross_split_checks"] = {
        "image_id_conflicts": id_conflicts[:20],
        "num_image_id_conflicts": len(id_conflicts),
        "file_name_conflicts": file_conflicts[:20],
        "num_file_name_conflicts": len(file_conflicts),
        "note": "empty conflict lists => no leakage between sampled splits",
    }
    if id_conflicts:
        log(f"WARNING: {len(id_conflicts)} image-id collision(s) across splits; "
            f"re-run with --id-policy remap to namespace them")
    if file_conflicts:
        log(f"WARNING: {len(file_conflicts)} duplicated file name(s) across splits (data leakage risk)")

    if not args.dry_run:
        dump_json(os.path.join(ann_dir, "coco_mini_stats.json"), global_stats)
        log("\n=== summary ===")
        for key, st in global_stats["splits"].items():
            log(f"{key}: images={st['num_images']} anns={st['num_annotations']} "
                f"cats={st['num_categories_present']} avg_ann/img={st['annotations_per_image']}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
