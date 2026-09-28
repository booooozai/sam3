#!/usr/bin/env python3
"""One-command local runner for the official SAM 3 COCO-O (APo) evaluation.

Sequentially launches the six per-domain COCO-O eval YAMLs under
``sam3/train/configs/coco_o_evals/`` and then aggregates the per-domain
bbox/segm AP into the COCO-O protocol number (mean over the six domains).
Image ids restart at 1 in every domain's annotation file, so domains are
evaluated separately and averaged -- the official COCO-O protocol.

Paper anchor (SAM 3 paper, Tab. 1): box APo = 55.7 on COCO-O.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# Fixed order; also the sub-folder layout under --output-root.
DOMAINS = (
    "cartoon",
    "handmake",
    "painting",
    "sketch",
    "tattoo",
    "weather",
)

# SAM 3 paper Tab. 1, Box Detection / COCO APo row for SAM 3.
PAPER_BOX_APO = 55.7


def read_last_stats(path: Path) -> dict:
    """Trainer appends one JSON object per validation run (JSONL)."""
    last = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    if not isinstance(last, dict):
        raise ValueError(f"No validation object in {path}")
    return last


def aggregate(output_root: Path) -> int:
    rows = []
    for dom in DOMAINS:
        stats_path = output_root / dom / "logs" / "val_stats.json"
        if not stats_path.is_file():
            print(f"[aggregate] MISSING {stats_path}", flush=True)
            continue
        stats = read_last_stats(stats_path)
        bbox_ap = segm_ap = None
        for key, value in stats.items():
            if key.endswith("/detection/coco_eval_bbox_AP"):
                bbox_ap = value
            elif key.endswith("/segmentation/coco_eval_segm_AP"):
                segm_ap = value
        rows.append((dom, bbox_ap, segm_ap))

    if not rows:
        print("[aggregate] no val_stats.json found; nothing to aggregate")
        return 1

    print("\n===== COCO-O per-domain AP (official SAM 3, r1008) =====")
    print(f"{'domain':<10} {'bbox AP':>8} {'segm AP':>8}")
    bbox_vals, segm_vals = [], []
    for dom, bbox_ap, segm_ap in rows:
        b = f"{100 * bbox_ap:.2f}" if bbox_ap is not None else "n/a"
        s = f"{100 * segm_ap:.2f}" if segm_ap is not None else "n/a"
        if bbox_ap is not None:
            bbox_vals.append(bbox_ap)
        if segm_ap is not None:
            segm_vals.append(segm_ap)
        print(f"{dom:<10} {b:>8} {s:>8}")

    if bbox_vals:
        mean_bbox = 100 * sum(bbox_vals) / len(bbox_vals)
        if segm_vals:
            mean_segm = 100 * sum(segm_vals) / len(segm_vals)
            print(f"{'MEAN':<10} {mean_bbox:>8.2f} {mean_segm:>8.2f}")
        else:
            # Expected: COCO-O is box-only (every GT "segmentation" field is
            # empty), so no segm AP exists -- matching paper Tab. 1, which
            # lists COCO-O only under Box Detection.
            print(f"{'MEAN':<10} {mean_bbox:>8.2f} {'n/a':>8}  (box-only benchmark)")
        print(
            f"\nBox APo (6-domain mean): {mean_bbox:.2f}  "
            f"(paper Tab.1 APo: {PAPER_BOX_APO}, diff {mean_bbox - PAPER_BOX_APO:+.2f})"
        )
        if len(bbox_vals) != len(DOMAINS):
            print("WARNING: some domains are missing bbox metrics; mean is partial.")
            return 1
    return 0 if len(bbox_vals) == len(DOMAINS) else 1


def main() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Run all six official SAM 3 COCO-O eval configs and aggregate APo."
    )
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--use-cluster", type=int, choices=(0,), default=0,
                        help="Local synchronous execution only; submit cluster jobs separately")
    parser.add_argument(
        "--gpus",
        type=str,
        default="2,3",
        help="comma-separated CUDA_VISIBLE_DEVICES for the local run",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            Path("/share/Pub_Datasets/chenshengbo/outputs")
            / "eval_official_sam3_coco_o_r1008"
        ),
    )
    parser.add_argument(
        "--smoke-images",
        type=int,
        default=None,
        help="if set, exports COCO_EVAL_NUM_IMAGES for a fast smoke run",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true", help="skip inference, only aggregate"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_gpus < 1 or (args.smoke_images is not None and args.smoke_images < 1):
        parser.error("--num-gpus and --smoke-images must be positive")
    args.output_root = args.output_root.resolve()

    if not args.aggregate_only:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        env["COCO_O_OUTPUT_ROOT"] = str(args.output_root)
        if args.smoke_images is not None:
            env["COCO_EVAL_NUM_IMAGES"] = str(args.smoke_images)
        print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)

        commands = [
            [
                sys.executable,
                str(repository_root / "sam3" / "train" / "train.py"),
                "-c",
                f"configs/coco_o_evals/sam3_coco_o_{dom}.yaml",
                "--use-cluster",
                str(args.use_cluster),
                "--num-gpus",
                str(args.num_gpus),
            ]
            for dom in DOMAINS
        ]
        for index, command in enumerate(commands, start=1):
            print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(
                    command, cwd=repository_root, env=env, check=True
                )

    if args.dry_run:
        print(f"Would aggregate results under {args.output_root}", flush=True)
        return
    if args.smoke_images is not None:
        print("SMOKE ONLY: subset metrics are not full COCO-O reproduction results.")
    sys.exit(aggregate(args.output_root))


if __name__ == "__main__":
    main()
