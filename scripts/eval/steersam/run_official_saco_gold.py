#!/usr/bin/env python3
"""One-command local runner for the unmodified official SA-Co/Gold configs.

This project-local wrapper keeps upstream ``scripts/eval/gold`` untouched.
It sequentially launches the seven official Gold inference YAMLs and then
calls the upstream ``scripts/eval/gold/eval_sam3.py`` aggregator.  It adds no
dataset conversion or metric logic.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Same seven-subset order as upstream scripts/eval/gold/eval_sam3.py.
GOLD_CONFIGS = (
    "sam3_gold_image_metaclip_nps",
    "sam3_gold_image_sa1b_nps",
    "sam3_gold_image_crowded",
    "sam3_gold_image_fg_food",
    "sam3_gold_image_fg_sports",
    "sam3_gold_image_attributes",
    "sam3_gold_image_wiki_common",
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Run all official SA-Co/Gold configs and aggregate official cgF1."
    )
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--use-cluster", type=int, choices=(0,), default=0,
                        help="Local synchronous execution only; submit cluster jobs separately")
    parser.add_argument(
        "--gt-folder",
        type=Path,
        default=repository_root / "data" / "sa_co_gold" / "gt-annotations",
    )
    parser.add_argument(
        "--pred-folder",
        type=Path,
        default=(
            Path("/share/Pub_Datasets/chenshengbo/outputs")
            / "eval_official_sam3_saco_gold_r1008"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_gpus < 1:
        parser.error("--num-gpus must be positive")
    args.gt_folder = args.gt_folder.resolve()
    args.pred_folder = args.pred_folder.resolve()

    required_gt = (
        "gold_metaclip_merged_a_release_test.json",
        "gold_sa1b_merged_a_release_test.json",
        "gold_crowded_merged_a_release_test.json",
        "gold_fg_food_merged_a_release_test.json",
        "gold_fg_sports_equipment_merged_a_release_test.json",
        "gold_attributes_merged_a_release_test.json",
        "gold_wiki_common_merged_a_release_test.json",
    )
    required_gt = tuple(name.replace("_a_release_", f"_{annotator}_release_")
                        for name in required_gt for annotator in ("a", "b", "c"))
    missing_gt = [args.gt_folder / name for name in required_gt if not (args.gt_folder / name).is_file()]
    if missing_gt:
        raise FileNotFoundError("Missing SA-Co/Gold GT: " + ", ".join(map(str, missing_gt)))

    commands = [
        [
            sys.executable,
            str(repository_root / "sam3" / "train" / "train.py"),
            "-c",
            f"configs/gold_image_evals/{name}.yaml",
            "--use-cluster",
            str(args.use_cluster),
            "--num-gpus",
            str(args.num_gpus),
        ]
        for name in GOLD_CONFIGS
    ]
    commands.append(
        [
            sys.executable,
            str(repository_root / "scripts" / "eval" / "gold" / "eval_sam3.py"),
            "--gt-folder",
            str(args.gt_folder),
            "--pred-folder",
            str(args.pred_folder),
        ]
    )

    env = os.environ.copy()
    env["SACO_GOLD_GT_ROOT"] = str(args.gt_folder)
    env["SACO_GOLD_OUTPUT_ROOT"] = str(args.pred_folder)
    print(f"GT: {args.gt_folder}; outputs: {args.pred_folder}", flush=True)
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=repository_root, env=env, check=True)


if __name__ == "__main__":
    main()
