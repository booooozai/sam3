# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Optional COCO prompt-pair diagnostics, not official benchmark metrics.

Standard COCO metrics use the original image/category ids.  CG-F1-style
metrics instead define one evaluation item per ``(image, prompt)`` pair so
negative prompts contribute true/false negatives/positives at image level.
This module builds that pair-level ground truth and adapts the existing
``CGF1Evaluator`` to the offline ``PredictionDumper`` interface.
No default formal COCO configuration enables these diagnostics. LVIS virtual
pair evaluation has been retired; use the dataset's original protocol instead.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

from sam3.eval.cgf1_eval import CGF1Evaluator


_BUILT_PAIR_GT_PATHS = set()


def build_prompt_pair_ground_truth(gt_path: str, output_path: str) -> str:
    """Expand COCO ground truth into exhaustive image-category pair records.

    Pair ids are one-based row-major ids over images sorted by id and
    categories sorted by id.  This is exactly the mapping used by
    ``COCOAllCategoryPairsFromJSON``.  All annotations use a dummy category id
    because ``CGF1Eval`` deliberately evaluates prompt pairs category-agnostic.
    The source category is retained as ``original_category_id`` for auditing.
    """
    with open(gt_path, "r") as f:
        source = json.load(f)

    images = sorted(source["images"], key=lambda image: image["id"])
    categories = sorted(source["categories"], key=lambda category: category["id"])
    category_ids = [category["id"] for category in categories]

    annotations_by_pair: Dict[tuple[int, int], list] = defaultdict(list)
    for annotation in source.get("annotations", []):
        annotations_by_pair[(annotation["image_id"], annotation["category_id"])].append(
            annotation
        )

    pair_images = []
    pair_annotations = []
    next_annotation_id = 1
    num_categories = len(category_ids)
    for image_index, image in enumerate(images):
        for category_index, category_id in enumerate(category_ids):
            pair_id = image_index * num_categories + category_index + 1
            pair_image = dict(image)
            pair_image.update(
                {
                    "id": pair_id,
                    "original_image_id": image["id"],
                    "original_category_id": category_id,
                    "is_instance_exhaustive": True,
                }
            )
            pair_images.append(pair_image)

            for annotation in annotations_by_pair[(image["id"], category_id)]:
                pair_annotation = dict(annotation)
                pair_annotation.update(
                    {
                        "id": next_annotation_id,
                        "image_id": pair_id,
                        "category_id": 1,
                        "original_category_id": category_id,
                    }
                )
                pair_annotations.append(pair_annotation)
                next_annotation_id += 1

    pair_ground_truth = {
        key: value
        for key, value in source.items()
        if key not in {"images", "annotations", "categories"}
    }
    pair_ground_truth.update(
        {
            "images": pair_images,
            "annotations": pair_annotations,
            "categories": [
                {
                    "id": 1,
                    "name": "prompt_target",
                    "supercategory": "prompt_target",
                }
            ],
            "prompt_pair_metadata": {
                "source_gt_path": os.path.abspath(gt_path),
                "num_source_images": len(images),
                "num_source_categories": num_categories,
                "pair_id_order": "sorted_image_id_x_sorted_category_id",
            },
        }
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w") as f:
        json.dump(pair_ground_truth, f)
    os.replace(temporary_output, output)
    return str(output)




class PromptPairCGF1Evaluator:
    """Run cgF1/IL-MCC/positive-F1 on exhaustive prompt-pair predictions."""

    def __init__(
        self,
        gt_path: str,
        pair_gt_path: str,
        iou_type: str = "segm",
        score_threshold: float = 0.5,
        verbose: bool = False,
    ):
        self.gt_path = gt_path
        self.pair_gt_path = pair_gt_path
        self.iou_type = iou_type
        self.score_threshold = float(score_threshold)
        self.verbose = verbose

    def _ensure_pair_ground_truth(self) -> None:
        absolute_output = os.path.abspath(self.pair_gt_path)
        if absolute_output in _BUILT_PAIR_GT_PATHS:
            return
        build_prompt_pair_ground_truth(self.gt_path, absolute_output)
        _BUILT_PAIR_GT_PATHS.add(absolute_output)

    def evaluate(self, dumped_file):
        self._ensure_pair_ground_truth()
        evaluator = CGF1Evaluator(
            gt_path=self.pair_gt_path,
            iou_type=self.iou_type,
            verbose=self.verbose,
        )
        for coco_eval in evaluator.coco_evals:
            coco_eval.threshold = self.score_threshold
        results = evaluator.evaluate(str(dumped_file))

        # Concise aliases for experiment tables.  Here pmF1 means the existing
        # positive_macro_F1 metric; retain the original keys as well.
        source_prefix = f"cgF1_eval_{self.iou_type}_"
        alias_prefix = f"prompt_pair_{self.iou_type}_"
        aliases = {
            "cgF1_50_95": "cgF1",
            "IL_MCC": "IL_MCC",
            "pmF1_50_95": "positive_macro_F1",
            "positive_micro_F1_50_95": "positive_micro_F1",
            "cgF1_50": "cgF1@0.5",
            "pmF1_50": "positive_macro_F1@0.5",
            "positive_micro_F1_50": "positive_micro_F1@0.5",
            "cgF1_75": "cgF1@0.75",
            "pmF1_75": "positive_macro_F1@0.75",
            "positive_micro_F1_75": "positive_micro_F1@0.75",
        }
        for alias, source_name in aliases.items():
            results[alias_prefix + alias] = results[source_prefix + source_name]
        return results
