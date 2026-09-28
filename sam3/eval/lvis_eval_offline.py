# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Official LVIS v1 evaluation from a merged prediction JSON file."""

import logging

import numpy as np
from lvis import LVIS, LVISEval, LVISResults

from sam3.train.utils.distributed import is_main_process


class LVISEvaluatorOfflineWithPredFile:
    """Run the official LVIS API for bbox or instance-mask predictions."""

    def __init__(self, gt_path: str, iou_type: str = "segm", max_dets: int = 300):
        if iou_type not in {"bbox", "segm"}:
            raise ValueError("LVIS iou_type must be 'bbox' or 'segm'.")
        if max_dets <= 0:
            raise ValueError("LVIS max_dets must be positive.")
        self.gt_path = gt_path
        self.iou_type = iou_type
        self.max_dets = int(max_dets)

    def evaluate(self, dumped_file):
        if not is_main_process():
            return {}

        logging.info("LVIS evaluator: loading ground truth and predictions")
        # lvis-api 0.5.3 still references the NumPy alias removed in 1.24.
        # Keep the compatibility shim local to evaluation rather than patching
        # the environment's site-packages.
        if "float" not in np.__dict__:
            np.float = float
        lvis_gt = LVIS(self.gt_path)
        lvis_dt = LVISResults(lvis_gt, str(dumped_file), max_dets=self.max_dets)
        evaluator = LVISEval(lvis_gt, lvis_dt, iou_type=self.iou_type)
        evaluator.params.max_dets = self.max_dets
        evaluator.run()
        evaluator.print_results()

        return {
            f"lvis_eval_{self.iou_type}_{name}": float(value)
            for name, value in evaluator.get_results().items()
        }
