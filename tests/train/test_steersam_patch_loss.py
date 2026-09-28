import math
import unittest

import torch
import torch.nn as nn

from sam3.model.model_misc import SAM3Output
from sam3.model.steersam_patch_head import SteerSAMPatchHead
from sam3.train.loss.loss_fns import CORE_LOSS_KEY
from sam3.train.loss.steersam_patch_loss import (
    build_patch_distribution_targets,
    SteerSAMPatchLocalizationLoss,
    SteerSAMPatchLossWrapper,
)


def _targets():
    # Pair 0 has two instances occupying the two top quadrants. Pair 1 has one
    # instance in the bottom-right quadrant.
    masks = torch.zeros(3, 4, 4, dtype=torch.bool)
    masks[0, :2, :2] = True
    masks[1, :2, 2:] = True
    masks[2, 2:, 2:] = True
    return {
        "masks": masks,
        "num_boxes": torch.tensor([2, 1]),
        "is_valid_mask": torch.tensor([True, True, True]),
    }


class TestPatchTargets(unittest.TestCase):
    def test_instances_are_unioned_per_positive_pair(self):
        distribution, occupancy, valid = build_patch_distribution_targets(
            _targets()["masks"],
            _targets()["num_boxes"],
            output_size=(2, 2),
            valid_instance_masks=_targets()["is_valid_mask"],
        )
        torch.testing.assert_close(
            occupancy,
            torch.tensor(
                [
                    [[1.0, 1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 1.0]],
                ]
            ),
        )
        torch.testing.assert_close(distribution.flatten(1).sum(1), torch.ones(2))
        torch.testing.assert_close(valid, torch.ones(2, dtype=torch.bool))

    def test_invalid_instances_are_ignored(self):
        targets = _targets()
        targets["is_valid_mask"][-1] = False
        distribution, occupancy, valid = build_patch_distribution_targets(
            targets["masks"],
            targets["num_boxes"],
            output_size=(2, 2),
            valid_instance_masks=targets["is_valid_mask"],
        )
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertEqual(occupancy[1].sum().item(), 0.0)
        self.assertEqual(distribution[1].sum().item(), 0.0)


class TestPatchHeadAndLoss(unittest.TestCase):
    def test_zero_initialized_head_shape_and_value(self):
        head = SteerSAMPatchHead(input_dim=8, zero_init=True)
        logits = head(torch.randn(2, 8, 3, 4))
        self.assertEqual(logits.shape, (2, 3, 4))
        torch.testing.assert_close(logits, torch.zeros_like(logits))

    def test_soft_ce_pmass_and_gradient(self):
        logits = torch.zeros(2, 2, 2, requires_grad=True)
        result = SteerSAMPatchLocalizationLoss()(logits, _targets())
        self.assertAlmostEqual(result["loss_steering_patch"].item(), math.log(4), 6)
        # Uniform mass: pair 0 has 2/4 foreground patches, pair 1 has 1/4.
        self.assertAlmostEqual(result["steering_pmass"].item(), 0.375, 6)
        result["loss_steering_patch"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_wrapper_composes_task_and_patch_losses(self):
        class ConstantTaskLoss(nn.Module):
            def forward(self, *_args, **_kwargs):
                return {
                    CORE_LOSS_KEY: torch.tensor(2.0),
                    "task_metric": torch.tensor(3.0),
                }

        logits = torch.zeros(2, 2, 2, requires_grad=True)
        outputs = SAM3Output(
            output=[[{"steering_patch_logits": logits}]],
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE,
        )
        wrapper = SteerSAMPatchLossWrapper(
            task_loss=ConstantTaskLoss(), patch_loss_weight=0.1
        )
        result = wrapper(outputs, [_targets()])
        self.assertAlmostEqual(result[CORE_LOSS_KEY].item(), 2.0 + 0.1 * math.log(4), 6)
        self.assertEqual(result["task_metric"].item(), 3.0)

    def test_matcher_free_validation_wrapper(self):
        logits = torch.zeros(2, 2, 2)
        outputs = SAM3Output(
            output=[[{"steering_patch_logits": logits}]],
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE,
        )
        result = SteerSAMPatchLossWrapper(task_loss=None)(outputs, [_targets()])
        self.assertIn(CORE_LOSS_KEY, result)
        self.assertIn("steering_pmass", result)


if __name__ == "__main__":
    unittest.main()
