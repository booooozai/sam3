"""CPU unit tests for trainable-only optimizer param groups (design 7.9).

Covers both the relaxed ``validate_param_group_params`` contract (frozen
extras allowed, trainable params must be covered) and ``construct_optimizer``
with the trainable-only allowlist the trainer passes for steering models.
"""

import unittest

import torch
import torch.nn as nn

from sam3.train.optim.optimizer import construct_optimizer, validate_param_group_params

OPTIMIZER_CONF = {"_target_": "torch.optim.AdamW", "lr": 1e-3}


def _toy_model():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(4, 4)
            self.backbone = nn.Linear(4, 4)
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    return Toy()


def _all_option(value=1e-3):
    return [
        {
            "scheduler": {
                "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                "value": value,
            }
        }
    ]


class TestValidateParamGroupParams(unittest.TestCase):
    def test_full_coverage_passes_with_frozen_params(self):
        model = _toy_model()
        params = list(model.parameters())
        validate_param_group_params([{"params": params}], model)

    def test_trainable_only_coverage_passes(self):
        model = _toy_model()
        validate_param_group_params(
            [{"params": [model.head.weight, model.head.bias]}], model
        )

    def test_missing_trainable_param_fails(self):
        model = _toy_model()
        with self.assertRaises(AssertionError):
            validate_param_group_params([{"params": [model.head.weight]}], model)

    def test_overlapping_groups_fail(self):
        model = _toy_model()
        params = list(model.parameters())
        with self.assertRaises(AssertionError):
            validate_param_group_params(
                [{"params": params}, {"params": [model.head.weight]}], model
            )

    def test_frozen_only_groups_fail(self):
        model = _toy_model()
        with self.assertRaises(AssertionError):
            validate_param_group_params([{"params": [model.backbone.weight]}], model)


class TestConstructOptimizerTrainableAllowlist(unittest.TestCase):
    def test_allowlist_excludes_frozen_params_from_groups(self):
        model = _toy_model()
        allowlist = {name for name, p in model.named_parameters() if p.requires_grad}
        opt = construct_optimizer(
            model,
            OPTIMIZER_CONF,
            {"lr": _all_option()},
            param_allowlist=allowlist,
        )
        grouped = {p for group in opt.optimizer.param_groups for p in group["params"]}
        self.assertEqual(grouped, {model.head.weight, model.head.bias})

        # No AdamW state is created for frozen params: only the head has grads.
        model.head.weight.grad = torch.zeros_like(model.head.weight)
        opt.step(0.0, 0)
        self.assertEqual(len(opt.optimizer.state), 1)

    def test_adapter_pattern_splits_lr_groups(self):
        class Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = nn.Parameter(torch.zeros(()))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(4, 4)
                self.trunk = nn.ModuleDict(
                    {"steering_adapters": nn.ModuleDict({"1": Adapter()})}
                )

        model = Model()
        allowlist = {n for n, p in model.named_parameters() if p.requires_grad}
        adapter_lr = [
            {
                "scheduler": {
                    "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                    "value": 1e-2,
                },
                "param_names": ["*steering_adapters*"],
            }
        ]
        opt = construct_optimizer(
            model,
            OPTIMIZER_CONF,
            {"lr": _all_option() + adapter_lr},
            param_allowlist=allowlist,
        )
        # Two groups: default-LR head params and adapter-LR adapter params.
        self.assertEqual(len(opt.optimizer.param_groups), 2)
        lr_by_param = {
            p: group["lr"]
            for group in opt.optimizer.param_groups
            for p in group["params"]
        }
        self.assertEqual(lr_by_param[model.trunk["steering_adapters"]["1"].alpha], 1e-2)
        self.assertEqual(lr_by_param[model.head.weight], 1e-3)

    def test_pattern_matching_only_frozen_params_fails_loudly(self):
        model = _toy_model()
        allowlist = {n for n, p in model.named_parameters() if p.requires_grad}
        with self.assertRaises(AssertionError):
            construct_optimizer(
                model,
                OPTIMIZER_CONF,
                {
                    "lr": _all_option()
                    + [
                        {
                            "scheduler": {
                                "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                                "value": 0.0,
                            },
                            "param_names": ["backbone.*"],
                        }
                    ]
                },
                param_allowlist=allowlist,
            )


if __name__ == "__main__":
    unittest.main()
