"""CPU integration test: build the SteerSAM image model end-to-end (no GPU, no checkpoint).

Builds the real model via ``build_sam3_image_model(enable_steering=True)`` at
resolution 336 (the ViT's native pretraining resolution), feeds it a positive-pair batch from
``COCOPositivePairFromJSON``, and checks the acceptance criteria from
steersam_design.md section 12 that are testable on CPU:

- the steered forward produces the expected output structure;
- gradients reach the steering adapters, patch head, and FPN while frozen
  backbone and post-FPN SAM3 task parameters get none;
- frozen backbone subtrees fall back to eval mode under ``model.train()``;
- the trainable-only optimizer param groups cover every trainable parameter
  exactly once and exclude the frozen backbones.

This test constructs the full-size model (~850M params of random weights) and
runs forward+backward on CPU; expect on the order of a minute.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image as PILImage

from sam3.model.model_misc import SAM3Output
from sam3.model.sam3_image import Sam3Image
from sam3.model.steering import GatedVisionLanguageAdapter
from sam3.model.steersam_image import SteerSAMImage
from sam3.train.data.coco_json_loaders import COCOPositivePairFromJSON
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
from sam3.train.optim.optimizer import construct_optimizer

BPE_PATH = (
    Path(__file__).resolve().parents[2] / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
)
RESOLUTION = 336  # native ViT pretrain resolution: 24x24 patch grid, no padding

OPTIMIZER_CONF = {"_target_": "torch.optim.AdamW", "lr": 1e-3, "weight_decay": 0.1}
OPTIM_OPTIONS = {
    "lr": [
        {
            "scheduler": {
                "_target_": "sam3.train.optim.schedulers.InverseSquareRootParamScheduler",
                "base_lr": 8e-6,
                "timescale": 20,
                "warmup_steps": 20,
                "cooldown_steps": 20,
            }
        },
        {
            "scheduler": {
                "_target_": "sam3.train.optim.schedulers.InverseSquareRootParamScheduler",
                "base_lr": 1e-4,
                "timescale": 20,
                "warmup_steps": 20,
                "cooldown_steps": 20,
            },
            "param_names": ["*steering_adapters*", "*steering_patch_head*"],
        },
    ],
    "weight_decay": [
        {
            "scheduler": {
                "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                "value": 0.1,
            }
        },
        {
            "scheduler": {
                "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                "value": 0.0,
            },
            "param_names": ["*bias*"],
            "module_cls_names": ["torch.nn.LayerNorm"],
        },
        {
            "scheduler": {
                "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                "value": 0.0,
            },
            "param_names": ["*steering_adapters*.alpha"],
        },
    ],
}


class _ToFixedTensor:
    def __call__(self, datapoint, epoch):
        del epoch
        for image in datapoint.images:
            image.data = torch.zeros(3, RESOLUTION, RESOLUTION)
        return datapoint


@unittest.skipUnless(BPE_PATH.exists(), "BPE vocabulary not found")
class TestSteerSAMBuildIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        root = Path(cls._tmpdir.name)
        data = {
            "images": [
                {"id": 10, "file_name": "image_10.jpg", "width": 100, "height": 200}
            ],
            "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
            "annotations": [
                {
                    "id": 101,
                    "image_id": 10,
                    "category_id": 1,
                    "bbox": [10, 20, 30, 40],
                    "iscrowd": 0,
                },
                {
                    "id": 102,
                    "image_id": 10,
                    "category_id": 2,
                    "bbox": [0, 0, 20, 20],
                    "iscrowd": 0,
                },
            ],
        }
        (root / "instances.json").write_text(json.dumps(data), encoding="utf-8")
        PILImage.new("RGB", (100, 200)).save(root / "image_10.jpg")

        from sam3.model_builder import build_sam3_image_model

        cls.model = build_sam3_image_model(
            bpe_path=str(BPE_PATH),
            device="cpu",
            eval_mode=False,
            checkpoint_path=None,
            load_from_HF=False,
            resolution=RESOLUTION,
            enable_steering=True,
            steering_layers=(7, 15, 23, 31),
            enable_patch_supervision=True,
            freeze_vision_backbone=True,
            freeze_vision_fpn=False,
            freeze_language_backbone=True,
            freeze_sam3_task_modules=True,
        )
        # TextTransformer leaves positional_embedding/text_projection as
        # torch.empty; real runs fill them from the SAM3 checkpoint. Without a
        # checkpoint they contain uninitialized garbage (often NaN), so give
        # them deterministic values for this test.
        with torch.no_grad():
            text_transformer = cls.model.backbone.language_backbone.encoder
            text_transformer.positional_embedding.normal_()
            text_transformer.text_projection.normal_()

        cls.dataset = Sam3ImageDataset(
            img_folder=str(root),
            ann_file=str(root / "instances.json"),
            transforms=[_ToFixedTensor()],
            max_ann_per_img=10,
            multiplier=1,
            training=True,
            coco_json_loader=COCOPositivePairFromJSON,
        )
        # Two positive pairs over the same source image: (img, cat), (img, dog).
        cls.batch = collate_fn_api(
            [cls.dataset[i] for i in range(len(cls.dataset))], dict_key="test"
        )["test"]

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _last_step_out(self, result):
        assert isinstance(result, SAM3Output)
        # With IterMode.LAST_STEP_PER_STAGE, indexing already resolves to the
        # last step dict of the stage.
        out = result[0]
        assert isinstance(out, dict)
        return out

    def test_batch_contract_holds(self):
        find_input = self.batch.find_inputs[0]
        self.assertEqual(len(self.batch.img_batch), 2)
        torch.testing.assert_close(find_input.img_ids, torch.arange(2))
        self.assertEqual(self.batch.find_text_batch, ["cat", "dog"])
        torch.testing.assert_close(find_input.text_ids, torch.tensor([0, 1]))

    def test_model_types_and_freeze_state(self):
        self.assertIsInstance(self.model, SteerSAMImage)
        self.assertTrue(self.model.steering_enabled)
        self.assertIsInstance(self.model, Sam3Image)
        # Exactly one adapter per requested block, all trainable.
        adapters = [
            m for m in self.model.modules() if isinstance(m, GatedVisionLanguageAdapter)
        ]
        self.assertEqual(len(adapters), 4)
        for adapter in adapters:
            self.assertTrue(all(p.requires_grad for p in adapter.parameters()))
        # Frozen language backbone and pretrained ViT trunk...
        self.assertFalse(
            any(
                p.requires_grad
                for p in self.model.backbone.language_backbone.parameters()
            )
        )
        trunk_params = {
            n: p.requires_grad
            for n, p in self.model.backbone.vision_backbone.trunk.named_parameters()
        }
        self.assertFalse(all(trunk_params.values()))
        self.assertTrue(
            all(
                grad_enabled
                for name, grad_enabled in trunk_params.items()
                if "steering_adapters" in name
            )
        )
        self.assertTrue(
            all(
                p.requires_grad
                for p in self.model.backbone.vision_backbone.convs.parameters()
            )
        )
        # Strategy 2 trains the FPN and patch head but freezes every pretrained
        # SAM3 module after the FPN.
        self.assertTrue(
            all(p.requires_grad for p in self.model.steering_patch_head.parameters())
        )
        frozen_task_modules = (
            self.model.geometry_encoder,
            self.model.transformer,
            self.model.dot_prod_scoring,
            self.model.segmentation_head,
        )
        for module in frozen_task_modules:
            self.assertFalse(any(p.requires_grad for p in module.parameters()))

        trainable_names = {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(
                name.startswith("backbone.vision_backbone.convs")
                or "steering_adapters" in name
                or name.startswith("steering_patch_head")
                for name in trainable_names
            )
        )
        self.assertEqual(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            24_588_549,
        )

    def test_train_mode_keeps_frozen_blocks_in_eval(self):
        self.model.train()
        # The frozen ViT blocks are in eval mode (no drop-path)...
        block = self.model.backbone.vision_backbone.trunk.blocks[0]
        self.assertFalse(block.training)
        # ...while trainable adapters/FPN stay in train mode and the fixed
        # downstream task stack remains in eval mode.
        adapter = self.model.backbone.vision_backbone.trunk.steering_adapters["7"]
        self.assertTrue(adapter.training)
        self.assertTrue(self.model.backbone.vision_backbone.convs.training)
        self.assertFalse(self.model.transformer.training)
        self.assertFalse(self.model.segmentation_head.training)

    def test_eval_mode_forward(self):
        # The trainer's val loop runs the model in eval mode; make sure the
        # steered forward works there too (no matching, single pass).
        self.model.eval()
        with torch.no_grad():
            result = self.model(self.batch)
        out = self._last_step_out(result)
        self.assertTrue(torch.isfinite(out["pred_logits"]).all())
        self.assertTrue(torch.isfinite(out["pred_boxes"]).all())
        self.model.train()

    def test_forward_backward_and_optimizer(self):
        self.model.train()
        result = self.model(self.batch)
        out = self._last_step_out(result)

        self.assertIn("pred_logits", out)
        self.assertIn("pred_boxes", out)
        self.assertIn("steering_patch_logits", out)
        self.assertTrue(torch.isfinite(out["pred_logits"]).all())
        self.assertTrue(torch.isfinite(out["pred_boxes"]).all())
        self.assertEqual(out["pred_logits"].shape[0], 2)
        self.assertEqual(out["pred_boxes"].shape[0], 2)

        loss = (
            out["pred_logits"].sum()
            + out["pred_boxes"].sum()
            + out["steering_patch_logits"].sum()
        )
        loss.backward()

        named_grads = {
            n: (p.grad is not None) for n, p in self.model.named_parameters()
        }
        adapter_grads = {
            n: g for n, g in named_grads.items() if "steering_adapters" in n
        }
        self.assertTrue(adapter_grads)
        self.assertTrue(all(adapter_grads.values()))
        frozen_no_grad = all(
            not g for n, g in named_grads.items() if "backbone.language_backbone" in n
        )
        self.assertTrue(frozen_no_grad)
        frozen_trunk_no_grad = all(
            not g
            for n, g in named_grads.items()
            if "backbone.vision_backbone.trunk" in n and "steering_adapters" not in n
        )
        self.assertTrue(frozen_trunk_no_grad)
        self.assertTrue(
            any(
                p.grad is not None
                for n, p in self.model.named_parameters()
                if n.startswith("backbone.vision_backbone.convs")
            )
        )
        self.assertTrue(
            all(p.grad is not None for p in self.model.steering_patch_head.parameters())
        )
        # The fixed task stack still propagates input gradients to the FPN and
        # adapters, but never accumulates parameter gradients of its own.
        for prefix in (
            "geometry_encoder",
            "transformer",
            "dot_prod_scoring",
            "segmentation_head",
        ):
            self.assertTrue(
                all(
                    p.grad is None
                    for name, p in self.model.named_parameters()
                    if name.startswith(prefix)
                )
            )

        # Trainable-only optimizer: every trainable param exactly once, no
        # frozen params, adapters on their own LR, gates without decay.
        allowlist = {n for n, p in self.model.named_parameters() if p.requires_grad}
        opt = construct_optimizer(
            self.model, OPTIMIZER_CONF, OPTIM_OPTIONS, param_allowlist=allowlist
        )
        grouped = [p for group in opt.optimizer.param_groups for p in group["params"]]
        trainable = [p for _, p in self.model.named_parameters() if p.requires_grad]
        self.assertEqual(len(grouped), len(trainable))
        self.assertEqual(set(grouped), set(trainable))
        self.assertFalse(
            any(
                not p.requires_grad
                for group in opt.optimizer.param_groups
                for p in group["params"]
            )
        )
        # LR routing: adapters and patch head sit in their own group(s),
        # unmixed with the FPN, and on a 1e-4 / 8e-6 = 12.5x LR scale once the
        # shared scheduler is advanced past warmup (lr at step 0 is 0 for
        # both groups because of warmup).
        steering_params = {
            id(p)
            for n, p in self.model.named_parameters()
            if "steering_adapters" in n or n.startswith("steering_patch_head")
        }
        groups_with_steering = [
            group
            for group in opt.optimizer.param_groups
            if any(id(p) in steering_params for p in group["params"])
        ]
        groups_without_steering = [
            group
            for group in opt.optimizer.param_groups
            if not any(id(p) in steering_params for p in group["params"])
        ]
        self.assertTrue(groups_with_steering)
        for group in groups_with_steering:
            self.assertTrue(all(id(p) in steering_params for p in group["params"]))
        opt.step_schedulers(0.0, 30)
        steering_lrs = {group["lr"] for group in groups_with_steering}
        self.assertEqual(len(steering_lrs), 1)
        expected_ratio = 1e-4 / 8e-6
        for steering_lr in steering_lrs:
            for other_group in groups_without_steering:
                self.assertAlmostEqual(
                    steering_lr / other_group["lr"], expected_ratio, places=5
                )


if __name__ == "__main__":
    unittest.main()
