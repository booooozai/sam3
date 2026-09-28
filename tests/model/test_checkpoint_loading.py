import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from sam3.model_builder import (
    _load_checkpoint,
    _validate_steersam_checkpoint_load,
    CheckpointLoadResult,
)


class _TinySteerSAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.inst_interactive_predictor = None
        self.segmentation_head = None
        self.backbone = nn.Module()
        self.backbone.attn = nn.Module()
        self.backbone.attn.register_buffer("freqs_cis", torch.ones(2, 2))
        self.steering_adapters = nn.Linear(2, 2)
        self.steering_patch_head = nn.Linear(2, 1)
        self.head = nn.Linear(2, 2)


class TestCheckpointLoading(unittest.TestCase):
    def test_dynamic_rope_and_adapter_missing_keys_are_allowed(self):
        model = _TinySteerSAM()
        checkpoint = {
            "model": {
                "detector.head.weight": torch.ones_like(model.head.weight),
                "detector.head.bias": torch.ones_like(model.head.bias),
                "detector.backbone.attn.freqs_cis": torch.zeros(4, 2),
                # Released checkpoints contain this optional image-only extra.
                "detector.backbone.vision_backbone.sam2_convs.0.weight": torch.ones(1),
                "detector.segmentation_head.optional_weight": torch.ones(1),
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            result = _load_checkpoint(model, str(checkpoint_path))

        self.assertEqual(result.skipped_dynamic_keys, ["backbone.attn.freqs_cis"])
        self.assertIn("backbone.attn.freqs_cis", result.missing_keys)
        self.assertIn("steering_adapters.weight", result.missing_keys)
        self.assertIn("steering_patch_head.weight", result.missing_keys)
        self.assertEqual(
            set(result.unexpected_keys),
            {
                "backbone.vision_backbone.sam2_convs.0.weight",
                "segmentation_head.optional_weight",
            },
        )
        _validate_steersam_checkpoint_load(result, model=model)

    def test_unknown_checkpoint_incompatibilities_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported incompatibilities"):
            _validate_steersam_checkpoint_load(
                CheckpointLoadResult(
                    missing_keys=["decoder.unknown_weight"],
                    unexpected_keys=["backbone.unknown_weight"],
                    skipped_dynamic_keys=[],
                )
            )

    def test_trainer_checkpoint_uses_direct_model_state_dict_names(self):
        model = _TinySteerSAM()
        checkpoint = {
            "model": {
                key: torch.full_like(value, 3)
                for key, value in model.state_dict().items()
            },
            "epoch": 7,
            "optimizer": {"state": {}, "param_groups": []},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            result = _load_checkpoint(model, str(checkpoint_path))

        # Resolution-dependent buffers are regenerated, while trained adapter
        # and head weights load without requiring a detector.* prefix.
        self.assertEqual(result.skipped_dynamic_keys, ["backbone.attn.freqs_cis"])
        self.assertEqual(result.unexpected_keys, [])
        torch.testing.assert_close(
            model.steering_adapters.weight,
            torch.full_like(model.steering_adapters.weight, 3),
        )
        torch.testing.assert_close(
            model.head.weight, torch.full_like(model.head.weight, 3)
        )
        _validate_steersam_checkpoint_load(result, model=model)


if __name__ == "__main__":
    unittest.main()
