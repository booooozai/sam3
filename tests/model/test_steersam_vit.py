"""CPU unit tests for SteerableViT (steersam_design.md 7.3/7.12)."""

import unittest

import torch

from sam3.model.steersam_vit import _validate_steering_layers, SteerableViT
from sam3.model.vitdet import ViT

TINY_VIT_KWARGS = dict(
    img_size=56,  # 4x4 patch grid at patch size 14
    pretrain_img_size=224,
    patch_size=14,
    embed_dim=64,
    depth=6,
    num_heads=4,
    mlp_ratio=2.0,
    norm_layer="LayerNorm",
    drop_path_rate=0.0,
    qkv_bias=True,
    use_abs_pos=True,
    tile_abs_pos=True,
    global_att_blocks=(2, 5),
    rel_pos_blocks=(),
    use_rope=True,
    use_interp_rope=True,
    window_size=2,
    pretrain_use_cls_token=True,
    retain_cls_token=False,
    ln_pre=True,
    ln_post=False,
    return_interm_layers=False,
    bias_patch_embed=False,
    use_act_checkpoint=False,
)


def _build(cls, **overrides):
    torch.manual_seed(7)
    return cls(**{**TINY_VIT_KWARGS, **overrides})


def _steering_inputs(batch=2, grid=4, dim=64, length=5):
    torch.manual_seed(3)
    x = torch.randn(batch, 3, 56, 56)
    text = torch.randn(batch, length, dim)
    padding = torch.zeros(batch, length, dtype=torch.bool)
    padding[0, 3:] = True
    return x, text, padding


class TestSteerableViT(unittest.TestCase):
    def test_steering_layers_validation(self):
        self.assertEqual(_validate_steering_layers((1, 3), 6), (1, 3))
        with self.assertRaises(ValueError):
            _validate_steering_layers((1, 6), 6)  # out of range
        with self.assertRaises(ValueError):
            _validate_steering_layers((3, 3), 6)  # duplicate
        with self.assertRaises(ValueError):
            _validate_steering_layers((3, 1), 6)  # not increasing

    def test_adapters_registered_only_on_requested_blocks(self):
        model = _build(
            SteerableViT,
            steering_layers=(1, 4),
            steering_num_heads=4,
            steering_head_dim=16,
        )
        self.assertEqual(tuple(model.steering_adapters.keys()), ("1", "4"))
        self.assertEqual(model.steering_layers, (1, 4))
        # Adapter parameter count: 4 projections of 64x64 plus norms and gate.
        adapter = model.steering_adapters["1"]
        n_params = sum(p.numel() for p in adapter.parameters())
        self.assertEqual(
            n_params,
            4 * 64 * 64  # to_q, to_k, to_v, to_out
            + 2 * 64  # vision LayerNorm
            + 1,  # alpha
        )

    def test_unconditioned_forward_matches_original_vit(self):
        vit = _build(ViT)
        steerable = _build(SteerableViT, steering_layers=(1, 4))
        vit.eval()
        steerable.eval()
        x, _, _ = _steering_inputs()
        torch.testing.assert_close(steerable(x), vit(x))

    def test_zero_gate_forward_matches_original_vit(self):
        vit = _build(ViT)
        steerable = _build(SteerableViT, steering_layers=(1, 4))
        vit.eval()
        steerable.eval()
        x, text, padding = _steering_inputs()
        # alpha is zero-initialized: steering input must not change anything.
        torch.testing.assert_close(steerable(x, text, padding), vit(x))
        # Same holds for an explicit zero steering factor with a trained gate.
        with torch.no_grad():
            for adapter in steerable.steering_adapters.values():
                adapter.alpha.fill_(0.9)
        torch.testing.assert_close(
            steerable(x, text, padding, steering_factor=0.0), vit(x)
        )

    def test_nonzero_gate_changes_output(self):
        steerable = _build(SteerableViT, steering_layers=(1, 4))
        steerable.eval()
        x, text, padding = _steering_inputs()
        baseline = steerable(x)[0]
        with torch.no_grad():
            for adapter in steerable.steering_adapters.values():
                adapter.alpha.fill_(0.5)
        steered = steerable(x, text, padding)[0]
        self.assertFalse(torch.allclose(baseline, steered))

    def test_patch_feature_output_matches_final_trunk_output(self):
        steerable = _build(SteerableViT, steering_layers=(1, 4))
        steerable.eval()
        x, text, padding = _steering_inputs()
        outputs, patch_features = steerable.forward_with_patch_features(
            x, text, padding
        )
        self.assertEqual(len(outputs), 1)
        self.assertEqual(patch_features.shape, (2, 64, 4, 4))
        torch.testing.assert_close(patch_features, outputs[-1])

    def test_gradient_flow_with_frozen_blocks(self):
        steerable = _build(SteerableViT, steering_layers=(1, 4))
        for p in steerable.parameters():
            p.requires_grad_(False)
        for adapter in steerable.steering_adapters.values():
            adapter.requires_grad_(True)
        steerable.eval()

        x, text, padding = _steering_inputs()
        out = steerable(x, text, padding)
        loss = sum(t.sum() for t in out)
        loss.backward()

        for name, p in steerable.named_parameters():
            if "steering_adapters" in name:
                self.assertIsNotNone(p.grad, name)
            else:
                self.assertIsNone(p.grad, name)

    def test_activation_checkpointing_path(self):
        steerable = _build(
            SteerableViT, steering_layers=(1, 4), use_act_checkpoint=True
        )
        steerable.train()  # checkpointing activates only in training mode
        x, text, padding = _steering_inputs()
        out = steerable(x, text, padding)
        sum(t.sum() for t in out).backward()
        adapter_grads = [
            p.grad
            for name, p in steerable.named_parameters()
            if "steering_adapters" in name
        ]
        self.assertTrue(all(g is not None for g in adapter_grads))
        self.assertTrue(all(torch.isfinite(g).all() for g in adapter_grads))


if __name__ == "__main__":
    unittest.main()
