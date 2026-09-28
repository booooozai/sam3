"""CPU unit tests for the SteerSAM gating adapter (steersam_design.md 7.1/7.12)."""

import unittest

import torch
import torch.nn as nn

from sam3.model.steering import (
    GatedVisionLanguageAdapter,
    set_frozen_submodules_to_eval,
)

P, H, W, L = 3, 4, 4, 6
DIM = 64


def _make_adapter():
    torch.manual_seed(0)
    return GatedVisionLanguageAdapter(
        vision_dim=DIM, text_dim=DIM, num_heads=4, head_dim=16, dropout=0.0
    )


def _make_inputs():
    torch.manual_seed(1)
    image = torch.randn(P, H, W, DIM)
    text = torch.randn(P, L, DIM)
    # One of the three samples has fully padded text.
    padding = torch.zeros(P, L, dtype=torch.bool)
    padding[0, 4:] = True
    padding[1, :] = True
    return image, text, padding


class TestGatedVisionLanguageAdapter(unittest.TestCase):
    def test_shape_and_dtype(self):
        adapter = _make_adapter()
        image, text, padding = _make_inputs()
        out = adapter(image, text, padding)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, image.dtype)

        # Pre-flattened tokens are supported too.
        out_flat = adapter(image.reshape(P, H * W, DIM), text, padding)
        self.assertEqual(out_flat.shape, (P, H * W, DIM))
        torch.testing.assert_close(out_flat, out.reshape(P, H * W, DIM))

    def test_zero_gate_is_exact_identity(self):
        adapter = _make_adapter()
        adapter.eval()
        image, text, padding = _make_inputs()
        self.assertEqual(float(adapter.alpha.item()), 0.0)
        out = adapter(image, text, padding)
        torch.testing.assert_close(out, image)

    def test_zero_steering_factor_is_exact_identity_even_with_trained_gate(self):
        adapter = _make_adapter()
        with torch.no_grad():
            adapter.alpha.fill_(0.7)
        image, text, padding = _make_inputs()
        out = adapter(image, text, padding, steering_factor=0.0)
        torch.testing.assert_close(out, image)

    def test_padding_value_invariance(self):
        """Changing ignored text tokens must not change the output."""
        adapter = _make_adapter()
        adapter.eval()
        image, text, padding = _make_inputs()
        out_a = adapter(image, text, padding)
        text_b = text.clone()
        text_b[padding] = torch.randn_like(text_b[padding]) * 10.0
        out_b = adapter(image, text_b, padding)
        torch.testing.assert_close(out_a, out_b)

    def test_all_padding_text_bypasses_without_nan(self):
        adapter = _make_adapter()
        with torch.no_grad():
            adapter.alpha.fill_(1.0)
        image, text, padding = _make_inputs()
        out = adapter(image, text, padding)  # sample 1 is fully padded
        self.assertTrue(torch.isfinite(out).all())
        # The fully padded sample keeps its unsteered tokens.
        torch.testing.assert_close(out[1], image[1])

    def test_all_padding_text_backward_stays_finite(self):
        adapter = _make_adapter()
        with torch.no_grad():
            adapter.alpha.fill_(1.0)
        image, text, padding = _make_inputs()
        out = adapter(image, text, padding)
        out.sum().backward()
        grads = [p.grad for p in adapter.parameters()]
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_none_mask_treats_all_text_as_valid(self):
        adapter = _make_adapter()
        with torch.no_grad():
            adapter.alpha.fill_(0.5)
        image, text, _ = _make_inputs()
        explicit = torch.zeros(text.shape[:2], dtype=torch.bool)
        out_none = adapter(image, text, None)
        out_valid = adapter(image, text, explicit)
        torch.testing.assert_close(out_none, out_valid)

    def test_nonzero_gate_is_prompt_sensitive(self):
        adapter = _make_adapter()
        with torch.no_grad():
            adapter.alpha.fill_(0.5)
        image, _, padding = _make_inputs()
        text_a = torch.randn(P, L, DIM)
        text_b = torch.randn(P, L, DIM)
        out_a = adapter(image, text_a, padding)
        out_b = adapter(image, text_b, padding)
        self.assertFalse(torch.allclose(out_a, out_b))
        # And the steered output differs from the baseline on valid samples.
        self.assertFalse(torch.allclose(out_a, image))

    def test_gradients_reach_all_adapter_parameters(self):
        adapter = _make_adapter()
        image, text, padding = _make_inputs()
        out = adapter(image, text, padding)
        out.sum().backward()
        for name in ["to_q", "to_k", "to_v", "to_out", "norm"]:
            module = getattr(adapter, name)
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None for g in grads), name)
        self.assertIsNotNone(adapter.alpha.grad)

    def test_supports_float16_half_precision_cpu(self):
        adapter = _make_adapter().half()
        image, text, padding = _make_inputs()
        out = adapter(image.half(), text.half(), padding)
        self.assertEqual(out.dtype, torch.float16)


class TestSetFrozenSubmodulesToEval(unittest.TestCase):
    def test_frozen_subtrees_eval_trainable_subtrees_train(self):
        frozen_block = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        for p in frozen_block.parameters():
            p.requires_grad_(False)
        adapter = GatedVisionLanguageAdapter(
            vision_dim=4, text_dim=4, num_heads=2, head_dim=2
        )
        model = nn.Sequential(frozen_block, adapter)

        model.train()
        self.assertTrue(frozen_block[0].training)
        set_frozen_submodules_to_eval(model)
        # The all-frozen block subtree is back in eval mode...
        self.assertFalse(frozen_block[0].training)
        self.assertFalse(frozen_block[1].training)
        # ...while the adapter subtree stays in train mode.
        self.assertTrue(adapter.training)
        self.assertTrue(adapter.to_q.training)

        model.train(False)
        self.assertFalse(adapter.training)

    def test_parameterless_dropout_in_trainable_subtree_stays_train(self):
        # Bare nn.Dropout modules own no parameters; they must inherit the
        # mode of their trainable context instead of being forced to eval
        # (otherwise task-head dropout would silently turn off in training).
        trainable_head = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))
        frozen_block = nn.Sequential(nn.Linear(4, 4))
        for p in frozen_block.parameters():
            p.requires_grad_(False)
        model = nn.Sequential(trainable_head, frozen_block)

        model.train()
        set_frozen_submodules_to_eval(model)
        self.assertTrue(trainable_head[0].training)
        self.assertTrue(trainable_head[1].training)
        self.assertFalse(frozen_block[0].training)

    def test_parameterless_dropout_inside_frozen_subtree_goes_eval(self):
        frozen_block = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))
        for p in frozen_block.parameters():
            p.requires_grad_(False)
        trainable_head = nn.Linear(4, 4)
        model = nn.Sequential(frozen_block, trainable_head)

        model.train()
        set_frozen_submodules_to_eval(model)
        self.assertFalse(frozen_block[0].training)
        # Eval propagates recursively from the frozen parameter-owning block.
        self.assertFalse(frozen_block[1].training)
        self.assertTrue(trainable_head.training)


if __name__ == "__main__":
    unittest.main()
