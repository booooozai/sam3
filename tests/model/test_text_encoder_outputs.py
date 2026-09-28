"""CPU unit tests for SteerSAMTextEncoder (steersam_design.md 7.2/7.12)."""

import unittest

import torch
from sam3.model.steersam_text_encoder import SteerSAMTextEncoder

from sam3.model.text_encoder_ve import TextTransformer, VETextEncoder


class _StubTokenizer:
    """Returns a fixed token layout with explicit padding on some rows."""

    def __call__(self, texts, context_length=None):
        batch = len(texts)
        tokens = torch.zeros(batch, context_length, dtype=torch.long)
        for i in range(batch):
            tokens[i, : 2 + i] = torch.arange(2 + i) + 1  # row i has 2+i valid tokens
        return tokens


def _build_encoder(cls):
    torch.manual_seed(5)
    encoder = cls(
        d_model=16,
        tokenizer=_StubTokenizer(),
        width=32,
        heads=2,
        layers=1,
        context_length=8,
        vocab_size=100,
    )
    # TextTransformer leaves positional_embedding and text_projection as
    # torch.empty (real weights come from the checkpoint). Initialize them so
    # tests are deterministic and free of uninitialized-memory NaNs.
    with torch.no_grad():
        encoder.encoder.positional_embedding.normal_()
        encoder.encoder.text_projection.normal_()
    encoder.eval()
    return encoder


class TestSteerSAMTextEncoder(unittest.TestCase):
    def test_encode_for_steering_matches_forward_outputs(self):
        encoder = _build_encoder(SteerSAMTextEncoder)
        texts = ["a cat", "a dog on the left", "rain"]

        mask, features, embeds = encoder(texts)
        out = encoder.encode_for_steering(texts)

        torch.testing.assert_close(out.padding_mask, mask)
        torch.testing.assert_close(out.features, features)
        torch.testing.assert_close(out.token_embeddings, embeds)

    def test_contextual_tokens_come_from_encoder_not_embedding_table(self):
        encoder = _build_encoder(SteerSAMTextEncoder)
        texts = ["a cat", "a dog on the left", "rain"]
        tokenized = encoder.tokenizer(texts, context_length=encoder.context_length)

        out = encoder.encode_for_steering(texts)

        _, text_memory = encoder.encoder(tokenized)
        torch.testing.assert_close(
            out.contextual_pre_resizer, text_memory.transpose(0, 1)
        )
        # The contextual output must not be the raw token embedding lookup.
        self.assertFalse(
            torch.allclose(out.contextual_pre_resizer, out.token_embeddings)
        )
        # And the late-fusion features are the resizer applied to it.
        torch.testing.assert_close(
            out.features, encoder.resizer(out.contextual_pre_resizer)
        )

    def test_padding_mask_semantics_true_is_padding(self):
        encoder = _build_encoder(SteerSAMTextEncoder)
        texts = ["a cat", "a dog on the left", "rain"]
        out = encoder.encode_for_steering(texts)
        # Row i has 2+i valid tokens; everything after is padding.
        for i, n_valid in enumerate([2, 3, 4]):
            self.assertFalse(out.padding_mask[i, :n_valid].any())
            self.assertTrue(out.padding_mask[i, n_valid:].all())

    def test_pre_encoded_text_is_rejected(self):
        encoder = _build_encoder(SteerSAMTextEncoder)
        fake_encoded = (
            torch.zeros(1, 8),
            torch.zeros(1, 8, 16),
            {"inputs_embeds": None},
        )
        with self.assertRaises(NotImplementedError):
            encoder.encode_for_steering(fake_encoded)

    def test_inherited_forward_unchanged(self):
        """The plain VETextEncoder protocol keeps working on the subclass."""
        encoder = _build_encoder(SteerSAMTextEncoder)
        reference = _build_encoder(VETextEncoder)
        texts = ["a cat", "a dog on the left", "rain"]
        out_sub = encoder(texts)
        out_ref = reference(texts)
        for a, b in zip(out_sub, out_ref):
            torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
