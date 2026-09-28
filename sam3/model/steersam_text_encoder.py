# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Text encoder subclass exposing contextual pre-resizer tokens for SteerSAM.

The original ``VETextEncoder.forward`` returns ``(padding_mask, resized
features, raw token embeddings)``. SteerSAM additionally needs the contextual
1024-d tokens produced by the language encoder *before* the 1024 -> 256
resizer, because those serve as the K/V of the steering adapters. This subclass
exposes them through a separate ``encode_for_steering`` method and leaves the
inherited ``forward`` (and the original SAM3 return protocol) untouched.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from sam3.model.text_encoder_ve import VETextEncoder


@dataclass
class TextEncoderOutput:
    """Structured output of ``SteerSAMTextEncoder.encode_for_steering``.

    ``features``/``contextual_pre_resizer``/``token_embeddings`` follow the
    SAM3 sequence-first convention (mirroring the tuple returned by
    ``VETextEncoder.forward``); the padding mask is batch-first like the
    parent's ``text_attention_mask``.
    """

    padding_mask: torch.Tensor  # (B, L) bool; True = padding/ignore
    features: torch.Tensor  # (L, B, d_model) resized features for late fusion
    contextual_pre_resizer: torch.Tensor  # (L, B, width) contextual tokens
    token_embeddings: torch.Tensor  # (L, B, width) pre-encoder token embeddings


class SteerSAMTextEncoder(VETextEncoder):
    """``VETextEncoder`` with an extra steering-oriented encoding entry point."""

    def encode_for_steering(
        self,
        text: Union[List[str], Tuple[torch.Tensor, torch.Tensor, dict]],
        input_boxes: Optional[List] = None,
        device: torch.device = None,
    ) -> TextEncoderOutput:
        if not isinstance(text[0], str):
            # Pre-encoded text arrives through the (mask, features, tokenized)
            # tuple contract, which carries no contextual 1024-d tokens. Using
            # the raw embeddings instead would silently steer with the wrong
            # semantic level, so refuse explicitly.
            raise NotImplementedError(
                "encode_for_steering requires raw strings; pre-encoded text "
                "caches do not contain the contextual pre-resizer tokens that "
                "steering adapters need."
            )
        assert input_boxes is None or len(input_boxes) == 0, "not supported"

        tokenized = self.tokenizer(text, context_length=self.context_length).to(
            device
        )  # (B, L)
        text_attention_mask = (tokenized != 0).bool()

        # Manually embed the tokens and run the contextual encoder, exactly as
        # VETextEncoder.forward does.
        inputs_embeds = self.encoder.token_embedding(tokenized)  # (B, L, width)
        _, text_memory = self.encoder(tokenized)  # (B, L, width)

        assert text_memory.shape[1] == inputs_embeds.shape[1]
        # SAM3 convention: True = padding/ignore.
        padding_mask = text_attention_mask.ne(1)
        text_memory = text_memory.transpose(0, 1)  # (L, B, width)
        features = self.resizer(text_memory)  # (L, B, d_model)

        return TextEncoderOutput(
            padding_mask=padding_mask,
            features=features,
            contextual_pre_resizer=text_memory,
            token_embeddings=inputs_embeds.transpose(0, 1),
        )
