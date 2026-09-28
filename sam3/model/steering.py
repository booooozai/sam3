# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Steering adapter for SteerSAM.

Implements the zero-gated image-to-text cross-attention adapter described in
``docs/steersam_design.md`` (section 7.1). The adapter is a standalone,
testable module: it does not depend on the dataset, the tokenizer, or the rest
of SAM3.

Mask convention (SAM3-wide): ``text_padding_mask`` is a bool tensor where
``True`` means padding/ignore. Inside the adapter it is inverted for
``scaled_dot_product_attention``, whose bool ``attn_mask`` uses ``True`` to
mean "allowed to attend".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GatedVisionLanguageAdapter(nn.Module):
    """Zero-gated cross-attention that steers vision tokens with text tokens.

    Vision tokens are the queries and contextual text tokens are the keys and
    values. The residual is scaled by ``tanh(alpha) * steering_factor`` with
    ``alpha`` initialized to zero, so at initialization the adapter is an exact
    identity and the model matches the frozen-backbone baseline.
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        text_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = inner_dim

        self.norm = nn.LayerNorm(vision_dim)
        self.to_q = nn.Linear(vision_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(text_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(text_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, vision_dim, bias=False)
        # Kept as a module so AMP/dropout bookkeeping stays uniform; p=0 by default.
        self.attn_drop = nn.Dropout(dropout)

        # Zero-initialized gate: tanh(0) == 0 makes the adapter an exact
        # identity at initialization. Weight decay must be disabled for this
        # parameter in the optimizer config.
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        text_padding_mask: Tensor | None,
        steering_factor: float | Tensor = 1.0,
    ) -> Tensor:
        """
        Args:
            image: ``(P, H, W, vision_dim)`` or pre-flattened ``(P, N, vision_dim)``
                vision tokens. Output has the same shape and dtype.
            text: ``(P, L, text_dim)`` contextual text tokens.
            text_padding_mask: ``(P, L)`` bool; ``True`` = padding/ignore.
                ``None`` treats every text token as valid.
            steering_factor: scalar (or per-sample ``(P,)`` tensor) multiplier
                applied on top of ``tanh(alpha)``; 0 gives the exact baseline.
        """
        if text_padding_mask is None:
            text_padding_mask = torch.zeros(
                text.shape[:2], dtype=torch.bool, device=text.device
            )
        elif text_padding_mask.dtype != torch.bool:
            text_padding_mask = text_padding_mask.bool()
        assert image.ndim in (
            3,
            4,
        ), "image must be (P, H, W, D) or pre-flattened (P, N, D)"

        # The projection output carries the (possibly autocast) dtype of the
        # vision tokens; text projections are cast to match so mixed dtypes
        # never reach SDPA.
        x = image.reshape(image.shape[0], -1, image.shape[-1])
        q = self.to_q(self.norm(x))
        k = self.to_k(text.to(dtype=q.dtype))
        v = self.to_v(text.to(dtype=q.dtype))

        P, N, _ = q.shape
        q = q.reshape(P, N, self.num_heads, self.head_dim).transpose(1, 2)
        L = k.shape[1]
        k = k.reshape(P, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(P, L, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA bool mask: True = allowed to attend. SAM3 padding masks are the
        # opposite (True = ignore), so invert and broadcast over heads/queries.
        valid_mask = (~text_padding_mask)[:, None, None, :]
        # A fully masked attention row makes softmax produce NaN in both the
        # forward and the backward pass. Unmask one position for such rows so
        # SDPA stays finite; their (meaningless) delta is zeroed below.
        all_padding = text_padding_mask.all(dim=-1)
        if all_padding.any():
            valid_mask = valid_mask.clone()
            valid_mask[all_padding, 0, 0, 0] = True

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=valid_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(P, N, self.inner_dim)
        delta = self.to_out(out)
        # Gate after the output projection, before the residual add.
        gate = torch.tanh(self.alpha)
        if isinstance(steering_factor, Tensor):
            gate = gate * steering_factor.reshape(-1, 1, 1)
        else:
            gate = gate * steering_factor
        delta = delta * gate

        # Samples whose text is fully padding have no steering signal; bypass
        # them so the residual keeps the input unchanged.
        if all_padding.any():
            delta = torch.where(
                all_padding[:, None, None], torch.zeros_like(delta), delta
            )

        return image + delta.reshape(image.shape)


def set_frozen_submodules_to_eval(model: nn.Module) -> None:
    """Put every all-frozen subtree in eval mode, leave trainable ones in train.

    Called after ``model.train()`` so frozen backbone blocks lose stochastic
    behavior (e.g. drop-path) while adapters and task heads stay in training
    mode. A module is switched to eval only when its subtree owns parameters
    and none of them are trainable; parameter-less modules (bare Dropout /
    DropPath) inherit their mode from the nearest parameter-owning ancestor,
    so dropout inside the trainable encoder/decoder stays active exactly as in
    the baseline.
    """
    for module in model.modules():
        has_trainable = False
        has_params = False
        for param in module.parameters(recurse=True):
            has_params = True
            if param.requires_grad:
                has_trainable = True
                break
        if has_params and not has_trainable:
            module.eval()
