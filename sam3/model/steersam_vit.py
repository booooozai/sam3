# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""ViT subclass with zero-gated steering adapters (SteerSAM).

``SteerableViT`` registers the adapters as real submodules (no monkey patching)
and inserts each adapter immediately before its ViT block, matching the
SteerViT ordering: text steers the tokens, then the frozen block processes
them. The original ``ViT`` class is untouched; ``enable_steering=False`` builds
never construct this class.
"""

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from sam3.model.steering import GatedVisionLanguageAdapter
from sam3.model.vitdet import get_abs_pos, ViT
from torch import Tensor


def _validate_steering_layers(steering_layers: Iterable[int], depth: int):
    layers = tuple(int(i) for i in steering_layers)
    if any(i < 0 or i >= depth for i in layers):
        raise ValueError(
            f"steering_layers {layers} out of range for a ViT with depth {depth}"
        )
    if len(set(layers)) != len(layers):
        raise ValueError(f"steering_layers {layers} contains duplicates")
    if list(layers) != sorted(layers):
        raise ValueError(f"steering_layers {layers} must be strictly increasing")
    return layers


class SteerableViT(ViT):
    """``ViT`` with one ``GatedVisionLanguageAdapter`` per configured block."""

    def __init__(
        self,
        *args,
        steering_layers: Iterable[int] = (7, 15, 23, 31),
        steering_num_heads: int = 16,
        steering_head_dim: int = 64,
        steering_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        layers = _validate_steering_layers(steering_layers, len(self.blocks))

        # The parent __init__ may have stored a torch.compile'd *parent*
        # forward as an instance attribute; that would shadow this subclass's
        # forward and silently skip the adapters. Compiled graphs are not
        # supported for the steering path, so drop the instance attribute.
        self.__dict__.pop("forward", None)

        vision_dim = self.patch_embed.proj.out_channels
        self.steering_layers = layers
        self.steering_adapters = nn.ModuleDict(
            {
                str(i): GatedVisionLanguageAdapter(
                    vision_dim=vision_dim,
                    text_dim=vision_dim,
                    num_heads=steering_num_heads,
                    head_dim=steering_head_dim,
                    dropout=steering_dropout,
                )
                for i in layers
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        steering_text: Optional[Tensor] = None,
        steering_padding_mask: Optional[Tensor] = None,
        steering_factor: float = 1.0,
    ) -> list[Tensor]:
        outputs, _ = self._forward_with_patch_features(
            x=x,
            steering_text=steering_text,
            steering_padding_mask=steering_padding_mask,
            steering_factor=steering_factor,
        )
        return outputs

    def forward_with_patch_features(
        self,
        x: torch.Tensor,
        steering_text: Optional[Tensor] = None,
        steering_padding_mask: Optional[Tensor] = None,
        steering_factor: float = 1.0,
    ) -> Tuple[list[Tensor], Tensor]:
        """Return normal trunk outputs plus the final pre-FPN patch feature.

        This method is exclusive to the SteerSAM path.  Keeping ``forward``'s
        original list return type preserves the image-backbone interface when
        patch supervision is disabled.
        """
        return self._forward_with_patch_features(
            x=x,
            steering_text=steering_text,
            steering_padding_mask=steering_padding_mask,
            steering_factor=steering_factor,
        )

    def _forward_with_patch_features(
        self,
        x: torch.Tensor,
        steering_text: Optional[Tensor] = None,
        steering_padding_mask: Optional[Tensor] = None,
        steering_factor: float = 1.0,
    ) -> Tuple[list[Tensor], Tensor]:
        assert isinstance(x, torch.Tensor), (
            "SteerableViT only supports a single image tensor per forward; "
            "list inputs are not part of the SteerSAM image pipeline."
        )
        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if self.retain_cls_token:
            # If cls_token is retained, we don't maintain spatial shape
            x = torch.cat([self.class_embedding, x.flatten(1, 2)], dim=1)
            s = 1

        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed,
                self.pretrain_use_cls_token,
                (h, w),
                self.retain_cls_token,
                tiling=self.tile_abs_pos,
            )

        x = self.ln_pre(x)

        use_steering = steering_text is not None and len(self.steering_adapters) > 0

        outputs = []
        final_patch_features = None
        for i, blk in enumerate(self.blocks):
            if use_steering and str(i) in self.steering_adapters:
                adapter = self.steering_adapters[str(i)]
                # Steering happens before the frozen block so the block's
                # self-attention propagates the steered tokens, matching
                # the SteerViT ordering. The adapter runs outside the
                # block's activation checkpoint: it is cheap (L <= 32 K/V
                # tokens) and its activations stay in memory, while the
                # expensive frozen block still recomputes on backward.
                x = adapter(
                    x,
                    steering_text,
                    steering_padding_mask,
                    steering_factor,
                )
            if self.use_act_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if (i == self.full_attn_ids[-1]) or (
                self.return_interm_layers and i in self.full_attn_ids
            ):
                if i == self.full_attn_ids[-1]:
                    x = self.ln_post(x)

                feats = x[:, s:]
                if feats.ndim == 4:
                    feats = feats.permute(0, 3, 1, 2)
                else:
                    assert feats.ndim == 3
                    h = w = math.sqrt(feats.shape[1])
                    feats = feats.reshape(
                        feats.shape[0], h, w, feats.shape[-1]
                    ).permute(0, 3, 1, 2)

                outputs.append(feats)
                if i == self.full_attn_ids[-1]:
                    final_patch_features = feats

        if final_patch_features is None:
            raise RuntimeError("ViT forward did not produce a final patch feature map.")
        return outputs, final_patch_features
