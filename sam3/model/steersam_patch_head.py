# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Patch-localization head for the image-only SteerSAM path.

The head is deliberately attached to the final prompt-conditioned ViT feature
*before* SimpleFPN.  A 1x1 convolution is exactly a shared ``Linear(D, 1)``
over spatial patches, matching the SteerViT localization head while keeping
the native ``NCHW`` layout used by SAM3.
"""

import torch
import torch.nn as nn


class SteerSAMPatchHead(nn.Module):
    """Predict one localization logit for every conditioned ViT patch."""

    def __init__(self, input_dim: int = 1024, zero_init: bool = True):
        super().__init__()
        self.projection = nn.Conv2d(input_dim, 1, kernel_size=1, bias=True)
        if zero_init:
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_features: Prompt-conditioned pre-FPN features ``(P,D,H,W)``.

        Returns:
            Patch logits with shape ``(P,H,W)``.
        """
        if patch_features.ndim != 4:
            raise ValueError(
                "patch_features must have shape (P,D,H,W), got "
                f"{tuple(patch_features.shape)}"
            )
        return self.projection(patch_features).squeeze(1)
