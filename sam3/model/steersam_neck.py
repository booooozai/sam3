# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Neck subclass that forwards steering context into a ``SteerableViT`` trunk.

``SteerSAMViTDetNeck`` reuses the SAM3/SAM2 FPN convs of the original
``Sam3DualViTDetNeck`` unchanged; only the conditioned forward is added. The
original image-only interface keeps working when ``steering_text is None``.
"""

from typing import List, Optional, Tuple

import torch

from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.steersam_vit import SteerableViT


class SteerSAMViTDetNeck(Sam3DualViTDetNeck):
    def forward(
        self,
        tensor_list,
        steering_text: Optional[torch.Tensor] = None,
        steering_padding_mask: Optional[torch.Tensor] = None,
        steering_factor: float = 1.0,
        return_patch_features: bool = False,
    ):
        if steering_text is None:
            if return_patch_features:
                raise ValueError(
                    "Pre-FPN patch features are only exposed for a conditioned "
                    "SteerSAM forward."
                )
            # Unconditioned call: exactly the original neck behavior.
            return super().forward(tensor_list)

        assert isinstance(self.trunk, SteerableViT), (
            "SteerSAMViTDetNeck requires a SteerableViT trunk to run "
            "conditioned forwards."
        )
        assert isinstance(tensor_list, torch.Tensor), (
            "The steering path only supports a single image tensor per "
            "forward, matching the SteerSAM image-pair data contract."
        )
        trunk_kwargs = dict(
            x=tensor_list,
            steering_text=steering_text,
            steering_padding_mask=steering_padding_mask,
            steering_factor=steering_factor,
        )
        if return_patch_features:
            xs, patch_features = self.trunk.forward_with_patch_features(**trunk_kwargs)
            return (*self._apply_fpn(xs), patch_features)

        xs = self.trunk(**trunk_kwargs)
        return self._apply_fpn(xs)

    def _apply_fpn(self, xs):
        """Run the SAM3 (and optional SAM2) FPN heads on trunk features.

        Mirrors the body of ``Sam3DualViTDetNeck.forward`` after the trunk call.
        """
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []
        x = xs[-1]  # simpleFPN
        for i in range(len(self.convs)):
            sam3_x_out = self.convs[i](x)
            sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](x)
                sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)
        return sam3_out, sam3_pos, sam2_out, sam2_pos
