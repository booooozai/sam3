# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""VL backbone subclass for SteerSAM's text-first, conditioned image forward.

``SteerSAMVLBackbone`` keeps the parent's separate ``forward_image`` /
``forward_text`` API intact and adds:

- ``forward_text_for_steering``: like ``forward_text`` but also returns the
  contextual pre-resizer 1024-d tokens needed by the steering adapters;
- ``forward_conditioned_image``: runs the vision trunk with text conditioning
  (one image per image-text pair, per the SteerSAM positive-pair contract).
"""

from typing import List, Optional

import torch

from sam3.model.act_ckpt_utils import activation_ckpt_wrapper
from sam3.model.steersam_neck import SteerSAMViTDetNeck
from sam3.model.steersam_text_encoder import SteerSAMTextEncoder
from sam3.model.vl_combiner import SAM3VLBackbone
from torch.nn.attention import sdpa_kernel, SDPBackend


class SteerSAMVLBackbone(SAM3VLBackbone):
    def forward_text_for_steering(
        self,
        captions: List[str],
        input_boxes: Optional[torch.Tensor] = None,
        additional_text: Optional[List[str]] = None,
        device="cuda",
    ):
        """Encode text for steering; mirrors ``forward_text`` plus contextual tokens.

        Returns a dict with the parent's ``language_features`` (L, B, 256),
        ``language_mask`` (B, L; True = padding) and ``language_embeds``
        (L, B, 1024), plus ``language_contextual`` (L, B, 1024): the encoder's
        contextual output before the 256-d resizer.
        """
        return activation_ckpt_wrapper(self._forward_text_for_steering_no_act_ckpt)(
            captions=captions,
            input_boxes=input_boxes,
            additional_text=additional_text,
            device=device,
            act_ckpt_enable=self.act_ckpt_whole_language_backbone and self.training,
        )

    def _forward_text_for_steering_no_act_ckpt(
        self,
        captions,
        input_boxes=None,
        additional_text=None,
        device="cuda",
    ):
        assert isinstance(
            self.language_backbone, SteerSAMTextEncoder
        ), "forward_text_for_steering requires a SteerSAMTextEncoder."
        output = {}

        text_to_encode = list(captions)
        if additional_text is not None:
            text_to_encode += additional_text

        sdpa_context = sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        )

        with sdpa_context:
            text_output = self.language_backbone.encode_for_steering(
                text_to_encode, input_boxes, device=device
            )

        if additional_text is not None:
            output["additional_text_features"] = text_output.features[
                :, -len(additional_text) :
            ]
            output["additional_text_mask"] = text_output.padding_mask[
                -len(additional_text) :
            ]

        n = len(captions)
        output["language_features"] = text_output.features[:, :n]
        output["language_mask"] = text_output.padding_mask[:n]
        output["language_embeds"] = text_output.token_embeddings[:, :n]
        output["language_contextual"] = text_output.contextual_pre_resizer[:, :n]
        return output

    def forward_conditioned_image(
        self,
        samples: torch.Tensor,
        steering_text: Optional[torch.Tensor] = None,
        steering_padding_mask: Optional[torch.Tensor] = None,
        steering_factor: float = 1.0,
        return_patch_features: bool = False,
    ):
        """Run the vision trunk conditioned on per-image steering text.

        ``samples`` holds one image per image-text pair (the SteerSAM
        positive-pair contract), so the returned FPN features are already
        prompt-specific and the downstream ``img_ids`` are ``arange(P)``.
        """
        return activation_ckpt_wrapper(self._forward_conditioned_image_no_act_ckpt)(
            samples=samples,
            steering_text=steering_text,
            steering_padding_mask=steering_padding_mask,
            steering_factor=steering_factor,
            return_patch_features=return_patch_features,
            act_ckpt_enable=self.act_ckpt_whole_vision_backbone and self.training,
        )

    def _forward_conditioned_image_no_act_ckpt(
        self,
        samples,
        steering_text=None,
        steering_padding_mask=None,
        steering_factor=1.0,
        return_patch_features=False,
    ):
        assert isinstance(
            self.vision_backbone, SteerSAMViTDetNeck
        ), "forward_conditioned_image requires a SteerSAMViTDetNeck."
        neck_outputs = self.vision_backbone.forward(
            samples,
            steering_text=steering_text,
            steering_padding_mask=steering_padding_mask,
            steering_factor=steering_factor,
            return_patch_features=return_patch_features,
        )
        if return_patch_features:
            (
                sam3_features,
                sam3_pos,
                sam2_features,
                sam2_pos,
                patch_features,
            ) = neck_outputs
        else:
            sam3_features, sam3_pos, sam2_features, sam2_pos = neck_outputs
            patch_features = None
        if self.scalp > 0:
            # Discard the lowest resolution features
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.scalp],
                    sam2_pos[: -self.scalp],
                )

        sam2_output = None
        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }

        sam3_src = sam3_features[-1]
        output = {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }
        if patch_features is not None:
            output["steering_patch_features"] = patch_features
        return output
