# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Top-level SteerSAM image model (text-first forward with steered trunk).

``SteerSAMImage`` swaps the original image-first late-fusion order for:

1. encode the (deduplicated) texts;
2. validate the positive-pair batch contract (one image row per pair,
   ``img_ids == arange(P)``), which the ``COCOPositivePairFromJSON`` loader
   guarantees;
3. run the vision trunk conditioned on each pair's text, producing
   prompt-specific FPN features;
4. reuse the parent's fusion encoder, decoder, heads and matching exactly.

``forward_grounding`` and everything downstream of ``backbone_out`` are
inherited unchanged: because ``img_ids`` is ``arange(P)``, the parent's
``_get_img_feats`` gather becomes an identity over the pair-batch features.
This class is only constructed when ``enable_steering`` is on; the original
``Sam3Image`` remains the baseline path.
"""

import torch
import torch.nn as nn

from sam3.model.geometry_encoders import Prompt
from sam3.model.model_misc import SAM3Output
from sam3.model.sam3_image import Sam3Image
from sam3.model.steering import set_frozen_submodules_to_eval
from sam3.train.data.collator import BatchedDatapoint


def validate_positive_pair_batch(images, img_ids, text_ids) -> None:
    """Check the SteerSAM positive-pair contract on a collated batch.

    The contract is produced by the data pipeline (e.g.
    ``COCOPositivePairFromJSON``): every record is a single image with a single
    positive prompt, so a batch has ``P == len(images)`` image rows and
    ``img_ids`` must be the identity mapping ``arange(P)``. Reject anything
    else instead of silently materializing or reusing image features.
    """
    pair_count = len(images)
    assert len(img_ids) == len(text_ids) == pair_count, (
        f"Pair batch mismatch: {pair_count} images, {len(img_ids)} img_ids, "
        f"{len(text_ids)} text_ids."
    )
    img_ids = torch.as_tensor(img_ids)
    expected = torch.arange(pair_count, device=img_ids.device)
    assert torch.equal(img_ids, expected), (
        "SteerSAM expects img_ids == arange(P) from the positive-pair data "
        f"contract; got {img_ids.tolist()}. A one-image-many-queries loader is "
        "being used with the steering pipeline."
    )


class SteerSAMImage(Sam3Image):
    # Read by the trainer to build a trainable-only optimizer param group
    # allowlist (design doc section 7.9).
    steering_enabled = True

    def __init__(
        self,
        *args,
        steering_factor: float = 1.0,
        frozen_pretrained_eval: bool = True,
        steering_patch_head: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.steering_factor = steering_factor
        self.frozen_pretrained_eval = frozen_pretrained_eval
        self.steering_patch_head = steering_patch_head

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.frozen_pretrained_eval:
            # Frozen backbone blocks must not flip back to train mode (and
            # re-enable drop-path) just because the top-level module did.
            set_frozen_submodules_to_eval(self)
        return self

    def forward(self, input: BatchedDatapoint):
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        num_frames = len(input.find_inputs)
        assert num_frames == 1

        # Text first: the contextual tokens feed both the steering adapters
        # (after pair gather) and, via the frozen resizer, the existing late
        # fusion path.
        text_outputs = self.backbone.forward_text_for_steering(
            input.find_text_batch, device=device
        )
        backbone_out.update(text_outputs)

        find_input = input.find_inputs[0]
        find_target = input.find_targets[0]

        # Positive-pair contract: the collated batch already contains one
        # image per pair, so no pair materialization (images[img_ids]) is
        # needed; only the text must be gathered from the deduplicated rows.
        validate_positive_pair_batch(
            input.img_batch, find_input.img_ids, find_input.text_ids
        )
        # (L, U, D) -> (P, L, D) for the seq-first contextual tokens, and
        # (U, L) -> (P, L) for the batch-first padding mask.
        pair_text = text_outputs["language_contextual"][
            :, find_input.text_ids
        ].transpose(0, 1)
        pair_padding_mask = text_outputs["language_mask"][find_input.text_ids]

        conditioned_image_out = self.backbone.forward_conditioned_image(
            input.img_batch,
            steering_text=pair_text,
            steering_padding_mask=pair_padding_mask,
            steering_factor=self.steering_factor,
            return_patch_features=self.steering_patch_head is not None,
        )
        patch_logits = None
        if self.steering_patch_head is not None:
            patch_features = conditioned_image_out.pop("steering_patch_features")
            patch_logits = self.steering_patch_head(patch_features)
        backbone_out.update(conditioned_image_out)

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )

        if find_input.input_points is not None and find_input.input_points.numel() > 0:
            print("Warning: Point prompts are ignored in PCS.")

        num_interactive_steps = 0 if self.training else self.num_interactive_steps_val
        geometric_prompt = Prompt(
            box_embeddings=find_input.input_boxes,
            box_mask=find_input.input_boxes_mask,
            box_labels=find_input.input_boxes_label,
        )

        # Init vars that are shared across the loop.
        stage_outs = []
        for cur_step in range(num_interactive_steps + 1):
            if cur_step > 0:
                # We sample interactive geometric prompts (boxes, points)
                geometric_prompt, _ = self.interactive_prompt_sampler.sample(
                    geo_prompt=geometric_prompt,
                    find_target=find_target,
                    previous_out=stage_outs[-1],
                )
            out = self.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=find_target,
                geometric_prompt=geometric_prompt.clone(),
            )
            # Patch localization is computed once per image-text pair. Attach
            # it only to the final interactive output so a multi-step loss does
            # not accidentally count the same auxiliary objective repeatedly.
            if cur_step == num_interactive_steps and patch_logits is not None:
                out["steering_patch_logits"] = patch_logits
            stage_outs.append(out)

        previous_stages_out.append(stage_outs)
        return previous_stages_out
