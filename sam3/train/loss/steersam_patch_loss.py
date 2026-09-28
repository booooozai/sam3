# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""SteerViT-style patch localization objective for image-only SteerSAM."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import CORE_LOSS_KEY


def build_patch_distribution_targets(
    instance_masks: torch.Tensor,
    num_instances: torch.Tensor,
    output_size: Tuple[int, int],
    valid_instance_masks: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert packed instance masks into per-query patch distributions.

    All valid instances belonging to one positive image-category pair are
    unioned first.  Average pooling then records fractional patch occupancy,
    and normalization over the spatial grid produces the soft distribution
    used by SteerViT's patch-wise cross entropy.

    Returns:
        target_distribution: ``(P,H_p,W_p)``, sums to one for valid pairs.
        patch_occupancy: ``(P,H_p,W_p)`` in ``[0,1]``.
        valid_pairs: ``(P,)``; false when no valid foreground mask is present.
    """
    if instance_masks is None:
        raise ValueError(
            "SteerSAM patch supervision requires segmentation masks; make sure "
            "the dataset and collator use load_segmentation/with_seg_masks=True."
        )
    if instance_masks.ndim != 3:
        raise ValueError(
            "instance_masks must have shape (sum(Kp),H,W), got "
            f"{tuple(instance_masks.shape)}"
        )
    if num_instances.ndim != 1:
        raise ValueError(
            f"num_instances must have shape (P,), got {tuple(num_instances.shape)}"
        )
    counts = [int(value) for value in num_instances.detach().cpu().tolist()]
    if sum(counts) != instance_masks.shape[0]:
        raise ValueError(
            "Packed mask/count mismatch: "
            f"sum(num_instances)={sum(counts)} but masks={instance_masks.shape[0]}."
        )

    if valid_instance_masks is None:
        valid_instance_masks = torch.ones(
            instance_masks.shape[0], dtype=torch.bool, device=instance_masks.device
        )
    else:
        valid_instance_masks = valid_instance_masks.to(
            device=instance_masks.device, dtype=torch.bool
        )
        if valid_instance_masks.ndim != 1 or len(valid_instance_masks) != len(
            instance_masks
        ):
            raise ValueError(
                "valid_instance_masks must have one entry per packed mask."
            )

    pair_unions = []
    mask_chunks = torch.split(instance_masks.bool(), counts)
    valid_chunks = torch.split(valid_instance_masks, counts)
    spatial_shape = instance_masks.shape[-2:]
    for masks, valid in zip(mask_chunks, valid_chunks):
        masks = masks[valid]
        if masks.shape[0] == 0:
            union = torch.zeros(
                spatial_shape, dtype=torch.bool, device=instance_masks.device
            )
        else:
            union = masks.any(dim=0)
        pair_unions.append(union)

    if not pair_unions:
        raise ValueError("Patch supervision received an empty pair batch.")

    union_masks = torch.stack(pair_unions, dim=0).unsqueeze(1).float()
    occupancy = F.adaptive_avg_pool2d(union_masks, output_size=output_size).squeeze(1)
    mass = occupancy.flatten(1).sum(dim=1)
    valid_pairs = mass > eps
    target_distribution = occupancy / mass.clamp_min(eps)[:, None, None]
    # Invalid pairs are skipped by the loss, but zero them as an explicit and
    # safe contract for callers/metrics.
    target_distribution = torch.where(
        valid_pairs[:, None, None],
        target_distribution,
        torch.zeros_like(target_distribution),
    )
    return target_distribution, occupancy, valid_pairs


class SteerSAMPatchLocalizationLoss(nn.Module):
    """Soft spatial cross entropy and PMASS for pre-FPN patch logits."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, patch_logits: torch.Tensor, targets: dict) -> dict:
        if patch_logits.ndim != 3:
            raise ValueError(
                "patch_logits must have shape (P,H_p,W_p), got "
                f"{tuple(patch_logits.shape)}"
            )
        if patch_logits.shape[0] != len(targets["num_boxes"]):
            raise ValueError(
                "Patch logit/target pair mismatch: "
                f"{patch_logits.shape[0]} logits vs "
                f"{len(targets['num_boxes'])} targets."
            )

        target_distribution, occupancy, valid_pairs = build_patch_distribution_targets(
            instance_masks=targets["masks"],
            num_instances=targets["num_boxes"],
            valid_instance_masks=targets.get("is_valid_mask"),
            output_size=patch_logits.shape[-2:],
            eps=self.eps,
        )
        # Compute the distribution loss in float32 even under bf16 autocast.
        flat_logits = patch_logits.float().flatten(1)
        flat_targets = target_distribution.to(flat_logits.dtype).flatten(1)
        per_pair_loss = -(flat_targets * F.log_softmax(flat_logits, dim=-1)).sum(dim=-1)

        if valid_pairs.any():
            loss = per_pair_loss[valid_pairs].mean()
            probabilities = F.softmax(flat_logits, dim=-1)
            foreground_support = occupancy.flatten(1) > 0
            pmass = (
                (probabilities * foreground_support.to(probabilities.dtype))
                .sum(dim=-1)[valid_pairs]
                .mean()
            )
        else:
            # Preserve a differentiable zero so DDP sees the patch-head output
            # even for a pathological batch without valid segmentation masks.
            loss = flat_logits.sum() * 0.0
            pmass = flat_logits.detach().sum() * 0.0

        return {
            "loss_steering_patch": loss,
            "steering_pmass": pmass,
            "steering_valid_pair_fraction": valid_pairs.float().mean(),
        }


class SteerSAMPatchLossWrapper(nn.Module):
    """Compose the original SAM3 objective with one patch loss per stage.

    ``task_loss=None`` gives a matcher-free validation criterion.  This is
    useful because SAM3 evaluation outputs omit Hungarian ``indices``, whereas
    patch localization has a direct pair-to-mask correspondence.
    """

    def __init__(
        self,
        task_loss: Optional[nn.Module] = None,
        patch_loss_weight: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        if patch_loss_weight < 0:
            raise ValueError("patch_loss_weight must be non-negative.")
        self.task_loss = task_loss
        self.patch_loss_weight = float(patch_loss_weight)
        self.patch_loss = SteerSAMPatchLocalizationLoss(eps=eps)

    def forward(self, find_stages: SAM3Output, find_targets) -> dict:
        total_losses = (
            self.task_loss(find_stages, find_targets)
            if self.task_loss is not None
            else {}
        )

        if find_stages.loss_stages is not None:
            find_targets = [find_targets[i] for i in find_stages.loss_stages]

        patch_results = []
        with SAM3Output.iteration_mode(
            find_stages, iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        ) as last_stage_outputs:
            if len(last_stage_outputs) != len(find_targets):
                raise ValueError("SteerSAM output/target stage count mismatch.")
            for outputs, targets in zip(last_stage_outputs, find_targets):
                if "steering_patch_logits" not in outputs:
                    raise KeyError(
                        "steering_patch_logits is missing. Build the model with "
                        "enable_patch_supervision=True."
                    )
                patch_results.append(
                    self.patch_loss(outputs["steering_patch_logits"], targets)
                )

        patch_loss = torch.stack(
            [result["loss_steering_patch"] for result in patch_results]
        ).mean()
        pmass = torch.stack(
            [result["steering_pmass"] for result in patch_results]
        ).mean()
        valid_fraction = torch.stack(
            [result["steering_valid_pair_fraction"] for result in patch_results]
        ).mean()

        if CORE_LOSS_KEY in total_losses:
            total_losses[CORE_LOSS_KEY] = (
                total_losses[CORE_LOSS_KEY] + self.patch_loss_weight * patch_loss
            )
        else:
            total_losses[CORE_LOSS_KEY] = self.patch_loss_weight * patch_loss

        if "loss_steering_patch" in total_losses:
            raise KeyError("task_loss already returned loss_steering_patch.")
        total_losses.update(
            {
                "loss_steering_patch": patch_loss,
                "steering_pmass": pmass,
                "steering_valid_pair_fraction": valid_fraction,
            }
        )
        return total_losses
