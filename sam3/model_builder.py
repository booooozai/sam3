# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import os
from dataclasses import dataclass
from typing import Optional

import pkg_resources
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr
from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.sam3_image import Sam3Image, Sam3ImageOnVideoMultiGPU
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from sam3.model.steering import GatedVisionLanguageAdapter
from sam3.model.steersam_backbone import SteerSAMVLBackbone
from sam3.model.steersam_image import SteerSAMImage
from sam3.model.steersam_neck import SteerSAMViTDetNeck
from sam3.model.steersam_patch_head import SteerSAMPatchHead
from sam3.model.steersam_text_encoder import SteerSAMTextEncoder
from sam3.model.steersam_vit import SteerableViT
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam.transformer import RoPEAttention


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(
    compile_mode=None,
    image_size=1008,
    enable_steering=False,
    steering_layers=(7, 15, 23, 31),
    steering_num_heads=16,
    steering_head_dim=64,
    steering_dropout=0.0,
):
    """Create ViT backbone for visual feature extraction."""
    vit_cls = SteerableViT if enable_steering else ViT
    return vit_cls(
        img_size=image_size,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
        **(
            dict(
                steering_layers=tuple(steering_layers),
                steering_num_heads=steering_num_heads,
                steering_head_dim=steering_head_dim,
                steering_dropout=steering_dropout,
            )
            if enable_steering
            else {}
        ),
    )


def _create_vit_neck(
    position_encoding,
    vit_backbone,
    enable_inst_interactivity=False,
    enable_steering=False,
):
    """Create ViT neck for feature pyramid."""
    neck_cls = SteerSAMViTDetNeck if enable_steering else Sam3DualViTDetNeck
    return neck_cls(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder, enable_steering=False):
    """Create visual-language backbone."""
    backbone_cls = SteerSAMVLBackbone if enable_steering else SAM3VLBackbone
    return backbone_cls(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder(resolution=1008) -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=resolution,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None):
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding()
    # Create CX block for fuser
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
    enable_steering=False,
    steering_factor=1.0,
    frozen_pretrained_eval=True,
    enable_patch_supervision=False,
    patch_head_zero_init=True,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    matcher = None
    if not eval_mode:
        from sam3.train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )
    common_params["matcher"] = matcher
    if enable_steering:
        steering_patch_head = (
            SteerSAMPatchHead(input_dim=1024, zero_init=patch_head_zero_init)
            if enable_patch_supervision
            else None
        )
        model = SteerSAMImage(
            steering_factor=steering_factor,
            frozen_pretrained_eval=frozen_pretrained_eval,
            steering_patch_head=steering_patch_head,
            **common_params,
        )
    else:
        model = Sam3Image(**common_params)

    return model


def _create_tracker_maskmem_backbone():
    """Create the SAM3 Tracker memory encoder."""
    # Position encoding for mask memory backbone
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    # Mask processing components
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone


def _create_tracker_transformer():
    """Create the SAM3 Tracker transformer components."""
    # Self attention
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )

    # Cross attention
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )

    # Encoder layer
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )

    # Encoder
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )

    # Transformer wrapper
    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer


def build_tracker(
    apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None
) -> Sam3TrackerPredictor:
    """
    Build the SAM3 Tracker module for video tracking.

    Returns:
        Sam3TrackerPredictor: Wrapped SAM3 Tracker module
    """

    # Create model components
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    # Create the Tracker module
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        # SAM parameters
        multimask_output_in_sam=True,
        # Evaluation
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        # Multimask
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        # Additional settings
        always_start_from_first_ann_frame=False,
        # Mask overlap
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        # SAM decoder settings
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )

    return model


def _create_text_encoder(bpe_path: str, enable_steering: bool = False):
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    encoder_cls = SteerSAMTextEncoder if enable_steering else VETextEncoder
    return encoder_cls(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_vision_backbone(
    compile_mode=None,
    enable_inst_interactivity=True,
    resolution=1008,
    enable_steering=False,
    steering_layers=(7, 15, 23, 31),
    steering_num_heads=16,
    steering_head_dim=64,
    steering_dropout=0.0,
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=resolution)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(
        compile_mode=compile_mode,
        image_size=resolution,
        enable_steering=enable_steering,
        steering_layers=steering_layers,
        steering_num_heads=steering_num_heads,
        steering_head_dim=steering_head_dim,
        steering_dropout=steering_dropout,
    )
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
        enable_steering=enable_steering,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer(
    has_presence_token: bool = True, resolution=1008
) -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder(resolution=resolution)

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


@dataclass(frozen=True)
class CheckpointLoadResult:
    """Compatibility information collected while loading a model checkpoint."""

    missing_keys: list[str]
    unexpected_keys: list[str]
    skipped_dynamic_keys: list[str]


def _load_checkpoint(model, checkpoint_path):
    """Load a checkpoint and report missing, unexpected and dynamic keys."""
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    has_released_detector_prefix = any(key.startswith("detector.") for key in ckpt)
    if has_released_detector_prefix:
        # Released SAM3 checkpoints wrap the image model under ``detector``
        # (and optionally the interactive model under ``tracker``).
        sam3_image_ckpt = {
            key.removeprefix("detector."): value
            for key, value in ckpt.items()
            if key.startswith("detector.")
        }
        if model.inst_interactive_predictor is not None:
            sam3_image_ckpt.update(
                {
                    key.replace(
                        "tracker.", "inst_interactive_predictor.model.", 1
                    ): value
                    for key, value in ckpt.items()
                    if key.startswith("tracker.")
                }
            )
    else:
        # Trainer checkpoints store the model's state dict directly under the
        # optional top-level ``model`` key.  Keeping these names unchanged lets
        # a standalone eval config load a trained SteerSAM checkpoint without
        # also restoring optimizer/epoch state.
        sam3_image_ckpt = dict(ckpt)

    # RoPE frequencies are registered buffers whose shape depends on the
    # runtime image resolution.  They are regenerated when constructing the
    # model, so a checkpoint created at 1008 cannot be loaded into a model
    # configured for 672 (and vice versa).
    skipped_dynamic_keys = [
        key for key in sam3_image_ckpt if key.endswith(".freqs_cis")
    ]
    for key in skipped_dynamic_keys:
        del sam3_image_ckpt[key]
    if skipped_dynamic_keys:
        print(
            "Skipping resolution-dependent RoPE buffers from checkpoint: "
            f"{skipped_dynamic_keys}"
        )

    incompatible_keys = model.load_state_dict(sam3_image_ckpt, strict=False)
    missing_keys = list(incompatible_keys.missing_keys)
    unexpected_keys = list(incompatible_keys.unexpected_keys)
    if missing_keys or unexpected_keys:
        print(
            f"loaded {checkpoint_path} and found "
            "missing and/or unexpected keys:\n"
            f"{missing_keys=}\n{unexpected_keys=}"
        )
    return CheckpointLoadResult(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        skipped_dynamic_keys=skipped_dynamic_keys,
    )


def _validate_steersam_checkpoint_load(
    result: CheckpointLoadResult, model: Optional[nn.Module] = None
) -> None:
    """Reject checkpoint incompatibilities outside SteerSAM's explicit allowlist."""

    skipped_dynamic_keys = set(result.skipped_dynamic_keys)
    allowed_missing_fragments = ["steering_adapters"]
    if model is not None and getattr(model, "steering_patch_head", None) is not None:
        # Released SAM3 and older SteerSAM checkpoints predate the auxiliary
        # patch head. Its freshly initialized parameters are expected missing
        # keys when patch supervision is explicitly enabled.
        allowed_missing_fragments.append("steering_patch_head")
    unsupported_missing = [
        key
        for key in result.missing_keys
        if not any(fragment in key for fragment in allowed_missing_fragments)
        and key not in skipped_dynamic_keys
    ]

    # The released detector checkpoint also contains the optional SAM2 neck.
    # Image-only SteerSAM deliberately builds without that neck, so those keys
    # are known checkpoint extras rather than an architecture mismatch.
    allowed_unexpected_prefixes = ["backbone.vision_backbone.sam2_convs."]
    if model is not None and getattr(model, "segmentation_head", None) is None:
        # The released checkpoint contains the optional segmentation head.
        # Its keys are expected extras only when the requested model genuinely
        # omits that head; a model that constructed it remains strictly checked.
        allowed_unexpected_prefixes.append("segmentation_head.")
    if model is not None and getattr(model, "steering_patch_head", None) is None:
        # Permit loading a trained SteerSAM checkpoint with the auxiliary head
        # into an inference build that intentionally disables that head.
        allowed_unexpected_prefixes.append("steering_patch_head.")
    unsupported_unexpected = [
        key
        for key in result.unexpected_keys
        if not key.startswith(tuple(allowed_unexpected_prefixes))
    ]

    if unsupported_missing or unsupported_unexpected:
        raise RuntimeError(
            "Loading a SAM3 checkpoint into SteerSAM found unsupported "
            "incompatibilities: "
            f"missing={unsupported_missing}, "
            f"unexpected={unsupported_unexpected}."
        )


def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    if device == "cuda":
        model = model.cuda()
    if eval_mode:
        model.eval()
    return model


def _log_trainable_parameters(model: nn.Module, enable_steering: bool) -> None:
    """Print trainable/frozen parameter counts, split by steering adapters."""
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    message = (
        f"Parameters: {n_total:,} total, {n_train:,} trainable "
        f"({100.0 * n_train / max(n_total, 1):.2f}%), "
        f"{n_total - n_train:,} frozen"
    )
    if enable_steering:
        n_adapter = sum(
            p.numel()
            for name, p in model.named_parameters()
            if "steering_adapters" in name
        )
        n_adapter_train = sum(
            p.numel()
            for name, p in model.named_parameters()
            if "steering_adapters" in name and p.requires_grad
        )
        message += f"; steering adapters {n_adapter_train:,}/{n_adapter:,} trainable"
        if n_adapter != n_adapter_train:
            message += " (WARNING: some adapter parameters are frozen)"
    print(message)


def _freeze_sam3_task_modules(model: nn.Module) -> None:
    """Freeze the pretrained SAM3 task stack after the vision FPN.

    SteerSAM's strategy-2 training policy keeps the prompt-conditioned visual
    front end trainable (steering adapters, optional patch head, and FPN) but
    uses the pretrained SAM3 task stack as a fixed readout.  Keeping this as a
    builder option, rather than hard-coding it in ``SteerSAMImage``, preserves
    the original SAM3 and full downstream-finetuning behavior by default.
    """
    module_names = (
        "geometry_encoder",
        "transformer",
        "dot_prod_scoring",
        "instance_dot_prod_scoring",
        "class_embed",
        "instance_class_embed",
        "segmentation_head",
    )
    for name in module_names:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            module.requires_grad_(False)


def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    freeze_vision_backbone=False,
    freeze_vision_fpn=True,
    freeze_language_backbone=False,
    freeze_sam3_task_modules=False,
    resolution=1008,
    enable_steering=False,
    steering_layers=(7, 15, 23, 31),
    steering_num_heads=16,
    steering_head_dim=64,
    steering_dropout=0.0,
    steering_factor=1.0,
    frozen_pretrained_eval=True,
    enable_patch_supervision=False,
    patch_head_zero_init=True,
    compile=False,
):
    """
    Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path to model checkpoint
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        freeze_vision_backbone: Disable gradients for the pretrained visual
            trunk. By default this also freezes its FPN neck.
        freeze_vision_fpn: When ``freeze_vision_backbone`` is enabled, also
            freeze the FPN neck. Set to ``False`` to train the FPN while keeping
            the pretrained ViT trunk frozen.
        freeze_language_backbone: Disable gradients for the language backbone.
        freeze_sam3_task_modules: Disable gradients for the pretrained SAM3
            modules after the vision FPN (geometry encoder, fusion
            encoder/decoder and their detection heads, scoring modules, and
            segmentation head). This is off by default. SteerSAM strategy 2
            enables it so only adapters, the optional patch head, and the FPN
            remain trainable.
        resolution: Input image resolution used by the visual backbone and decoder.
        enable_steering: Build the SteerSAM image model (steering adapters in
            the vision trunk, text-first pair-batch forward). Off by default;
            the original SAM3 classes and behavior are used otherwise.
        steering_layers: ViT block indices (0-based, before which) that receive
            a steering adapter when enable_steering is on.
        steering_num_heads: Heads per steering adapter.
        steering_head_dim: Head dimension per steering adapter.
        steering_dropout: Dropout inside steering adapters.
        steering_factor: Multiplier on the tanh(alpha) gate at run time.
        frozen_pretrained_eval: When steering is enabled, keep all-frozen
            subtrees in eval mode during training (disables e.g. drop-path on
            the frozen ViT blocks).
        enable_patch_supervision: Add the SteerViT-style pre-FPN patch
            localization head. Requires ``enable_steering=True`` and is off by
            default so the original SAM3 path remains unchanged.
        patch_head_zero_init: Initialize the shared patch projection to zero,
            matching SteerViT's localization head initialization.
        compile_mode: To enable compilation, set to "default"

    Returns:
        A SAM3 image model
    """
    steering_layers = tuple(int(i) for i in steering_layers)
    if enable_patch_supervision and not enable_steering:
        raise ValueError(
            "enable_patch_supervision requires enable_steering=True because "
            "the head consumes prompt-conditioned pre-FPN ViT features."
        )
    if enable_steering and compile:
        raise NotImplementedError(
            "torch.compile is not supported for the SteerSAM steering path; "
            "build with compile=False."
        )
    if enable_steering and enable_inst_interactivity:
        # The SAM1-task interactive predictor shares the vision trunk but is
        # outside the SteerSAM design scope (steersam_design.md section 1);
        # refuse the untested combination instead of silently mis-steering it.
        raise NotImplementedError(
            "enable_inst_interactivity is not supported together with "
            "enable_steering."
        )
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    # Create visual components
    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(
        compile_mode=compile_mode,
        enable_inst_interactivity=enable_inst_interactivity,
        resolution=resolution,
        enable_steering=enable_steering,
        steering_layers=steering_layers,
        steering_num_heads=steering_num_heads,
        steering_head_dim=steering_head_dim,
        steering_dropout=steering_dropout,
    )

    # Create text components
    text_encoder = _create_text_encoder(bpe_path, enable_steering=enable_steering)

    # Create visual-language backbone
    backbone = _create_vl_backbone(
        vision_encoder, text_encoder, enable_steering=enable_steering
    )

    # Create transformer components
    transformer = _create_sam3_transformer(resolution=resolution)

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
        enable_steering=enable_steering,
        steering_factor=steering_factor,
        frozen_pretrained_eval=frozen_pretrained_eval,
        enable_patch_supervision=enable_patch_supervision,
        patch_head_zero_init=patch_head_zero_init,
    )
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint_result = _load_checkpoint(model, checkpoint_path)
        if enable_steering:
            _validate_steersam_checkpoint_load(checkpoint_result, model=model)

    # Freeze after checkpoint loading so the pretrained backbone remains intact
    # while autograd can omit its backward graph during fine-tuning.
    if freeze_vision_backbone:
        if freeze_vision_fpn:
            model.backbone.vision_backbone.requires_grad_(False)
        else:
            model.backbone.vision_backbone.trunk.requires_grad_(False)
    if freeze_language_backbone:
        model.backbone.language_backbone.requires_grad_(False)
    if freeze_sam3_task_modules:
        _freeze_sam3_task_modules(model)
    if enable_steering:
        # Freezing the vision backbone also catches the adapters nested in the
        # trunk; re-enable them explicitly so the steering path can train.
        n_adapter_modules = 0
        for module in model.modules():
            if isinstance(module, GatedVisionLanguageAdapter):
                module.requires_grad_(True)
                n_adapter_modules += 1
        if n_adapter_modules == 0:
            raise RuntimeError(
                "enable_steering is on but no steering adapters were found in "
                "the model."
            )
        _log_trainable_parameters(model, enable_steering=True)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model


def download_ckpt_from_hf():
    SAM3_MODEL_ID = "facebook/sam3"
    SAM3_CKPT_NAME = "sam3.pt"
    SAM3_CFG_NAME = "config.json"
    _ = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CFG_NAME)
    checkpoint_path = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CKPT_NAME)
    return checkpoint_path


def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    """
    Build SAM3 dense tracking model.

    Args:
        checkpoint_path: Optional path to checkpoint file
        bpe_path: Path to the BPE tokenizer file

    Returns:
        Sam3VideoInferenceWithInstanceInteractivity: The instantiated dense tracking model
    """
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    # Build Tracker module
    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)

    # Build Detector components
    visual_neck = _create_vision_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
    segmentation_head: UniversalSegmentationHead = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    # Create main dot product scoring
    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
    )

    # Build Detector module
    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
    )

    # Build the main SAM3 video model
    if apply_temporal_disambiguation:
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=16,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )
    else:
        # a version without any heuristics for ablation studies
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=0,
            hotstart_unmatch_thresh=0,
            hotstart_dup_thresh=0,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=0,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )

    # Load checkpoint if provided
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt, strict=strict_state_dict_loading
        )
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    model.to(device=device)
    return model


def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )
