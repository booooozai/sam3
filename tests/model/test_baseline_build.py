"""CPU regression: ``enable_steering=False`` keeps the original SAM3 classes.

Design doc (steersam_design.md section 12, criterion 1): with the switch off,
the builder must construct the original SAM3 modules, register no steering
adapters anywhere, and keep the plain ``Sam3Image`` behavior. This test builds
the real (full-size) model once on CPU, checks the class selection, and frees
it immediately.

Run on CPU only; no checkpoint is loaded.
"""

import gc
import unittest
from pathlib import Path

from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.sam3_image import Sam3Image
from sam3.model.steering import GatedVisionLanguageAdapter
from sam3.model.steersam_backbone import SteerSAMVLBackbone
from sam3.model.steersam_image import SteerSAMImage
from sam3.model.steersam_neck import SteerSAMViTDetNeck
from sam3.model.steersam_text_encoder import SteerSAMTextEncoder
from sam3.model.steersam_vit import SteerableViT
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone

BPE_PATH = (
    Path(__file__).resolve().parents[2] / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
)


@unittest.skipUnless(BPE_PATH.exists(), "BPE vocabulary not found")
class TestBaselineBuildUnchanged(unittest.TestCase):
    def test_patch_supervision_requires_steering(self):
        from sam3.model_builder import build_sam3_image_model

        with self.assertRaisesRegex(ValueError, "requires enable_steering"):
            build_sam3_image_model(
                checkpoint_path=None,
                load_from_HF=False,
                device="cpu",
                enable_steering=False,
                enable_patch_supervision=True,
            )

    def test_enable_steering_false_builds_original_classes(self):
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            bpe_path=str(BPE_PATH),
            device="cpu",
            eval_mode=True,
            checkpoint_path=None,
            load_from_HF=False,
            resolution=336,
            enable_segmentation=False,
        )
        try:
            self.assertIsInstance(model, Sam3Image)
            self.assertNotIsInstance(model, SteerSAMImage)
            self.assertFalse(getattr(model, "steering_enabled", False))
            self.assertIsInstance(model.backbone, SAM3VLBackbone)
            self.assertNotIsInstance(model.backbone, SteerSAMVLBackbone)
            self.assertIsInstance(model.backbone.vision_backbone, Sam3DualViTDetNeck)
            self.assertNotIsInstance(model.backbone.vision_backbone, SteerSAMViTDetNeck)
            trunk = model.backbone.vision_backbone.trunk
            self.assertIsInstance(trunk, ViT)
            self.assertNotIsInstance(trunk, SteerableViT)
            self.assertIsInstance(model.backbone.language_backbone, VETextEncoder)
            self.assertNotIsInstance(
                model.backbone.language_backbone, SteerSAMTextEncoder
            )
            # The new strategy-2 task-stack freeze is opt-in. A normal SAM3
            # build must keep its pretrained downstream modules trainable.
            self.assertTrue(
                all(p.requires_grad for p in model.transformer.parameters())
            )
            self.assertTrue(
                all(p.requires_grad for p in model.dot_prod_scoring.parameters())
            )
            # No adapter is registered anywhere in the graph.
            self.assertEqual(
                [
                    m
                    for m in model.modules()
                    if isinstance(m, GatedVisionLanguageAdapter)
                ],
                [],
            )
        finally:
            del model
            gc.collect()


if __name__ == "__main__":
    unittest.main()
