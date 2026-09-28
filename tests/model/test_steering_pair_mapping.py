"""CPU unit tests for the SteerSAM positive-pair batch contract (design 7.7)."""

import unittest

import torch

from sam3.model.steersam_image import validate_positive_pair_batch


class TestValidatePositivePairBatch(unittest.TestCase):
    def test_identity_mapping_passes(self):
        images = torch.zeros(3, 3, 8, 8)
        img_ids = torch.arange(3)
        text_ids = torch.tensor([0, 1, 1])
        validate_positive_pair_batch(images, img_ids, text_ids)

    def test_repeated_image_ids_are_rejected(self):
        # A one-image-many-queries loader would produce repeated img_ids.
        images = torch.zeros(3, 3, 8, 8)
        img_ids = torch.tensor([0, 0, 1])
        text_ids = torch.arange(3)
        with self.assertRaises(AssertionError):
            validate_positive_pair_batch(images, img_ids, text_ids)

    def test_length_mismatch_is_rejected(self):
        images = torch.zeros(3, 3, 8, 8)
        with self.assertRaises(AssertionError):
            validate_positive_pair_batch(images, torch.arange(2), torch.arange(3))
        with self.assertRaises(AssertionError):
            validate_positive_pair_batch(images, torch.arange(3), torch.arange(2))

    def test_single_pair_batch(self):
        validate_positive_pair_batch(
            torch.zeros(1, 3, 8, 8),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )


if __name__ == "__main__":
    unittest.main()
