import json
import tempfile
import unittest
from functools import partial
from pathlib import Path

import torch
from PIL import Image as PILImage
from sam3.eval.prompt_pair_eval import build_prompt_pair_ground_truth
from sam3.train.data.coco_json_loaders import (
    COCO_FROM_JSON,
    COCOAllCategoryPairsFromJSON,
    COCOPositivePairFromJSON,
)

from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset


class _ToFixedTensor:
    def __call__(self, datapoint, epoch):
        del epoch
        for image in datapoint.images:
            image.data = torch.zeros(3, 8, 8)
        return datapoint


class TestCOCOPositivePairFromJSON(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.annotation_file = Path(self._tmpdir.name) / "instances.json"
        data = {
            "images": [
                {"id": 10, "file_name": "image_10.jpg", "width": 100, "height": 200},
                {"id": 20, "file_name": "image_20.jpg", "width": 80, "height": 60},
                {"id": 30, "file_name": "image_30.jpg", "width": 50, "height": 50},
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
                {"id": 3, "name": "person"},
            ],
            "annotations": [
                {
                    "id": 101,
                    "image_id": 10,
                    "category_id": 1,
                    "bbox": [10, 20, 30, 40],
                    "iscrowd": 0,
                },
                {
                    "id": 102,
                    "image_id": 10,
                    "category_id": 1,
                    "bbox": [50, 60, 10, 20],
                    "iscrowd": 0,
                },
                {
                    "id": 103,
                    "image_id": 10,
                    "category_id": 2,
                    "bbox": [0, 0, 20, 20],
                    "iscrowd": 0,
                },
                {
                    "id": 104,
                    "image_id": 30,
                    "category_id": 2,
                    "bbox": [5, 5, 10, 10],
                    "iscrowd": 0,
                },
            ],
        }
        self.annotation_file.write_text(json.dumps(data), encoding="utf-8")
        for image in data["images"]:
            PILImage.new(
                "RGB", (image["width"], image["height"]), color=(0, 0, 0)
            ).save(Path(self._tmpdir.name) / image["file_name"])

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_builds_one_record_per_positive_image_category_pair(self):
        loader = COCOPositivePairFromJSON(str(self.annotation_file))

        self.assertEqual(list(loader.getDatapointIds()), [0, 1, 2])

        queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(0)
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["query_text"], "cat")
        self.assertEqual(queries[0]["original_cat_id"], 1)
        self.assertEqual(queries[0]["object_ids_output"], [0, 1])
        self.assertEqual(len(annotations), 2)
        torch.testing.assert_close(
            annotations[0]["bbox"],
            torch.tensor([0.1, 0.1, 0.3, 0.2], dtype=torch.float32),
        )

        dog_queries, dog_annotations = loader.loadQueriesAndAnnotationsFromDatapoint(1)
        self.assertEqual(dog_queries[0]["query_text"], "dog")
        self.assertEqual(len(dog_annotations), 1)

        second_image_queries, _ = loader.loadQueriesAndAnnotationsFromDatapoint(2)
        self.assertEqual(second_image_queries[0]["query_text"], "dog")
        self.assertEqual(
            loader.loadImagesFromDatapoint(2)[0]["file_name"], "image_30.jpg"
        )

    def test_original_loader_still_groups_positive_categories(self):
        loader = COCO_FROM_JSON(
            str(self.annotation_file),
            include_negatives=False,
            category_chunk_size=None,
        )

        self.assertEqual(loader.getDatapointIds(), [0, 2])
        queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(0)
        self.assertEqual([query["query_text"] for query in queries], ["cat", "dog"])
        self.assertEqual(len(annotations), 3)

    def test_single_category_chunks_cover_all_image_category_pairs(self):
        loader = COCO_FROM_JSON(
            str(self.annotation_file),
            include_negatives=True,
            category_chunk_size=1,
        )

        # Three images x three categories. Every record contains exactly one
        # query, including categories that are absent from the image.
        self.assertEqual(loader.getDatapointIds(), list(range(9)))
        for datapoint_id in loader.getDatapointIds():
            queries, _ = loader.loadQueriesAndAnnotationsFromDatapoint(datapoint_id)
            self.assertEqual(len(queries), 1)

        negative_queries, negative_annotations = (
            loader.loadQueriesAndAnnotationsFromDatapoint(2)
        )
        self.assertEqual(negative_queries[0]["query_text"], "person")
        self.assertEqual(negative_queries[0]["object_ids_output"], [])
        self.assertEqual(negative_annotations, [])

    def test_max_images_limits_source_images_before_category_chunk_expansion(self):
        loader = COCO_FROM_JSON(
            str(self.annotation_file),
            include_negatives=True,
            category_chunk_size=2,
            max_images=2,
        )

        # Two source images, each with two category chunks.  A smoke limit is
        # applied before expansion, hence all three categories are preserved
        # for both images.
        self.assertEqual(len(loader._raw_data), 2)
        self.assertEqual(loader.getDatapointIds(), [0, 1, 2, 3])
        self.assertEqual(
            [query["original_cat_id"] for query in loader.loadQueriesAndAnnotationsFromDatapoint(0)[0]],
            [1, 2],
        )
        self.assertEqual(
            [query["original_cat_id"] for query in loader.loadQueriesAndAnnotationsFromDatapoint(1)[0]],
            [3],
        )
        self.assertEqual(loader.loadImagesFromDatapoint(2)[0]["original_img_id"], 20)

    def test_max_images_rejects_non_positive_values(self):
        with self.assertRaisesRegex(ValueError, "max_images"):
            COCO_FROM_JSON(str(self.annotation_file), max_images=0)

    def test_dataset_and_collator_keep_one_image_row_per_pair(self):
        dataset = Sam3ImageDataset(
            img_folder=self._tmpdir.name,
            ann_file=str(self.annotation_file),
            transforms=[_ToFixedTensor()],
            max_ann_per_img=10,
            multiplier=1,
            training=True,
            coco_json_loader=COCOPositivePairFromJSON,
        )

        collated = collate_fn_api(
            [dataset[index] for index in range(len(dataset))],
            dict_key="test",
        )["test"]

        self.assertEqual(tuple(collated.img_batch.shape), (3, 3, 8, 8))
        torch.testing.assert_close(
            collated.find_inputs[0].img_ids, torch.tensor([0, 1, 2])
        )
        self.assertEqual(collated.find_text_batch, ["cat", "dog"])
        torch.testing.assert_close(
            collated.find_inputs[0].text_ids, torch.tensor([0, 1, 1])
        )
        torch.testing.assert_close(
            collated.find_targets[0].num_boxes, torch.tensor([2, 1, 1])
        )

    def test_all_category_validation_pairs_keep_the_same_batch_contract(self):
        dataset = Sam3ImageDataset(
            img_folder=self._tmpdir.name,
            ann_file=str(self.annotation_file),
            transforms=[_ToFixedTensor()],
            max_ann_per_img=10,
            multiplier=1,
            training=False,
            coco_json_loader=partial(
                COCO_FROM_JSON,
                include_negatives=True,
                category_chunk_size=1,
            ),
        )

        collated = collate_fn_api(
            [dataset[index] for index in range(len(dataset))],
            dict_key="test",
        )["test"]

        self.assertEqual(tuple(collated.img_batch.shape), (9, 3, 8, 8))
        torch.testing.assert_close(collated.find_inputs[0].img_ids, torch.arange(9))
        self.assertEqual(len(collated.find_inputs[0].text_ids), 9)
        self.assertEqual((collated.find_targets[0].num_boxes == 0).sum().item(), 6)

    def test_exhaustive_eval_loader_assigns_stable_prompt_pair_ids(self):
        loader = COCOAllCategoryPairsFromJSON(str(self.annotation_file))

        self.assertEqual(list(loader.getDatapointIds()), list(range(9)))
        for datapoint_id in loader.getDatapointIds():
            metadata = loader.loadImagesFromDatapoint(datapoint_id)[0]
            self.assertEqual(metadata["coco_img_id"], datapoint_id + 1)

        # Row-major ordering over sorted image/category ids.
        self.assertEqual(loader.loadImagesFromDatapoint(7)[0]["original_img_id"], 30)
        queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(7)
        self.assertEqual(queries[0]["original_cat_id"], 2)
        self.assertEqual(len(annotations), 1)

    def test_prompt_pair_ground_truth_matches_eval_loader_ids(self):
        output_path = Path(self._tmpdir.name) / "prompt_pairs.json"
        build_prompt_pair_ground_truth(str(self.annotation_file), str(output_path))
        pair_gt = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(pair_gt["images"]), 9)
        self.assertEqual(
            [image["id"] for image in pair_gt["images"]], list(range(1, 10))
        )
        self.assertTrue(
            all(image["is_instance_exhaustive"] for image in pair_gt["images"])
        )
        self.assertEqual(
            pair_gt["categories"],
            [
                {
                    "id": 1,
                    "name": "prompt_target",
                    "supercategory": "prompt_target",
                }
            ],
        )
        # image 10/cat, image 10/dog, image 30/dog map to pair ids 1, 2, 8.
        self.assertEqual(
            [annotation["image_id"] for annotation in pair_gt["annotations"]],
            [1, 1, 2, 8],
        )
        self.assertTrue(
            all(annotation["category_id"] == 1 for annotation in pair_gt["annotations"])
        )


if __name__ == "__main__":
    unittest.main()
