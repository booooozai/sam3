import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image as PILImage

from sam3.train.data.collator import collate_fn_api
from sam3.train.data.coco_json_loaders import (
    COCO_FROM_JSON,
    COCOPositivePairFromJSON,
)
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


if __name__ == "__main__":
    unittest.main()
