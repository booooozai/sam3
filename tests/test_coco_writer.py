import json
import tempfile
import unittest
from pathlib import Path

from sam3.eval.coco_writer import PredictionDumper
from sam3.eval.prompt_pair_eval import PromptPairCGF1Evaluator


class TestPredictionDumper(unittest.TestCase):
    def test_online_topk_bounds_results_per_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dumper = PredictionDumper(
                dump_dir=tmpdir,
                postprocessor=None,
                maxdets=2,
                iou_type="bbox",
                online_topk=True,
            )
            dumper._dump(
                [
                    {
                        "image_id": 10,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                        "score": score,
                    }
                    for score in (0.1, 0.9, 0.5)
                ]
            )
            dumper._dump(
                [
                    {
                        "image_id": 20,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                        "score": 0.2,
                    }
                ]
            )

        self.assertEqual(len(dumper.dump), 0)
        self.assertEqual(len(dumper._online_heaps[10]), 2)
        self.assertEqual(
            sorted(element.val["score"] for element in dumper._online_heaps[10]),
            [0.5, 0.9],
        )
        self.assertEqual(len(dumper._online_heaps[20]), 1)

    def test_prompt_pair_evaluator_reports_cgf1_ilmcc_and_pmf1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_path = tmpdir / "instances.json"
            pair_gt_path = tmpdir / "prompt_pairs.json"
            prediction_path = tmpdir / "predictions.json"
            gt_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 10,
                                "file_name": "image.jpg",
                                "width": 20,
                                "height": 20,
                            }
                        ],
                        "categories": [
                            {"id": 1, "name": "cat"},
                            {"id": 2, "name": "dog"},
                        ],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 10,
                                "category_id": 1,
                                "bbox": [0, 0, 10, 10],
                                "area": 100,
                                "iscrowd": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            # Pair 1 is the positive cat prompt; pair 2 is a true-negative dog
            # prompt and intentionally has no prediction.
            prediction_path.write_text(
                json.dumps(
                    [
                        {
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [0, 0, 10, 10],
                            "score": 0.9,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            evaluator = PromptPairCGF1Evaluator(
                gt_path=str(gt_path),
                pair_gt_path=str(pair_gt_path),
                iou_type="bbox",
            )
            results = evaluator.evaluate(str(prediction_path))

        self.assertGreater(results["prompt_pair_bbox_cgF1_50_95"], 0.99)
        self.assertGreater(results["prompt_pair_bbox_IL_MCC"], 0.99)
        self.assertGreater(results["prompt_pair_bbox_pmF1_50_95"], 0.99)


if __name__ == "__main__":
    unittest.main()
