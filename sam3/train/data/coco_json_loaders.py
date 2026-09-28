# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from pycocotools import mask as mask_util


# ============================================================================
# Utility Functions
# ============================================================================


def convert_boxlist_to_normalized_tensor(box_list, image_width, image_height):
    """
    Converts a list of bounding boxes to a normalized PyTorch tensor.

    Args:
        box_list (list of list or tuples): Each box is [x_min, y_min, x_max, y_max].
        image_width (int or float): Width of the image.
        image_height (int or float): Height of the image.

    Returns:
        torch.Tensor: Normalized tensor of shape (N, 4), values in [0, 1].
    """
    boxes = torch.tensor(box_list, dtype=torch.float32)
    boxes[:, [0, 2]] /= image_width  # x_min, x_max
    boxes[:, [1, 3]] /= image_height  # y_min, y_max
    boxes = boxes.clamp(0, 1)
    return boxes


def load_coco_and_group_by_image(json_path: str) -> Tuple[List[Dict], Dict[int, str]]:
    """
    Load COCO JSON file and group annotations by image.

    Args:
        json_path (str): Path to COCO JSON file.

    Returns:
        Tuple containing:
            - List of dicts with 'image' and 'annotations' keys
            - Dict mapping category IDs to category names
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    sorted_image_ids = sorted(images.keys())

    grouped = []
    for image_id in sorted_image_ids:
        image_info = images[image_id]
        grouped.append(
            {"image": image_info, "annotations": anns_by_image.get(image_id, [])}
        )

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    return grouped, cat_id_to_name


def ann_to_rle(segm, im_info: Dict) -> Dict:
    """
    Convert annotation which can be polygons or uncompressed RLE to RLE.

    Args:
        segm: Segmentation data (polygon list or RLE dict)
        im_info (dict): Image info containing 'height' and 'width'

    Returns:
        RLE encoded segmentation
    """
    h, w = im_info["height"], im_info["width"]

    if isinstance(segm, list):
        # Polygon - merge all parts into one mask RLE code
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif isinstance(segm["counts"], list):
        # Uncompressed RLE
        rle = mask_util.frPyObjects(segm, h, w)
    else:
        # Already RLE
        rle = segm

    return rle


# ============================================================================
# COCO Training API
# ============================================================================


class COCO_FROM_JSON:
    """
    COCO training API for loading box-only annotations from JSON.
    Groups all annotations per image and creates queries per category.
    """

    def __init__(
        self,
        annotation_file,
        prompts=None,
        include_negatives=True,
        category_chunk_size=None,
        max_images: Optional[int] = None,
    ):
        """
        Initialize the COCO training API.

        Args:
            annotation_file (str): Path to COCO JSON annotation file
            prompts: Optional custom prompts for categories
            include_negatives (bool): Whether to include negative examples (categories with no instances)
        """
        self._raw_data, self._cat_idx_to_text = load_coco_and_group_by_image(
            annotation_file
        )
        if max_images is not None:
            max_images = int(max_images)
            if max_images <= 0:
                raise ValueError(
                    f"max_images must be a positive integer or None, got {max_images}"
                )
            # ``_raw_data`` is deterministically sorted by original COCO image
            # id.  Limit at this level, before category chunks are expanded, so
            # a smoke evaluation retains every category prompt for each chosen
            # source image rather than truncating arbitrary image-chunk rows.
            self._raw_data = self._raw_data[:max_images]
        self.max_images = max_images
        self._sorted_cat_ids = sorted(list(self._cat_idx_to_text.keys()))
        self.prompts = None
        self.include_negatives = include_negatives
        self.category_chunk_size = (
            category_chunk_size
            if category_chunk_size is not None
            else len(self._sorted_cat_ids)
        )
        self.category_chunks = [
            self._sorted_cat_ids[i : i + self.category_chunk_size]
            for i in range(0, len(self._sorted_cat_ids), self.category_chunk_size)
        ]
        if prompts is not None:
            prompts = eval(prompts)
            self.prompts = {}
            for loc_dict in prompts:
                self.prompts[int(loc_dict["id"])] = loc_dict["name"]
            assert len(self.prompts) == len(
                self._sorted_cat_ids
            ), "Number of prompts must match number of categories"

    def getDatapointIds(self):
        """Return datapoints that can produce at least one find query.

        When negative queries are disabled, images without annotations (or a
        category chunk without any annotation) would otherwise produce an
        empty datapoint.  ``Sam3ImageDataset`` rejects those datapoints with
        ``No find queries``.  Filter them here so the dataset does not sample
        known-invalid indices.
        """
        if self.include_negatives:
            return list(range(len(self._raw_data) * len(self.category_chunks)))

        datapoint_ids = []
        for image_idx, record in enumerate(self._raw_data):
            annotated_categories = {
                annotation["category_id"] for annotation in record["annotations"]
            }
            for chunk_idx, category_chunk in enumerate(self.category_chunks):
                if annotated_categories.intersection(category_chunk):
                    datapoint_ids.append(
                        image_idx * len(self.category_chunks) + chunk_idx
                    )
        return datapoint_ids

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        """
        Load queries and annotations for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            Tuple of (queries, annotations) lists
        """
        img_idx = idx // len(self.category_chunks)
        chunk_idx = idx % len(self.category_chunks)
        cat_chunk = self.category_chunks[chunk_idx]

        return self._load_queries_and_annotations_for_categories(img_idx, cat_chunk)

    def _load_queries_and_annotations_for_categories(self, img_idx, category_ids):
        """Build find queries and targets for selected categories of one image."""

        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,  # Single image per datapoint
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        annot_template = {
            "image_id": 0,
            "bbox": None,  # Normalized bbox in xywh
            "area": None,  # Unnormalized area
            "segmentation": None,  # RLE encoded
            "object_id": None,
            "is_crowd": None,
            "id": None,
        }

        raw_annotations = self._raw_data[img_idx]["annotations"]
        image_info = self._raw_data[img_idx]["image"]
        width, height = image_info["width"], image_info["height"]

        # Group annotations by category
        cat_id_to_anns = defaultdict(list)
        for ann in raw_annotations:
            cat_id_to_anns[ann["category_id"]].append(ann)

        annotations_by_cat_sorted = [
            (cat_id, cat_id_to_anns[cat_id]) for cat_id in category_ids
        ]

        for cat_id, anns in annotations_by_cat_sorted:
            if len(anns) == 0 and not self.include_negatives:
                continue

            cur_ann_ids = []

            # Create annotations for this category
            for ann in anns:
                annotation = annot_template.copy()
                annotation["id"] = len(annotations)
                annotation["object_id"] = annotation["id"]
                # LVIS follows the COCO annotation layout but omits iscrowd.
                annotation["is_crowd"] = ann.get("iscrowd", 0)

                normalized_boxes = convert_boxlist_to_normalized_tensor(
                    [ann["bbox"]], width, height
                )
                bbox = normalized_boxes[0]

                annotation["area"] = (bbox[2] * bbox[3]).item()
                annotation["bbox"] = bbox

                if (
                    "segmentation" in ann
                    and ann["segmentation"] is not None
                    and ann["segmentation"] != []
                ):
                    annotation["segmentation"] = ann_to_rle(
                        ann["segmentation"], im_info=image_info
                    )

                annotations.append(annotation)
                cur_ann_ids.append(annotation["id"])

            # Create query for this category
            query = query_template.copy()
            query["id"] = len(queries)
            query["original_cat_id"] = cat_id
            query["query_text"] = (
                self._cat_idx_to_text[cat_id]
                if self.prompts is None
                else self.prompts[cat_id]
            )
            query["object_ids_output"] = cur_ann_ids
            queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        """
        Load image information for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            List containing image info dict
        """
        img_idx = idx // len(self.category_chunks)
        return self._load_image_from_image_index(img_idx)

    def _load_image_from_image_index(self, img_idx):
        """Build the image metadata entry for a raw image index."""
        img_data = self._raw_data[img_idx]["image"]
        images = [
            {
                "id": 0,
                "file_name": img_data["file_name"],
                "original_img_id": img_data["id"],
                "coco_img_id": img_data["id"],
            }
        ]
        return images


class COCOPositivePairFromJSON(COCO_FROM_JSON):
    """Expose each positive image-category pair as one dataset record.

    An image containing multiple annotated categories appears once per distinct
    positive category. All instances of that category remain grouped under one
    find query. The index is built directly from annotations, so its size scales
    with the number of positive image-category pairs rather than the Cartesian
    product of images and the category vocabulary.

    This loader intentionally does not expose ``include_negatives`` or
    ``category_chunk_size``. It is the positive-pair data contract used by the
    SteerSAM image training pipeline.
    """

    def __init__(self, annotation_file, prompts=None):
        super().__init__(
            annotation_file=annotation_file,
            prompts=prompts,
            include_negatives=False,
            category_chunk_size=None,
        )

        known_categories = set(self._cat_idx_to_text)
        self._positive_pairs = []
        for image_idx, record in enumerate(self._raw_data):
            annotated_categories = {
                annotation["category_id"] for annotation in record["annotations"]
            }
            unknown_categories = annotated_categories - known_categories
            if unknown_categories:
                raise ValueError(
                    "Annotations reference categories missing from the category "
                    f"table: {sorted(unknown_categories)}"
                )
            self._positive_pairs.extend(
                (image_idx, category_id) for category_id in sorted(annotated_categories)
            )

    def getDatapointIds(self):
        """Return contiguous ids for all positive image-category pairs."""
        return range(len(self._positive_pairs))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        image_idx, category_id = self._positive_pairs[idx]
        queries, annotations = self._load_queries_and_annotations_for_categories(
            image_idx, [category_id]
        )
        if len(queries) != 1 or len(annotations) == 0:
            raise RuntimeError(
                "A positive-pair record must contain exactly one query and at "
                "least one annotation."
            )
        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        image_idx, _ = self._positive_pairs[idx]
        return self._load_image_from_image_index(image_idx)




class LVISVerifiedCategoriesFromJSON(COCO_FROM_JSON):
    """Load exactly the LVIS categories verified for each validation image.

    LVIS uses federated annotations: a missing annotation is a true negative
    only when its category appears in ``neg_category_ids``; categories outside
    the positive/negative verified sets must be ignored.  Querying the union of
    annotated positive categories and explicit negative categories is therefore
    both sufficient for official LVIS evaluation and substantially cheaper than
    evaluating all 1,203 categories on every image.

    One datapoint contains one image and all of its verified categories.  LVIS
    v1 validation contains at most a small number of verified categories per
    image, so this also avoids repeatedly encoding the same image.
    """

    def __init__(self, annotation_file, prompts=None):
        super().__init__(
            annotation_file=annotation_file,
            prompts=prompts,
            include_negatives=True,
            category_chunk_size=None,
        )
        if self.prompts is None:
            self._cat_idx_to_text = {
                category_id: name.replace("_", " ")
                for category_id, name in self._cat_idx_to_text.items()
            }

        known_categories = set(self._cat_idx_to_text)
        self._verified_categories = {}
        for image_idx, record in enumerate(self._raw_data):
            image_info = record["image"]
            positive_categories = {
                int(annotation["category_id"])
                for annotation in record["annotations"]
            }
            negative_categories = {
                int(category_id)
                for category_id in image_info.get("neg_category_ids", [])
            }
            overlap = positive_categories.intersection(negative_categories)
            if overlap:
                raise ValueError(
                    f"LVIS image {image_info['id']} marks categories as both "
                    f"positive and negative: {sorted(overlap)}."
                )
            verified = positive_categories.union(negative_categories)
            unknown = verified - known_categories
            if unknown:
                raise ValueError(
                    f"LVIS image {image_info['id']} references unknown categories: "
                    f"{sorted(unknown)}."
                )
            if verified:
                self._verified_categories[image_idx] = sorted(verified)

    def getDatapointIds(self):
        return sorted(self._verified_categories)

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        queries, annotations = self._load_queries_and_annotations_for_categories(
            idx, self._verified_categories[idx]
        )
        image_info = self._raw_data[idx]["image"]
        not_exhaustive = {
            int(category_id)
            for category_id in image_info.get("not_exhaustive_category_ids", [])
        }
        for query in queries:
            exhaustive = query["original_cat_id"] not in not_exhaustive
            query["is_exhaustive"] = exhaustive
            query["is_pixel_exhaustive"] = exhaustive
        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        image_info = self._raw_data[idx]["image"]
        # LVIS val is federated over images originating from both COCO
        # train2017 and val2017.  Preserve that split directory from coco_url;
        # using a single val2017 root silently drops a large fraction of LVIS.
        coco_split = os.path.basename(os.path.dirname(image_info["coco_url"]))
        return [
            {
                "id": 0,
                "file_name": os.path.join(
                    coco_split, os.path.basename(image_info["coco_url"])
                ),
                "original_img_id": image_info["id"],
                "coco_img_id": image_info["id"],
            }
        ]




class COCOAllCategoryPairsFromJSON(COCO_FROM_JSON):
    """Expose every image-category pair with a stable pair-level evaluation id.

    This loader is intended for standalone, exhaustive evaluation of a
    prompt-conditioned image model.  Every record contains one image and one
    category prompt, including categories that are absent from the image.  The
    original COCO image/category ids are preserved for standard AP, while
    ``coco_img_id`` is replaced by the one-based datapoint id so prompt-pair
    metrics can treat ``(image, category)`` as the evaluation unit.

    The ordering is deterministic: images are sorted by image id and categories
    are sorted by category id, matching ``build_prompt_pair_ground_truth`` in
    ``sam3.eval.prompt_pair_eval``.
    """

    def __init__(self, annotation_file, prompts=None, max_images=None):
        super().__init__(
            annotation_file=annotation_file,
            prompts=prompts,
            include_negatives=True,
            category_chunk_size=1,
            max_images=max_images,
        )

    def loadImagesFromDatapoint(self, idx):
        images = super().loadImagesFromDatapoint(idx)
        assert len(images) == 1
        images[0]["coco_img_id"] = int(idx) + 1
        return images


# ============================================================================
# SAM3 Evaluation APIs
# ============================================================================


class SAM3_EVAL_API_FROM_JSON_NP:
    """
    SAM3 evaluation API for loading noun phrase queries from JSON.
    """

    def __init__(self, annotation_file):
        """
        Initialize the SAM3 evaluation API.

        Args:
            annotation_file (str): Path to SAM3 JSON annotation file
        """
        with open(annotation_file, "r") as f:
            data = json.load(f)
        self._image_data = data["images"]

    def getDatapointIds(self):
        """Return all datapoint indices."""
        return list(range(len(self._image_data)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        """
        Load queries and annotations for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            Tuple of (queries, annotations) lists
        """
        cur_img_data = self._image_data[idx]
        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        # Create query
        query = query_template.copy()
        query["id"] = len(queries)
        query["original_cat_id"] = int(cur_img_data["queried_category"])
        query["query_text"] = cur_img_data["text_input"]
        query["object_ids_output"] = []
        queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        """
        Load image information for a specific datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            List containing image info dict
        """
        img_data = self._image_data[idx]
        images = [
            {
                "id": 0,
                "file_name": img_data["file_name"],
                "original_img_id": img_data["id"],
                "coco_img_id": img_data["id"],
            }
        ]
        return images


class SAM3_VEVAL_API_FROM_JSON_NP:
    """
    SAM3 video evaluation API for loading noun phrase queries from JSON.
    """

    def __init__(self, annotation_file):
        """
        Initialize the SAM3 video evaluation API.

        Args:
            annotation_file (str): Path to SAM3 video JSON annotation file
        """
        with open(annotation_file, "r") as f:
            data = json.load(f)

        assert "video_np_pairs" in data, "Incorrect data format"

        self._video_data = data["videos"]
        self._video_id_to_np_ids = defaultdict(list)
        self._cat_id_to_np = {}

        for cat_dict in data["categories"]:
            self._cat_id_to_np[cat_dict["id"]] = cat_dict["name"]

        for video_np_dict in data["video_np_pairs"]:
            self._video_id_to_np_ids[video_np_dict["video_id"]].append(
                video_np_dict["category_id"]
            )
            assert (
                self._cat_id_to_np[video_np_dict["category_id"]]
                == video_np_dict["noun_phrase"]
            ), "Category name does not match text input"

    def getDatapointIds(self):
        """Return all datapoint indices."""
        return list(range(len(self._video_data)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        """
        Load queries and annotations for a specific video datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            Tuple of (queries, annotations) lists
        """
        cur_vid_data = self._video_data[idx]
        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        all_np_ids = self._video_id_to_np_ids[cur_vid_data["id"]]

        for np_id in all_np_ids:
            text_input = self._cat_id_to_np[np_id]

            for i, image_path in enumerate(cur_vid_data["file_names"]):
                query = query_template.copy()
                query["id"] = len(queries)
                query["original_cat_id"] = np_id
                query["query_text"] = text_input
                query["image_id"] = i
                query["query_processing_order"] = i
                query["object_ids_output"] = []
                queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
        """
        Load image information for a specific video datapoint.

        Args:
            idx (int): Datapoint index

        Returns:
            List containing image info dicts for all frames
        """
        video_data = self._video_data[idx]
        images = [
            {
                "id": i,
                "file_name": file_name,
                "original_img_id": video_data["id"],
                "coco_img_id": video_data["id"],
            }
            for i, file_name in enumerate(video_data["file_names"])
        ]
        return images
