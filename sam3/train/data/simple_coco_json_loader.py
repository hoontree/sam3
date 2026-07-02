# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Simple COCO JSON Loader for basic image + mask segmentation
Creates annotations on-the-fly from mask images
"""

import json
import os
from typing import Dict, List

import numpy as np
from PIL import Image as PILImage
from pycocotools import mask as mask_util


class SimpleMaskCocoLoader:
    """
    Simple COCO-style loader that reads images and their corresponding masks.

    Expected JSON format:
    {
        "images": [
            {"id": 0, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 1, "file_name": "image2.jpg", "width": 800, "height": 600},
            ...
        ],
        "categories": [
            {"id": 1, "name": "object"}
        ]
    }

    Masks are automatically loaded from masks/ directory.
    """

    def __init__(self, annotation_file: str, masks_dir: str = None):
        """
        Args:
            annotation_file: Path to JSON file with image list
            masks_dir: Directory containing mask images (if None, assumes masks/ next to images/)
        """
        with open(annotation_file, "r") as f:
            self.data = json.load(f)

        self.images = {img["id"]: img for img in self.data["images"]}
        self.categories = self.data.get("categories", [{"id": 1, "name": "object"}])
        self._cat_id_to_name = {cat["id"]: cat["name"] for cat in self.categories}
        self._sorted_cat_ids = sorted([cat["id"] for cat in self.categories])

        # Determine masks directory
        if masks_dir is None:
            data_root = os.path.dirname(os.path.dirname(annotation_file))
            self.masks_dir = os.path.join(data_root, "masks")
        else:
            self.masks_dir = masks_dir

        print(f"Loaded {len(self.images)} images from {annotation_file}")
        print(f"Looking for masks in: {self.masks_dir}")

    def getDatapointIds(self) -> List[int]:
        """Return all image IDs."""
        return list(self.images.keys())

    def loadImagesFromDatapoint(self, datapoint_id: int) -> List[Dict]:
        """Load image metadata for a datapoint."""
        if datapoint_id not in self.images:
            return []

        img_data = self.images[datapoint_id].copy()

        # Ensure required metadata fields are present
        if "original_img_id" not in img_data:
            img_data["original_img_id"] = img_data["id"]
        if "coco_img_id" not in img_data:
            img_data["coco_img_id"] = img_data["id"]

        return [img_data]

    def loadQueriesAndAnnotationsFromDatapoint(self, datapoint_id: int):
        """
        Load annotations from mask file for a datapoint.

        Returns:
            queries: List of query dictionaries
            annotations: List of annotation dictionaries with masks
        """
        if datapoint_id not in self.images:
            return [], []

        img_info = self.images[datapoint_id]
        img_name = os.path.basename(img_info["file_name"])
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        annotations = []

        # Load mask if exists
        if os.path.exists(mask_path):
            mask = PILImage.open(mask_path)
            mask_np = np.array(mask)

            # Get image dimensions
            h, w = img_info.get("height", mask_np.shape[0]), img_info.get(
                "width", mask_np.shape[1]
            )

            # Parse mask into instances
            annotations = self._parse_mask_to_annotations(mask_np, datapoint_id, h, w)

        # Create queries for each category
        queries = []
        for cat_id in self._sorted_cat_ids:
            # For now, we assume all annotations in this simple loader belong to all categories
            # or we are in a single-category setup.
            # In simple mask loading, we usually have one mask per image representing one category.
            # If there are multiple categories in JSON, we might need a more complex mapping.
            # But here we at least populate the query_text correctly.
            query = {
                "id": len(queries),
                "query_text": self._cat_id_to_name[cat_id],
                "image_id": datapoint_id,
                "input_box": None,
                "input_box_label": None,
                "object_ids_output": [
                    ann["id"] for ann in annotations if ann.get("category_id") == cat_id
                ],
                "is_exhaustive": True,
                "is_pixel_exhaustive": True,
                "query_processing_order": 0,
                "original_cat_id": cat_id,
            }
            # Only add query if it has annotations or if we want to include negatives (implicitly True here)
            queries.append(query)

        return queries, annotations

    def _parse_mask_to_annotations(
        self, mask_np: np.ndarray, image_id: int, img_height: int, img_width: int
    ) -> List[Dict]:
        """
        Parse mask into COCO-style annotations.

        Args:
            mask_np: Mask array (H, W)
            image_id: Image ID
            img_height: Image height
            img_width: Image width

        Returns:
            List of annotation dictionaries
        """
        annotations = []

        # Detect mask type
        unique_vals = np.unique(mask_np)

        # Binary mask (0 and 255 or 0 and 1)
        if len(unique_vals) <= 2:
            if 255 in unique_vals:
                instance_ids = [255]
            elif 1 in unique_vals:
                instance_ids = [1]
            else:
                return []
        else:
            # Multi-instance mask
            instance_ids = unique_vals[unique_vals > 0]

        # Process each instance
        for ann_id, instance_id in enumerate(instance_ids):
            instance_mask = (mask_np == instance_id).astype(np.uint8)

            if instance_mask.sum() == 0:
                continue

            # Compute bbox
            rows = np.any(instance_mask, axis=1)
            cols = np.any(instance_mask, axis=0)

            if not rows.any() or not cols.any():
                continue

            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]

            # Store bbox in absolute pixel coordinates [xmin, ymin, xmax, ymax]
            # This is expected by SAM3's normalization transforms
            bbox = [
                float(xmin),
                float(ymin),
                float(xmax),
                float(ymax),
            ]

            area = float(instance_mask.sum())

            # Encode mask to RLE for efficient storage
            # pycocotools expects Fortran-order uint8 array
            instance_mask_fortran = np.asfortranarray(instance_mask.astype(np.uint8))
            rle = mask_util.encode(instance_mask_fortran)

            # If rle is a list (can happen if multiple masks are encoded), take the first one
            if isinstance(rle, list):
                rle = rle[0]

            # We keep rle["counts"] as is (likely bytes) for runtime performance.
            # Convert to string only if you intend to save this directly to JSON.

            # Assign category ID
            # If the mask value (instance_id) itself is one of the category IDs, use it.
            # Otherwise, use the first category ID available.
            cat_id = self._sorted_cat_ids[0] if self._sorted_cat_ids else 1
            if instance_id in self._sorted_cat_ids:
                cat_id = int(instance_id)

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": area,
                "segmentation": rle,
                "iscrowd": 0,
                "is_crowd": False,
                "object_id": ann_id,
                "frame_index": 0,
                "source": "mask_file",
            }
            annotations.append(ann)

        return annotations


def create_simple_annotation_file(images_dir: str, output_json: str):
    """
    Utility function to create a simple annotation JSON from an images directory.

    Args:
        images_dir: Directory containing images
        output_json: Path to output JSON file
    """
    from PIL import Image

    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for idx, fname in enumerate(sorted(os.listdir(images_dir))):
        if os.path.splitext(fname.lower())[1] not in valid_extensions:
            continue

        img_path = os.path.join(images_dir, fname)
        try:
            with Image.open(img_path) as img:
                width, height = img.size

            images.append(
                {
                    "id": idx,
                    "file_name": fname,
                    "width": width,
                    "height": height,
                    "original_img_id": idx,
                    "coco_img_id": idx,
                }
            )
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    data = {
        "images": images,
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [],  # Empty, will be loaded from masks
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Created annotation file: {output_json}")
    print(f"Total images: {len(images)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create simple COCO annotation file")
    parser.add_argument("--images_dir", required=True, help="Directory with images")
    parser.add_argument("--output_json", required=True, help="Output JSON path")

    args = parser.parse_args()
    create_simple_annotation_file(args.images_dir, args.output_json)
