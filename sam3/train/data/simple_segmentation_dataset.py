# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Simple Segmentation Dataset for SAM3 Fine-tuning
Supports training with only images and mask images (binary or multi-class)
"""

import os
from typing import Optional, Callable, List

import torch
import numpy as np
from PIL import Image as PILImage

from .sam3_image_dataset import Sam3ImageDataset, Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy


class SimpleSegmentationDataset(Sam3ImageDataset):
    """
    Simple dataset for image + mask segmentation fine-tuning.
    
    Expects data structure:
    root/
        images/
            img1.jpg
            img2.jpg
            ...
        masks/
            img1.png  # Binary or multi-class masks
            img2.png
            ...
    
    Mask format:
    - Binary: 0=background, 255=foreground
    - Multi-class: 0=background, 1,2,3...=different instances/classes
    """
    
    def __init__(
        self,
        img_folder: str,
        ann_file: str,  # JSON file mapping images to masks
        transforms,
        max_ann_per_img: int = 100,
        multiplier: int = 1,
        training: bool = True,
        load_segmentation: bool = True,
        max_train_queries: int = 81,
        max_val_queries: int = 300,
        **kwargs
    ):
        # Call parent with basic initialization
        super().__init__(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            max_ann_per_img=max_ann_per_img,
            multiplier=multiplier,
            training=training,
            load_segmentation=load_segmentation,
            max_train_queries=max_train_queries,
            max_val_queries=max_val_queries,
            **kwargs
        )
    
    def _load_datapoint(self, index: int) -> Datapoint:
        """Load a single datapoint with image and mask."""
        id = self.ids[index].item()
        pil_images, img_metadata = self._load_images(id)
        queries, annotations = self.coco.loadQueriesAndAnnotationsFromDatapoint(id)
        
        # Load and process mask
        img_path = img_metadata[0]["file_name"]
        mask_path = self._get_mask_path(img_path)
        
        if os.path.exists(mask_path):
            mask = PILImage.open(mask_path)
            mask_np = np.array(mask)
            
            # Convert mask to instances with bboxes
            annotations = self._mask_to_annotations(mask_np, img_metadata[0])
            
            # Create a simple query for segmentation
            queries = self._create_segmentation_queries(annotations, img_metadata[0])
        
        return self.load_queries(pil_images, annotations, queries, img_metadata)
    
    def _get_mask_path(self, img_path: str) -> str:
        """Get corresponding mask path from image path."""
        img_name = os.path.basename(img_path)
        # Change extension to .png for mask
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(os.path.dirname(self.root), 'masks', mask_name)
        return mask_path
    
    def _mask_to_annotations(self, mask_np: np.ndarray, img_metadata: dict) -> List[dict]:
        """
        Convert mask to COCO-style annotations with bounding boxes.
        
        Args:
            mask_np: Numpy array of the mask (H, W)
            img_metadata: Image metadata dictionary
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        # Handle binary masks (0, 255)
        if mask_np.max() <= 255 and len(np.unique(mask_np)) <= 2:
            mask_binary = (mask_np > 0).astype(np.uint8)
            unique_ids = [1] if mask_binary.sum() > 0 else []
        else:
            # Multi-class masks
            unique_ids = np.unique(mask_np)
            unique_ids = unique_ids[unique_ids > 0]  # Remove background
        
        for idx, instance_id in enumerate(unique_ids):
            # Extract instance mask
            instance_mask = (mask_np == instance_id).astype(np.uint8)
            
            if instance_mask.sum() == 0:
                continue
            
            # Compute bounding box
            rows = np.any(instance_mask, axis=1)
            cols = np.any(instance_mask, axis=0)
            
            if not rows.any() or not cols.any():
                continue
                
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            # COCO format: [x, y, width, height] normalized
            # Note: xmax and ymax are inclusive indices, so we need +1 for width/height
            h, w = mask_np.shape
            bbox = [
                float(xmin) / w,
                float(ymin) / h,
                float(xmax - xmin + 1) / w,
                float(ymax - ymin + 1) / h
            ]
            
            area = float(instance_mask.sum())
            
            # Create annotation
            ann = {
                "id": idx,
                "image_id": img_metadata["id"],
                "category_id": 1,  # Single class for now
                "bbox": bbox,
                "area": area,
                "segmentation": instance_mask,  # Binary mask
                "iscrowd": 0,
                "is_crowd": False,
                "object_id": idx,
                "frame_index": 0,
                "source": "manual_mask"
            }
            annotations.append(ann)
        
        return annotations
    
    def _create_segmentation_queries(self, annotations: List[dict], img_metadata: dict) -> List[dict]:
        """
        Create simple segmentation queries for all instances.
        
        Args:
            annotations: List of annotation dictionaries
            img_metadata: Image metadata
            
        Returns:
            List of query dictionaries
        """
        queries = []
        
        if len(annotations) == 0:
            # Create a dummy query for images with no annotations
            query = {
                "id": 0,
                "query_text": "",  # No text prompt
                "image_id": img_metadata["id"],
                "input_box": None,
                "input_box_label": None,
                "object_ids_output": [],
                "is_exhaustive": True,
                "is_pixel_exhaustive": True,
                "query_processing_order": 0,
                "original_cat_id": 1,
            }
            queries.append(query)
        else:
            # Create one query that covers all instances
            query = {
                "id": 0,
                "query_text": "",  # No text prompt for simple segmentation
                "image_id": img_metadata["id"],
                "input_box": None,  # No box prompt
                "input_box_label": None,
                "object_ids_output": [ann["id"] for ann in annotations],
                "is_exhaustive": True,
                "is_pixel_exhaustive": True,
                "query_processing_order": 0,
                "original_cat_id": 1,
            }
            queries.append(query)
        
        return queries
