#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM (Segment Anything Model) integration for refining humerus contours.
Uses the approximate contour from traditional methods as a prompt for SAM.
"""

import numpy as np
from typing import Tuple, Optional, Union
import cv2

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment-anything not installed. SAM refinement will not be available.")
    print("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")


class SAMContourRefiner:
    """
    Refines humerus contours using Meta's Segment Anything Model (SAM).
    """
    
    def __init__(self, model_type: str = "vit_b", checkpoint_path: Optional[str] = None):
        """
        Initialize the SAM refiner.
        
        Args:
            model_type: SAM model type ('vit_h', 'vit_l', or 'vit_b')
            checkpoint_path: Path to the SAM checkpoint file
        """
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything package is required for SAM refinement")
        
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        
        if checkpoint_path:
            self._load_model()
    
    def _load_model(self):
        """Load the SAM model."""
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.predictor = SamPredictor(sam)
        print(f"SAM model loaded: {self.model_type}")
    
    def set_image(self, image: np.ndarray):
        """
        Set the image for SAM to process.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
        """
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded. Provide checkpoint_path.")
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image
        
        # Normalize to 0-255 uint8
        if image_rgb.dtype != np.uint8:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        
        # Store original shape for later rescaling
        self.original_shape = image.shape[:2]
        
        self.predictor.set_image(image_rgb)
    
    def refine_with_box(self, box: np.ndarray, center_point: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Refine contour using a bounding box prompt with optional center point.
        
        Args:
            box: Bounding box [x_min, y_min, x_max, y_max]
            center_point: Optional center point [x, y] to guide segmentation to interior
            
        Returns:
            Tuple of (refined_mask, confidence_score)
        """
        # If we have a center point, use it as a positive prompt to select interior
        if center_point is not None:
            point_coords = np.array([center_point])
            point_labels = np.array([1])  # 1 = positive (inside object)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True
            )
        else:
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=True  # Get multiple mask proposals
            )
        
        # Return the mask with highest confidence
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
    
    def refine_with_points(self, 
                          positive_points: np.ndarray,
                          negative_points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Refine contour using point prompts.
        
        Args:
            positive_points: Points inside the object (N, 2) array of [x, y]
            negative_points: Points outside the object (M, 2) array of [x, y]
            
        Returns:
            Tuple of (refined_mask, confidence_score)
        """
        if negative_points is not None:
            point_coords = np.vstack([positive_points, negative_points])
            point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
        else:
            point_coords = positive_points
            point_labels = np.ones(len(positive_points))
        
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
    
    def refine_with_mask(self, 
                        approximate_mask: np.ndarray,
                        use_box: bool = True) -> Tuple[np.ndarray, float]:
        """
        Refine contour using an approximate mask as prompt.
        This is the most useful mode for improving existing contours.
        
        Args:
            approximate_mask: Binary mask from traditional methods (H, W)
            use_box: Also use bounding box from the mask as additional prompt
            
        Returns:
            Tuple of (refined_mask, confidence_score)
        """
        # Convert mask to proper format
        mask_input = approximate_mask.astype(np.float32)
        
        # Optionally extract bounding box from mask
        box = None
        if use_box:
            box = self._mask_to_box(approximate_mask)
        
        masks, scores, _ = self.predictor.predict(
            mask_input=mask_input[None, :, :],  # Add batch dimension
            box=box,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
    
    def refine_with_contour(self,
                           contour: np.ndarray,
                           image_shape: Tuple[int, int],
                           method: str = 'points_negative') -> Tuple[np.ndarray, float]:
        """
        Refine a contour using SAM.
        
        Args:
            contour: Input contour as (N, 2) array of [y, x] coordinates
            image_shape: Shape of the image (height, width)
            method: Refinement method - 'mask', 'box', 'points', or 'points_negative'
            
        Returns:
            Tuple of (refined_contour, confidence_score)
        """
        # Calculate center point to guide SAM to interior region
        center = np.mean(contour, axis=0)
        center_point = np.array([center[1], center[0]])  # Convert [y,x] to [x,y]
        
        if method == 'mask':
            # Convert contour to mask
            mask = self._contour_to_mask(contour, image_shape)
            refined_mask, score = self.refine_with_mask(mask)
            
        elif method == 'box':
            # Get bounding box from contour + use center point
            # This combination helps SAM capture the full interior region
            box = self._contour_to_box(contour)
            
            # Expand box by 15% to give SAM more freedom
            width = box[2] - box[0]
            height = box[3] - box[1]
            expansion = 0.15
            box[0] -= width * expansion
            box[1] -= height * expansion
            box[2] += width * expansion
            box[3] += height * expansion
            
            # Clip to image bounds
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(image_shape[1], box[2])
            box[3] = min(image_shape[0], box[3])
            
            masks, scores, _ = self.predictor.predict(
                point_coords=center_point.reshape(1, 2),
                point_labels=np.array([1]),
                box=box,
                multimask_output=True
            )
            
            # Mask 0 typically has best coverage of interior region with box prompt
            best_idx = 0 if len(masks) > 0 else np.argmax(scores)
            refined_mask = masks[best_idx]
            score = scores[best_idx]
            
        elif method == 'points':
            # Use center point as positive prompt for interior
            # This generates 3 masks, usually mask 2 has best interior segmentation
            positive_points = center_point.reshape(1, 2)  # Shape: (1, 2)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=positive_points,
                point_labels=np.array([1]),
                multimask_output=True
            )
            
            # Prefer mask 2 (highest confidence) for interior segmentation
            best_idx = 2 if len(masks) > 2 else np.argmax(scores)
            refined_mask = masks[best_idx]
            score = scores[best_idx]
            
        elif method == 'points_negative':
            # Best method: center point + negative points on contour border
            # Sample points from the approximate contour as negative prompts
            num_negative = 8
            negative_indices = np.linspace(0, len(contour)-1, num_negative, dtype=int)
            negative_points = contour[negative_indices]
            negative_points_xy = negative_points[:, [1, 0]]  # Convert [y,x] to [x,y]
            
            # Combine positive (center) and negative (border) points
            all_points = np.vstack([center_point.reshape(1, 2), negative_points_xy])
            all_labels = np.array([1] + [0]*len(negative_points_xy))
            
            # Predict with SAM
            masks, scores, _ = self.predictor.predict(
                point_coords=all_points,
                point_labels=all_labels,
                multimask_output=True
            )
            
            # Select best mask (usually index 1 or 2 has best interior segmentation)
            # Prefer masks with high confidence
            best_idx = np.argmax(scores)
            refined_mask = masks[best_idx]
            score = scores[best_idx]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert mask back to contour
        refined_contour = self._mask_to_contour(refined_mask)
        
        return refined_contour, score
    
    @staticmethod
    def _contour_to_mask(contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Convert contour to binary mask."""
        mask = np.zeros(shape, dtype=np.uint8)
        # Contour is in [y, x] format, need to swap for cv2
        contour_xy = contour[:, [1, 0]].astype(np.int32)
        cv2.fillPoly(mask, [contour_xy], 1)
        return mask
    
    @staticmethod
    def _mask_to_contour(mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to contour."""
        # Find contours using OpenCV
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            return np.array([])
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Convert from OpenCV format (N, 1, 2) [x, y] to (N, 2) [y, x]
        contour = largest_contour.squeeze()
        if len(contour.shape) == 1:
            contour = contour.reshape(-1, 2)
        
        # Swap x, y to y, x
        contour = contour[:, [1, 0]]
        
        return contour
    
    @staticmethod
    def _contour_to_box(contour: np.ndarray) -> np.ndarray:
        """Convert contour to bounding box [x_min, y_min, x_max, y_max]."""
        y_coords = contour[:, 0]
        x_coords = contour[:, 1]
        return np.array([
            x_coords.min(),
            y_coords.min(),
            x_coords.max(),
            y_coords.max()
        ])
    
    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to bounding box."""
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return None
        return np.array([
            cols.min(),
            rows.min(),
            cols.max(),
            rows.max()
        ])


def refine_contour_with_sam(
    image: np.ndarray,
    approximate_contour: np.ndarray,
    sam_checkpoint: str,
    model_type: str = "vit_b",
    method: str = 'mask'
) -> Tuple[np.ndarray, float]:
    """
    Convenience function to refine a contour using SAM.
    
    Args:
        image: Input image (grayscale or RGB)
        approximate_contour: Initial contour from traditional methods (N, 2) [y, x]
        sam_checkpoint: Path to SAM checkpoint file
        model_type: SAM model type ('vit_h', 'vit_l', or 'vit_b')
        method: Refinement method - 'mask', 'box', or 'points'
        
    Returns:
        Tuple of (refined_contour, confidence_score)
    """
    if not SAM_AVAILABLE:
        print("SAM not available, returning original contour")
        return approximate_contour, 0.0
    
    # Initialize refiner
    refiner = SAMContourRefiner(model_type=model_type, checkpoint_path=sam_checkpoint)
    
    # Set image
    refiner.set_image(image)
    
    # Refine contour
    refined_contour, score = refiner.refine_with_contour(
        approximate_contour,
        image.shape[:2],
        method=method
    )
    
    return refined_contour, score
