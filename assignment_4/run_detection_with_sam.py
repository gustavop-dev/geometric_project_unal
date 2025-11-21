#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced humerus detection pipeline using SAM for contour refinement.
Combines traditional methods (Hough + Snake) with SAM for better accuracy.
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from humerus_detection import run_pipeline
from humerus_detection.io import dicom_utils
from humerus_detection.preprocessing import image_processing
from humerus_detection.contour import detection, outliers
from humerus_detection.contour.sam_refinement import SAMContourRefiner, SAM_AVAILABLE
from humerus_detection.spline import fitting
from humerus_detection.visualization import plotting


def run_sam_enhanced_pipeline(dicom_directory: str, 
                              output_directory: str, 
                              sam_checkpoint: str,
                              sam_model: str = "vit_b",
                              show_images: bool = False) -> None:
    """
    Run the humerus detection pipeline with SAM refinement.
    
    Args:
        dicom_directory: Directory containing the DICOM files
        output_directory: Directory where to save the results
        sam_checkpoint: Path to SAM checkpoint file
        sam_model: SAM model type ('vit_h', 'vit_l', or 'vit_b')
        show_images: Whether to display the images during processing
    """
    if not SAM_AVAILABLE:
        print("ERROR: SAM is not available. Please install segment-anything:")
        print("pip install git+https://github.com/facebookresearch/segment-anything.git")
        return
    
    print(f"Running SAM-enhanced humerus detection pipeline:")
    print(f"  - DICOM directory: {dicom_directory}")
    print(f"  - Output directory: {output_directory}")
    print(f"  - SAM model: {sam_model}")
    print(f"  - SAM checkpoint: {sam_checkpoint}")
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Initialize SAM refiner
    print("Loading SAM model...")
    refiner = SAMContourRefiner(model_type=sam_model, checkpoint_path=sam_checkpoint)
    
    # Process all DICOM files
    files = sorted([f for f in os.listdir(dicom_directory) if f.endswith('.dcm')])
    print(f"Found {len(files)} DICOM files to process\n")
    
    previous_contour = None
    
    for i, file in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {file}")
        full_path = os.path.join(dicom_directory, file)
        
        # Step 1: Read DICOM file
        dataset, pixel_array = dicom_utils.read_dicom(full_path)
        if dataset is None or pixel_array is None:
            print(f"  Error reading {file}")
            continue
        
        base_name = os.path.splitext(os.path.basename(full_path))[0]
        
        # Step 2: Traditional detection (Hough + Snake)
        print("  [1/3] Traditional detection...")
        traditional_contour = detection.detect_advanced_humerus_contour(
            pixel_array, 
            previous_contour=previous_contour, 
            file_name=base_name,
            slice_height=dicom_utils.get_slice_height(dataset)
        )
        
        # If no contour detected, skip
        if len(traditional_contour) == 0:
            print(f"  No humerus detected in {base_name}")
            previous_contour = None
            
            # Save empty image
            fig = plt.figure(figsize=(10, 8))
            plt.imshow(pixel_array, cmap='gray')
            plt.title(f"Axial slice - No humerus detected")
            plt.axis('off')
            plt.tight_layout()
            plotting.save_results(fig, output_directory, base_name, "sam_enhanced")
            plt.close(fig)
            continue
        
        # Step 3: SAM refinement
        print("  [2/3] SAM refinement...")
        refiner.set_image(pixel_array)
        
        confidence = 0.0
        try:
            # Use points method: only center point, SAM mask 2 for interior segmentation
            refined_contour, confidence = refiner.refine_with_contour(
                traditional_contour,
                pixel_array.shape[:2],
                method='points'  # M1: Simple and effective - only center point
            )
            
            print(f"  SAM confidence: {confidence:.3f}")
            
            # If SAM failed or low confidence, use traditional contour
            if len(refined_contour) == 0 or confidence < 0.5:
                print(f"  Low SAM confidence, using traditional contour")
                refined_contour = traditional_contour
            
        except Exception as e:
            print(f"  SAM refinement failed: {e}")
            refined_contour = traditional_contour
        
        # Step 4: Apply B-spline smoothing
        print("  [3/3] B-spline fitting...")
        smoothed_contour = outliers.detect_and_correct_outliers(refined_contour, is_spline=False)
        
        smoothing = 10.0 * 2.5
        degree = 3
        x_spline, y_spline = fitting.apply_bspline(smoothed_contour, degree, smoothing)
        
        if x_spline is None or y_spline is None:
            print(f"  Error applying B-spline to {base_name}")
            continue
        
        # Step 5: Visualize comparison
        fig = plt.figure(figsize=(15, 5))
        
        # Original with traditional contour
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(pixel_array, cmap='gray')
        ax1.plot(traditional_contour[:, 1], traditional_contour[:, 0], 'r-', linewidth=2, label='Traditional')
        ax1.set_title('Traditional (Hough + Snake)')
        ax1.axis('off')
        ax1.legend()
        
        # SAM refined
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(pixel_array, cmap='gray')
        ax2.plot(refined_contour[:, 1], refined_contour[:, 0], 'g-', linewidth=2, label='SAM refined')
        ax2.set_title(f'SAM Refined (conf: {confidence:.2f})')
        ax2.axis('off')
        ax2.legend()
        
        # Final with B-spline
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(pixel_array, cmap='gray')
        ax3.plot(refined_contour[:, 1], refined_contour[:, 0], 'r-', linewidth=1, alpha=0.5, label='SAM contour')
        ax3.plot(x_spline, y_spline, 'b-', linewidth=2, label='B-spline')
        ax3.set_title(f'Final Result (smoothing {smoothing})')
        ax3.axis('off')
        ax3.legend()
        
        plt.tight_layout()
        plotting.save_results(fig, output_directory, base_name, "sam_enhanced")
        
        if not show_images:
            plt.close(fig)
        else:
            plt.show()
        
        # Update previous contour
        previous_contour = refined_contour
        print(f"  ✓ Completed\n")
    
    print(f"SAM-enhanced processing completed. Results saved to {output_directory}")


def main():
    """
    Main function to parse arguments and run the SAM-enhanced pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Detect and model the humerus using traditional methods + SAM refinement."
    )
    
    parser.add_argument(
        "--dicom_dir", 
        type=str, 
        default="assignment_4/axial_sections",
        help="Directory containing the DICOM files"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="assignment_4/sam_results",
        help="Directory where to save the results"
    )
    
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        required=True,
        help="Path to SAM checkpoint file (e.g., sam_vit_b_01ec64.pth)"
    )
    
    parser.add_argument(
        "--sam_model",
        type=str,
        default="vit_b",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type (vit_h=huge, vit_l=large, vit_b=base)"
    )
    
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Show images during processing"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("SAM-Enhanced Humerus Detection Pipeline")
    print("=" * 60)
    print(f"DICOM directory: {args.dicom_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"SAM checkpoint: {args.sam_checkpoint}")
    print(f"SAM model: {args.sam_model}")
    print(f"Show images: {args.show}")
    print("=" * 60)
    
    # Record start time
    start_time = time.time()
    
    # Run the pipeline
    run_sam_enhanced_pipeline(
        args.dicom_dir, 
        args.output_dir, 
        args.sam_checkpoint,
        args.sam_model,
        args.show
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
