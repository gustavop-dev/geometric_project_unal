#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from humerus_detection.io import dicom_utils
from humerus_detection.contour import detection
from humerus_detection.contour.sam_refinement import SAMContourRefiner

# Probar con I01, I15, I18
for img_name in ["I01", "I15", "I18"]:
    print(f"\n{'='*60}")
    print(f"Procesando {img_name}")
    print('='*60)
    
    dataset, pixel_array = dicom_utils.read_dicom(f"/home/cerrotico/unal/geometric_project_unal/assignment_4/axial_sections/{img_name}.dcm")
    
    # Detección tradicional
    traditional_contour = detection.detect_advanced_humerus_contour(pixel_array, file_name=img_name)
    
    if len(traditional_contour) == 0:
        print(f"No se detectó húmero en {img_name}")
        continue
    
    # SAM
    refiner = SAMContourRefiner(model_type="vit_b", checkpoint_path="/home/cerrotico/unal/geometric_project_unal/assignment_4/models/sam_vit_b_01ec64.pth")
    refiner.set_image(pixel_array)
    
    # Refinar con método points (M1: solo centro, mask 2)
    refined_contour, confidence = refiner.refine_with_contour(
        traditional_contour,
        pixel_array.shape[:2],
        method='points'
    )
    
    print(f"SAM confidence: {confidence:.3f}")
    print(f"Contorno tradicional: {len(traditional_contour)} puntos")
    print(f"Contorno SAM: {len(refined_contour)} puntos")
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(pixel_array, cmap='gray')
    axes[0].plot(traditional_contour[:, 1], traditional_contour[:, 0], 'r-', linewidth=2)
    axes[0].set_title('Traditional (Hough+Snake)')
    axes[0].axis('off')
    
    axes[1].imshow(pixel_array, cmap='gray')
    axes[1].plot(refined_contour[:, 1], refined_contour[:, 0], 'g-', linewidth=2)
    axes[1].set_title(f'SAM M1 (conf: {confidence:.2f})')
    axes[1].axis('off')
    
    axes[2].imshow(pixel_array, cmap='gray')
    axes[2].plot(traditional_contour[:, 1], traditional_contour[:, 0], 'r-', linewidth=1, alpha=0.5, label='Traditional')
    axes[2].plot(refined_contour[:, 1], refined_contour[:, 0], 'g-', linewidth=2, label='SAM')
    axes[2].set_title('Comparación')
    axes[2].axis('off')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'/home/cerrotico/unal/geometric_project_unal/assignment_4/result_{img_name}.png', dpi=150)
    print(f"Guardado: result_{img_name}.png")

print("\n✓ Completado")
