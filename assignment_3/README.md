# Assignment 3: B-Spline for Medical Image Contours

This project implements quadratic B-spline interpolation for contours in medical images, specifically for MRI scans of the shoulder (rotator cuff).

## Context

Shoulder MRIs are very common in Colombia for diagnosing pain caused by tears in the rotator cuff muscle complex. In this project, we focus on axial slices, which typically come in series of 15-20 cuts.

## Objectives

The main objective is to define a quadratic B-spline that interpolates a series of points that delineate the contour of the rotator cuff in a magnetic resonance image. For this:

1. We define a sequence of intervals for the functions x(t) and y(t)
2. We determine the positions of the parameters t in each interval
3. We implement a non-uniform parameterization based on chord length between points to handle non-equidistant points

## Main Features

- **Adaptive parameterization**: The algorithm uses chord length parameterization to adapt to the non-uniform distribution of points
- **Derivative calculation**: Tangent vectors can be calculated and visualized along the curve
- **2D interpolation**: Ability to interpolate any set of 2D points
- **Closed curves**: Support for closed curves, important for representing anatomical contours
- **Angular sorting**: Option to sort points angularly to prevent self-intersections
- **3D visualization**: Advanced 3D visualization of curvature and other properties

## Implementation

The implementation is structured as follows:

1. `QuadraticBSpline`: Class that implements a quadratic/cubic B-spline to interpolate points in 2D
2. Helper functions for loading data, visualizing results, and evaluating interpolation quality
3. Simplified visualization scripts using scipy.interpolate directly

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- SciPy

## Usage

The project includes three main scripts for different types of visualizations:

### 1. Simple Curve Visualization

Generates a clean visualization of the B-spline curve with original points:

```bash
python assignment_3/simple_curve.py
```

Options:
```
--points_file (default='ptos.mat'): Path to the .mat file with contour points
--degree (default=3): B-spline degree (3=cubic, 2=quadratic)
--smoothing (default=0.0): Smoothing parameter (0.0 for exact interpolation)
--save_only: Only save the plot without displaying it
--output_dir: Directory to save output images (default is assignment_3/output/simple/)
--no_sort_angular: Do not sort points angularly around center of mass
```

Example:
```bash
python assignment_3/simple_curve.py --degree 3 --smoothing 10 --save_only
```

### 2. Comparison Grid and Animation

Creates a comparison grid of different B-spline degrees and smoothing values, or generates animation frames:

```bash
python assignment_3/compare_bsplines.py --mode grid
```

For animation frames:
```bash
python assignment_3/compare_bsplines.py --mode animation
```

Options:
```
--points_file (default='ptos.mat'): Path to the .mat file with contour points
--output_dir: Directory to save output images (default is assignment_3/output/)
--no_sort_angular: Do not sort points angularly
--mode (choices=['grid', 'animation'], default='grid'): Operation mode
--degree (default=3): B-spline degree for animation mode
--smoothing (default=0.0): Smoothing parameter for animation mode
```

Example:
```bash
python assignment_3/compare_bsplines.py --mode animation --degree 3 --smoothing 20
```

### 3. 3D Visualization

Creates 3D visualizations where the Z-axis represents curvature or distance:

```bash
python assignment_3/visualize_3d.py
```

Options:
```
--points_file (default='ptos.mat'): Path to the .mat file with contour points
--output_dir: Directory to save output images (default is assignment_3/output/)
--degree (default=3): B-spline degree
--smoothing (default=10.0): Smoothing parameter
--no_sort_angular: Do not sort points angularly
--z_scale (default=50.0): Scale factor for the Z-axis
--surface_type (choices=['curvature', 'distance'], default='curvature'): Type of 3D surface
```

Example:
```bash
python assignment_3/visualize_3d.py --surface_type curvature --z_scale 50 --smoothing 10
```

## Output

All output images are organized in subdirectories:
- Simple curves: `assignment_3/output/simple/`
- Comparison grids: `assignment_3/output/comparison_grid/`
- Animation frames: `assignment_3/output/animation_TIMESTAMP/`
- 3D visualizations: `assignment_3/output/3d_visualization_TYPE_TIMESTAMP/`

## Technical Features

- **Cubic B-splines**: Provide C² continuity along the entire curve
- **Chord length parameterization**: Assigns parameter values proportional to the distance between consecutive points
- **Derivative visualization**: Visualizes the direction and magnitude of the tangent at selected points
- **Self-intersection prevention**: Angular sorting and other methods to prevent curve self-intersections
- **Adaptively smooth curves**: Smoothing parameter to control curve exactness vs. smoothness

## Future Improvements

- Implement methods to optimize control point placement
- Add support for higher-degree B-splines
- Develop curvature analysis tools
- Implement methods to calculate areas and perimeters based on the curve 