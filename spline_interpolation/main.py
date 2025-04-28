"""
Main script for spline interpolation.

This script allows running different examples of implemented interpolation methods.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from spline_interpolation.linear.spline import LinearSpline, example_rocket as linear_rocket_example
from spline_interpolation.quadratic.spline import QuadraticSpline, example_rocket as quadratic_rocket_example
from spline_interpolation.utils.compare import compare_rocket_example

def main():
    """
    Main function to run spline interpolation examples.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Spline Interpolation')
    parser.add_argument('--method', choices=['linear', 'quadratic', 'compare', 'all'], 
                        default='all', help='Interpolation method to use')
    parser.add_argument('--eval_point', type=float, default=16.0,
                        help='Evaluation point for the rocket example (time in seconds)')
    parser.add_argument('--custom', action='store_true',
                        help='Use custom data instead of the rocket example')
    parser.add_argument('--x', type=float, nargs='+',
                        help='x values for custom data')
    parser.add_argument('--y', type=float, nargs='+',
                        help='y values for custom data')
    
    args = parser.parse_args()
    
    # Check if custom data will be used
    if args.custom:
        if args.x is None or args.y is None or len(args.x) != len(args.y) or len(args.x) < 2:
            print("Error: At least 2 points (x, y) of the same size are required for custom interpolation")
            return
        
        x = np.array(args.x)
        y = np.array(args.y)
        
        if args.method == 'linear' or args.method == 'all':
            linear_spline = LinearSpline(x, y)
            linear_spline.plot()
            
        if args.method == 'quadratic' or args.method == 'all':
            if len(x) < 3:
                print("Error: At least 3 points are required for quadratic interpolation")
            else:
                quadratic_spline = QuadraticSpline(x, y)
                quadratic_spline.plot(show_derivatives=True)
                
        if args.method == 'compare' or args.method == 'all':
            if len(x) < 3:
                print("Error: At least 3 points are required for comparison (includes quadratic)")
            else:
                x_eval = args.eval_point if args.eval_point >= x.min() and args.eval_point <= x.max() else None
                from spline_interpolation.utils.compare import compare_splines
                compare_splines(x, y, x_eval, show_derivatives=True)
    else:
        # Use the rocket example
        eval_point = args.eval_point
        
        if args.method == 'linear' or args.method == 'all':
            print("\n=== Rocket Example with Linear Spline ===")
            linear_rocket_example()
            
        if args.method == 'quadratic' or args.method == 'all':
            print("\n=== Rocket Example with Quadratic Spline ===")
            quadratic_rocket_example()
            
        if args.method == 'compare' or args.method == 'all':
            print("\n=== Comparison of Splines in the Rocket Example ===")
            compare_rocket_example(eval_point)


def run_test_runge():
    """
    Run an example to demonstrate the Runge phenomenon, as mentioned in the document.
    
    This example shows why splines are preferable to high-order polynomials.
    """
    # Generate data based on Runge's function: f(x) = 1/(1 + 25x²)
    x = np.linspace(-1, 1, 6)  # 6 equidistant points
    y = 1 / (1 + 25 * x**2)
    
    # Calculate a polynomial interpolation of order 5
    from numpy.polynomial.polynomial import Polynomial
    from numpy.polynomial.polynomial import polyfit
    
    # Fit a polynomial of degree 5
    p_coeffs = polyfit(x, y, 5)
    p = Polynomial(p_coeffs)
    
    # Create points for plotting
    x_fine = np.linspace(-1, 1, 1000)
    y_true = 1 / (1 + 25 * x_fine**2)
    y_poly = p(x_fine)
    
    # Create spline interpolation
    linear_spline = LinearSpline(x, y)
    y_linear = linear_spline(x_fine)
    
    quadratic_spline = QuadraticSpline(x, y)
    y_quadratic = quadratic_spline(x_fine)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(x_fine, y_true, 'k-', label='Original Function: 1/(1+25x²)', linewidth=2)
    plt.plot(x_fine, y_poly, 'r--', label='Polynomial of Degree 5', linewidth=2)
    plt.plot(x_fine, y_linear, 'b-', label='Linear Spline', linewidth=2)
    plt.plot(x_fine, y_quadratic, 'g-', label='Quadratic Spline', linewidth=2)
    plt.plot(x, y, 'ko', label='Data Points', markersize=8)
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Runge Phenomenon: Comparison of Interpolation Methods')
    
    # Show error relative to the original function
    plt.figure(figsize=(12, 6))
    plt.plot(x_fine, np.abs(y_poly - y_true), 'r-', label='Polynomial Error', linewidth=2)
    plt.plot(x_fine, np.abs(y_linear - y_true), 'b-', label='Linear Spline Error', linewidth=2)
    plt.plot(x_fine, np.abs(y_quadratic - y_true), 'g-', label='Quadratic Spline Error', linewidth=2)
    
    plt.grid(True)
    plt.legend()
    plt.yscale('log')  # Logarithmic scale for better visualization
    plt.xlabel('x')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Interpolation Error')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    # To run the Runge phenomenon example, uncomment the following line:
    # run_test_runge() 