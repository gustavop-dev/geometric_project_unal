"""
Utilities for comparing different spline interpolation methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from spline_interpolation.linear.spline import LinearSpline
from spline_interpolation.quadratic.spline import QuadraticSpline

def compare_splines(x, y, x_eval=None, title="Comparison of Spline Interpolation Methods", 
                   show_derivatives=False):
    """
    Compare different spline interpolation methods on the same dataset.
    
    Args:
        x (array-like): x-coordinates of the data points.
        y (array-like): y-coordinates of the data points.
        x_eval (float, optional): Specific point to evaluate and compare methods.
        title (str): Title for the plot.
        show_derivatives (bool): If True, show the derivatives of the splines.
    """
    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Create the interpolators
    linear_spline = LinearSpline(x, y)
    quadratic_spline = QuadraticSpline(x, y)
    
    # Create points for plotting
    x_plot = np.linspace(x.min(), x.max(), 1000)
    y_linear = linear_spline(x_plot)
    y_quadratic = quadratic_spline(x_plot)
    
    # Set up the plot
    fig = plt.figure(figsize=(15, 10))
    
    if show_derivatives:
        plt.subplot(211)
        
    # Plot the splines
    plt.plot(x_plot, y_linear, 'b-', label='Linear Spline', linewidth=2)
    plt.plot(x_plot, y_quadratic, 'g-', label='Quadratic Spline', linewidth=2)
    plt.plot(x, y, 'ro', label='Data Points', markersize=8)
    
    # If an evaluation point is specified, show it
    if x_eval is not None:
        y_linear_eval = linear_spline(x_eval)
        y_quadratic_eval = quadratic_spline(x_eval)
        
        plt.plot(x_eval, y_linear_eval, 'b*', markersize=10, 
                label=f'Linear @ x={x_eval}: {y_linear_eval:.2f}')
        plt.plot(x_eval, y_quadratic_eval, 'g*', markersize=10, 
                label=f'Quadratic @ x={x_eval}: {y_quadratic_eval:.2f}')
    
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    
    # If requested, show the derivatives
    if show_derivatives:
        plt.subplot(212)
        
        # We can't calculate derivatives for the linear spline across the entire curve
        # but we can show the jumps at the data points
        for i in range(len(x) - 1):
            mid_x = (x[i] + x[i+1]) / 2
            plt.plot([x[i], x[i+1]], [linear_spline.slopes[i], linear_spline.slopes[i]], 'b-', linewidth=2)
            if i < len(x) - 2:
                plt.plot([x[i+1], x[i+1]], 
                         [linear_spline.slopes[i], linear_spline.slopes[i+1]], 
                         'b--', linewidth=1)
        
        # Plot the quadratic spline derivative
        y_deriv_quadratic = quadratic_spline.derivative(x_plot)
        plt.plot(x_plot, y_deriv_quadratic, 'g-', label='Quadratic Spline Derivative', linewidth=2)
        
        # If an evaluation point is specified, show its derivative
        if x_eval is not None:
            deriv_quadratic_eval = quadratic_spline.derivative(x_eval)
            
            # Determine the slope of the linear spline at x_eval
            idx = np.searchsorted(x, x_eval, side='right') - 1
            if idx < 0:
                idx = 0
            elif idx >= len(linear_spline.slopes):
                idx = len(linear_spline.slopes) - 1
                
            deriv_linear_eval = linear_spline.slopes[idx]
            
            plt.plot(x_eval, deriv_linear_eval, 'b*', markersize=10, 
                    label=f'Linear Deriv @ x={x_eval}: {deriv_linear_eval:.2f}')
            plt.plot(x_eval, deriv_quadratic_eval, 'g*', markersize=10, 
                    label=f'Quadratic Deriv @ x={x_eval}: {deriv_quadratic_eval:.2f}')
        
        plt.grid(True)
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('Derivative')
        plt.title('Derivatives of the Splines')
    
    plt.tight_layout()
    plt.show()


def compare_rocket_example(eval_point=16):
    """
    Compare different interpolation methods using the rocket example.
    
    Args:
        eval_point (float): Point to evaluate (time in seconds).
    """
    # Rocket data: time and velocity
    t = np.array([0, 10, 15, 20, 22.5, 30])
    v = np.array([0, 227.04, 362.78, 517.35, 602.97, 901.67])
    
    # Compare splines on the rocket data
    compare_splines(t, v, eval_point, 
                   title=f"Comparison of Splines for Rocket Velocity (t = {eval_point}s)",
                   show_derivatives=True)


if __name__ == "__main__":
    compare_rocket_example() 