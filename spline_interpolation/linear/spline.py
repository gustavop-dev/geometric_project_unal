"""
Linear Spline Interpolation Module

This module implements linear spline interpolation based on the approach described in
the "Spline Method of Interpolation" document.
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearSpline:
    """
    Class for linear spline interpolation.
    
    Linear spline interpolation connects consecutive data points with straight lines.
    """
    
    def __init__(self, x, y):
        """
        Initialize the linear spline interpolator with data points.
        
        Args:
            x (array-like): x-coordinates of the data points.
            y (array-like): y-coordinates of the data points.
        """
        # Convert to numpy arrays
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        
        # Verify that the data is sorted by x
        if not np.all(np.diff(self.x) > 0):
            idx = np.argsort(self.x)
            self.x = self.x[idx]
            self.y = self.y[idx]
            
        # Calculate slopes between consecutive points
        self.slopes = np.zeros(len(self.x) - 1)
        for i in range(len(self.x) - 1):
            self.slopes[i] = (self.y[i+1] - self.y[i]) / (self.x[i+1] - self.x[i])
    
    def __call__(self, x_new):
        """
        Evaluate the linear spline at the given points.
        
        Args:
            x_new (float or array-like): Points where to evaluate the spline.
            
        Returns:
            float or ndarray: Interpolated values.
        """
        x_new = np.asarray(x_new)
        scalar_input = False
        if x_new.ndim == 0:
            x_new = x_new[np.newaxis]
            scalar_input = True
            
        y_new = np.zeros_like(x_new)
        
        for i, x_val in enumerate(x_new):
            if x_val < self.x[0] or x_val > self.x[-1]:
                raise ValueError(f"Point {x_val} is outside the interpolation range [{self.x[0]}, {self.x[-1]}]")
                
            # Handle the case where x_val equals the last point exactly
            if x_val == self.x[-1]:
                y_new[i] = self.y[-1]
                continue
                
            # Find the corresponding interval
            idx = np.searchsorted(self.x, x_val, side='right') - 1
            
            # Apply the linear interpolation formula
            y_new[i] = self.y[idx] + self.slopes[idx] * (x_val - self.x[idx])
            
        return y_new[0] if scalar_input else y_new
    
    def plot(self, num_points=1000, show_points=True):
        """
        Plot the linear spline.
        
        Args:
            num_points (int): Number of points for the plot.
            show_points (bool): If True, show the original data points.
        """
        x_plot = np.linspace(self.x[0], self.x[-1], num_points)
        y_plot = self(x_plot)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, 'b-', label='Linear Spline')
        
        if show_points:
            plt.plot(self.x, self.y, 'ro', label='Data Points')
            
        plt.grid(True)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Spline Interpolation')
        plt.show()


def example_rocket():
    """
    Implementation of Example 1 from the document about the rocket.
    
    The example calculates the rocket's velocity at t = 16 seconds using linear splines.
    """
    # Rocket data: time and velocity
    t = np.array([0, 10, 15, 20, 22.5, 30])
    v = np.array([0, 227.04, 362.78, 517.35, 602.97, 901.67])
    
    # Create the linear spline interpolator
    spline = LinearSpline(t, v)
    
    # Calculate the velocity at t = 16 seconds
    v_16 = spline(16)
    
    print(f"Velocity at t = 16 seconds: {v_16:.2f} m/s")
    
    # Plot the result
    plt.figure(figsize=(10, 6))
    t_plot = np.linspace(0, 30, 1000)
    v_plot = spline(t_plot)
    
    plt.plot(t_plot, v_plot, 'b-', label='Linear Spline')
    plt.plot(t, v, 'ro', label='Data Points')
    plt.plot(16, v_16, 'g*', markersize=10, label=f't = 16s, v = {v_16:.2f} m/s')
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Rocket Velocity vs Time - Linear Spline Interpolation')
    plt.show()


if __name__ == "__main__":
    example_rocket() 
