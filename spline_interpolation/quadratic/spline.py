"""
Quadratic Spline Interpolation Module

This module implements quadratic spline interpolation based on the approach described in
the "Spline Method of Interpolation" document.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

class QuadraticSpline:
    """
    Class for quadratic spline interpolation.
    
    Quadratic splines fit a quadratic polynomial between each pair of consecutive data points,
    maintaining first derivative continuity at interior points.
    """
    
    def __init__(self, x, y):
        """
        Initialize the quadratic spline interpolator with data points.
        
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
            
        n = len(self.x) - 1  # Number of splines
        
        # Initialize coefficients a, b, c for each spline
        self.a = np.zeros(n)
        self.b = np.zeros(n)
        self.c = np.zeros(n)
        
        # Build the equation system
        # For n splines we have 3n coefficients and need 3n equations:
        # - 2n equations: each spline passes through 2 points
        # - (n-1) equations: first derivative continuity at interior points
        # - 1 equation: we assume the first spline is linear (a[0] = 0)
        
        # Matrix of the system and right-hand side vector
        A = np.zeros((3*n, 3*n))
        B = np.zeros(3*n)
        
        # Apply conditions that each spline passes through two consecutive points
        eq = 0
        for i in range(n):
            # First point: a_i*x_i^2 + b_i*x_i + c_i = y_i
            A[eq, 3*i] = self.x[i]**2
            A[eq, 3*i+1] = self.x[i]
            A[eq, 3*i+2] = 1
            B[eq] = self.y[i]
            eq += 1
            
            # Second point: a_i*x_{i+1}^2 + b_i*x_{i+1} + c_i = y_{i+1}
            A[eq, 3*i] = self.x[i+1]**2
            A[eq, 3*i+1] = self.x[i+1]
            A[eq, 3*i+2] = 1
            B[eq] = self.y[i+1]
            eq += 1
        
        # Apply first derivative continuity conditions at interior points
        for i in range(n-1):
            # 2*a_i*x_{i+1} + b_i = 2*a_{i+1}*x_{i+1} + b_{i+1}
            A[eq, 3*i] = 2*self.x[i+1]
            A[eq, 3*i+1] = 1
            A[eq, 3*(i+1)] = -2*self.x[i+1]
            A[eq, 3*(i+1)+1] = -1
            B[eq] = 0
            eq += 1
        
        # Apply the condition that the first spline is linear (a[0] = 0)
        A[eq, 0] = 1
        B[eq] = 0
        
        # Solve the system to find the coefficients
        coeffs = np.linalg.solve(A, B)
        
        # Extract coefficients a, b, c for each spline
        for i in range(n):
            self.a[i] = coeffs[3*i]
            self.b[i] = coeffs[3*i+1]
            self.c[i] = coeffs[3*i+2]
    
    def __call__(self, x_new):
        """
        Evaluate the quadratic spline at the given points.
        
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
            
            # Apply the corresponding quadratic polynomial
            y_new[i] = self.a[idx] * x_val**2 + self.b[idx] * x_val + self.c[idx]
            
        return y_new[0] if scalar_input else y_new
    
    def derivative(self, x_new):
        """
        Calculate the first derivative of the quadratic spline at the given points.
        
        Args:
            x_new (float or array-like): Points where to evaluate the derivative.
            
        Returns:
            float or ndarray: Derivative values.
        """
        x_new = np.asarray(x_new)
        scalar_input = False
        if x_new.ndim == 0:
            x_new = x_new[np.newaxis]
            scalar_input = True
            
        y_deriv = np.zeros_like(x_new)
        
        for i, x_val in enumerate(x_new):
            if x_val < self.x[0] or x_val > self.x[-1]:
                raise ValueError(f"Point {x_val} is outside the interpolation range [{self.x[0]}, {self.x[-1]}]")
                
            # Handle the case where x_val equals the last point exactly
            if x_val == self.x[-1]:
                # Use the last interval's coefficients to calculate the derivative at the last point
                idx = len(self.a) - 1
                y_deriv[i] = 2 * self.a[idx] * x_val + self.b[idx]
                continue
                
            # Find the corresponding interval
            idx = np.searchsorted(self.x, x_val, side='right') - 1
            
            # Derivative of the quadratic polynomial: 2*a*x + b
            y_deriv[i] = 2 * self.a[idx] * x_val + self.b[idx]
            
        return y_deriv[0] if scalar_input else y_deriv
    
    def integrate(self, a, b):
        """
        Integrate the quadratic spline over the interval [a, b].
        
        Args:
            a (float): Lower limit of integration.
            b (float): Upper limit of integration.
            
        Returns:
            float: Value of the integral.
        """
        if a < self.x[0] or b > self.x[-1]:
            raise ValueError(f"Integration interval [{a}, {b}] is outside the interpolation range [{self.x[0]}, {self.x[-1]}]")
        
        # Find the subintervals containing a and b
        idx_a = np.searchsorted(self.x, a, side='right') - 1
        idx_b = np.searchsorted(self.x, b, side='right') - 1
        
        result = 0.0
        
        # Special case: a and b are in the same subinterval
        if idx_a == idx_b:
            result += self._integrate_segment(a, b, idx_a)
        else:
            # Integrate from a to the end of the first subinterval
            result += self._integrate_segment(a, self.x[idx_a + 1], idx_a)
            
            # Integrate complete intermediate subintervals
            for i in range(idx_a + 1, idx_b):
                result += self._integrate_segment(self.x[i], self.x[i + 1], i)
            
            # Integrate from the beginning of the last subinterval to b
            result += self._integrate_segment(self.x[idx_b], b, idx_b)
            
        return result
    
    def _integrate_segment(self, a, b, idx):
        """
        Integrate a segment of the quadratic spline.
        
        Args:
            a (float): Lower limit of integration.
            b (float): Upper limit of integration.
            idx (int): Index of the spline to integrate.
            
        Returns:
            float: Value of the segment integral.
        """
        # Integrate a*x^2 + b*x + c from a to b
        a_coef, b_coef, c_coef = self.a[idx], self.b[idx], self.c[idx]
        return (a_coef * (b**3 - a**3) / 3 + b_coef * (b**2 - a**2) / 2 + c_coef * (b - a))
    
    def plot(self, num_points=1000, show_points=True, show_derivatives=False):
        """
        Plot the quadratic spline and optionally its derivative.
        
        Args:
            num_points (int): Number of points for the plot.
            show_points (bool): If True, show the original data points.
            show_derivatives (bool): If True, show the first derivative.
        """
        x_plot = np.linspace(self.x[0], self.x[-1], num_points)
        y_plot = self(x_plot)
        
        plt.figure(figsize=(12, 8))
        
        # Plot the spline
        plt.subplot(211)
        plt.plot(x_plot, y_plot, 'b-', label='Quadratic Spline')
        
        if show_points:
            plt.plot(self.x, self.y, 'ro', label='Data Points')
            
        plt.grid(True)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Quadratic Spline Interpolation')
        
        # Plot the derivative if requested
        if show_derivatives:
            plt.subplot(212)
            y_deriv = self.derivative(x_plot)
            plt.plot(x_plot, y_deriv, 'g-', label='First Derivative')
            plt.grid(True)
            plt.legend()
            plt.xlabel('x')
            plt.ylabel("y'")
            plt.title('First Derivative of Quadratic Spline')
        
        plt.tight_layout()
        plt.show()


def example_rocket():
    """
    Implementation of Example 2 from the document about the rocket.
    
    The example calculates:
    a) The rocket's velocity at t = 16 seconds using quadratic splines.
    b) The distance covered by the rocket between t = 11s and t = 16s.
    c) The rocket's acceleration at t = 16s.
    """
    # Rocket data: time and velocity
    t = np.array([0, 10, 15, 20, 22.5, 30])
    v = np.array([0, 227.04, 362.78, 517.35, 602.97, 901.67])
    
    # Create the quadratic spline interpolator
    spline = QuadraticSpline(t, v)
    
    # Part a: Calculate the velocity at t = 16 seconds
    v_16 = spline(16)
    print(f"a) Velocity at t = 16 seconds: {v_16:.2f} m/s")
    
    # Part b: Calculate the distance covered between t = 11s and t = 16s
    distance = spline.integrate(11, 16)
    print(f"b) Distance covered between t = 11s and t = 16s: {distance:.2f} m")
    
    # Part c: Calculate the acceleration at t = 16s
    accel_16 = spline.derivative(16)
    print(f"c) Acceleration at t = 16 seconds: {accel_16:.2f} m/s²")
    
    # Plot the results
    plt.figure(figsize=(15, 12))
    
    # Velocity vs time
    plt.subplot(311)
    t_plot = np.linspace(0, 30, 1000)
    v_plot = spline(t_plot)
    
    plt.plot(t_plot, v_plot, 'b-', label='Quadratic Spline')
    plt.plot(t, v, 'ro', label='Data Points')
    plt.plot(16, v_16, 'g*', markersize=10, label=f't = 16s, v = {v_16:.2f} m/s')
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Rocket Velocity vs Time - Quadratic Spline Interpolation')
    
    # Acceleration vs time
    plt.subplot(312)
    a_plot = spline.derivative(t_plot)
    
    plt.plot(t_plot, a_plot, 'r-', label='Acceleration')
    plt.plot(16, accel_16, 'g*', markersize=10, label=f't = 16s, a = {accel_16:.2f} m/s²')
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Rocket Acceleration vs Time')
    
    # Visualize the distance covered
    plt.subplot(313)
    # Shade the area under the curve between t = 11s and t = 16s
    t_segment = np.linspace(11, 16, 100)
    v_segment = spline(t_segment)
    
    plt.plot(t_plot, v_plot, 'b-', label='Velocity')
    plt.fill_between(t_segment, 0, v_segment, alpha=0.3, color='blue', 
                    label=f'Distance = {distance:.2f} m')
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Distance covered between t = 11s and t = 16s')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_rocket() 