"""
Greco 2022: Trimmed Approximate Maximum Likelihood Estimation (AMLE)

Implements a robust circle fitting algorithm using trimmed optimization.
The method automatically discards a proportion (gamma) of outliers by
minimizing the sum of squared distances for only the h closest points.

Reference: Greco et al. (2022)

Main function: greco_2022(edgels, **kwargs)
"""

import numpy as np
from scipy.optimize import minimize


def _geometric_distance(points, center, radius):
    """
    Calculate the geometric distance of points from a circle.
    
    Geometric distance is the perpendicular distance from each point
    to the circle (not to the center).
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates
    center : numpy.ndarray
        Array of shape (2,) containing circle center (cx, cy)
    radius : float
        Circle radius
        
    Returns
    -------
    distances : numpy.ndarray
        Array of geometric distances from each point to the circle
    """
    # Distance from point to center
    distances_to_center = np.sqrt(np.sum((points - center)**2, axis=1))
    
    # Geometric distance to circle = |distance_to_center - radius|
    return np.abs(distances_to_center - radius)


def _objective_function(params, points, h):
    """
    Objective function for trimmed optimization.
    
    Computes the sum of squared distances for the h closest points only.
    This provides robustness by automatically trimming outliers.
    
    Parameters
    ----------
    params : numpy.ndarray
        Array [cx, cy, radius] containing circle parameters
    points : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates
    h : int
        Number of closest points to consider (trimming level)
        
    Returns
    -------
    float
        Sum of squared distances for the h closest points
    """
    center_x, center_y, radius = params
    center = np.array([center_x, center_y])
    
    # Calculate geometric distances
    distances = _geometric_distance(points, center, radius)
    
    # Sort distances and sum only the h smallest (trim outliers)
    sorted_distances = np.sort(distances)
    trimmed_sum = np.sum(sorted_distances[:h]**2)
    
    return trimmed_sum


def greco_2022(edgels, gamma=0.25, max_iterations=100, tol=1e-6):
    """
    Fit a circle to edge points using Trimmed AMLE method.
    
    This algorithm uses trimmed optimization where only the h = n*(1-gamma)
    closest points are used in the objective function. This automatically
    discards outliers without explicit detection.
    
    Algorithm steps:
    1. Initialize center as mean of points, radius as mean distance
    2. Iteratively minimize sum of squared distances for h closest points
    3. Repeat until convergence
    
    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    gamma : float, optional
        Trimming level (proportion of points to discard as outliers)
        Default: 0.25 (discard 25% of points)
        Range: 0.0 (no trimming) to 0.5 (discard half)
    max_iterations : int, optional
        Maximum number of optimization iterations (default: 100)
    tol : float, optional
        Convergence tolerance for parameter changes (default: 1e-6)
        
    Returns
    -------
    center : numpy.ndarray
        Array [cx, cy] of circle center coordinates
    radius : float
        Circle radius
        
    Notes
    -----
    The trimming level gamma determines robustness vs. efficiency:
    - gamma=0.0: No trimming (standard least squares, sensitive to outliers)
    - gamma=0.25: Moderate robustness (default, handles 25% outliers)
    - gamma=0.40: High robustness (handles 40% outliers)
    - gamma=0.5: Maximum practical trimming (50% outliers)
    
    The algorithm uses Nelder-Mead simplex optimization, which is
    derivative-free and robust but may be slower than gradient methods.
    
    Examples
    --------
    >>> # Circle with 30% outliers
    >>> theta = np.linspace(0, 2*np.pi, 70)
    >>> circle = np.column_stack([50 + 20*np.cos(theta), 50 + 20*np.sin(theta)])
    >>> outliers = np.random.rand(30, 2) * 100
    >>> edgels = np.vstack([circle, outliers])
    >>> center, radius = greco_2022(edgels, gamma=0.35)
    >>> print(f"Center: {center}, Radius: {radius:.2f}")
    """
    # Input validation
    if len(edgels) < 3:
        return np.array([-1, -1]), -1
    
    n = len(edgels)
    
    # Validate gamma
    if not 0 <= gamma < 0.5:
        raise ValueError(f"gamma must be in [0, 0.5), got {gamma}")
    
    # Calculate number of points to use (after trimming)
    h = int(n * (1 - gamma))
    h = max(h, 3)  # Ensure at least 3 points
    
    # Initial estimate using all points
    center_init = np.mean(edgels, axis=0)
    distances_init = np.sqrt(np.sum((edgels - center_init)**2, axis=1))
    radius_init = np.mean(distances_init)
    
    params = np.array([center_init[0], center_init[1], radius_init])
    
    # Iterative optimization
    for iteration in range(max_iterations):
        old_params = params.copy()
        
        # Minimize the objective function using Nelder-Mead
        result = minimize(
            _objective_function, 
            params, 
            args=(edgels, h), 
            method='Nelder-Mead',
            options={'xatol': tol, 'fatol': tol}
        )
        params = result.x
        
        # Check convergence
        param_change = np.max(np.abs(params - old_params))
        if param_change < tol:
            break
    
    # Extract final parameters
    center_x, center_y, radius = params
    center = np.array([center_x, center_y])
    
    return center, float(radius)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        # Run built-in tests
        print("Testing Greco 2022 (Trimmed AMLE Circle Fitting)")
        print("=" * 60)
        
        # Test 1: Perfect circle
        print("\nTest 1: Perfect circle (no outliers)")
        theta = np.linspace(0, 2*np.pi, 100)
        true_center = (50, 50)
        true_radius = 20
        edgels = np.column_stack([
            true_center[0] + true_radius * np.cos(theta),
            true_center[1] + true_radius * np.sin(theta)
        ])
        
        center, radius = greco_2022(edgels, gamma=0.0)  # No trimming needed
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 2: Noisy circle
        print("\nTest 2: Noisy circle")
        noise = np.random.randn(100, 2) * 1.5
        edgels_noisy = edgels + noise
        
        center, radius = greco_2022(edgels_noisy, gamma=0.1)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 3: Circle with 20% outliers
        print("\nTest 3: Circle with 20% outliers (gamma=0.25)")
        theta_circle = np.linspace(0, 2*np.pi, 80)
        circle_points = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_circle),
            true_center[1] + true_radius * np.sin(theta_circle)
        ])
        outliers = np.random.rand(20, 2) * 100
        edgels_outliers = np.vstack([circle_points, outliers])
        
        center, radius = greco_2022(edgels_outliers, gamma=0.25)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 4: Circle with 40% outliers (stress test)
        print("\nTest 4: Stress test - 40% outliers (gamma=0.45)")
        theta_circle = np.linspace(0, 2*np.pi, 60)
        circle_points = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_circle),
            true_center[1] + true_radius * np.sin(theta_circle)
        ])
        outliers = np.random.rand(40, 2) * 100
        edgels_extreme = np.vstack([circle_points, outliers])
        
        center, radius = greco_2022(edgels_extreme, gamma=0.45)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 5: Small example
        print("\nTest 5: Small dataset")
        small_edgels = np.array([
            [2, 0], [4, 0], [6, 2], [6, 4],
            [4, 6], [2, 6], [0, 4], [0, 2]
        ], dtype=float)
        
        center, radius = greco_2022(small_edgels, gamma=0.0)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        print(f"Expected: center≈(3, 3), radius≈3.16")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
    else:
        # Load edgels from CSV file
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        center, radius = greco_2022(edgels)
        print(f"center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")