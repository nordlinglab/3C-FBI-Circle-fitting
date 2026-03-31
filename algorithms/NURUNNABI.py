"""
Nurunnabi: Fast Hyper-Accurate Circle Fitting with LTS

Implements Nurunnabi's fast circle fitting algorithm combining hyperaccurate
algebraic fitting with Least Trimmed Squares (LTS) for robustness.

Reference: Nurunnabi et al.

Main function: nurunnabi(edgels, **kwargs)
"""

import numpy as np


def _fast_hyper_fit(points):
    """
    Fast hyperaccurate circle fit using direct algebraic method.
    
    This method uses centered coordinates and solves a 2x2 linear system
    instead of a full eigenvalue problem, making it very fast while
    maintaining high accuracy.
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates
        
    Returns
    -------
    center : numpy.ndarray
        Circle center [cx, cy]
    radius : float
        Circle radius
        
    Notes
    -----
    Falls back to mean-based estimate if the linear system is singular
    (which happens when points are perfectly aligned).
    """
    x = points[:, 0]
    y = points[:, 1]
    z = x**2 + y**2
    
    # Compute mean values
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Center the data (improves numerical stability)
    u = x - mean_x
    v = y - mean_y
    
    # Compute required sums for the linear system
    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suv = np.sum(u * v)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    
    # Solve 2x2 linear system: A * center_offset = b
    A = np.array([[Suu, Suv], 
                  [Suv, Svv]])
    b = np.array([Suuu + Suvv, Svvv + Suuv]) / 2
    
    try:
        # Solve for center offset in centered coordinates
        center_offset = np.linalg.solve(A, b)
        
        # Convert back to original coordinates
        center = np.array([center_offset[0] + mean_x, 
                          center_offset[1] + mean_y])
        
        # Compute radius as RMS distance from center
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        radius = np.sqrt(np.mean(distances**2))
        
        return center, float(radius)
        
    except np.linalg.LinAlgError:
        # Fallback if matrix is singular (collinear points)
        center = np.array([mean_x, mean_y])
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        radius = np.mean(distances)
        return center, float(radius)


def nurunnabi(edgels, h_proportion=0.7, max_iter=25):
    """
    Fit a circle to edge points using Nurunnabi's fast LTS method.
    
    This algorithm combines fast hyperaccurate algebraic fitting with
    Least Trimmed Squares (LTS) for robustness. It repeatedly:
    1. Samples 3 random points
    2. Fits an initial circle
    3. Selects h closest points (trimming outliers)
    4. Refits using only the h closest points
    5. Keeps the fit with lowest total squared error
    
    Algorithm steps:
    1. Sample 3 random points as initial hypothesis
    2. Fit circle using fast hyperaccurate method
    3. Find h points closest to this circle
    4. Refit circle using these h points
    5. Compute error on all points
    6. Repeat and keep best result
    
    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    h_proportion : float, optional
        Proportion of points to consider as inliers (default: 0.7)
        Range: 0.5 (50% inliers) to 1.0 (all points)
        Higher values = less robust but uses more data
        Lower values = more robust but may underfit
    max_iter : int, optional
        Maximum number of random sampling iterations (default: 25)
        
    Returns
    -------
    center : numpy.ndarray
        Array [cx, cy] of circle center coordinates
    radius : float
        Circle radius
        
    Notes
    -----
    The algorithm is very fast due to:
    - Fast algebraic fitting (no eigenvalue decomposition)
    - Small number of iterations (typically 25 is sufficient)
    - Efficient numpy operations
    
    The LTS approach provides robustness by:
    - Automatically identifying and excluding outliers
    - Not requiring explicit outlier detection
    - Being robust to up to (1 - h_proportion) * 100% outliers
    
    Recommended h_proportion values:
    - 0.5-0.6: Very robust (handles 40-50% outliers)
    - 0.7: Balanced (default, handles 30% outliers)
    - 0.8-0.9: Less robust but more accurate on clean data
    - 1.0: No trimming (not recommended, use algebraic fit instead)
    
    Performance characteristics:
    - Very fast: ~0.1-1ms for 100 points
    - Good accuracy on clean data
    - Robust to moderate outliers
    - Works well on partial arcs
    
    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> edgels = np.column_stack([50 + 20*np.cos(theta), 50 + 20*np.sin(theta)])
    >>> center, radius = nurunnabi(edgels, h_proportion=0.7, max_iter=25)
    >>> print(f"Center: {center}, Radius: {radius:.2f}")
    """
    # Input validation
    if len(edgels) < 3:
        return np.array([-1, -1]), -1
    
    n_points = len(edgels)
    
    # Calculate number of inliers to use
    h = int(np.ceil(h_proportion * n_points))
    h = max(h, 3)  # Ensure at least 3 points
    
    best_error = np.inf
    best_center = None
    best_radius = None
    
    for iteration in range(max_iter):
        # Sample 3 random points for initial hypothesis
        sample_idx = np.random.choice(n_points, 3, replace=False)
        sample = edgels[sample_idx]
        
        # Initial fit on 3 points
        center, radius = _fast_hyper_fit(sample)
        
        # Compute geometric distances from all points to circle
        distances_to_center = np.sqrt((edgels[:, 0] - center[0])**2 + 
                                      (edgels[:, 1] - center[1])**2)
        residuals = np.abs(distances_to_center - radius)
        
        # Select h points with smallest residuals (trim outliers)
        closest_indices = np.argsort(residuals)[:h]
        closest_points = edgels[closest_indices]
        
        # Refit using only the h closest points
        center, radius = _fast_hyper_fit(closest_points)
        
        # Compute total squared error on all points
        distances_to_center = np.sqrt((edgels[:, 0] - center[0])**2 + 
                                      (edgels[:, 1] - center[1])**2)
        squared_errors = (distances_to_center - radius)**2
        total_error = np.sum(squared_errors)
        
        # Update best result if this is better
        if total_error < best_error:
            best_error = total_error
            best_center = center
            best_radius = radius
    
    return best_center, best_radius


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        # Run built-in tests
        print("Testing Nurunnabi (Fast Hyper-Accurate Circle Fitting)")
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
        
        center, radius = nurunnabi(edgels, h_proportion=1.0, max_iter=25)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.6f}, Radius error: {abs(radius-true_radius):.6f}")
        
        # Test 2: Noisy circle
        print("\nTest 2: Noisy circle")
        noise = np.random.randn(100, 2) * 1.5
        edgels_noisy = edgels + noise
        
        center, radius = nurunnabi(edgels_noisy, h_proportion=0.8, max_iter=25)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 3: Circle with 20% outliers
        print("\nTest 3: Circle with 20% outliers (h_proportion=0.7)")
        theta_circle = np.linspace(0, 2*np.pi, 80)
        circle_points = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_circle),
            true_center[1] + true_radius * np.sin(theta_circle)
        ])
        outliers = np.random.rand(20, 2) * 100
        edgels_outliers = np.vstack([circle_points, outliers])
        
        center, radius = nurunnabi(edgels_outliers, h_proportion=0.7, max_iter=50)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 4: Small example
        print("\nTest 4: Small dataset")
        small_edgels = np.array([
            [2, 0], [4, 0], [6, 2], [6, 4],
            [4, 6], [2, 6], [0, 4], [0, 2]
        ], dtype=float)
        
        center, radius = nurunnabi(small_edgels, h_proportion=1.0, max_iter=25)
        print(f"Detected: center=({center[0]:.4f}, {center[1]:.4f}), radius={radius:.4f}")
        print(f"Expected: center≈(3, 3), radius≈3.1623")
        
        # Test 5: Effect of h_proportion
        print("\nTest 5: Testing h_proportion (30% outliers)")
        
        print("  With h_proportion=0.5 (very robust):")
        center, radius = nurunnabi(edgels_outliers, h_proportion=0.5, max_iter=50)
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"    Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        print("  With h_proportion=0.7 (balanced):")
        center, radius = nurunnabi(edgels_outliers, h_proportion=0.7, max_iter=50)
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"    Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        print("  With h_proportion=0.9 (less robust):")
        center, radius = nurunnabi(edgels_outliers, h_proportion=0.9, max_iter=50)
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"    Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 6: Speed test
        print("\nTest 6: Speed comparison")
        import time
        
        # Generate larger dataset
        theta_large = np.linspace(0, 2*np.pi, 500)
        edgels_large = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_large),
            true_center[1] + true_radius * np.sin(theta_large)
        ])
        
        start = time.time()
        center, radius = nurunnabi(edgels_large, h_proportion=0.7, max_iter=25)
        elapsed = time.time() - start
        print(f"  500 points, 25 iterations: {elapsed*1000:.2f} ms")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
    else:
        # Load edgels from CSV file
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        center, radius = nurunnabi(edgels)
        print(f"center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")