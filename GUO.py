"""
Guo 2019: Robust Circle Fitting with Iterative Outlier Removal

Implements a robust circle fitting algorithm that combines Taubin's algebraic
fit with iterative outlier detection using Median Absolute Deviation (MAD).

Reference: Guo et al. (2019)

Main function: guo_2019(edgels, **kwargs)
"""

import numpy as np
import scipy.linalg as la


def _taubin_fit(points):
    """
    Fit a circle using Taubin's algebraic method.
    
    Solves the generalized eigenvalue problem for algebraic circle fitting.
    This method is more stable than basic least squares.
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates
        
    Returns
    -------
    center : tuple of float
        Circle center (cx, cy)
    radius : float
        Circle radius
    """
    xi, yi = points[:, 0], points[:, 1]
    zi = xi**2 + yi**2
    
    # Construct design matrix Z = [zi, xi, yi, 1]
    Z = np.column_stack([zi, xi, yi, np.ones_like(xi)])
    
    # Solve generalized eigenvalue problem: A * v = λ * B * v
    A = np.dot(Z.T, Z)
    B = np.eye(4)
    
    evals, evecs = la.eig(A, B)
    
    # Select eigenvector with minimum eigenvalue
    idx = np.argmin(evals)
    coefs = np.real(evecs[:, idx])
    
    # Extract circle parameters from coefficients
    # Circle equation: coefs[0]*(x²+y²) + coefs[1]*x + coefs[2]*y + coefs[3] = 0
    a = -coefs[1] / (2 * coefs[0])
    b = -coefs[2] / (2 * coefs[0])
    R = np.sqrt((coefs[1]**2 + coefs[2]**2 - 4 * coefs[0] * coefs[3]) / 
                (4 * coefs[0]**2))
    
    return (a, b), R


def _calculate_signed_distances(points, center, radius):
    """
    Calculate signed distances from points to circle.
    
    Positive distances indicate points outside the circle,
    negative distances indicate points inside.
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates
    center : tuple of float
        Circle center (cx, cy)
    radius : float
        Circle radius
        
    Returns
    -------
    distances : numpy.ndarray
        Signed distances from each point to the circle
    """
    xi, yi = points[:, 0], points[:, 1]
    cx, cy = center
    
    # Distance from point to center minus radius
    distances = np.sqrt((xi - cx)**2 + (yi - cy)**2) - radius
    
    return distances


def guo_2019(edgels, max_iterations=10, threshold=3):
    """
    Fit a circle to edge points using Guo 2019 robust method.
    
    This algorithm iteratively:
    1. Fits a circle using Taubin's algebraic method
    2. Identifies outliers using Median Absolute Deviation (MAD)
    3. Removes outliers and repeats until convergence
    
    The MAD-based outlier detection is more robust than standard deviation
    and works well even with 50% outliers.
    
    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    max_iterations : int, optional
        Maximum number of outlier removal iterations (default: 10)
    threshold : float, optional
        MAD threshold multiplier for outlier detection (default: 3)
        Points with |distance - median| > threshold * MAD are outliers
        
    Returns
    -------
    center : numpy.ndarray
        Array [cx, cy] of circle center coordinates
    radius : float
        Circle radius
        
    Notes
    -----
    The algorithm stops early if no outliers are detected in an iteration.
    
    The MAD (Median Absolute Deviation) is computed as:
        MAD = median(|distances - median(distances)|) / 0.6745
    The factor 0.6745 makes MAD comparable to standard deviation for normal distributions.
    
    Typical threshold values:
    - threshold=2.5: More aggressive outlier removal
    - threshold=3.0: Balanced (default, ~99.7% confidence for normal data)
    - threshold=3.5: More conservative
    
    Examples
    --------
    >>> # Circle with outliers
    >>> theta = np.linspace(0, 2*np.pi, 50)
    >>> circle_points = np.column_stack([20*np.cos(theta), 20*np.sin(theta)])
    >>> outliers = np.random.rand(10, 2) * 50
    >>> edgels = np.vstack([circle_points, outliers])
    >>> center, radius = guo_2019(edgels)
    >>> print(f"Center: {center}, Radius: {radius:.2f}")
    """
    # Input validation
    if len(edgels) < 3:
        return np.array([-1, -1]), -1
    
    points = np.array(edgels, dtype=float)
    
    for iteration in range(max_iterations):
        # Fit circle to current inlier set
        center, radius = _taubin_fit(points)
        
        # Calculate signed distances from points to fitted circle
        distances = _calculate_signed_distances(points, center, radius)
        
        # Compute robust statistics using MAD
        median_distance = np.median(distances)
        mad = np.median(np.abs(distances - median_distance)) / 0.6745
        
        # Identify outliers: points far from median distance
        outliers = np.abs(distances - median_distance) >= threshold * mad
        
        # Stop if no outliers found
        if not np.any(outliers):
            break
        
        # Remove outliers for next iteration
        points = points[~outliers]
        
        # Safety check: ensure enough points remain
        if len(points) < 3:
            # Restore previous valid fit
            break
    
    # Return final result
    center_array = np.array([center[0], center[1]])
    return center_array, radius


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        # Run built-in tests
        print("Testing Guo 2019 (Robust Circle Fitting)")
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
        
        center, radius = guo_2019(edgels)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 2: Noisy circle
        print("\nTest 2: Noisy circle")
        noise = np.random.randn(100, 2) * 2.0
        edgels_noisy = edgels + noise
        
        center, radius = guo_2019(edgels_noisy)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 3: Circle with 20% outliers
        print("\nTest 3: Circle with 20% outliers")
        theta_circle = np.linspace(0, 2*np.pi, 80)
        circle_points = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_circle),
            true_center[1] + true_radius * np.sin(theta_circle)
        ])
        outliers = np.random.rand(20, 2) * 100
        edgels_outliers = np.vstack([circle_points, outliers])
        
        center, radius = guo_2019(edgels_outliers)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 4: Example from original code
        print("\nTest 4: Original example with intentional outliers")
        data_points = np.array([
            [0.5, 1.5], [2.1, 2.3], [2.7, 3.1], [1.5, 2.0],
            [1.8, 1.8], [3.0, 3.3], [2.5, 2.9], [3.5, 4.1],
            [10.0, 10.0], [11.0, 11.0]  # outliers
        ])
        center, radius = guo_2019(data_points)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        print(f"(Outliers at (10,10) and (11,11) should be rejected)")
        
        # Test 5: 50% outliers (stress test)
        print("\nTest 5: Stress test - 50% outliers")
        theta_circle = np.linspace(0, 2*np.pi, 50)
        circle_points = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_circle),
            true_center[1] + true_radius * np.sin(theta_circle)
        ])
        outliers = np.random.rand(50, 2) * 100
        edgels_extreme = np.vstack([circle_points, outliers])
        
        center, radius = guo_2019(edgels_extreme, threshold=3)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
    else:
        # Load edgels from CSV file
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        center, radius = guo_2019(edgels)
        print(f"center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")