"""
CIBICA: Circle Detection using Ballot Inspection and Least Squares Refinement

Implements the CIBICA (Circle Inspection using Ballot Inspection and Circle Alignment)
algorithm which combines random triplet sampling, median-based voting, and least
squares refinement.

Main function: cibica(edgels, **kwargs)
"""

import numpy as np
from itertools import combinations
import random
from scipy.spatial.distance import cdist


def _vectorized_circle_from_triplets(p1, p2, p3, rmin=5, rmax=40):
    """
    Calculate circle parameters from arrays of point triplets.
    
    Vectorized implementation that processes multiple triplets simultaneously
    and filters results based on geometric constraints.
    
    Parameters
    ----------
    p1, p2, p3 : numpy.ndarray
        Arrays of shape (n, 2) containing (x, y) coordinates of point triplets
    rmin, rmax : float, optional
        Minimum and maximum allowed radius (default: 5, 40)
        
    Returns
    -------
    cx, cy : numpy.ndarray
        x and y coordinates of valid circle centers
    radius : numpy.ndarray
        Radii of valid circles
    """
    # Determine image bounds from input points
    xmax = max(p1[:, 0].max(), p2[:, 0].max(), p3[:, 0].max())
    ymax = max(p1[:, 1].max(), p2[:, 1].max(), p3[:, 1].max())
    
    # Calculate circle parameters using determinant method
    temp = p2[:, 0]**2 + p2[:, 1]**2
    bc = (p1[:, 0]**2 + p1[:, 1]**2 - temp) / 2
    cd = (temp - p3[:, 0]**2 - p3[:, 1]**2) / 2
    det = (p1[:, 0] - p2[:, 0]) * (p2[:, 1] - p3[:, 1]) - \
          (p2[:, 0] - p3[:, 0]) * (p1[:, 1] - p2[:, 1])
    
    # Create mask for non-degenerate triangles
    mask = np.abs(det) > 1e-6
    
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([])
    
    # Calculate centers
    cx = np.zeros_like(det)
    cy = np.zeros_like(det)
    
    cx[mask] = (bc[mask] * (p2[mask, 1] - p3[mask, 1]) - 
                cd[mask] * (p1[mask, 1] - p2[mask, 1])) / det[mask]
    cy[mask] = ((p1[mask, 0] - p2[mask, 0]) * cd[mask] - 
                (p2[mask, 0] - p3[mask, 0]) * bc[mask]) / det[mask]
    
    # Apply boundary constraints (center must be within reasonable bounds)
    mask &= (cx >= -rmax) & (cy >= -rmax)
    mask &= (cx <= xmax + rmax) & (cy <= ymax + rmax)
    
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([])
    
    # Calculate radius (distance from center to first point)
    radius = np.zeros_like(det)
    radius[mask] = np.sqrt((cx[mask] - p1[mask, 0])**2 + 
                           (cy[mask] - p1[mask, 1])**2)
    
    # Apply radius constraints
    mask &= (radius >= rmin) & (radius <= rmax)
    
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([])
    
    return cx[mask], cy[mask], radius[mask]


def _median_3d_weighted_by_radius(x, y, r):
    """
    Find the mode in 3D parameter space with radius-weighted voting.
    
    Points are rounded to integers and votes are weighted by 1/radius,
    giving preference to consistent small circles over large circles.
    
    Parameters
    ----------
    x, y, r : numpy.ndarray
        Arrays of circle parameters (centers and radii)
        
    Returns
    -------
    tuple
        (x_mode, y_mode, r_mode) - most common parameter combination
    """
    # Round to integer bins for voting
    X = np.round(np.column_stack([x, y, r]), 0).astype(int)
    
    # Convert to tuples for counting
    coord_tuples = [tuple(point) for point in X]
    
    # Get unique coordinates preserving order
    unique_coords = sorted(set(coord_tuples), key=lambda x: coord_tuples.index(x))
    
    # Count occurrences of each coordinate
    counts = [coord_tuples.count(coord) for coord in unique_coords]
    
    # Weight by inverse radius (prefer smaller circles)
    weighted_counts = [count / coord[2] if coord[2] > 0 else 0 
                      for count, coord in zip(counts, unique_coords)]
    
    # Find coordinate with maximum weighted count
    max_idx = np.argmax(weighted_counts)
    x_out, y_out, r_out = unique_coords[max_idx]
    
    return x_out, y_out, r_out


def _least_squares_circle(x, y):
    """
    Fit circle using standard least squares method.
    
    Solves the linear system for the circle equation:
    x² + y² + ax + by + c = 0
    
    Parameters
    ----------
    x, y : numpy.ndarray
        Arrays of point coordinates
        
    Returns
    -------
    xc, yc : float
        Circle center coordinates
    r : float
        Circle radius
    residual : float
        Sum of squared geometric distances
    """
    # Construct design matrix
    A = np.ones((len(x), 3))
    A[:, 0] = 2 * x
    A[:, 1] = 2 * y
    b = x**2 + y**2
    
    # Solve normal equations
    X = np.linalg.solve(A.T @ A, A.T @ b)
    
    xc, yc = X[0], X[1]
    r = np.sqrt(X[2] + xc**2 + yc**2)
    
    # Compute residual
    distances = np.sqrt((x - xc)**2 + (y - yc)**2)
    residual = np.sum((distances - r)**2)
    
    return xc, yc, r, residual


def cibica(edgels, Nmax=500, rmin=5, rmax=40, inlier_threshold=1.5):
    """
    Fit a circle to edge points using CIBICA method.
    
    CIBICA (Circle Inspection using Ballot Inspection and Circle Alignment)
    combines multiple techniques for robust circle detection:
    
    Algorithm steps:
    1. Random sampling: Select N random triplets of edge points
    2. Circle fitting: Fit circles to each triplet
    3. Ballot inspection: Find mode in (cx, cy, r) space with radius weighting
    4. Inlier selection: Select points within threshold of the mode circle
    5. Refinement: Refit using least squares on inliers only
    
    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    Nmax : int, optional
        Maximum number of random triplets to sample (default: 500)
    rmin, rmax : float, optional
        Expected radius range for filtering (default: 5, 40)
    inlier_threshold : float, optional
        Distance threshold for selecting inlier points (default: 1.5)
        Points within this distance from the mode circle are used for refinement
        
    Returns
    -------
    center : numpy.ndarray
        Array [xc, yc] of circle center coordinates
    radius : float
        Circle radius
        
    Notes
    -----
    The algorithm is particularly effective due to:
    - Radius-weighted voting prevents bias toward large circles
    - Least squares refinement improves accuracy on inliers
    - Robust to outliers through voting mechanism
    
    The two-stage approach (voting + refinement) combines robustness
    with accuracy better than either method alone.
    
    Recommended parameter ranges:
    - Nmax: 200-1000 (more samples = more robust but slower)
    - rmin, rmax: Set based on expected circle size in pixels
    - inlier_threshold: 1-3 pixels (tighter = more selective)
    
    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> edgels = np.column_stack([50 + 20*np.cos(theta), 50 + 20*np.sin(theta)])
    >>> center, radius = cibica(edgels, Nmax=500)
    >>> print(f"Center: {center}, Radius: {radius:.2f}")
    """
    # Input validation
    if len(edgels) < 3:
        return np.array([-1, -1]), -1
    
    n_points = len(edgels)
    
    # Generate all possible triplet combinations
    all_triplets = list(combinations(np.arange(n_points), 3))
    N = min(Nmax, len(all_triplets))
    
    # Randomly sample N triplets
    random_sample = np.array(random.sample(all_triplets, N))
    
    # Extract point triplets
    p1 = edgels[random_sample[:, 0]]
    p2 = edgels[random_sample[:, 1]]
    p3 = edgels[random_sample[:, 2]]
    
    # Fit circles to all triplets (vectorized)
    cx, cy, radius = _vectorized_circle_from_triplets(p1, p2, p3, rmin=rmin, rmax=rmax)
    
    # Check if any valid circles were found
    if len(cx) == 0:
        return np.array([-1, -1]), -1
    
    # Find mode in parameter space (radius-weighted voting)
    xc_mode, yc_mode, r_mode = _median_3d_weighted_by_radius(cx, cy, radius)
    
    # Select inlier points near the mode circle
    mode_center = np.array([[xc_mode, yc_mode]])
    distances_to_center = cdist(mode_center, edgels)[0]
    geometric_distances = np.abs(distances_to_center - r_mode)
    
    inlier_mask = geometric_distances < inlier_threshold
    inlier_points = edgels[inlier_mask]
    
    # Check if enough inliers found
    if len(inlier_points) < 3:
        # Fall back to mode result if insufficient inliers
        return np.array([xc_mode, yc_mode]), float(r_mode)
    
    # Refine using least squares on inliers
    xc_refined, yc_refined, r_refined, _ = _least_squares_circle(
        inlier_points[:, 0], 
        inlier_points[:, 1]
    )
    
    # Round to reasonable precision
    xc_refined = np.round(xc_refined, 3)
    yc_refined = np.round(yc_refined, 3)
    r_refined = np.round(r_refined, 3)
    
    return np.array([xc_refined, yc_refined]), float(r_refined)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        # Run built-in tests
        print("Testing CIBICA (Circle Inspection using Ballot Inspection)")
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
        
        center, radius = cibica(edgels, Nmax=500, rmin=10, rmax=30)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 2: Noisy circle
        print("\nTest 2: Noisy circle")
        noise = np.random.randn(100, 2) * 1.5
        edgels_noisy = edgels + noise
        
        center, radius = cibica(edgels_noisy, Nmax=1000, rmin=10, rmax=30)
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
        
        center, radius = cibica(edgels_outliers, Nmax=1000, rmin=10, rmax=30)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 4: Small example
        print("\nTest 4: Small dataset")
        small_edgels = np.array([
            [2, 0], [4, 0], [6, 2], [6, 4],
            [4, 6], [2, 6], [0, 4], [0, 2]
        ], dtype=float)
        
        center, radius = cibica(small_edgels, Nmax=100, rmin=1, rmax=10)
        print(f"Detected: center=({center[0]:.4f}, {center[1]:.4f}), radius={radius:.4f}")
        print(f"Expected: center≈(3, 3), radius≈3.1623")
        
        # Test 5: Partial arc
        print("\nTest 5: Partial arc (90 degrees)")
        theta_arc = np.linspace(0, np.pi/2, 50)
        edgels_arc = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_arc),
            true_center[1] + true_radius * np.sin(theta_arc)
        ])
        
        center, radius = cibica(edgels_arc, Nmax=500, rmin=10, rmax=30)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
    else:
        # Load edgels from CSV file
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        center, radius = cibica(edgels)
        print(f"center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")