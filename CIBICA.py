"""
CIBICA: Circle fitting using triplet combinations with median estimation

Reference:
    This implementation uses geometric circle fitting through three points.
    The median-based approach provides robustness to outliers.
"""

import numpy as np
import random
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy import stats


def vectorized_XYR(p1, p2, p3, xmax=50, ymax=50):
    """
    Vectorized circle fitting through triplets of points.
    
    This is the core CIBICA function that fits circles through multiple
    triplets simultaneously and filters invalid results.
    
    Parameters
    ----------
    p1, p2, p3 : numpy.ndarray
        Arrays of points, shape (N, 2) where N is number of triplets
    xmax, ymax : int
        Maximum image dimensions for filtering
        
    Returns
    -------
    cx, cy, radius : numpy.ndarray
        Arrays of circle centers and radii after filtering
    """
    temp = p2[:, 0] * p2[:, 0] + p2[:, 1] * p2[:, 1]
    bc = (p1[:, 0] * p1[:, 0] + p1[:, 1] * p1[:, 1] - temp) / 2
    cd = (temp - p3[:, 0] * p3[:, 0] - p3[:, 1] * p3[:, 1]) / 2
    det = (p1[:, 0] - p2[:, 0]) * (p2[:, 1] - p3[:, 1]) - (p2[:, 0] - p3[:, 0]) * (p1[:, 1] - p2[:, 1])

    # Remove collinear points
    zeros = np.where(det == 0)
    p1 = np.delete(p1, zeros, 0)
    p2 = np.delete(p2, zeros, 0)
    p3 = np.delete(p3, zeros, 0)
    bc = np.delete(bc, zeros, 0)
    cd = np.delete(cd, zeros, 0)
    det = np.delete(det, zeros, 0)
    
    cx = (bc * (p2[:, 1] - p3[:, 1]) - cd * (p1[:, 1] - p2[:, 1])) / det
    cy = ((p1[:, 0] - p2[:, 0]) * cd - (p2[:, 0] - p3[:, 0]) * bc) / det

    # Filter out-of-bounds circles
    deletes = np.where(cx < 0 - 20)
    cx = np.delete(cx, deletes, 0)
    cy = np.delete(cy, deletes, 0)
    p1 = np.delete(p1, deletes, 0)
    
    deletes = np.where(cy < 0 - 20)
    cx = np.delete(cx, deletes, 0)
    cy = np.delete(cy, deletes, 0)
    p1 = np.delete(p1, deletes, 0)
    
    deletes = np.where(cx > xmax + 20)
    cx = np.delete(cx, deletes, 0)
    cy = np.delete(cy, deletes, 0)
    p1 = np.delete(p1, deletes, 0)
    
    deletes = np.where(cy > ymax + 20)
    cx = np.delete(cx, deletes, 0)
    cy = np.delete(cy, deletes, 0)
    p1 = np.delete(p1, deletes, 0)

    # Calculate radius and filter invalid sizes
    radius = np.sqrt((cx - p1[:, 0])**2 + (cy - p1[:, 1])**2)
    
    big_radius = np.where(radius > 30)
    radius = np.delete(radius, big_radius, 0)
    cx = np.delete(cx, big_radius, 0)
    cy = np.delete(cy, big_radius, 0)

    small_radius = np.where(radius < 4)
    radius = np.delete(radius, small_radius, 0)
    cx = np.delete(cx, small_radius, 0)
    cy = np.delete(cy, small_radius, 0)
    
    return cx, cy, radius


def median_3d(x, y, r, xmax=100, ymax=100):
    """
    Calculate median in 3D space (x, y, r) using mode-based approach.
    
    Parameters
    ----------
    x, y, r : numpy.ndarray
        Arrays of circle parameters
    xmax, ymax : int
        Maximum dimensions for encoding
        
    Returns
    -------
    x_out, y_out, r_out : float
        Median circle parameters
    """
    X = np.round(np.c_[r, x, y], 0)
    c = np.array([[ymax * xmax], [ymax], [1]])
    identifier = X.dot(c)
    
    data = stats.mode(identifier, keepdims=True)
    mode = int(data.mode.ravel()[0])
    
    y_out = mode % ymax
    aux = (mode - y_out) / ymax
    x_out = aux % xmax
    r_out = (aux - x_out) / xmax
    
    return x_out, y_out, r_out


def LS_circle(x, y):
    """
    Least squares circle fitting.
    
    Parameters
    ----------
    x, y : numpy.ndarray
        Coordinates of points on the circle
        
    Returns
    -------
    xc, yc : float
        Circle center coordinates
    r : float
        Circle radius
    residu : float
        Sum of squared residuals
    """
    A = np.ones((len(x), 3))
    A[:, 0] = 2 * x
    A[:, 1] = 2 * y
    b = x**2 + y**2
    X = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))
    xc, yc = X[0], X[1]
    r = np.sqrt(X[2] + xc**2 + yc**2)
    residu = np.sum((np.sqrt((x - xc)**2 + (y - yc)**2) - r)**2)
    return xc, yc, r, residu


def CIBICA(coord, n_triplets=500, xmax=50, ymax=50, refinement=True):
    """
    Circle detection using CIBICA method.
    
    Main function that implements the complete CIBICA algorithm:
    1. Sample random triplets of edge points
    2. Fit circles through each triplet
    3. Estimate final circle using median
    4. Optionally refine using least squares
    
    Parameters
    ----------
    coord : numpy.ndarray
        Edge point coordinates, shape (N, 2) as [[row,col], [row,col], ...]
    n_triplets : int
        Number of random triplets to sample (default: 10000)
    xmax, ymax : int
        Image dimensions for filtering (default: 50, 50)
    refinement : bool
        Whether to apply least squares refinement (default: True)
        
    Returns
    -------
    x, y, r : float
        Circle center (x, y) and radius r
        
    Examples
    --------
    >>> from preprocessing import extract_edgels
    >>> edgels, _ = extract_edgels(image, method='green_level', threshold=76)
    >>> x, y, r = CIBICA(edgels, n_triplets=10000)
    """
    if len(coord) < 3:
        return np.nan, np.nan, np.nan
    
    # Generate combinations and sample
    combi = list(combinations(np.arange(len(coord)), 3))
    N = min(n_triplets, len(combi))
    RandomSample = np.array(random.sample(combi, N))
    
    # Get triplet coordinates
    p1 = coord[RandomSample[:, 0]]
    p2 = coord[RandomSample[:, 1]]
    p3 = coord[RandomSample[:, 2]]
    
    # Fit circles
    cx, cy, radius = vectorized_XYR(p1, p2, p3, xmax, ymax)
    
    # Get median estimate
    XYR = median_3d(cx, cy, radius, xmax, ymax)
    
    if not refinement:
        return XYR[1], XYR[0], XYR[2]  # Return as x, y, r
    
    # Least squares refinement
    coord2 = [(XYR[0], XYR[1])]
    distances = cdist(coord2, coord)
    near = np.where(np.abs(cdist(coord2, coord) - XYR[2]) < 1.5)
    circle_points = coord[near[1]]
    
    if len(circle_points) >= 3:
        xl, yl, rl, res = np.round(LS_circle(circle_points[:, 0], circle_points[:, 1]), 3)
        return yl, xl, rl  # Return as x, y, r
    else:
        return XYR[1], XYR[0], XYR[2]  # Return as x, y, r

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        # Run built-in tests
        print("Testing CIBICA")
        print("=" * 60)
        
        # Test 1: Perfect circle
        print("\nTest 1: Perfect circle")
        theta = np.linspace(0, 2*np.pi, 100)
        true_center = (25, 25)
        true_radius = 10
        edgels = np.column_stack([
            true_center[0] + true_radius * np.cos(theta),
            true_center[1] + true_radius * np.sin(theta)
        ])
        
        x, y, r = CIBICA(edgels, n_triplets=500, xmax=50, ymax=50)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({x:.2f}, {y:.2f}), radius={r:.2f}")
        error = np.sqrt((x-true_center[0])**2 + (y-true_center[1])**2)
        print(f"Error: {error:.4f} pixels")
        
        # Test 2: Noisy circle
        print("\nTest 2: Noisy circle")
        noise = np.random.randn(100, 2) * 0.5
        edgels_noisy = edgels + noise
        
        x, y, r = CIBICA(edgels_noisy, n_triplets=1000, xmax=50, ymax=50)
        print(f"Detected: center=({x:.2f}, {y:.2f}), radius={r:.2f}")
        error = np.sqrt((x-true_center[0])**2 + (y-true_center[1])**2)
        print(f"Error: {error:.4f} pixels")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
    else:
        # Load edgels from CSV file
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        x, y, r = CIBICA(edgels, n_triplets=10000, xmax=100, ymax=100)
        print(f"center=({x:.2f}, {y:.2f}), radius={r:.2f}")