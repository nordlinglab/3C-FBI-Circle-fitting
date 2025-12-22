"""
CIBICA Circle Detection Module
Cleaned version with essential functions only.
"""

import numpy as np
import random
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy import stats

def vectorized_XYR(p1, p2, p3, xmax=50, ymax=50, rmin=5, rmax=40, minval=0):
    """
    Optimized version of circle calculation from three points.
    Uses single mask array for filtering and fewer memory allocations.
    
    Parameters:
    -----------
    p1, p2, p3 : numpy.ndarray
        Arrays of shape (n,2) containing x,y coordinates of points
    xmax, ymax : float
        Maximum allowed x and y coordinates
    rmin, rmax : float
        Minimum and maximum allowed radius
    minval : float
        Minimum allowed coordinate value
        
    Returns:
    --------
    cx, cy : numpy.ndarray
        x and y coordinates of valid circle centers
    radius : numpy.ndarray
        Radii of valid circles
    """
    # Initial calculations
    temp = p2[:, 0]**2 + p2[:, 1]**2
    bc = (p1[:, 0]**2 + p1[:, 1]**2 - temp) / 2
    cd = (temp - p3[:, 0]**2 - p3[:, 1]**2) / 2
    det = (p1[:, 0] - p2[:, 0]) * (p2[:, 1] - p3[:, 1]) - (p2[:, 0] - p3[:, 0]) * (p1[:, 1] - p2[:, 1])

    # Create initial mask for non-zero determinant
    mask = abs(det) > 0.001

    if not np.any(mask):
        return np.array((-1, -1)), np.array((-1, -1)), np.array((-1, -1))

    # Calculate centers only for valid determinants
    cx = np.zeros_like(det)
    cy = np.zeros_like(det)
    
    cx[mask] = (bc[mask] * (p2[mask, 1] - p3[mask, 1]) - cd[mask] * (p1[mask, 1] - p2[mask, 1])) / det[mask]
    cy[mask] = ((p1[mask, 0] - p2[mask, 0]) * cd[mask] - (p2[mask, 0] - p3[mask, 0]) * bc[mask]) / det[mask]

    # Update mask with boundary conditions
    mask &= (cx >= minval) & (cy >= minval)
    mask &= (cx <= xmax + 20) & (cy <= ymax + 20)

    if not np.any(mask):
        return np.array((-1, -1)), np.array((-1, -1)), np.array((-1, -1))

    # Calculate radius only for remaining valid points
    radius = np.zeros_like(det)
    radius[mask] = np.sqrt((cx[mask] - p1[mask, 0])**2 + (cy[mask] - p1[mask, 1])**2)

    # Update mask with radius conditions
    mask &= (radius >= rmin) & (radius <= rmax)

    if not np.any(mask):
        return np.array((-1, -1)), np.array((-1, -1)), np.array((-1, -1))

    # Return only the valid results
    return cx[mask], cy[mask], radius[mask]


def median_3d(x, y, r, xmax=500, ymax=500):
    """
    Median in 3D using mode calculation.
    
    Parameters:
    -----------
    x, y, r : numpy arrays
        Coordinates and radii
    xmax, ymax : int
        Maximum x and y values for encoding
        
    Returns:
    --------
    x_out, y_out, r_out : float
        Mode values for x, y, and radius
    """
    X = np.round(np.c_[r, x, y], 0)
    c = np.array([[ymax * xmax], [ymax], [1]])
    identifier = X.dot(c)
    
    data = stats.mode(identifier)
    
    # Handle different scipy versions
    try:
        # Newer scipy versions (>= 1.9.0)
        mode = data.mode[0]
    except:
        # Older scipy versions
        try:
            mode = data[0][0][0]
        except:
            mode = data[0][0]
    
    y_out = mode % ymax
    aux = (mode - y_out) / ymax
    x_out = aux % xmax
    r_out = (aux - x_out) / xmax
    
    return x_out, y_out, r_out

def LS_circle(x, y):
    """
    Least squares circle fitting.
    
    Parameters:
    -----------
    x : array-like
        x coordinates of points
    y : array-like
        y coordinates of points

    Returns:
    --------
    xc : float
        x coordinate of center
    yc : float
        y coordinate of center
    r : float
        radius of circle
    residu : float
        residual value (sum of squared distances)
    """
    A = np.ones((len(x), 3))
    A[:, 0] = 2 * x
    A[:, 1] = 2 * y
    b = x**2 + y**2
    
    X = np.linalg.solve(A.T @ A, A.T @ b)  # More stable than inv()
    xc, yc = X[0], X[1]
    r = np.sqrt(X[2] + xc**2 + yc**2)
    
    # Calculate residual
    residu = np.sum((np.sqrt((x - xc)**2 + (y - yc)**2) - r)**2)
    
    return xc, yc, r, residu


def CIBICA(edgels, Nmax=500):
    """
    Perform CIBICA (CIrcle BAsed CAmera calibration) operation on edge points.
    
    Parameters:
    -----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing the x and y coordinates of edge pixels
    Nmax : int
        Maximum number of random triplets to sample (default 500)
    
    Returns:
    --------
    center : numpy.ndarray
        Array [xc, yc] with circle center coordinates
    rc : float
        Radius of the detected circle
    """
    x, y = edgels[:, 0], edgels[:, 1]
    NumberOfPixels = len(y)
    coord = np.column_stack((x, y))
    
    # Generate all possible triplets
    allTriplets = list(combinations(np.arange(NumberOfPixels), 3))
    N = min(Nmax, len(allTriplets))
    
    # Random sampling of triplets
    RandomSample = np.array(random.sample(allTriplets, N))
    
    # Get points for each triplet
    p1 = coord[RandomSample[:, 0]]
    p2 = coord[RandomSample[:, 1]]
    p3 = coord[RandomSample[:, 2]]
    
    # Calculate circles from triplets
    cx, cy, radius = vectorized_XYR(p1, p2, p3)
    
    # Find the median circle
    XYR = median_3d(cx, cy, radius)
    
    # Find points near the median circle
    coord2 = [(XYR[0], XYR[1])]
    near = np.where(np.abs(cdist(coord2, coord) - XYR[2]) < 1.5)
    circle_points = coord[near[1]]
    
    # Refine circle using least squares
    xc, yc, rc, _ = np.round(LS_circle(circle_points[:, 0], circle_points[:, 1]), 3)
    
    return np.array([xc, yc]), rc


if __name__ == "__main__":
    print("CIBICA Circle Detection Module")
    print("=" * 40)
    print("Essential functions:")
    print("  - CIBICA: Main circle detection function")
    print("  - vectorized_XYR: Circle calculation from point triplets")
    print("  - median_3d: Mode-based median in 3D space")
    print("  - median_3d_R: Radius-weighted median in 3D")
    print("  - LS_circle: Least squares circle fitting")
    print("\nExample usage:")
    print("  edgels = np.array([[x1, y1], [x2, y2], ...])")
    print("  center, radius = CIBICA(edgels, Nmax=500)")
