"""
Circle Detection using Fast Ballot Inspection (FBI) method.

This module implements the CCC_FBI (Circle Center Calculation - Fast Ballot Inspection)
algorithm for detecting circles in edge data. The algorithm uses:
1. Random sampling of edge point triplets
2. Circle fitting from three points
3. Voting in parameter space (Hough-like approach)
4. 3D convolution for refinement

Main function: ccc_fbi(edgels, **kwargs)
"""

import numpy as np
from collections import Counter
from itertools import combinations
import random
from numba import jit

def vectorized_XYR(p1, p2, p3, xmax=50, ymax=50, rmin=5, rmax=40, minval=0):
    """
    Calculate circle parameters from three points using vectorized operations.
    
    Fits circles to sets of three points and filters results based on geometric constraints.
    Uses the determinant method for circle center calculation.
    
    Parameters
    ----------
    p1, p2, p3 : numpy.ndarray
        Arrays of shape (n, 2) containing (x, y) coordinates of point triplets
    xmax, ymax : float, optional
        Maximum allowed x and y coordinates (default: 50)
    rmin, rmax : float, optional
        Minimum and maximum allowed radius (default: 5, 40)
    minval : float, optional
        Minimum allowed coordinate value (default: 0)
        
    Returns
    -------
    cx, cy : numpy.ndarray
        x and y coordinates of valid circle centers
    radius : numpy.ndarray
        Radii of valid circles
        
    Notes
    -----
    Returns (-1, -1), (-1, -1), (-1, -1) if no valid circles found.
    """
    # Calculate intermediate values for circle center formula
    temp = p2[:, 0]**2 + p2[:, 1]**2
    bc   = (p1[:, 0]**2 + p1[:, 1]**2 - temp) / 2
    cd   = (temp - p3[:, 0]**2 - p3[:, 1]**2) / 2
    det  = (p1[:, 0] - p2[:, 0]) * (p2[:, 1] - p3[:, 1]) - \
           (p2[:, 0] - p3[:, 0]) * (p1[:, 1] - p2[:, 1])

    # Create mask for non-degenerate triangles (non-zero determinant)
    mask = np.abs(det) > 0.001

    if not np.any(mask):
        return np.array([-1, -1]), np.array([-1, -1]), np.array([-1, -1])

    # Calculate circle centers
    cx = np.zeros_like(det)
    cy = np.zeros_like(det)
    
    cx[mask] = (bc[mask] * (p2[mask, 1] - p3[mask, 1]) - 
                cd[mask] * (p1[mask, 1] - p2[mask, 1])) / det[mask]
    cy[mask] = ((p1[mask, 0] - p2[mask, 0]) * cd[mask] - 
                (p2[mask, 0] - p3[mask, 0]) * bc[mask]) / det[mask]

    # Apply boundary constraints
    mask &= (cx >= minval) & (cy >= minval)
    mask &= (cx <= xmax + 20) & (cy <= ymax + 20)

    if not np.any(mask):
        return np.array([-1, -1]), np.array([-1, -1]), np.array([-1, -1])

    # Calculate radii
    radius = np.zeros_like(det)
    radius[mask] = np.sqrt((cx[mask] - p1[mask, 0])**2 + 
                           (cy[mask] - p1[mask, 1])**2)

    # Apply radius constraints
    mask &= (radius >= rmin) & (radius <= rmax)

    if not np.any(mask):
        return np.array([-1, -1]), np.array([-1, -1]), np.array([-1, -1])

    return cx[mask], cy[mask], radius[mask]

def find_top_3d_modes(points, top_n=5):
    """
    Find the most common 3D points (modes) in parameter space.
    
    Parameters
    ----------
    points : array-like
        Array of 3D points (cx, cy, r) where each point represents circle parameters
    top_n : int, optional
        Number of most common points to return (default: 5)
        
    Returns
    -------
    points_list : list of tuples
        List of the top n most common 3D points
    frequencies_list : list of int
        Corresponding frequencies for each point
    """
    # Convert points to tuples and count occurrences
    most_common = Counter(map(tuple, points)).most_common(top_n)
    
    # Unzip points and frequencies
    if most_common:
        points_list, frequencies_list = zip(*most_common)
    else:
        points_list, frequencies_list = [], []
    
    return list(points_list), list(frequencies_list)

@jit(nopython=True)
def conv3D_sum_numba(data_volume, x0, y0, z0, tol=1):
    """
    Compute 3D convolution sum around a point (Numba-accelerated).
    
    Calculates the weighted average of coordinates within a cubic neighborhood.
    
    Parameters
    ----------
    data_volume : numpy.ndarray
        3D array containing voting weights
    x0, y0, z0 : int
        Center coordinates for convolution
    tol : int, optional
        Radius of convolution kernel (default: 1)
        
    Returns
    -------
    total_sum : float
        Sum of all values in the neighborhood
    weighted_point : tuple of float
        Weighted average (x, y, z) coordinates
    """
    total_sum = 0.0
    weighted_sum_x = 0.0
    weighted_sum_y = 0.0
    weighted_sum_z = 0.0
    
    for dx in range(-tol, tol + 1):
        for dy in range(-tol, tol + 1):
            for dz in range(-tol, tol + 1):
                x, y, z = x0 + dx, y0 + dy, z0 + dz
                if (0 <= x < data_volume.shape[0] and 
                    0 <= y < data_volume.shape[1] and 
                    0 <= z < data_volume.shape[2]):
                    value = data_volume[x, y, z]
                    total_sum += value
                    weighted_sum_x += x * value
                    weighted_sum_y += y * value
                    weighted_sum_z += z * value
    
    if total_sum > 0:
        return total_sum, (weighted_sum_x / total_sum, 
                          weighted_sum_y / total_sum, 
                          weighted_sum_z / total_sum)
    return 0.0, (float(x0), float(y0), float(z0))

def MaxConv3D(data_volume, neighbor_points, tol=1):
    """
    Find maximum convolution response in 3D volume.
    
    Evaluates convolution at multiple candidate points and returns the
    weighted center of the point with maximum voting.
    
    Parameters
    ----------
    data_volume : numpy.ndarray
        3D array containing voting weights
    neighbor_points : numpy.ndarray
        Array of candidate points to evaluate
    tol : int, optional
        Convolution kernel radius (default: 1)
        
    Returns
    -------
    tuple of float
        Weighted average coordinates (x, y, z) at maximum convolution
    """
    data_volume = np.asarray(data_volume)
    points = np.asarray(neighbor_points)
    
    # Compute convolutions for all candidate points
    results = [conv3D_sum_numba(data_volume, int(x), int(y), int(z), tol) 
               for x, y, z in points]
    
    # Find point with maximum voting
    max_idx = np.argmax([sum_val for sum_val, _ in results])
    
    return results[max_idx][1]

def add_neighboring_points(points):
    """
    Generate all 26-connected neighbors for a set of 3D points.
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 3) containing 3D points
        
    Returns
    -------
    numpy.ndarray
        Array containing original points plus unique neighbors
    """
    points = points.tolist()
    points_set = set(tuple(point) for point in points)
    
    neighbors = []
    for point in points:
        x, y, z = point
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    new_point = (x + dx, y + dy, z + dz)
                    if new_point not in points_set:
                        neighbors.append(new_point)
                        points_set.add(new_point)
    
    all_points = np.array(points + neighbors)
    return all_points

def ccc_fbi(edgels, Nmax=5000, xmax=50, ymax=50, rmin=1, rmax=40, 
                   top_n=5, ConvDist=2):
    """
    Fit a circle to edge points using Fast Ballot Inspection (FBI) method.
    
    This algorithm combines random sampling, voting in parameter space, and
    local refinement to robustly detect circles in noisy edge data.
    
    Algorithm steps:
    1. Randomly sample triplets of edge points
    2. Fit circles to each triplet
    3. Accumulate votes in (cx, cy, r) parameter space
    4. Find top voting modes
    5. Refine using 3D convolution
    
    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    Nmax : int, optional
        Maximum number of random triplets to sample (default: 5000)
    xmax, ymax : float, optional
        Expected maximum x and y coordinates (default: 50)
    rmin, rmax : float, optional
        Expected radius range (default: 1, 40)
    top_n : int, optional
        Number of top voting modes to consider (default: 5)
    ConvDist : int, optional
        Convolution kernel radius for refinement (default: 2)
        
    Returns
    -------
    center : numpy.ndarray
        Array [xc, yc] of circle center coordinates
    radius : float
        Circle radius
        
    Notes
    -----
    Returns ([-1, -1], -1) if circle detection fails.
    
    Examples
    --------
    >>> # Generate synthetic circle
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> edgels = np.column_stack([50 + 20*np.cos(theta), 50 + 20*np.sin(theta)])
    >>> center, radius = ccc_fbi(edgels)
    >>> print(f"Center: {center}, Radius: {radius}")
    """
    # Input validation
    if isinstance(edgels, bool) or len(edgels) < 3:
        return np.array([-1, -1]), -1
    
    # Generate random combinations of edge point triplets
    combi = list(combinations(np.arange(len(edgels)), 3))
    N = min(Nmax, len(combi))
    randomSample = np.array(random.sample(combi, N))
    
    # Extract point triplets
    p1 = edgels[randomSample[:,0]]
    p2 = edgels[randomSample[:,1]]
    p3 = edgels[randomSample[:,2]]
    
    # Fit circles to all triplets
    cx, cy, r = vectorized_XYR(p1, p2, p3, xmax=xmax, ymax=ymax, 
                               rmin=rmin, rmax=rmax, minval=0)
    # Check if valid circles were found
    if len(cx) == 0 or cx[0] == -1:
        return np.array([-1, -1]), -1
    
    # Quantize parameters for voting
    cx_int = np.round(cx).astype(np.int32)
    cy_int = np.round(cy).astype(np.int32)
    r_int  = np.round(r).astype(np.int32)
    points = np.column_stack((cx_int, cy_int, r_int))
    
    # Find top voting modes in parameter space
    top_common_points, freq = find_top_3d_modes(points, top_n=top_n)
    
    if not top_common_points:
        return np.array([-1, -1]), -1
    
    # Generate neighborhood around top modes
    TCP = np.array(top_common_points)
    neighbor_points = add_neighboring_points(TCP)

    # Create 3D voting volume
    xmax_vol, ymax_vol, rmax_vol = np.max(TCP, axis=0)
    data_volume = np.zeros((xmax_vol + 5, ymax_vol + 5, rmax_vol + 5))
    
    # Fill volume with voting weights
    for coord, weight in zip(top_common_points, freq):
        data_volume[coord[0], coord[1], coord[2]] = weight

    # Refine using 3D convolution
    weighted_point = MaxConv3D(data_volume, neighbor_points, tol=ConvDist)
    print(weighted_point)
    xc, yc, rc = np.round(weighted_point, 4)
    
    return np.array([xc, yc]), rc

# Main entry point for testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        print("Testing Circle Detection FBI Algorithm")
        print("=" * 60)
        
        # Test 1: Perfect circle
        print("\nTest 1: Perfect circle")
        theta = np.linspace(0, 2*np.pi, 100)
        true_center = (50, 50)
        true_radius = 20.5
        edgels_perfect = np.column_stack([
            true_center[0] + true_radius * np.cos(theta),
            true_center[1] + true_radius * np.sin(theta)
        ])
        
        center, radius = ccc_fbi(edgels_perfect, xmax=100, ymax=100, 
                                        rmin=10, rmax=30)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        
        # Test 2: Noisy circle
        print("\nTest 2: Noisy circle")
        noise = np.random.randn(100, 2) * 2.0
        edgels_noisy = edgels_perfect + noise
        
        center, radius = ccc_fbi(edgels_noisy, xmax=100, ymax=100, 
                                        rmin=10, rmax=30, Nmax=2000)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        
        # Test 3: Partial circle with outliers
        print("\nTest 3: Partial circle (90 degrees) with outliers")
        theta_partial = np.linspace(0, np.pi/2, 50)
        edgels_partial = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_partial),
            true_center[1] + true_radius * np.sin(theta_partial)
        ])
        # Add outliers
        outliers = np.random.rand(20, 2) * 100
        edgels_with_outliers = np.vstack([edgels_partial, outliers])
        
        center, radius = ccc_fbi(edgels_with_outliers, xmax=100, ymax=100, 
                                        rmin=10, rmax=30, Nmax=3000)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        
        # Test 4: Edge case - too few points
        print("\nTest 4: Edge case - too few points")
        edgels_few = np.array([[10, 10], [20, 20]])
        center, radius = ccc_fbi(edgels_few)
        print(f"Result: center={center}, radius={radius} (should be invalid)")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
    else:
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        center, radius = ccc_fbi(edgels)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")