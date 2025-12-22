"""
RFCA: Robust Fitting of Circle Arcs (Ladron 2011)

Implements a robust circle fitting algorithm using iterative median-based
optimization with adaptive step size.

Reference: Ladron et al. (2011)

Main function: rfca(edgels, **kwargs)
"""

import numpy as np


def rfca(edgels, lambda_0=0.5, lambda_k_min=0.001, factor1=1.2, 
               factor2=0.9, max_iterations=1000):
    """
    Fit a circle to edge points using Robust Fitting of Circle Arcs (RFCA).
    
    This algorithm uses iterative median-based optimization with adaptive
    step size to robustly fit circles in the presence of outliers.
    
    Algorithm steps:
    1. Initialize center as mean of points
    2. Compute median distance as radius estimate
    3. Classify points as inside/outside/on-circle
    4. Update center using gradient descent with adaptive step
    5. Repeat until convergence
    
    Parameters
    ----------
    edgels : numpy.ndarray
        Array of shape (n, 2) containing (x, y) coordinates of edge points
    lambda_0 : float, optional
        Initial step size (default: 0.5)
    lambda_k_min : float, optional
        Minimum step size for stopping criterion (default: 0.001)
    factor1 : float, optional
        Factor to increase step size on improvement (default: 1.2)
    factor2 : float, optional
        Factor to decrease step size on no improvement (default: 0.9)
    max_iterations : int, optional
        Maximum number of iterations (default: 1000)
        
    Returns
    -------
    center : numpy.ndarray
        Array [xc, yc] of circle center coordinates
    radius : float
        Circle radius (median distance from center to points)
        
    Notes
    -----
    The algorithm is robust to outliers through the use of median statistics
    and adaptive step sizing.
    
    Examples
    --------
    >>> theta = np.linspace(0, 2*np.pi, 100)
    >>> edgels = np.column_stack([50 + 20*np.cos(theta), 50 + 20*np.sin(theta)])
    >>> center, radius = rfca(edgels)
    >>> print(f"Center: {center}, Radius: {radius:.2f}")
    """
    # Input validation
    if len(edgels) < 3:
        return np.array([-1, -1]), -1
    
    # Step 1: Initialize center as mean of points
    a, b = np.mean(edgels, axis=0)
    lambda_k = lambda_0
    
    for iteration in range(max_iterations):
        # Step 2: Compute distances from current center and median radius
        distances = np.sqrt(np.sum((edgels - np.array([a, b]))**2, axis=1))
        r = np.median(distances)
        
        # Step 3: Classify points as Inside, Outside, or on-Circle
        I = np.where(distances < r)[0]
        O = np.where(distances > r)[0]
        C = np.where(distances == r)[0]
        
        # Step 4: Compute directional cosines and sines
        cos_theta = (edgels[:, 0] - a) / distances
        sin_theta = (edgels[:, 1] - b) / distances
        
        # Step 5: Compute partial derivatives of error function
        E_a_plus = np.sum(cos_theta[I]) - np.sum(cos_theta[O]) + np.sum(np.abs(cos_theta[C]))
        E_a_minus = np.sum(cos_theta[I]) - np.sum(cos_theta[O]) - np.sum(np.abs(cos_theta[C]))
        E_b_plus = np.sum(sin_theta[I]) - np.sum(sin_theta[O]) + np.sum(np.abs(sin_theta[C]))
        E_b_minus = np.sum(sin_theta[I]) - np.sum(sin_theta[O]) - np.sum(np.abs(sin_theta[C]))
        
        # Step 6: Determine descent direction
        alpha = 1 if E_a_minus >= 0 else (0 if E_a_plus <= 0 else 0.5)
        beta = 1 if E_b_minus >= 0 else (0 if E_b_plus <= 0 else 0.5)
        
        d1 = -(alpha * E_a_minus + (1 - alpha) * E_a_plus)
        d2 = -(beta * E_b_minus + (1 - beta) * E_b_plus)
        
        # Compute new center candidate
        a_new = a + lambda_k * d1
        b_new = b + lambda_k * d2
        
        # Step 7: Check for improvement
        new_distances = np.sqrt(np.sum((edgels - np.array([a_new, b_new]))**2, axis=1))
        new_r = np.median(new_distances)
        new_error = np.sum(np.abs(new_distances - new_r))
        current_error = np.sum(np.abs(distances - r))
        
        if new_error < current_error:
            # Accept new center and increase step size
            a, b = a_new, b_new
            lambda_k = min(lambda_k * factor1, lambda_0)
        else:
            # Reject new center and decrease step size
            lambda_k *= factor2
        
        # Step 8: Check stopping criterion
        if lambda_k < lambda_k_min:
            break
    
    # Compute final radius
    final_distances = np.sqrt(np.sum((edgels - np.array([a, b]))**2, axis=1))
    final_r = np.median(final_distances)
    
    return np.array([a, b]), final_r


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        # Run built-in tests
        print("Testing RFCA (Robust Fitting of Circle Arcs)")
        print("=" * 60)
        
        # Test 1: Perfect circle
        print("\nTest 1: Perfect circle")
        theta = np.linspace(0, 2*np.pi, 100)
        true_center = (50, 50)
        true_radius = 20.5
        edgels = np.column_stack([
            true_center[0] + true_radius * np.cos(theta),
            true_center[1] + true_radius * np.sin(theta)
        ])
        
        center, radius = rfca(edgels)
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 2: Noisy circle
        print("\nTest 2: Noisy circle")
        noise = np.random.randn(100, 2) * 2.0
        edgels_noisy = edgels + noise
        
        center, radius = rfca(edgels_noisy)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 3: Circle with outliers
        print("\nTest 3: Circle with 20% outliers")
        # 80 points on circle
        theta_partial = np.linspace(0, 2*np.pi, 80)
        edgels_circle = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_partial),
            true_center[1] + true_radius * np.sin(theta_partial)
        ])
        # 20 random outliers
        outliers = np.random.rand(20, 2) * 100
        edgels_outliers = np.vstack([edgels_circle, outliers])
        
        center, radius = rfca(edgels_outliers)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 4: Partial arc (90 degrees)
        print("\nTest 4: Partial arc (90 degrees)")
        theta_arc = np.linspace(0, np.pi/2, 50)
        edgels_arc = np.column_stack([
            true_center[0] + true_radius * np.cos(theta_arc),
            true_center[1] + true_radius * np.sin(theta_arc)
        ])
        
        center, radius = rfca(edgels_arc)
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
        print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
    else:
        # Load edgels from CSV file
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        center, radius = rfca(edgels)
        print(f"center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")