"""
Hough Transform Circle Detection

Implements OpenCV's Hough Circle Transform for circle detection.
This is a classical computer vision method that works directly on images.

Main function: hough(image, **kwargs)
"""

import numpy as np
import cv2


def hough(image, dp=1, minDist=300, param1=50, param2=8, 
          minRadius=5, maxRadius=40, return_all=False):
    """
    Detect circles in an image using Hough Transform.
    
    Uses OpenCV's HoughCircles implementation which applies Canny edge
    detection internally and then uses the Hough gradient method to
    find circles.
    
    Parameters
    ----------
    image : numpy.ndarray or str
        Input image. Can be:
        - Path to image file (str)
        - Grayscale image (2D numpy array)
        - Color image (3D numpy array, will be converted to grayscale)
        - Binary edge image (values 0-1, will be scaled to 0-255)
    dp : float, optional
        Inverse ratio of accumulator resolution to image resolution (default: 1)
        dp=1: Accumulator has same resolution as input image
        dp=2: Accumulator has half the resolution (faster but less precise)
    minDist : float, optional
        Minimum distance between detected circle centers (default: 300)
        Prevents detecting multiple circles at nearly the same location
    param1 : float, optional
        Upper threshold for internal Canny edge detector (default: 50)
        Lower threshold is param1/2
        Higher values → fewer edges → fewer false circles
    param2 : float, optional
        Accumulator threshold for circle detection (default: 8)
        Lower values → more circles detected (including false positives)
        Higher values → fewer circles (only strong candidates)
        Typical range: 8-30
    minRadius : int, optional
        Minimum circle radius in pixels (default: 5)
    maxRadius : int, optional
        Maximum circle radius in pixels (default: 40)
        Set to 0 to detect circles of any size
    return_all : bool, optional
        If True, return all detected circles (default: False)
        If False, return only the best circle (first in sorted list)
        
    Returns
    -------
    center : numpy.ndarray
        Array [cx, cy] of circle center coordinates
        Returns [-1, -1] if no circle detected
    radius : float
        Circle radius
        Returns -1 if no circle detected
    all_circles : numpy.ndarray (only if return_all=True)
        Array of shape (n_circles, 3) containing [cx, cy, r] for all circles
        
    Notes
    -----
    The Hough Transform is particularly effective for:
    - Clean, well-defined circles
    - Images with good contrast
    - Known approximate circle size (set minRadius/maxRadius appropriately)
    
    It may struggle with:
    - Partial circles (arcs)
    - Very noisy images
    - Overlapping circles (adjust minDist)
    
    Parameter tuning guide:
    - Too many false circles → Increase param2, increase minDist
    - Missing circles → Decrease param2, decrease param1
    - Wrong size circles → Adjust minRadius/maxRadius
    - Multiple detections → Increase minDist
    
    Examples
    --------
    >>> # From image file path
    >>> center, radius = hough('circle.png')
    
    >>> # From numpy array (grayscale image)
    >>> img = cv2.imread('circle.png', cv2.IMREAD_GRAYSCALE)
    >>> center, radius = hough(img)
    
    >>> # From edge image
    >>> edges = cv2.Canny(img, 50, 150)
    >>> center, radius = hough(edges, param2=10)
    
    >>> # From binary edge image (preprocessing output)
    >>> edge_img = preprocess_image(img, method='GL76', return_edgels=False)
    >>> center, radius = hough(edge_img, minRadius=5, maxRadius=40)
    
    >>> # Get all detected circles
    >>> center, radius, all_circles = hough(img, return_all=True)
    """
    # Handle different input types
    if isinstance(image, str):
        # Load from file path
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Could not load image from path: {image}")
    
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            # Color image - convert to grayscale
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
        
        elif len(image.shape) == 2:
            # Already grayscale or binary edge image
            gray = image.copy()
        
        else:
            raise ValueError(f"Unsupported array shape: {image.shape}")
        
        # Scale to 0-255 if binary (0-1 range)
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
    
    else:
        raise TypeError(f"Input must be str (file path) or numpy.ndarray, got {type(image)}")
    
    # Input validation
    if gray is None or gray.size == 0:
        if return_all:
            return np.array([-1, -1]), -1, np.array([])
        return np.array([-1, -1]), -1
    
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
    
    # Check if any circles were detected
    if circles is None:
        if return_all:
            return np.array([-1, -1]), -1, np.array([])
        return np.array([-1, -1]), -1
    
    # Reshape circles array
    circles = circles[0, :]  # Shape: (n_circles, 3) where each row is [x, y, r]
    
    # Get best circle (first is strongest)
    best_circle = circles[0]
    
    center = np.array([best_circle[0], best_circle[1]])
    radius = float(best_circle[2])
    
    if return_all:
        return center, radius, circles
    else:
        return center, radius


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        # Run built-in tests
        print("Testing Hough Transform Circle Detection")
        print("=" * 60)
        
        # Test 1: Synthetic circle image
        print("\nTest 1: Circle from grayscale image")
        img_size = 200
        test_img = np.zeros((img_size, img_size), dtype=np.uint8)
        true_center = (100, 100)
        true_radius = 40
        cv2.circle(test_img, true_center, true_radius, 255, 2)
        
        center, radius = hough(test_img, minDist=100, param2=10, 
                              minRadius=30, maxRadius=50)
        
        print(f"True: center={true_center}, radius={true_radius}")
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        
        if center[0] != -1:
            error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
            print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 2: Binary edge image (0-1 values)
        print("\nTest 2: Binary edge image (0-1 values)")
        binary_edges = (test_img / 255.0)  # Convert to 0-1
        center, radius = hough(binary_edges, minRadius=30, maxRadius=50, param2=10)
        
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        
        if center[0] != -1:
            error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
            print(f"Center error: {error:.4f}, Radius error: {abs(radius-true_radius):.4f}")
        
        # Test 3: Color image
        print("\nTest 3: Color image (auto-converts to grayscale)")
        color_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        center, radius = hough(color_img, minRadius=30, maxRadius=50, param2=10)
        
        print(f"Detected: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        
        # Test 4: Multiple circles
        print("\nTest 4: Multiple circles detection")
        test_img_multi = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.circle(test_img_multi, (50, 50), 20, 255, 2)
        cv2.circle(test_img_multi, (150, 150), 30, 255, 2)
        
        center, radius, all_circles = hough(test_img_multi, minDist=50, param2=10,
                                           minRadius=15, maxRadius=40, return_all=True)
        
        print(f"Best circle: center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")
        print(f"Total circles detected: {len(all_circles)}")
        for i, circle in enumerate(all_circles):
            print(f"  Circle {i+1}: center=({circle[0]:.1f}, {circle[1]:.1f}), radius={circle[2]:.1f}")
        
        # Test 5: Parameter sensitivity
        print("\nTest 5: Parameter sensitivity (different param2 values)")
        for param2_val in [5, 10, 15, 20]:
            center, radius = hough(test_img, param2=param2_val, 
                                  minRadius=30, maxRadius=50)
            
            if center[0] != -1:
                error = np.sqrt((center[0]-true_center[0])**2 + (center[1]-true_center[1])**2)
                print(f"  param2={param2_val:2d}: error={error:.2f}, radius={radius:.1f}")
            else:
                print(f"  param2={param2_val:2d}: No circle detected")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
        print("\nSupported input types:")
        print("  ✓ Image file path (str)")
        print("  ✓ Grayscale image (2D numpy array)")
        print("  ✓ Color image (3D numpy array)")
        print("  ✓ Binary edge image (0-1 values)")
        
    else:
        # Load image from file
        image_path = sys.argv[1]
        center, radius = hough(image_path)
        
        if center[0] == -1:
            print("No circle detected")
        else:
            print(f"center=({center[0]:.2f}, {center[1]:.2f}), radius={radius:.2f}")s