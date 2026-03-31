"""
Preprocessing for Black Sphere Circle Detection

Applies green background filtering using two methods:
1. Green Level (GL): HSV thresholding with varying saturation/value levels
2. Median (Med): Median filtering followed by adaptive HSV thresholding

Main function: preprocess_image(bs_image, gb_image, method, return_edgels)
"""

import numpy as np
import cv2


def auto_canny(image, sigma=0.33):
    """
    Automatic Canny edge detection with automatic threshold selection.
    
    Parameters
    ----------
    image : numpy.ndarray
        Grayscale input image
    sigma : float, optional
        Sigma value for threshold calculation (default: 0.33)
        
    Returns
    -------
    edges : numpy.ndarray
        Binary edge map
    """
    # Compute median of pixel intensities
    v = np.median(image)
    
    # Compute lower and upper thresholds
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, lower, upper)
    
    return edges


def frames_to_edgepoints(edge_image):
    """
    Convert binary edge image to array of edge point coordinates.
    
    Parameters
    ----------
    edge_image : numpy.ndarray
        Binary edge image (0 or 1 values)
        
    Returns
    -------
    edgels : numpy.ndarray or bool
        Array of shape (n, 2) with (x, y) coordinates, or False if no edges
    """
    # Find edge points
    y, x = np.where(edge_image == 1)
    
    if len(x) == 0:
        return False
    
    # Return as (x, y) coordinates
    edgels = np.column_stack((x, y))
    
    return edgels


def _preprocess_green_level(bs_image, green_level):
    """
    Preprocess using green level filtering method.
    
    Applies HSV thresholding with fixed hue range and variable saturation/value.
    
    Parameters
    ----------
    bs_image : numpy.ndarray
        Black sphere ROI image (BGR format)
    green_level : int
        Lower threshold for saturation and value (70-88 typical)
        
    Returns
    -------
    edge_image : numpy.ndarray
        Binary edge image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(bs_image, cv2.COLOR_BGR2HSV)
    
    # Define green color range
    # Hue: 36-86 (green in HSV)
    # Saturation: green_level to 255
    # Value: green_level to 255
    lower_green = np.array([36, green_level, green_level], dtype=np.uint8)
    upper_green = np.array([86, 255, 255], dtype=np.uint8)
    
    # Create green mask
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply Canny edge detection
    edges = auto_canny(green_mask)
    
    # Normalize to 0-1
    edge_image = edges / 255
    
    return edge_image


def _preprocess_median(bs_image, gb_image, median_size):
    """
    Preprocess using median filtering method.
    
    Applies median blur, then uses green background statistics to create
    adaptive HSV thresholding.
    
    Parameters
    ----------
    bs_image : numpy.ndarray
        Black sphere ROI image (BGR format)
    gb_image : numpy.ndarray
        Green background ROI image (BGR format)
    median_size : int
        Size of median filter kernel (must be odd, 3-19 typical)
        
    Returns
    -------
    edge_image : numpy.ndarray
        Binary edge image
    """
    # Apply median filter to black sphere image
    median_filtered = cv2.medianBlur(bs_image, median_size)
    
    # Convert to HSV
    hsv = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2HSV)
    
    # Analyze green background to determine adaptive thresholds
    gb_hsv = cv2.cvtColor(gb_image, cv2.COLOR_BGR2HSV)
    
    # Flatten spatial dimensions
    rows, cols = gb_hsv.shape[:2]
    green_pixels = gb_hsv.reshape(rows * cols, 3)
    
    # Compute statistics for each HSV channel
    green_std = np.std(green_pixels, axis=0)
    green_min = np.min(green_pixels, axis=0)
    green_max = np.max(green_pixels, axis=0)
    
    # Adaptive thresholds: mean ± 5*std
    lower_bound = np.maximum(0, np.floor(green_min - 5 * green_std))
    upper_bound = np.minimum(255, np.ceil(green_max + 5 * green_std))
    
    # Create green mask using adaptive bounds
    green_mask = cv2.inRange(hsv, lower_bound.astype(np.uint8), 
                             upper_bound.astype(np.uint8))
    
    # Apply Canny edge detection
    edges = auto_canny(green_mask)
    
    # Normalize to 0-1
    edge_image = edges / 255
    
    return edge_image


def preprocess_image(bs_image, gb_image=None, method='GL70', return_edgels=True):
    """
    Preprocess black sphere image to extract edges or edge points.
    
    Supports two preprocessing methods:
    1. Green Level (GL): HSV color filtering with fixed thresholds
    2. Median (Med): Median filtering with adaptive thresholding
    
    Parameters
    ----------
    bs_image : numpy.ndarray or str
        Black sphere ROI image (BGR format) or path to image file
    gb_image : numpy.ndarray or str, optional
        Green background ROI image (BGR format) or path to image file
        Required only for 'Med' methods
    method : str, optional
        Preprocessing method name (default: 'GL70')
        Format: 'GL{level}' where level in [70, 72, 74, ..., 86]
                'Med{size}' where size in [3, 5, 7, ..., 19]
        Examples: 'GL70', 'GL80', 'Med3', 'Med11'
    return_edgels : bool, optional
        If True, returns edge point coordinates (default)
        If False, returns binary edge image
        
    Returns
    -------
    result : numpy.ndarray or bool
        If return_edgels=True: Array of shape (n, 2) with (x, y) coordinates
                               Returns False if no edges detected
        If return_edgels=False: Binary edge image (values 0 or 1)
        
    Raises
    ------
    ValueError
        If method format is invalid or parameters are out of range
        If gb_image is not provided for 'Med' methods
        
    Examples
    --------
    >>> # Green level method
    >>> bs_img = cv2.imread('black_sphere_ROI/frame_001.png')
    >>> edgels = preprocess_image(bs_img, method='GL76')
    >>> print(edgels.shape)  # (n_edges, 2)
    
    >>> # Median method (requires green background)
    >>> gb_img = cv2.imread('green_back_ROI/frame_001.png')
    >>> edgels = preprocess_image(bs_img, gb_img, method='Med9')
    
    >>> # Get edge image instead of coordinates
    >>> edge_img = preprocess_image(bs_img, method='GL80', return_edgels=False)
    >>> plt.imshow(edge_img, cmap='gray')
    
    Notes
    -----
    Available methods:
    - Green Level: GL70, GL72, GL74, GL76, GL78, GL80, GL82, GL84, GL86
    - Median: Med3, Med5, Med7, Med9, Med11, Med13, Med15, Med17, Med19
    
    The green level controls selectivity:
    - Lower values (GL70): More permissive, captures more green
    - Higher values (GL86): More selective, only bright saturated green
    
    The median size controls smoothing:
    - Small sizes (Med3): Less smoothing, preserves detail
    - Large sizes (Med19): More smoothing, removes noise
    """
    # Load images if paths provided
    if isinstance(bs_image, str):
        bs_image = cv2.imread(bs_image)
        if bs_image is None:
            raise ValueError(f"Could not load image: {bs_image}")
    
    if isinstance(gb_image, str):
        gb_image = cv2.imread(gb_image)
        if gb_image is None:
            raise ValueError(f"Could not load image: {gb_image}")
    
    # Parse method string
    if method.startswith('GL'):
        # Green Level method
        try:
            green_level = int(method[2:])
        except ValueError:
            raise ValueError(f"Invalid green level in method '{method}'. "
                           "Expected format: 'GL70', 'GL72', ..., 'GL86'")
        
        if green_level < 70 or green_level > 88 or green_level % 2 != 0:
            raise ValueError(f"Green level must be even and in range [70, 86], got {green_level}")
        
        edge_image = _preprocess_green_level(bs_image, green_level)
    
    elif method.startswith('Med'):
        # Median method
        try:
            median_size = int(method[3:])
        except ValueError:
            raise ValueError(f"Invalid median size in method '{method}'. "
                           "Expected format: 'Med3', 'Med5', ..., 'Med19'")
        
        if median_size < 3 or median_size > 19 or median_size % 2 != 1:
            raise ValueError(f"Median size must be odd and in range [3, 19], got {median_size}")
        
        if gb_image is None:
            raise ValueError("Green background image required for median methods")
        
        edge_image = _preprocess_median(bs_image, gb_image, median_size)
    
    else:
        raise ValueError(f"Unknown method '{method}'. Must start with 'GL' or 'Med'")
    
    # Return based on flag
    if return_edgels:
        return frames_to_edgepoints(edge_image)
    else:
        return edge_image


# Convenience functions for batch processing
def get_all_method_names():
    """
    Get list of all available preprocessing method names.
    
    Returns
    -------
    list of str
        All method names in order: GL methods first, then Med methods
    """
    green_level_range = range(70, 88, 2)
    median_range = range(3, 20, 2)
    
    gl_names = [f'GL{i}' for i in green_level_range]
    med_names = [f'Med{i}' for i in median_range]
    
    return gl_names + med_names


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        # Command line usage
        bs_path = sys.argv[1]
        method = sys.argv[2]
        gb_path = sys.argv[3] if len(sys.argv) > 3 else None
        
        edgels = preprocess_image(bs_path, gb_path, method=method)
        
        if edgels is False:
            print(f"No edges detected with method {method}")
        else:
            print(f"Detected {len(edgels)} edge points")
            print(f"Edgels shape: {edgels.shape}")
            
            # Save to CSV
            output_file = f'edgels_{method}.csv'
            np.savetxt(output_file, edgels, delimiter=',', fmt='%d')
            print(f"Saved to {output_file}")
    
    else:
        # Demo usage
        print("Preprocessing Demo")
        print("=" * 60)
        print("\nAvailable methods:")
        methods = get_all_method_names()
        print(f"  Green Level: {methods[:9]}")
        print(f"  Median: {methods[9:]}")
        
        print("\nUsage:")
        print("  python preprocessing.py <bs_image> <method> [gb_image]")
        print("\nExamples:")
        print("  python preprocessing.py black_sphere_ROI/img.png GL76")
        print("  python preprocessing.py black_sphere_ROI/img.png Med9 green_back_ROI/img.png")
        
        print("\nProgrammatic usage:")
        print("  from preprocessing import preprocess_image")
        print("  edgels = preprocess_image(bs_img, method='GL80')")
        print("  edgels = preprocess_image(bs_img, gb_img, method='Med11')")