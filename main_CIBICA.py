"""
Main Script for Circle Detection Comparison Study

This script reproduces the results from the article comparing CIBICA and 
Hough Transform methods for circle detection.

Usage:
    python main_CIBICA.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from itertools import combinations
import random
import math as m
from scipy.spatial.distance import cdist

from CIBICA import CIBICA, vectorized_XYR, median_3d, LS_circle
from HOUGH import HOUGH
from preprocessing import *


def jaccard_circles(x1, y1, r1, x2, y2, r2, show=False):
    """
    Calculate Jaccard Index between two circles.
    
    Parameters
    ----------
    x1, y1, r1 : float
        Ground truth circle center and radius
    x2, y2, r2 : float
        Estimated circle center and radius
    show : bool
        Whether to print debug information
        
    Returns
    -------
    jaccard : float
        Jaccard Index (intersection over union)
        
    Reference
    ---------
    https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6
    """
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    if d == 0:
        return min((r1 / r2)**2, (r2 / r1)**2)
    else:
        d1 = (d**2 + r1**2 - r2**2) / (2 * d)
        d2 = d - d1
        R = max((r1, r2))
        r = min((r1, r2))
        
        if d >= r1 + r2:
            if show:
                print('Case 1 - Zero intersection')
            output = 0
        elif d <= R - r:
            if show:
                print('Case 2 - Small inside big')
            output = (r / R)**2
        else:
            if show:
                print('Case 3 - Some intersection')
            alpha1 = 2 * m.acos(d1 / r1)
            alpha2 = 2 * m.acos(d2 / r2)
            intersection = 0.5 * r1**2 * (alpha1 - m.sin(alpha1)) + 0.5 * r2**2 * (alpha2 - m.sin(alpha2))
            union = m.pi * (R**2 + r**2) - intersection
            output = intersection / union
        
        return output


def create_synthetic_test_data(n_images=10):
    """Create synthetic test images and ground truth"""
    np.random.seed(42)
    
    ground_truth = []
    images = []
    
    for i in range(n_images):
        # Random circle parameters
        true_x = np.random.randint(15, 35)
        true_y = np.random.randint(15, 35)
        true_r = np.random.randint(8, 15)
        
        # Create image
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        
        # Draw circle with noise
        angles = np.linspace(0, 2 * np.pi, 100)
        for angle in angles:
            px = int(true_y + true_r * np.cos(angle) + np.random.normal(0, 0.5))
            py = int(true_x + true_r * np.sin(angle) + np.random.normal(0, 0.5))
            if 0 <= px < 50 and 0 <= py < 50:
                cv2.circle(img, (px, py), 1, (0, 255, 0), -1)
        
        images.append(img)
        ground_truth.append({'X': true_x, 'Y': true_y, 'R': true_r})
    
    return images, pd.DataFrame(ground_truth)


def run_experiments(images, ground_truth):
    """
    Run experiments following the notebook's approach.
    
    Parameters
    ----------
    images : list
        List of test images
    ground_truth : pandas.DataFrame
        Ground truth circle parameters
        
    Returns
    -------
    results : dict
        Dictionary containing Jaccard indices for both methods
    """
    configs = get_preprocessing_configs()
    N_of_files = len(images)
    N_of_methods = len(configs)
    
    # Initialize result arrays (following notebook structure)
    Jaccard_Hough = np.zeros((N_of_files, N_of_methods))
    Jaccard_CIBICA = np.zeros((N_of_files, N_of_methods))
    Jaccard_CIBICA_LSC = np.zeros((N_of_files, N_of_methods))
    
    X_ground_truth = ground_truth['X'].to_numpy()
    Y_ground_truth = ground_truth['Y'].to_numpy()
    R_ground_truth = ground_truth['R'].to_numpy()
    
    print(f"Running experiments: {N_of_files} images x {N_of_methods} methods")
    print("=" * 70)
    
    # Process each image
    for i, img in enumerate(images):
        XGT = X_ground_truth[i]
        YGT = Y_ground_truth[i]
        RGT = R_ground_truth[i]
        
        # Get dimensions
        xmax = img.shape[1]
        ymax = img.shape[0]
        
        # Process each preprocessing method
        for j, config in enumerate(configs):
            # Preprocess
            if config['method'] == 'green_level':
                GreenMask, GreenCanny, coord = preprocess_green_level(
                    img, config['green_level']
                )
            else:
                # For median filter, use same image as reference
                GreenMask, GreenCanny, coord = preprocess_green_level(
                    img, 76  # Use default green level
                )
            
            # HOUGH method
            try:
                Hough = cv2.HoughCircles(
                    GreenMask, cv2.HOUGH_GRADIENT, 1, 300,
                    param1=50, param2=8, minRadius=5, maxRadius=25
                )
                
                if Hough is not None:
                    Hough = np.uint16(np.around(Hough))
                    HoughVal = Hough[0, 0, :]
                    x2, y2, r2 = HoughVal[0], HoughVal[1], HoughVal[2]
                    Jaccard_Hough[i, j] = jaccard_circles(XGT, YGT, RGT, x2, y2, r2)
            except:
                pass
            
            # CIBICA method
            if len(coord) >= 3:
                try:
                    # Sample triplets
                    Nmax = 500
                    combi = list(combinations(np.arange(len(coord)), 3))
                    N = min(Nmax, len(combi))
                    RandomSample = np.array(random.sample(combi, N))
                    
                    p1 = coord[RandomSample[:, 0]]
                    p2 = coord[RandomSample[:, 1]]
                    p3 = coord[RandomSample[:, 2]]
                    
                    # Fit circles
                    cx, cy, radius = vectorized_XYR(p1, p2, p3, xmax, ymax)
                    
                    # Median estimate (without refinement)
                    XYR = median_3d(cx, cy, radius, xmax, ymax)
                    Jaccard_CIBICA[i, j] = jaccard_circles(XGT, YGT, RGT, XYR[1], XYR[0], XYR[2])
                    
                    # Least squares refinement
                    coord2 = [(XYR[0], XYR[1])]
                    distances = cdist(coord2, coord)
                    near = np.where(np.abs(cdist(coord2, coord) - XYR[2]) < 1.5)
                    circle_points = coord[near[1]]
                    
                    if len(circle_points) >= 3:
                        xl, yl, rl, res = np.round(LS_circle(circle_points[:, 0], circle_points[:, 1]), 3)
                        Jaccard_CIBICA_LSC[i, j] = jaccard_circles(XGT, YGT, RGT, yl, xl, rl)
                except:
                    pass
        
        if (i + 1) % 2 == 0:
            print(f"Processed {i + 1}/{N_of_files} images...")
    
    return {
        'Jaccard_Hough': Jaccard_Hough,
        'Jaccard_CIBICA': Jaccard_CIBICA,
        'Jaccard_CIBICA_LSC': Jaccard_CIBICA_LSC,
        'config_names': [c['name'] for c in configs]
    }


def plot_results(results, output_path='Figure_Comparison.png'):
    """Generate comparison plots"""
    config_names = results['config_names']
    
    # Calculate Jaccard Distance (1 - Jaccard Index)
    hough_distance = 1 - np.mean(results['Jaccard_Hough'], 0)
    cibica_distance = 1 - np.mean(results['Jaccard_CIBICA'], 0)
    
    method_indices = np.arange(len(config_names))
    figureSize = (15, 6)
    pltFontSize = 12
    
    # ===== Plot 1: Jaccard Distance Comparison with Fill =====
    fig, ax = plt.subplots(figsize=figureSize)
    
    # Plot lines
    ax.plot(method_indices, cibica_distance, color="green", linewidth=2.5, label="CIBICA")
    ax.plot(method_indices, hough_distance, color="red", linewidth=2.5, label="Hough")
    
    # Fill areas
    ax.fill_between(method_indices, cibica_distance, hough_distance, 
                     where=(cibica_distance < hough_distance), 
                     interpolate=True, color="green", alpha=0.25, label="CIBICA better")
    ax.fill_between(method_indices, cibica_distance, hough_distance, 
                     where=(cibica_distance >= hough_distance), 
                     interpolate=True, color="red", alpha=0.25, label="Hough better")
    
    ax.set_xlabel('Preprocessing Method', fontsize=pltFontSize, fontweight='bold')
    ax.set_ylabel('Jaccard Distance', fontsize=pltFontSize, fontweight='bold')
    ax.set_title('Jaccard Distance Comparison: Hough vs CIBICA', 
                 fontsize=pltFontSize + 2, fontweight='bold')
    ax.set_xticks(method_indices)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=pltFontSize)
    
    plt.tight_layout()
    plt.savefig('Figure_Distance_Comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: Figure_Distance_Comparison.png")
    plt.close()
    
    # ===== Plot 2: Distance Ratio =====
    fig, ax = plt.subplots(figsize=figureSize)
    
    # Calculate ratio (avoid division by zero)
    ratio = np.divide(hough_distance, cibica_distance, 
                      out=np.ones_like(hough_distance), 
                      where=cibica_distance!=0)
    
    ax.plot(method_indices, ratio, color="blue", linewidth=3)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal performance')
    
    ax.set_xlabel('Preprocessing Method', fontsize=pltFontSize, fontweight='bold')
    ax.set_ylabel('Distance Ratio', fontsize=pltFontSize, fontweight='bold')
    ax.set_title('Jaccard Distance Ratio: Hough/CIBICA', 
                 fontsize=pltFontSize + 2, fontweight='bold')
    ax.set_xticks(method_indices)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=pltFontSize)
    
    plt.tight_layout()
    plt.savefig('Figure_Distance_Ratio.png', dpi=300, bbox_inches='tight')
    print("Saved: Figure_Distance_Ratio.png")
    plt.close()




def run_experiments_with_real_data():
    """
    Run experiments using actual images from ROI folders.
    
    Returns
    -------
    results : dict
        Dictionary containing Jaccard indices for both methods
    """
    # Load ground truth
    ground_truth = pd.read_csv('Ground_Truth.csv')
    filenames = ground_truth['Filename'].tolist()
    
    # Define all preprocessing parameters
    green_levels = list(range(70, 88, 2))  # [70, 72, 74, 76, 78, 80, 82, 84, 86]
    median_sizes = list(range(3, 21, 2))   # [3, 5, 7, 9, 11, 13, 15, 17, 19]
    
    # Combine into configs
    configs = []
    for level in green_levels:
        configs.append({'name': f'Green level {level}', 'method': 'green_level', 'param': level})
    for size in median_sizes:
        configs.append({'name': f'Median {size}x{size}', 'method': 'median_filter', 'param': size})
    
    n_images = len(filenames)
    n_configs = len(configs)
    
    # Initialize results
    Jaccard_Hough = np.zeros((n_images, n_configs))
    Jaccard_CIBICA = np.zeros((n_images, n_configs))
    
    print(f"Processing {n_images} images x {n_configs} configs...")
    print("=" * 70)
    
    # Process each image
    for i, filename in enumerate(filenames):
        XGT = ground_truth.iloc[i]['X']
        YGT = ground_truth.iloc[i]['Y']
        RGT = ground_truth.iloc[i]['R']
        
        # Load image to get dimensions
        import cv2
        img_path = f'black_sphere_ROI/{filename}.png'
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue
            
        xmax, ymax = img.shape[1], img.shape[0]
        
        # Test each preprocessing config
        for j, config in enumerate(configs):
            try:
                # CIBICA - get edgels
                edgels = preprocess_image(
                    filename, 
                    method=config['method'],
                    param=config['param'],
                    hough=False
                )
                
                if len(edgels) >= 3:
                    x_c, y_c, r_c = CIBICA(edgels, n_triplets=500, xmax=xmax, ymax=ymax)
                    if not np.isnan(x_c):
                        Jaccard_CIBICA[i, j] = jaccard_circles(XGT, YGT, RGT, x_c, y_c, r_c)
                
                # HOUGH - get mask
                mask = preprocess_image(
                    filename,
                    method=config['method'],
                    param=config['param'],
                    hough=True
                )
                
                x_h, y_h, r_h = HOUGH(mask, minDist=300, param2=8, minRadius=5, maxRadius=25)
                if x_h > 0:
                    Jaccard_Hough[i, j] = jaccard_circles(XGT, YGT, RGT, x_h, y_h, r_h)
                    
            except Exception as e:
                print(f"Error on {filename}, config {config['name']}: {e}")
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{n_images} images...")
    
    return {
        'Jaccard_Hough': Jaccard_Hough,
        'Jaccard_CIBICA': Jaccard_CIBICA,
        'config_names': [c['name'] for c in configs]
    }
def main():
    """Main function"""
    print("=" * 70)
    print("Circle Detection Comparison: CIBICA vs Hough Transform")
    print("=" * 70)
    print()
    
    # Run experiments with real data
    results = run_experiments_with_real_data()
    print()
    
    # Plot results
    print("Generating figure...")
    plot_results(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    hough_mean = np.mean(results['Jaccard_Hough'])
    cibica_mean = np.mean(results['Jaccard_CIBICA'])
    
    print(f"Mean Jaccard Index (Hough):  {hough_mean:.4f}")
    print(f"Mean Jaccard Index (CIBICA): {cibica_mean:.4f}")
    
    if cibica_mean > hough_mean:
        improvement = ((cibica_mean - hough_mean) / hough_mean) * 100
        print(f"CIBICA improvement: {improvement:.2f}%")
    
    print("\nDone!")
    print("=" * 70)


if __name__ == "__main__":
    main()
