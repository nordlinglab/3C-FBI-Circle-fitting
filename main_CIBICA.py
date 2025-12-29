"""
compare_cibica_vs_hough.py

Compare CIBICA algorithm vs Hough Transform across all preprocessing methods.
Uses Jaccard Index as the primary evaluation metric.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Import with CORRECT module and function names based on uploaded files
from CIBICA import cibica  # Main CIBICA function
from HOUGH import hough    # Hough transform function
from preprocessing import preprocess_image

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def calculate_jaccard_index(center1, radius1, center2, radius2, img_shape=(200, 200)):
    """Calculate Jaccard Index between two circles."""
    if radius2 == -1 or center2[0] == -1:
        return 0.0
    
    y, x = np.ogrid[:img_shape[0], :img_shape[1]]
    mask1 = (x - center1[0])**2 + (y - center1[1])**2 <= radius1**2
    mask2 = (x - center2[0])**2 + (y - center2[1])**2 <= radius2**2
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0.0


def create_synthetic_test_data(n_images=5):
    """Create synthetic test images with known ground truth."""
    test_data = []
    np.random.seed(42)
    
    for i in range(n_images):
        img_size = 200
        
        # Create green background
        bs_img = np.ones((img_size, img_size, 3), dtype=np.uint8)
        bs_img[:, :] = [0, 200, 0]  # BGR: Green background
        
        # Add black circle
        center = (100 + np.random.randint(-20, 20), 
                 100 + np.random.randint(-20, 20))
        radius = 30 + np.random.randint(-10, 10)
        cv2.circle(bs_img, center, radius, (0, 0, 0), -1)
        
        # Add noise
        noise = np.random.normal(0, 10, bs_img.shape).astype(np.int16)
        bs_img = np.clip(bs_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Create green background sample
        gb_img = np.ones((50, 50, 3), dtype=np.uint8)
        gb_img[:, :] = [0, 200, 0]
        
        test_data.append((bs_img, gb_img, center, radius, f'test_{i+1}'))
    
    return test_data


def run_comparison():
    """Main comparison function for CIBICA vs Hough Transform."""
    print("=" * 80)
    print("CIRCLE DETECTION COMPARISON: CIBICA vs HOUGH TRANSFORM")
    print("=" * 80)
    
    # Define preprocessing methods
    gl_methods = [f"GL{i}" for i in range(70, 87, 2)]
    med_methods = [f"Med{i}" for i in [3, 5, 7, 9, 11, 13, 15, 17, 19]]
    all_methods = gl_methods + med_methods
    
    # Create test data
    print("\nCreating synthetic test data...")
    test_data = create_synthetic_test_data(n_images=5)
    print(f"Created {len(test_data)} test images")
    
    results = []
    total_tests = len(test_data) * len(all_methods) * 2
    current_test = 0
    
    for bs_img, gb_img, true_center, true_radius, img_name in test_data:
        print(f"\nProcessing {img_name}...")
        print(f"  Ground truth: center={true_center}, radius={true_radius}")
        
        for method in all_methods:
            current_test += 2
            print(f"  Progress: {current_test}/{total_tests} - {method}...", end='\r')
            
            # Get edge points for CIBICA
            if method.startswith('GL'):
                edge_result = preprocess_image(
                    bs_image=bs_img,
                    gb_image=None,
                    method=method,
                    return_edgels=True
                )
            else:
                edge_result = preprocess_image(
                    bs_image=bs_img,
                    gb_image=gb_img,
                    method=method,
                    return_edgels=True
                )
            
            # Get edge image for Hough
            if method.startswith('GL'):
                edge_img = preprocess_image(
                    bs_image=bs_img,
                    gb_image=None,
                    method=method,
                    return_edgels=False
                )
            else:
                edge_img = preprocess_image(
                    bs_image=bs_img,
                    gb_image=gb_img,
                    method=method,
                    return_edgels=False
                )
            
            # Handle preprocessing failures
            if edge_result is False or (isinstance(edge_result, np.ndarray) and len(edge_result) == 0):
                edgels = np.array([[0, 0]])
            else:
                edgels = edge_result
                
            if edge_img is False:
                edge_img = np.zeros((200, 200), dtype=np.uint8)
            
            # Test CIBICA
            start_time = time.time()
            try:
                cibica_center, cibica_radius = cibica(edgels, Nmax=500)
            except:
                cibica_center, cibica_radius = np.array([-1, -1]), -1
            cibica_time = time.time() - start_time
            
            cibica_jaccard = calculate_jaccard_index(
                true_center, true_radius, cibica_center, cibica_radius,
                img_shape=bs_img.shape[:2]
            )
            
            results.append({
                'image': img_name,
                'preprocessing': method,
                'algorithm': 'CIBICA',
                'jaccard': cibica_jaccard,
                'runtime': cibica_time,
                'success': cibica_center[0] != -1
            })
            
            # Test Hough Transform
            start_time = time.time()
            try:
                hough_center, hough_radius = hough(edge_img, minRadius=20, maxRadius=50)
            except:
                hough_center, hough_radius = np.array([-1, -1]), -1
            hough_time = time.time() - start_time
            
            hough_jaccard = calculate_jaccard_index(
                true_center, true_radius, hough_center, hough_radius,
                img_shape=bs_img.shape[:2]
            )
            
            results.append({
                'image': img_name,
                'preprocessing': method,
                'algorithm': 'Hough',
                'jaccard': hough_jaccard,
                'runtime': hough_time,
                'success': hough_center[0] != -1
            })
    
    print("\n")
    df_results = pd.DataFrame(results)
    df_results.to_csv('cibica_vs_hough_results.csv', index=False)
    print("Results saved to cibica_vs_hough_results.csv")
    
    return df_results


def generate_visualizations(df_results):
    """Generate comprehensive comparison plots."""
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Overall comparison
    ax1 = plt.subplot(2, 3, 1)
    algo_means = df_results.groupby('algorithm')['jaccard'].mean()
    colors = ['#2E7D32', '#1565C0']
    bars = ax1.bar(algo_means.index, algo_means.values, color=colors)
    ax1.set_ylabel('Mean Jaccard Index')
    ax1.set_title('Overall Performance: CIBICA vs Hough')
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 2. Per-preprocessing method comparison
    ax2 = plt.subplot(2, 3, 2)
    prep_methods = sorted(df_results['preprocessing'].unique())
    
    cibica_scores = [df_results[(df_results['preprocessing'] == m) & 
                                (df_results['algorithm'] == 'CIBICA')]['jaccard'].mean() 
                     for m in prep_methods]
    hough_scores = [df_results[(df_results['preprocessing'] == m) & 
                               (df_results['algorithm'] == 'Hough')]['jaccard'].mean() 
                    for m in prep_methods]
    
    x = np.arange(len(prep_methods))
    width = 0.35
    
    ax2.bar(x - width/2, cibica_scores, width, label='CIBICA', color='#2E7D32', alpha=0.8)
    ax2.bar(x + width/2, hough_scores, width, label='Hough', color='#1565C0', alpha=0.8)
    
    ax2.set_xlabel('Preprocessing Method')
    ax2.set_ylabel('Mean Jaccard Index')
    ax2.set_title('Performance by Preprocessing Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(prep_methods, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax3 = plt.subplot(2, 3, 3)
    df_results.boxplot(column='jaccard', by='algorithm', ax=ax3)
    ax3.set_ylabel('Jaccard Index')
    ax3.set_title('Distribution of Jaccard Scores')
    plt.sca(ax3)
    plt.xticks(rotation=0)
    
    # 4. Success rate
    ax4 = plt.subplot(2, 3, 4)
    success_rates = df_results.groupby('algorithm')['success'].mean() * 100
    bars = ax4.bar(success_rates.index, success_rates.values, color=colors)
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Detection Success Rate')
    ax4.set_ylim([0, 105])
    ax4.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 5. Runtime comparison
    ax5 = plt.subplot(2, 3, 5)
    runtime_means = df_results.groupby('algorithm')['runtime'].mean()
    bars = ax5.bar(runtime_means.index, runtime_means.values, color=colors)
    ax5.set_ylabel('Average Runtime (seconds)')
    ax5.set_title('Runtime Performance')
    ax5.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom')
    
    # 6. Heatmap
    ax6 = plt.subplot(2, 3, 6)
    pivot_data = df_results.pivot_table(values='jaccard', 
                                        index='preprocessing', 
                                        columns='algorithm', 
                                        aggfunc='mean')
    
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0, vmax=1, ax=ax6, cbar_kws={'label': 'Jaccard Index'})
    ax6.set_title('Performance Heatmap')
    
    plt.tight_layout()
    plt.savefig('cibica_vs_hough_comparison.png', dpi=300, bbox_inches='tight')
    print("Plots saved to cibica_vs_hough_comparison.png")
    
    # Print summary
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    
    for algo in ['CIBICA', 'Hough']:
        algo_data = df_results[df_results['algorithm'] == algo]
        print(f"\n{algo}:")
        print(f"  Mean Jaccard: {algo_data['jaccard'].mean():.4f}")
        print(f"  Std Dev: {algo_data['jaccard'].std():.4f}")
        print(f"  Success Rate: {algo_data['success'].mean()*100:.1f}%")
        print(f"  Mean Runtime: {algo_data['runtime'].mean():.4f}s")
    
    plt.show()


if __name__ == "__main__":
    # Run comparison
    results_df = run_comparison()
    
    # Generate visualizations
    generate_visualizations(results_df)
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)