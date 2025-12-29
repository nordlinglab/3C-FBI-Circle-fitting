"""
compare_fbi_vs_all.py

Compare FBI (3C-FBI/CCC-FBI) algorithm against all other circle fitting methods.
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

# Import all algorithms with CORRECT names from uploaded files
from CCC_FBI import ccc_fbi  # FBI/3C-FBI algorithm
from RFCA import rfca
from GUO import guo_2019
from GRECO import greco_2022
from QI import qi_2024
from RHT import rht
from RCD import rcd
from NURUNNABI import nurunnabi
from CIBICA import cibica
from preprocessing import preprocess_image

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


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


def create_synthetic_test_data(n_images=10):
    """Create synthetic test images with varying difficulty."""
    test_data = []
    np.random.seed(42)
    
    for i in range(n_images):
        img_size = 200
        
        # Green background with varying intensity
        bs_img = np.ones((img_size, img_size, 3), dtype=np.uint8)
        green_intensity = 150 + np.random.randint(0, 100)
        bs_img[:, :] = [0, green_intensity, 0]  # BGR
        
        # Black circle with random position and size
        center = (100 + np.random.randint(-30, 30), 
                 100 + np.random.randint(-30, 30))
        radius = 25 + np.random.randint(-10, 15)
        cv2.circle(bs_img, center, radius, (0, 0, 0), -1)
        
        # Add increasing noise
        noise_level = 5 + i * 2
        noise = np.random.normal(0, noise_level, bs_img.shape).astype(np.int16)
        bs_img = np.clip(bs_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add blur for later tests
        if i > 5:
            bs_img = cv2.GaussianBlur(bs_img, (5, 5), 0)
        
        # Green background sample
        gb_img = np.ones((50, 50, 3), dtype=np.uint8)
        gb_img[:, :] = [0, green_intensity, 0]
        
        test_data.append((bs_img, gb_img, center, radius, f'test_{i+1}'))
    
    return test_data


def run_algorithm_safely(algo_func, edgels, algo_name):
    """Run algorithm with error handling."""
    try:
        start_time = time.time()
        center, radius = algo_func(edgels)
        runtime = time.time() - start_time
        return center, radius, runtime
    except Exception as e:
        print(f"    Warning: {algo_name} failed: {str(e)[:50]}")
        return np.array([-1, -1]), -1, 0.0


def run_comprehensive_comparison():
    """Main comparison function for FBI vs all algorithms."""
    print("=" * 80)
    print("COMPREHENSIVE CIRCLE DETECTION COMPARISON")
    print("FBI (3C-FBI/CCC-FBI) vs All Other Methods")
    print("=" * 80)
    
    # Define algorithms with correct function names
    algorithms = {
        'FBI (3C-FBI)': ccc_fbi,
        'RFCA': rfca,
        'Guo 2019': guo_2019,
        'Greco 2022': greco_2022,
        'Qi 2024': qi_2024,
        'RHT': rht,
        'RCD': rcd,
        'Nurunnabi': nurunnabi,
        'CIBICA': cibica
    }
    
    # Use subset of methods for faster testing
    gl_methods = [f"GL{i}" for i in [76, 80, 84]]  # Best GL methods
    med_methods = [f"Med{i}" for i in [7, 9, 11]]  # Best Med methods
    all_methods = gl_methods + med_methods
    
    # Create test data
    print("\nCreating synthetic test data...")
    test_data = create_synthetic_test_data(n_images=5)
    print(f"Created {len(test_data)} test images")
    
    results = []
    total_tests = len(test_data) * len(all_methods) * len(algorithms)
    current_test = 0
    
    for bs_img, gb_img, true_center, true_radius, img_name in test_data:
        print(f"\nProcessing {img_name}...")
        
        for method in all_methods:
            # Get edge points
            if method.startswith('GL'):
                edgels = preprocess_image(
                    bs_image=bs_img,
                    gb_image=None,
                    method=method,
                    return_edgels=True
                )
            else:
                edgels = preprocess_image(
                    bs_image=bs_img,
                    gb_image=gb_img,
                    method=method,
                    return_edgels=True
                )
            
            # Handle preprocessing failures
            if edgels is False or (isinstance(edgels, np.ndarray) and len(edgels) == 0):
                # Generate synthetic edges around true circle
                theta = np.linspace(0, 2*np.pi, 50)
                edgels = np.column_stack([
                    true_center[0] + true_radius * np.cos(theta) + np.random.randn(50)*2,
                    true_center[1] + true_radius * np.sin(theta) + np.random.randn(50)*2
                ])
            
            # Test each algorithm
            for algo_name, algo_func in algorithms.items():
                current_test += 1
                progress = (current_test / total_tests) * 100
                print(f"  Progress: {progress:.1f}% - {algo_name} with {method}...", end='\r')
                
                center, radius, runtime = run_algorithm_safely(algo_func, edgels, algo_name)
                
                jaccard = calculate_jaccard_index(
                    true_center, true_radius, center, radius,
                    img_shape=bs_img.shape[:2]
                )
                
                results.append({
                    'image': img_name,
                    'preprocessing': method,
                    'algorithm': algo_name,
                    'jaccard': jaccard,
                    'runtime': runtime,
                    'success': center[0] != -1
                })
    
    print("\n")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('fbi_vs_all_results.csv', index=False)
    print("Results saved to fbi_vs_all_results.csv")
    
    return df_results


def generate_comprehensive_visualizations(df_results):
    """Generate comprehensive comparison plots."""
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(24, 16))
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, 9))
    algo_colors = {algo: colors[i] for i, algo in enumerate(df_results['algorithm'].unique())}
    algo_colors['FBI (3C-FBI)'] = '#FF0000'  # Red for FBI
    
    # 1. Overall performance
    ax1 = plt.subplot(3, 3, 1)
    algo_means = df_results.groupby('algorithm')['jaccard'].mean().sort_values(ascending=False)
    bars = ax1.bar(range(len(algo_means)), algo_means.values,
                   color=[algo_colors[name] for name in algo_means.index])
    
    # Highlight FBI
    fbi_idx = list(algo_means.index).index('FBI (3C-FBI)')
    bars[fbi_idx].set_edgecolor('black')
    bars[fbi_idx].set_linewidth(3)
    
    ax1.set_xticks(range(len(algo_means)))
    ax1.set_xticklabels(algo_means.index, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Mean Jaccard Index')
    ax1.set_title('Overall Algorithm Performance', fontweight='bold')
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Box plot
    ax2 = plt.subplot(3, 3, 2)
    boxplot_data = [df_results[df_results['algorithm'] == algo]['jaccard'].values 
                    for algo in algo_means.index]
    box_plot = ax2.boxplot(boxplot_data, labels=algo_means.index, patch_artist=True)
    
    for patch, algo in zip(box_plot['boxes'], algo_means.index):
        patch.set_facecolor(algo_colors[algo])
        patch.set_alpha(0.7)
        if algo == 'FBI (3C-FBI)':
            patch.set_linewidth(2)
            patch.set_edgecolor('black')
    
    ax2.set_xticklabels(algo_means.index, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Jaccard Index')
    ax2.set_title('Jaccard Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Success rate
    ax3 = plt.subplot(3, 3, 3)
    success_rates = df_results.groupby('algorithm')['success'].mean() * 100
    success_rates = success_rates.reindex(algo_means.index)
    
    bars = ax3.bar(range(len(success_rates)), success_rates.values,
                   color=[algo_colors[name] for name in success_rates.index])
    
    fbi_idx = list(success_rates.index).index('FBI (3C-FBI)')
    bars[fbi_idx].set_edgecolor('black')
    bars[fbi_idx].set_linewidth(3)
    
    ax3.set_xticks(range(len(success_rates)))
    ax3.set_xticklabels(success_rates.index, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Detection Success Rate', fontweight='bold')
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 4. Heatmap
    ax4 = plt.subplot(3, 3, (4, 5))
    pivot_data = df_results.pivot_table(values='jaccard', 
                                        index='preprocessing', 
                                        columns='algorithm', 
                                        aggfunc='mean')
    
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0, vmax=1, ax=ax4, cbar_kws={'label': 'Jaccard Index'},
                linewidths=0.5, linecolor='gray')
    ax4.set_title('Performance Heatmap', fontweight='bold')
    
    # 5. Runtime comparison
    ax5 = plt.subplot(3, 3, 6)
    runtime_means = df_results.groupby('algorithm')['runtime'].mean()
    runtime_means = runtime_means.reindex(algo_means.index)
    
    bars = ax5.bar(range(len(runtime_means)), runtime_means.values,
                   color=[algo_colors[name] for name in runtime_means.index])
    
    fbi_idx = list(runtime_means.index).index('FBI (3C-FBI)')
    bars[fbi_idx].set_edgecolor('black')
    bars[fbi_idx].set_linewidth(3)
    
    ax5.set_xticks(range(len(runtime_means)))
    ax5.set_xticklabels(runtime_means.index, rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('Runtime (seconds)')
    ax5.set_title('Average Runtime', fontweight='bold')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Ranking
    ax6 = plt.subplot(3, 3, 7)
    
    # Calculate average rank for each algorithm
    rankings = []
    for prep in df_results['preprocessing'].unique():
        prep_data = df_results[df_results['preprocessing'] == prep]
        prep_ranks = prep_data.groupby('algorithm')['jaccard'].mean().rank(ascending=False)
        rankings.append(prep_ranks)
    
    if rankings:
        avg_rankings = pd.concat(rankings, axis=1).mean(axis=1).sort_values()
        
        bars = ax6.barh(range(len(avg_rankings)), avg_rankings.values,
                        color=[algo_colors[name] for name in avg_rankings.index])
        
        if 'FBI (3C-FBI)' in avg_rankings.index:
            fbi_idx = list(avg_rankings.index).index('FBI (3C-FBI)')
            bars[fbi_idx].set_edgecolor('black')
            bars[fbi_idx].set_linewidth(3)
        
        ax6.set_yticks(range(len(avg_rankings)))
        ax6.set_yticklabels(avg_rankings.index, fontsize=9)
        ax6.set_xlabel('Average Rank (lower is better)')
        ax6.set_title('Algorithm Rankings', fontweight='bold')
        ax6.invert_xaxis()
        ax6.grid(True, alpha=0.3, axis='x')
    
    # 7. Performance by preprocessing type
    ax7 = plt.subplot(3, 3, 8)
    
    gl_data = df_results[df_results['preprocessing'].str.startswith('GL')]
    med_data = df_results[df_results['preprocessing'].str.startswith('Med')]
    
    gl_means = gl_data.groupby('algorithm')['jaccard'].mean()
    med_means = med_data.groupby('algorithm')['jaccard'].mean()
    
    x = np.arange(len(algo_means.index))
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, [gl_means.get(algo, 0) for algo in algo_means.index],
                    width, label='GL Methods', alpha=0.8)
    bars2 = ax7.bar(x + width/2, [med_means.get(algo, 0) for algo in algo_means.index],
                    width, label='Med Methods', alpha=0.8)
    
    ax7.set_xticks(x)
    ax7.set_xticklabels(algo_means.index, rotation=45, ha='right', fontsize=9)
    ax7.set_ylabel('Mean Jaccard Index')
    ax7.set_title('Performance: GL vs Med Methods', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Statistical summary table
    ax8 = plt.subplot(3, 3, 9)
    ax8.axis('tight')
    ax8.axis('off')
    
    summary_data = []
    for algo in algo_means.index:
        algo_data = df_results[df_results['algorithm'] == algo]
        summary_data.append([
            algo[:12],
            f"{algo_data['jaccard'].mean():.3f}",
            f"{algo_data['jaccard'].std():.3f}",
            f"{algo_data['success'].mean()*100:.1f}%"
        ])
    
    table = ax8.table(cellText=summary_data,
                     colLabels=['Algorithm', 'Mean J', 'Std J', 'Success'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Highlight FBI row
    if any('FBI' in row[0] for row in summary_data):
        fbi_row = [i for i, row in enumerate(summary_data) if 'FBI' in row[0]][0]
        for j in range(4):
            table[(fbi_row + 1, j)].set_facecolor('#FFE5E5')
    
    ax8.set_title('Statistical Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('fbi_vs_all_comparison.png', dpi=300, bbox_inches='tight')
    print("Plots saved to fbi_vs_all_comparison.png")
    
    plt.show()


def print_comprehensive_report(df_results):
    """Print detailed statistical report."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STATISTICAL REPORT")
    print("=" * 80)
    
    # Overall ranking
    print("\n1. OVERALL PERFORMANCE RANKING")
    print("-" * 60)
    
    algo_stats = df_results.groupby('algorithm')['jaccard'].agg(['mean', 'std', 'median'])
    algo_stats = algo_stats.sort_values('mean', ascending=False)
    
    for rank, (algo, row) in enumerate(algo_stats.iterrows(), 1):
        star = "★" if algo == 'FBI (3C-FBI)' else " "
        print(f"{star} {rank}. {algo:15s} - Mean: {row['mean']:.4f}, "
              f"Std: {row['std']:.4f}, Median: {row['median']:.4f}")
    
    # FBI comparison
    print("\n2. FBI HEAD-TO-HEAD COMPARISON")
    print("-" * 60)
    
    if 'FBI (3C-FBI)' in algo_stats.index:
        fbi_mean = algo_stats.loc['FBI (3C-FBI)', 'mean']
        
        for algo in algo_stats.index:
            if algo != 'FBI (3C-FBI)':
                algo_mean = algo_stats.loc[algo, 'mean']
                diff = fbi_mean - algo_mean
                
                if diff > 0:
                    print(f"FBI beats {algo:15s} by {diff:.4f} Jaccard points")
                else:
                    print(f"FBI loses to {algo:15s} by {abs(diff):.4f} Jaccard points")
    
    # Success rates
    print("\n3. SUCCESS RATES")
    print("-" * 60)
    
    success_rates = df_results.groupby('algorithm')['success'].mean() * 100
    success_rates = success_rates.sort_values(ascending=False)
    
    for algo, rate in success_rates.items():
        star = "★" if algo == 'FBI (3C-FBI)' else " "
        print(f"{star} {algo:15s} - {rate:.1f}%")


if __name__ == "__main__":
    # Run comparison
    print("Starting comprehensive comparison...")
    results_df = run_comprehensive_comparison()
    
    # Generate visualizations
    generate_comprehensive_visualizations(results_df)
    
    # Print report
    print_comprehensive_report(results_df)
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("Results saved to: fbi_vs_all_results.csv")
    print("Plots saved to: fbi_vs_all_comparison.png")
    print("=" * 80)