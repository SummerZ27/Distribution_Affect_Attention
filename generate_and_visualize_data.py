"""
Generate all time series data and save to files, plus create visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_generators import get_all_generators


def generate_and_save_all_data(series_length=2000, seed=42, output_dir='generated_data'):
    """
    Generate all time series data and save to files.
    
    Returns:
        Dictionary mapping generator names to their raw series
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generators = get_all_generators(length=series_length, seed=seed)
    all_data = {}
    
    print("Generating all time series data...")
    for generator_name, generator in generators.items():
        print(f"  Generating {generator_name}...")
        series = generator.generate()
        all_data[generator_name] = series
        
        # Save as CSV
        df = pd.DataFrame({
            'time': np.arange(len(series)),
            'value': series
        })
        csv_path = os.path.join(output_dir, f'{generator_name}.csv')
        df.to_csv(csv_path, index=False)
        print(f"    Saved to {csv_path}")
        
        # Also save as numpy file
        np_path = os.path.join(output_dir, f'{generator_name}.npy')
        np.save(np_path, series)
        print(f"    Saved to {np_path}")
    
    # Save metadata
    metadata = {
        'series_length': series_length,
        'seed': seed,
        'generators': list(generators.keys()),
        'description': {
            'HeavyTailedAR1': 'AR(1) process with Student-t noise (fat-tailed outliers)',
            'GARCH11': 'Time-varying volatility with clustering',
            'JumpDiffusion': 'Drift-diffusion with random massive jumps',
            'RegimeSwitching': 'Mean shifts based on hidden Markov state',
            'TrendBreaks': 'Linear trend with abrupt slope changes',
            'RandomWalk': 'Non-stationary process with no mean reversion',
            'SeasonTrendOutliers': 'Sinusoidal wave with trend and sparse outliers',
            'MultiSeasonality': 'Complex overlapping cycles',
            'OneOverFNoise': '1/f noise with persistent correlations',
        }
    }
    
    import json
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_path}")
    
    return all_data


def visualize_all_data(all_data, output_dir='generated_data', show_plot=False):
    """
    Create visualizations for all generated time series.
    """
    n_generators = len(all_data)
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Create 3x3 grid
    for idx, (generator_name, series) in enumerate(all_data.items(), 1):
        ax = plt.subplot(3, 3, idx)
        
        # Plot full series
        ax.plot(series, linewidth=0.8, alpha=0.7)
        ax.set_title(f'{generator_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(series):.3f}\nStd: {np.std(series):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('All Generated Time Series Data', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    viz_path = os.path.join(output_dir, 'all_time_series_overview.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"\nOverview visualization saved to {viz_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Create individual detailed plots
    print("\nCreating individual visualizations...")
    for generator_name, series in all_data.items():
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Full series
        axes[0].plot(series, linewidth=0.8, alpha=0.8, color='steelblue')
        axes[0].set_title(f'{generator_name} - Full Series', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('Value', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # First 500 points (zoomed view)
        zoom_length = min(500, len(series))
        axes[1].plot(series[:zoom_length], linewidth=1.2, alpha=0.8, color='coral')
        axes[1].set_title(f'{generator_name} - First {zoom_length} Points (Zoomed)', 
                          fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('Value', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = (
            f'Length: {len(series)}\n'
            f'Mean: {np.mean(series):.4f}\n'
            f'Std: {np.std(series):.4f}\n'
            f'Min: {np.min(series):.4f}\n'
            f'Max: {np.max(series):.4f}'
        )
        axes[0].text(0.98, 0.98, stats_text, transform=axes[0].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        # Save individual plot
        individual_path = os.path.join(output_dir, f'{generator_name}_visualization.png')
        plt.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {generator_name} visualization")
    
    # Create comparison plot with first 500 points of each
    print("\nCreating comparison plot...")
    fig, ax = plt.subplots(figsize=(18, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    zoom_length = min(500, min(len(s) for s in all_data.values()))
    
    for (generator_name, series), color in zip(all_data.items(), colors):
        ax.plot(series[:zoom_length], label=generator_name, linewidth=1.5, alpha=0.7, color=color)
    
    ax.set_title(f'Comparison: First {zoom_length} Points of All Time Series', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    comparison_path = os.path.join(output_dir, 'comparison_first_500_points.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {comparison_path}")


def main():
    """Main function to generate and visualize all data."""
    print("="*60)
    print("Generating and Visualizing All Time Series Data")
    print("="*60)
    
    # Generate all data
    all_data = generate_and_save_all_data(series_length=2000, seed=42)
    
    # Create visualizations
    visualize_all_data(all_data)
    
    print("\n" + "="*60)
    print("All data generated and saved!")
    print(f"Check the 'generated_data/' directory for:")
    print("  - CSV files: <GeneratorName>.csv")
    print("  - NumPy files: <GeneratorName>.npy")
    print("  - Visualizations: all_time_series_overview.png")
    print("  - Individual plots: <GeneratorName>_visualization.png")
    print("  - Comparison plot: comparison_first_500_points.png")
    print("="*60)


if __name__ == '__main__':
    main()

