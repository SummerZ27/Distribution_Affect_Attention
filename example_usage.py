"""
Example script demonstrating how to use individual components.
"""

import numpy as np
import matplotlib.pyplot as plt
from data_generators import HeavyTailedAR1, GARCH11, get_all_generators
from transformer_model import TransformerForecaster
from trainer import TimeSeriesDataset, get_attention_weights_direct
from visualization import plot_attention_heatmap, analyze_attention_patterns
import torch


def example_generate_data():
    """Example: Generate time series data."""
    print("Example 1: Generating time series data")
    print("-" * 50)
    
    # Generate heavy-tailed AR(1) process
    generator = HeavyTailedAR1(phi=0.7, df=3.0, length=500, seed=42)
    series = generator.generate()
    
    print(f"Generated series length: {len(series)}")
    print(f"Mean: {np.mean(series):.4f}, Std: {np.std(series):.4f}")
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(series[:200])
    plt.title("Heavy-Tailed AR(1) Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.savefig("example_series.png", dpi=150, bbox_inches='tight')
    print("Saved plot to example_series.png")
    plt.close()
    
    return series


def example_create_dataset():
    """Example: Create dataset from time series."""
    print("\nExample 2: Creating dataset")
    print("-" * 50)
    
    generator = GARCH11(length=1000, seed=42)
    series = generator.generate()
    
    # Normalize
    series = (series - np.mean(series)) / np.std(series)
    
    # Create dataset
    dataset = TimeSeriesDataset(series, input_len=100, output_len=20)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Input shape: {dataset[0][0].shape}")
    print(f"Output shape: {dataset[0][1].shape}")
    
    return dataset


def example_model_usage():
    """Example: Create and use transformer model."""
    print("\nExample 3: Transformer model")
    print("-" * 50)
    
    # Create model
    model = TransformerForecaster(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        input_len=100,
        output_len=20
    )
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 100)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def example_attention_extraction():
    """Example: Extract and visualize attention weights."""
    print("\nExample 4: Attention extraction")
    print("-" * 50)
    
    # Generate data
    generator = HeavyTailedAR1(length=500, seed=42)
    series = generator.generate()
    series = (series - np.mean(series)) / np.std(series)
    
    # Create dataset
    dataset = TimeSeriesDataset(series, input_len=100, output_len=20)
    sample_input = dataset[0][0].unsqueeze(0)  # Add batch dimension
    
    # Create model (use a small one for quick example)
    model = TransformerForecaster(
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=128,
        input_len=100,
        output_len=20
    )
    
    # Extract attention (even without training, we can see the structure)
    attention_weights = get_attention_weights_direct(model, sample_input)
    
    print(f"Number of layers: {len(attention_weights)}")
    print(f"Attention shape (layer 0): {attention_weights[0].shape}")
    
    # Analyze patterns
    metrics = analyze_attention_patterns(attention_weights, layer_idx=0, head_idx=0, sample_idx=0)
    print("\nAttention metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Plot attention map
    plot_attention_heatmap(
        attention_weights,
        layer_idx=0,
        head_idx=0,
        sample_idx=0,
        title="Example Attention Map (Untrained Model)",
        save_path="example_attention.png"
    )
    print("\nSaved attention map to example_attention.png")
    plt.close()


if __name__ == '__main__':
    print("=" * 50)
    print("Attention Map Analysis - Example Usage")
    print("=" * 50)
    
    # Run examples
    example_generate_data()
    example_create_dataset()
    example_model_usage()
    example_attention_extraction()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)
    print("\nTo run full experiments, use:")
    print("  python main_experiment.py")
    print("  or")
    print("  python main.py")

