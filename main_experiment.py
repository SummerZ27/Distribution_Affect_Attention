"""
Main experiment script for exploring attention maps across different input distributions.
"""

import os
import numpy as np
import torch
import torch.utils.data as data
from data_generators import get_all_generators
from transformer_model import TransformerForecaster
from trainer import TimeSeriesDataset, train_model, get_attention_weights_direct
from visualization import (
    plot_attention_heatmap, plot_attention_analysis_summary, 
    plot_entropy_analysis, analyze_attention_patterns
)
import json
from tqdm import tqdm


# Fixed hyperparameters
CONFIG = {
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'input_len': 100,  # L: past points
    'output_len': 20,   # H: future points
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.001,
    'series_length': 2000,  # Total length of generated series
    'train_split': 0.7,
    'val_split': 0.15,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def prepare_data(generator, config, save_raw=False, generator_name=None, data_dir='generated_data'):
    """Generate and split data."""
    series = generator.generate()
    
    # Save raw data before normalization if requested
    if save_raw and generator_name:
        os.makedirs(data_dir, exist_ok=True)
        raw_path = os.path.join(data_dir, f'{generator_name}_raw.npy')
        np.save(raw_path, series)
    
    # Normalize
    series_mean = np.mean(series)
    series_std = np.std(series)
    series = (series - series_mean) / (series_std + 1e-8)
    
    # Split
    n = len(series)
    train_end = int(n * config['train_split'])
    val_end = train_end + int(n * config['val_split'])
    
    train_series = series[:train_end]
    val_series = series[train_end:val_end]
    test_series = series[val_end:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        train_series, config['input_len'], config['output_len']
    )
    val_dataset = TimeSeriesDataset(
        val_series, config['input_len'], config['output_len']
    )
    test_dataset = TimeSeriesDataset(
        test_series, config['input_len'], config['output_len']
    )
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    return train_loader, val_loader, test_loader, {
        'mean': series_mean,
        'std': series_std
    }


def run_experiment(generator_name, generator, config, results_dir='results'):
    """Run experiment for a single generator."""
    print(f"\n{'='*60}")
    print(f"Experiment: {generator_name}")
    print(f"{'='*60}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    generator_dir = os.path.join(results_dir, generator_name)
    os.makedirs(generator_dir, exist_ok=True)
    
    # Prepare data
    print("Generating data...")
    train_loader, val_loader, test_loader, norm_params = prepare_data(
        generator, config, save_raw=True, generator_name=generator_name, 
        data_dir=os.path.join(results_dir, '..', 'generated_data')
    )
    
    # Initialize model
    print("Initializing model...")
    model = TransformerForecaster(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        input_len=config['input_len'],
        output_len=config['output_len']
    )
    
    # Train model
    print("Training model...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=config['epochs'],
        lr=config['lr'],
        device=config['device']
    )
    
    # Save training curves
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training Curves: {generator_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(generator_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Extract attention weights
    print("Extracting attention weights...")
    # Use a batch from test set
    test_batch = next(iter(test_loader))[0][:5]  # Use first 5 samples
    test_batch = test_batch.to(config['device'])
    
    attention_weights = get_attention_weights_direct(model, test_batch, device=config['device'])
    
    # Visualize attention maps
    print("Creating visualizations...")
    
    # Summary heatmaps
    plot_attention_analysis_summary(
        attention_weights, generator_name,
        save_dir=generator_dir
    )
    
    # Entropy analysis
    plot_entropy_analysis(
        attention_weights, generator_name,
        save_path=os.path.join(generator_dir, 'entropy_analysis.png')
    )
    
    # Individual attention maps for each layer/head
    n_layers = len(attention_weights)
    n_heads = attention_weights[0].shape[1]
    n_samples = min(3, attention_weights[0].shape[0])  # Visualize first 3 samples
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            for sample_idx in range(n_samples):
                plot_attention_heatmap(
                    attention_weights,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    sample_idx=sample_idx,
                    title=f'{generator_name} - Layer {layer_idx}, Head {head_idx}, Sample {sample_idx}',
                    save_path=os.path.join(
                        generator_dir,
                        f'attention_L{layer_idx}_H{head_idx}_S{sample_idx}.png'
                    )
                )
                plt.close()
    
    # Compute attention pattern metrics
    print("Computing attention metrics...")
    metrics = {}
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            layer_head_metrics = []
            for sample_idx in range(attention_weights[0].shape[0]):
                pattern_metrics = analyze_attention_patterns(
                    attention_weights, layer_idx, head_idx, sample_idx
                )
                layer_head_metrics.append(pattern_metrics)
            
            # Average across samples
            avg_metrics = {}
            for key in layer_head_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in layer_head_metrics])
            
            metrics[f'layer_{layer_idx}_head_{head_idx}'] = avg_metrics
    
    # Save metrics (convert numpy types to Python native types for JSON)
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    metrics_serializable = convert_to_serializable(metrics)
    with open(os.path.join(generator_dir, 'attention_metrics.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(config['device']), targets.to(config['device'])
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    test_loss = test_loss / len(test_loader)
    
    # Save results summary
    summary = {
        'generator_name': generator_name,
        'test_loss': test_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'metrics': metrics,
        'config': config
    }
    
    with open(os.path.join(generator_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Results saved to {generator_dir}")
    
    return summary


def main():
    """Run experiments for all generators."""
    print("="*60)
    print("Attention Map Analysis: Input Distribution Effects")
    print("="*60)
    
    # Set random seeds
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])
    
    # Update generator lengths
    generators = get_all_generators(length=CONFIG['series_length'], seed=CONFIG['seed'])
    
    # Run experiments
    all_results = {}
    for generator_name, generator in tqdm(generators.items(), desc="Running experiments"):
        try:
            result = run_experiment(generator_name, generator, CONFIG)
            all_results[generator_name] = result
        except Exception as e:
            print(f"Error in {generator_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison summary
    print("\n" + "="*60)
    print("Creating comparison summary...")
    print("="*60)
    
    comparison_data = []
    for gen_name, result in all_results.items():
        comparison_data.append({
            'generator': gen_name,
            'test_loss': result['test_loss'],
            'final_val_loss': result['final_val_loss'],
        })
    
    # Save comparison
    with open('results/comparison_summary.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print("\nAll experiments completed!")
    print(f"Results saved in 'results/' directory")


if __name__ == '__main__':
    main()

