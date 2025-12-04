"""
Visualization and analysis tools for attention maps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch


def plot_attention_heatmap(attention_weights, layer_idx=0, head_idx=0, sample_idx=0, 
                          title=None, save_path=None):
    """
    Plot attention heatmap.
    
    Args:
        attention_weights: List of (batch_size, nhead, seq_len, seq_len) tensors
        layer_idx: Which layer to visualize
        head_idx: Which attention head to visualize
        sample_idx: Which sample in the batch to visualize
        title: Plot title
        save_path: Path to save figure
    """
    # Extract attention for specific layer, head, and sample
    attn = attention_weights[layer_idx][sample_idx, head_idx].detach().cpu().numpy()
    
    seq_len = attn.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(attn, cmap='viridis', aspect='auto', origin='lower')
    
    ax.set_xlabel('Time Being Attended To', fontsize=12)
    ax.set_ylabel('Time Doing the Attending', fontsize=12)
    
    if title is None:
        title = f'Attention Map - Layer {layer_idx}, Head {head_idx}'
    ax.set_title(title, fontsize=14)
    
    plt.colorbar(im, ax=ax, label='Attention Strength')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def compute_attention_entropy(attention_weights, layer_idx=0, head_idx=0, sample_idx=0):
    """
    Compute entropy of attention distribution for each position.
    Higher entropy = more uniform attention (less focused).
    Lower entropy = more focused attention.
    
    Returns: (seq_len,) array of entropies
    """
    attn = attention_weights[layer_idx][sample_idx, head_idx].detach().cpu().numpy()
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    attn = attn + eps
    attn = attn / attn.sum(axis=1, keepdims=True)  # Normalize rows
    
    # Compute entropy for each row (each attending position)
    entropy = -np.sum(attn * np.log(attn), axis=1)
    
    return entropy


def compute_attention_sparsity(attention_weights, layer_idx=0, head_idx=0, sample_idx=0):
    """
    Compute sparsity of attention (Gini coefficient).
    Higher sparsity = more concentrated attention.
    Lower sparsity = more uniform attention.
    Returns: (seq_len,) array of sparsity values
    """
    attn = attention_weights[layer_idx][sample_idx, head_idx].detach().cpu().numpy()
    
    # Normalize rows
    attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-10)
    
    # Sort each row
    attn_sorted = np.sort(attn, axis=1)
    n = attn.shape[1]
    
    # Compute Gini coefficient for each row
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * attn_sorted, axis=1)) / (n * np.sum(attn_sorted, axis=1)) - (n + 1) / n
    
    return gini


def detect_diagonal_pattern(attention_weights, layer_idx=0, head_idx=0, sample_idx=0, 
                           diagonal_width=3):
    """
    Detect diagonal patterns in attention (local attention).
    Returns: strength of diagonal pattern (higher = more diagonal)
    """
    attn = attention_weights[layer_idx][sample_idx, head_idx].detach().cpu().numpy()
    seq_len = attn.shape[0]
    
    diagonal_strength = 0.0
    for i in range(seq_len):
        for offset in range(-diagonal_width, diagonal_width + 1):
            j = i + offset
            if 0 <= j < seq_len:
                diagonal_strength += attn[i, j]
    
    # Normalize by number of positions
    diagonal_strength = diagonal_strength / seq_len
    return diagonal_strength


def detect_horizontal_pattern(attention_weights, layer_idx=0, head_idx=0, sample_idx=0):
    """
    Detect horizontal patterns (attending to specific time points).
    Returns: variance across rows (lower = more horizontal)
    """
    attn = attention_weights[layer_idx][sample_idx, head_idx].detach().cpu().numpy()
    
    # Compute column sums (how much each time point is attended to)
    column_sums = attn.sum(axis=0)
    
    # High variance = some time points much more attended to (horizontal pattern)
    return np.var(column_sums)


def detect_vertical_pattern(attention_weights, layer_idx=0, head_idx=0, sample_idx=0):
    """
    Detect vertical patterns (specific attending positions focus heavily).
    Returns: variance across columns (lower = more vertical)
    """
    attn = attention_weights[layer_idx][sample_idx, head_idx].detach().cpu().numpy()
    
    # Compute row sums (how much each attending position contributes)
    row_sums = attn.sum(axis=1)
    
    # High variance = some positions attend much more (vertical pattern)
    return np.var(row_sums)


def analyze_attention_patterns(attention_weights, layer_idx=0, head_idx=0, sample_idx=0):
    """
    Comprehensive analysis of attention patterns.
    Returns: Dictionary with various pattern metrics
    """
    attn = attention_weights[layer_idx][sample_idx, head_idx].detach().cpu().numpy()
    
    seq_len = attn.shape[0]
    
    # Normalize
    attn_norm = attn / (attn.sum(axis=1, keepdims=True) + 1e-10)
    
    results = {
        'entropy_mean': np.mean(compute_attention_entropy(attention_weights, layer_idx, head_idx, sample_idx)),
        'entropy_std': np.std(compute_attention_entropy(attention_weights, layer_idx, head_idx, sample_idx)),
        'sparsity_mean': np.mean(compute_attention_sparsity(attention_weights, layer_idx, head_idx, sample_idx)),
        'sparsity_std': np.std(compute_attention_sparsity(attention_weights, layer_idx, head_idx, sample_idx)),
        'diagonal_strength': detect_diagonal_pattern(attention_weights, layer_idx, head_idx, sample_idx),
        'horizontal_pattern': detect_horizontal_pattern(attention_weights, layer_idx, head_idx, sample_idx),
        'vertical_pattern': detect_vertical_pattern(attention_weights, layer_idx, head_idx, sample_idx),
        'max_attention': np.max(attn),
        'mean_attention': np.mean(attn),
        'attention_concentration': np.sum(attn_norm ** 2) / seq_len,  # Higher = more concentrated
    }
    
    return results


def plot_attention_analysis_summary(attention_weights, generator_name, save_dir=None):
    """
    Create comprehensive visualization of attention patterns.
    """
    n_layers = len(attention_weights)
    n_heads = attention_weights[0].shape[1]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 5 * n_layers))
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            ax_idx = layer_idx * n_heads + head_idx + 1
            ax = plt.subplot(n_layers, n_heads, ax_idx)
            
            # Average across batch
            attn = attention_weights[layer_idx][:, head_idx, :, :].mean(dim=0).detach().cpu().numpy()
            
            im = ax.imshow(attn, cmap='viridis', aspect='auto', origin='lower')
            ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=10)
            ax.set_xlabel('Attended To')
            ax.set_ylabel('Attending')
            plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Attention Maps: {generator_name}', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/attention_summary_{generator_name}.png', dpi=150, bbox_inches='tight')
    
    return fig


def plot_entropy_analysis(attention_weights, generator_name, save_path=None):
    """
    Plot entropy analysis across layers and heads.
    """
    n_layers = len(attention_weights)
    n_heads = attention_weights[0].shape[1]
    n_samples = attention_weights[0].shape[0]
    
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 3 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if n_heads == 1:
        axes = axes.reshape(-1, 1)
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx]
            
            entropies = []
            for sample_idx in range(n_samples):
                entropy = compute_attention_entropy(attention_weights, layer_idx, head_idx, sample_idx)
                entropies.append(entropy)
            
            entropies = np.array(entropies)
            mean_entropy = entropies.mean(axis=0)
            std_entropy = entropies.std(axis=0)
            
            x = np.arange(len(mean_entropy))
            ax.plot(x, mean_entropy, 'b-', label='Mean')
            ax.fill_between(x, mean_entropy - std_entropy, mean_entropy + std_entropy, alpha=0.3)
            ax.set_title(f'L{layer_idx}H{head_idx}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Entropy')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Attention Entropy: {generator_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

