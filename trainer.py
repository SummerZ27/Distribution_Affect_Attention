"""
Training utilities for transformer time series forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(self, series, input_len, output_len):
        """
        Args:
            series: 1D numpy array of time series
            input_len: Length of input sequence (L)
            output_len: Length of output sequence (H)
        """
        self.series = series
        self.input_len = input_len
        self.output_len = output_len
        
        # Create sliding windows
        self.samples = []
        for i in range(len(series) - input_len - output_len + 1):
            input_seq = series[i:i+input_len]
            output_seq = series[i+input_len:i+input_len+output_len]
            self.samples.append((input_seq, output_seq))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, output_seq = self.samples[idx]
        return torch.FloatTensor(input_seq), torch.FloatTensor(output_seq)


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """
    Train the transformer model.
    
    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    
    model = model.to(device)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
    
    return train_losses, val_losses


def extract_attention_weights(model, inputs, device='cpu'):
    """
    Extract attention weights from the model for given inputs.
    
    Args:
        model: Trained transformer model
        inputs: (batch_size, input_len) tensor
    
    Returns:
        attention_weights: List of (batch_size, nhead, seq_len, seq_len) tensors
                          One per layer
    """
    model.eval()
    model = model.to(device)
    inputs = inputs.to(device)
    
    # Hook to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # output is a tuple: (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            attn_weights = output[1]  # (batch_size, nhead, seq_len, seq_len)
            attention_weights.append(attn_weights)
    
    # Register hooks on self-attention layers
    hooks = []
    for layer in model.transformer_encoder.layers:
        hook = layer.self_attn.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(inputs, return_attention=False)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_weights


def get_attention_weights_direct(model, inputs, device='cpu'):
    """
    Directly extract attention weights by modifying forward pass.
    This is a more reliable method.
    """
    model.eval()
    model = model.to(device)
    inputs = inputs.to(device)
    
    batch_size = inputs.size(0)
    input_len = inputs.size(1)
    
    # Project input
    x = inputs.unsqueeze(-1)
    x = model.input_projection(x)
    x = x.transpose(0, 1)  # (input_len, batch_size, d_model)
    x = model.pos_encoder(x)
    
    # Extract attention from each layer
    attention_weights_all = []
    
    for layer in model.transformer_encoder.layers:
        residual = x
        x_norm = layer.norm1(x)
        
        # x_norm is (seq_len, batch, d_model) - this is correct for batch_first=False
        # Use the self_attn module directly without transposing
        # MultiheadAttention with batch_first=False expects (seq_len, batch, d_model)
        attn_output, attn_weights = layer.self_attn(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=False
        )
        # attn_weights: (batch_size, nhead, seq_len, seq_len) when batch_first=False
        # This is the correct shape we want!
        attention_weights_all.append(attn_weights)
        
        # Continue forward pass (attn_output is already (seq_len, batch, d_model))
        x = residual + layer.dropout1(attn_output)
        residual = x
        x = layer.norm2(x)
        x2 = layer.linear2(layer.dropout(F.relu(layer.linear1(x))))
        x = residual + layer.dropout2(x2)
    
    return attention_weights_all

