"""
Transformer model for time series forecasting with attention visualization capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerForecaster(nn.Module):
    """
    Transformer model for time series forecasting.
    Given L past points, predicts H future points.
    """
    
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, 
                 dropout=0.1, input_len=100, output_len=20):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.input_len = input_len
        self.output_len = output_len
        
        # Input projection: 1D time series -> d_model
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: d_model -> 1D forecast
        self.output_projection = nn.Linear(d_model, 1)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch_size, input_len) - input time series
            return_attention: If True, return attention weights
        
        Returns:
            output: (batch_size, output_len) - predicted future values
            attention_weights: List of attention weights if return_attention=True
        """
        batch_size = x.size(0)
        
        # Reshape: (batch_size, input_len) -> (batch_size, input_len, 1)
        x = x.unsqueeze(-1)
        
        # Project to d_model: (batch_size, input_len, 1) -> (batch_size, input_len, d_model)
        x = self.input_projection(x)
        
        # Transpose for transformer: (input_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Store attention weights if requested
        attention_weights = []
        if return_attention:
            # We need to manually extract attention from each layer
            for layer in self.transformer_encoder.layers:
                # Create a wrapper to capture attention
                pass  # Will handle this differently
        
        # Apply transformer encoder
        # For attention extraction, we'll use a custom forward pass
        if return_attention:
            x, attn_weights = self._forward_with_attention(x)
        else:
            x = self.transformer_encoder(x)
            attn_weights = None
        
        # Take the last input_len positions and project to output
        # We'll use the last few positions to generate forecasts
        # For simplicity, we'll use a decoder-like approach
        # Take the last position's representation and expand it
        
        # Use the last position to initialize decoder input
        decoder_input = x[-1:, :, :]  # (1, batch_size, d_model)
        
        # Generate output_len predictions autoregressively or in parallel
        # For simplicity, we'll use a linear projection approach
        # More sophisticated: use a decoder transformer
        
        # Simple approach: project each position to future
        # We'll use the last output_len positions
        if x.size(0) >= self.output_len:
            future_repr = x[-self.output_len:, :, :]  # (output_len, batch_size, d_model)
        else:
            # Repeat the last representation
            last_repr = x[-1:, :, :]  # (1, batch_size, d_model)
            future_repr = last_repr.repeat(self.output_len, 1, 1)
        
        # Project to output: (output_len, batch_size, d_model) -> (output_len, batch_size, 1)
        output = self.output_projection(future_repr)
        
        # Reshape: (output_len, batch_size, 1) -> (batch_size, output_len)
        output = output.squeeze(-1).transpose(0, 1)
        
        if return_attention:
            return output, attn_weights
        return output
    
    def _forward_with_attention(self, x):
        """Forward pass that captures attention weights."""
        attention_weights_all = []
        
        for layer in self.transformer_encoder.layers:
            # Standard forward through the layer
            residual = x
            x_norm = layer.norm1(x)
            
            # Transpose for multihead attention: (seq_len, batch, d_model) -> (batch, seq_len, d_model)
            x_norm_t = x_norm.transpose(0, 1)
            
            # Self-attention
            x2_t, attn_weights = layer.self_attn(
                x_norm_t, x_norm_t, x_norm_t,
                need_weights=True,
                average_attn_weights=False
            )
            # attn_weights: (batch_size, nhead, seq_len, seq_len)
            attention_weights_all.append(attn_weights)
            
            # Transpose back: (batch, seq_len, d_model) -> (seq_len, batch, d_model)
            x2 = x2_t.transpose(0, 1)
            
            x = residual + layer.dropout1(x2)
            residual = x
            x = layer.norm2(x)
            x2 = layer.linear2(layer.dropout(F.relu(layer.linear1(x))))
            x = residual + layer.dropout2(x2)
        
        return x, attention_weights_all

