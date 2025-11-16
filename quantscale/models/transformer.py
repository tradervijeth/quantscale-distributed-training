"""
QuantScale: Distributed Training Framework for Financial Time Series

Temporal Fusion Transformer for financial time series prediction.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.

    Adds position information to the input embeddings using sine and cosine functions.

    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        """Initialize positional encoding."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Handle both even and odd d_model dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # Add batch dimension

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        """Initialize multi-head attention."""
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.

        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # Final linear projection
        output = self.out_linear(context)

        return output, attention_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Args:
        d_model: Dimension of the model
        d_ff: Dimension of feed-forward layer
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1) -> None:
        """Initialize feed-forward network."""
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """Initialize transformer encoder layer."""
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for financial time series prediction.

    A transformer-based architecture designed for time series forecasting with
    multi-head attention, positional encoding, and residual connections.

    Args:
        input_dim: Number of input features
        hidden_dim: Dimension of hidden layers
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        output_dim: Number of output features (default: 1)
        dropout: Dropout probability
        max_seq_length: Maximum sequence length

    Example:
        >>> model = TemporalFusionTransformer(
        ...     input_dim=50,
        ...     hidden_dim=256,
        ...     num_heads=8,
        ...     num_layers=4
        ... )
        >>> x = torch.randn(32, 100, 50)  # (batch, seq_len, features)
        >>> output = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        output_dim: int = 1,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
    ) -> None:
        """Initialize Temporal Fusion Transformer."""
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input embedding
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_length, dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    num_heads=num_heads,
                    d_ff=hidden_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

        logger.info(
            f"Initialized TemporalFusionTransformer with "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"num_heads={num_heads}, num_layers={num_layers}"
        )

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer layers
        for layer in self.encoder_layers:
            x = layer(x, mask)

        # Use last timestep for prediction
        x = x[:, -1, :]

        # Output projection
        output = self.output_projection(x)

        return output

    def get_attention_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> list:
        """
        Get attention weights from all layers for visualization.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            List of attention weight tensors from each layer
        """
        attention_weights = []

        x = self.input_projection(x)
        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            attn_output, attn_weights = layer.self_attention(x, x, x, mask)
            attention_weights.append(attn_weights)
            x = x + layer.dropout1(attn_output)
            x = layer.norm1(x)
            ff_output = layer.feed_forward(x)
            x = x + layer.dropout2(ff_output)
            x = layer.norm2(x)

        return attention_weights
