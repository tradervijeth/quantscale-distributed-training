"""
QuantScale: Distributed Training Framework for Financial Time Series

Bidirectional LSTM with attention for financial time series prediction.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """
    Attention mechanism for LSTM outputs.

    Computes weighted sum of LSTM hidden states based on learned attention weights.

    Args:
        hidden_dim: Dimension of LSTM hidden states
    """

    def __init__(self, hidden_dim: int) -> None:
        """Initialize attention layer."""
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vector.

        Args:
            lstm_output: LSTM output of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Calculate attention scores
        attention_scores = self.attention_weights(lstm_output)  # (batch, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, seq_len)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)

        # Compute weighted sum of LSTM outputs
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), lstm_output
        ).squeeze(1)  # (batch, hidden_dim)

        return context_vector, attention_weights


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM with attention mechanism for time series prediction.

    Uses bidirectional LSTM layers to capture both forward and backward temporal
    dependencies, with an attention mechanism to focus on important time steps.

    Args:
        input_dim: Number of input features
        hidden_dim: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        output_dim: Number of output features (default: 1)
        dropout: Dropout probability between LSTM layers
        use_attention: Whether to use attention mechanism (default: True)
        bidirectional: Whether to use bidirectional LSTM (default: True)

    Example:
        >>> model = BidirectionalLSTM(
        ...     input_dim=50,
        ...     hidden_dim=128,
        ...     num_layers=2,
        ...     use_attention=True
        ... )
        >>> x = torch.randn(32, 100, 50)  # (batch, seq_len, features)
        >>> output = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        use_attention: bool = True,
        bidirectional: bool = True,
    ) -> None:
        """Initialize Bidirectional LSTM."""
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_attention = use_attention
        self.bidirectional = bidirectional

        # Determine LSTM output dimension
        self.lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention mechanism (optional)
        if use_attention:
            self.attention = AttentionLayer(self.lstm_output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

        logger.info(
            f"Initialized BidirectionalLSTM with "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, bidirectional={bidirectional}, "
            f"use_attention={use_attention}"
        )

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

        for module in self.fc_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Optional initial hidden state and cell state tuple

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)

        # LSTM forward pass
        lstm_output, hidden = self.lstm(x, hidden)
        # lstm_output shape: (batch_size, seq_len, lstm_output_dim)

        # Apply attention or use last timestep
        if self.use_attention:
            context_vector, attention_weights = self.attention(lstm_output)
            # Store attention weights for visualization
            self.last_attention_weights = attention_weights
        else:
            # Use last timestep output
            context_vector = lstm_output[:, -1, :]

        # Apply dropout
        context_vector = self.dropout(context_vector)

        # Output layers
        output = self.fc_layers(context_vector)

        return output

    def _init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (hidden_state, cell_state)
        """
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
        )
        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
        )
        return (h0, c0)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from the last forward pass.

        Returns:
            Attention weights tensor or None if attention is not used

        Example:
            >>> output = model(x)
            >>> attention_weights = model.get_attention_weights()
        """
        if self.use_attention and hasattr(self, "last_attention_weights"):
            return self.last_attention_weights
        return None


class StackedLSTM(nn.Module):
    """
    Stacked LSTM with residual connections.

    Args:
        input_dim: Number of input features
        hidden_dim: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        output_dim: Number of output features
        dropout: Dropout probability
        use_residual: Whether to use residual connections
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.2,
        use_residual: bool = True,
    ) -> None:
        """Initialize Stacked LSTM."""
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual

        # First LSTM layer (different input dimension)
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(
            nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0,
            )
        )

        # Additional LSTM layers
        for _ in range(num_layers - 1):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0,
                )
            )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        logger.info(
            f"Initialized StackedLSTM with "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, use_residual={use_residual}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # First layer
        output, _ = self.lstm_layers[0](x)
        output = self.dropout(output)

        # Subsequent layers with optional residual connections
        for i in range(1, self.num_layers):
            residual = output
            output, _ = self.lstm_layers[i](output)
            output = self.dropout(output)

            # Add residual connection
            if self.use_residual:
                output = output + residual

        # Use last timestep
        output = output[:, -1, :]

        # Output layer
        output = self.fc(output)

        return output
