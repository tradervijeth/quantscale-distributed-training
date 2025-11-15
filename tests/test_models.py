"""
QuantScale: Distributed Training Framework for Financial Time Series

Unit tests for neural network models.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import pytest
import torch

from quantscale.models.transformer import TemporalFusionTransformer
from quantscale.models.lstm import BidirectionalLSTM, StackedLSTM


class TestTemporalFusionTransformer:
    """Test cases for TemporalFusionTransformer."""

    def test_model_creation(self):
        """Test model instantiation."""
        model = TemporalFusionTransformer(
            input_dim=10,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )

        assert model is not None
        assert model.input_dim == 10
        assert model.hidden_dim == 64

    def test_forward_pass(self):
        """Test forward pass."""
        model = TemporalFusionTransformer(
            input_dim=10,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )

        # Create dummy input (batch_size=4, seq_len=20, input_dim=10)
        x = torch.randn(4, 20, 10)

        # Forward pass
        output = model(x)

        # Check output shape (batch_size=4, output_dim=1)
        assert output.shape == (4, 1)

    def test_different_output_dim(self):
        """Test model with different output dimension."""
        model = TemporalFusionTransformer(
            input_dim=10,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            output_dim=3,
        )

        x = torch.randn(4, 20, 10)
        output = model(x)

        assert output.shape == (4, 3)

    def test_attention_weights(self):
        """Test attention weight extraction."""
        model = TemporalFusionTransformer(
            input_dim=10,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )

        x = torch.randn(4, 20, 10)

        # Get attention weights
        attention_weights = model.get_attention_weights(x)

        assert len(attention_weights) == 2  # num_layers
        assert attention_weights[0].shape[0] == 4  # batch_size

    def test_invalid_hidden_dim(self):
        """Test error handling for invalid hidden dimension."""
        with pytest.raises(ValueError):
            # hidden_dim not divisible by num_heads
            model = TemporalFusionTransformer(
                input_dim=10,
                hidden_dim=65,
                num_heads=4,
            )


class TestBidirectionalLSTM:
    """Test cases for BidirectionalLSTM."""

    def test_model_creation(self):
        """Test model instantiation."""
        model = BidirectionalLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
        )

        assert model is not None
        assert model.input_dim == 10
        assert model.hidden_dim == 64

    def test_forward_pass(self):
        """Test forward pass."""
        model = BidirectionalLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
        )

        x = torch.randn(4, 20, 10)
        output = model(x)

        assert output.shape == (4, 1)

    def test_with_attention(self):
        """Test model with attention mechanism."""
        model = BidirectionalLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            use_attention=True,
        )

        x = torch.randn(4, 20, 10)
        output = model(x)

        assert output.shape == (4, 1)

        # Check attention weights are available
        attention_weights = model.get_attention_weights()
        assert attention_weights is not None
        assert attention_weights.shape == (4, 20)  # (batch_size, seq_len)

    def test_without_attention(self):
        """Test model without attention mechanism."""
        model = BidirectionalLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            use_attention=False,
        )

        x = torch.randn(4, 20, 10)
        output = model(x)

        assert output.shape == (4, 1)

        # Attention weights should be None
        attention_weights = model.get_attention_weights()
        assert attention_weights is None

    def test_unidirectional(self):
        """Test unidirectional LSTM."""
        model = BidirectionalLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            bidirectional=False,
        )

        x = torch.randn(4, 20, 10)
        output = model(x)

        assert output.shape == (4, 1)

    def test_different_output_dim(self):
        """Test model with different output dimension."""
        model = BidirectionalLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            output_dim=5,
        )

        x = torch.randn(4, 20, 10)
        output = model(x)

        assert output.shape == (4, 5)


class TestStackedLSTM:
    """Test cases for StackedLSTM."""

    def test_model_creation(self):
        """Test model instantiation."""
        model = StackedLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=3,
        )

        assert model is not None
        assert model.input_dim == 10
        assert model.hidden_dim == 64
        assert model.num_layers == 3

    def test_forward_pass(self):
        """Test forward pass."""
        model = StackedLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=3,
        )

        x = torch.randn(4, 20, 10)
        output = model(x)

        assert output.shape == (4, 1)

    def test_with_residual(self):
        """Test model with residual connections."""
        model = StackedLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=3,
            use_residual=True,
        )

        x = torch.randn(4, 20, 10)
        output = model(x)

        assert output.shape == (4, 1)

    def test_without_residual(self):
        """Test model without residual connections."""
        model = StackedLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=3,
            use_residual=False,
        )

        x = torch.randn(4, 20, 10)
        output = model(x)

        assert output.shape == (4, 1)


class TestModelGradients:
    """Test gradient flow in models."""

    def test_transformer_gradients(self):
        """Test gradients flow through transformer."""
        model = TemporalFusionTransformer(
            input_dim=10,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )

        x = torch.randn(4, 20, 10)
        target = torch.randn(4, 1)

        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_lstm_gradients(self):
        """Test gradients flow through LSTM."""
        model = BidirectionalLSTM(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
        )

        x = torch.randn(4, 20, 10)
        target = torch.randn(4, 1)

        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
