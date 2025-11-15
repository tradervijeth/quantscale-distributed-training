"""
QuantScale: Distributed Training Framework for Financial Time Series

Unit tests for data loading and processing.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from quantscale.data.loaders import FinancialDataLoader
from quantscale.data.processors import FinancialDataProcessor


class TestFinancialDataLoader:
    """Test cases for FinancialDataLoader."""

    @pytest.fixture
    def sample_data(self):
        """Create sample financial data for testing."""
        np.random.seed(42)
        n_samples = 1000

        data = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
            "volume": np.random.randint(1000, 10000, n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "target": np.random.randn(n_samples) * 0.01,
        })

        return data

    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            return f.name

    @pytest.fixture
    def temp_parquet_file(self, sample_data):
        """Create temporary Parquet file with sample data."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            sample_data.to_parquet(f.name, index=False)
            return f.name

    def test_loader_creation_csv(self, temp_csv_file):
        """Test loader creation with CSV file."""
        loader = FinancialDataLoader(
            temp_csv_file,
            sequence_length=10,
            prediction_horizon=1,
            target="target",
        )

        assert loader is not None
        assert len(loader) > 0
        assert loader.sequence_length == 10

    def test_loader_creation_parquet(self, temp_parquet_file):
        """Test loader creation with Parquet file."""
        loader = FinancialDataLoader(
            temp_parquet_file,
            sequence_length=10,
            prediction_horizon=1,
            target="target",
        )

        assert loader is not None
        assert len(loader) > 0

    def test_get_item(self, temp_csv_file):
        """Test getting items from loader."""
        loader = FinancialDataLoader(
            temp_csv_file,
            sequence_length=10,
            prediction_horizon=1,
            target="target",
        )

        x, y = loader[0]

        assert x.shape[0] == 10  # sequence_length
        assert x.shape[1] == 4  # number of features (excluding target)
        assert y.shape[0] == 1  # single target value

    def test_normalization(self, temp_csv_file):
        """Test data normalization."""
        loader = FinancialDataLoader(
            temp_csv_file,
            sequence_length=10,
            prediction_horizon=1,
            target="target",
            normalize=True,
        )

        assert loader.mean_ is not None
        assert loader.std_ is not None

        # Check normalization parameters exist for each feature
        assert len(loader.mean_) == 4
        assert len(loader.std_) == 4

    def test_inverse_transform(self, temp_csv_file):
        """Test inverse transformation."""
        loader = FinancialDataLoader(
            temp_csv_file,
            sequence_length=10,
            prediction_horizon=1,
            target="target",
            normalize=True,
        )

        # Get normalized data
        x, _ = loader[0]
        normalized_data = x.numpy()

        # Inverse transform
        original_scale = loader.inverse_transform(normalized_data)

        # Check shape is preserved
        assert original_scale.shape == normalized_data.shape

    def test_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            FinancialDataLoader(
                "nonexistent_file.csv",
                sequence_length=10,
                prediction_horizon=1,
            )

    def test_invalid_sequence_length(self, temp_csv_file):
        """Test error handling for invalid sequence length."""
        with pytest.raises(ValueError):
            FinancialDataLoader(
                temp_csv_file,
                sequence_length=0,
                prediction_horizon=1,
            )


class TestFinancialDataProcessor:
    """Test cases for FinancialDataProcessor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample financial data for testing."""
        np.random.seed(42)
        n_samples = 500

        data = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
            "volume": np.random.randint(1000, 10000, n_samples),
            "high": 100 + np.cumsum(np.random.randn(n_samples) * 0.5) + 1,
            "low": 100 + np.cumsum(np.random.randn(n_samples) * 0.5) - 1,
        })

        return data

    def test_processor_creation(self, sample_data):
        """Test processor creation."""
        processor = FinancialDataProcessor(sample_data, price_col="close")

        assert processor is not None
        assert len(processor.data) == len(sample_data)

    def test_add_returns(self, sample_data):
        """Test adding return features."""
        processor = FinancialDataProcessor(sample_data, price_col="close")
        processor.add_returns(periods=[1, 5], log_returns=True)

        assert "log_return_1" in processor.data.columns
        assert "log_return_5" in processor.data.columns

    def test_add_volatility(self, sample_data):
        """Test adding volatility features."""
        processor = FinancialDataProcessor(sample_data, price_col="close")
        processor.add_volatility(windows=[5, 10])

        assert "volatility_5" in processor.data.columns
        assert "volatility_10" in processor.data.columns

    def test_add_moving_averages(self, sample_data):
        """Test adding moving average features."""
        processor = FinancialDataProcessor(sample_data, price_col="close")
        processor.add_moving_averages(windows=[5, 10])

        assert "sma_5" in processor.data.columns
        assert "ema_5" in processor.data.columns
        assert "sma_10" in processor.data.columns
        assert "ema_10" in processor.data.columns

    def test_add_momentum_indicators(self, sample_data):
        """Test adding momentum indicators."""
        processor = FinancialDataProcessor(sample_data, price_col="close")
        processor.add_momentum_indicators()

        assert "rsi" in processor.data.columns
        assert "macd" in processor.data.columns
        assert "macd_signal" in processor.data.columns

    def test_train_val_test_split(self, sample_data):
        """Test data splitting."""
        processor = FinancialDataProcessor(sample_data, price_col="close")
        train, val, test = processor.train_val_test_split(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        # Check sizes
        total_size = len(train) + len(val) + len(test)
        assert total_size == len(sample_data)

        # Check ratios (approximately)
        assert abs(len(train) / total_size - 0.7) < 0.01
        assert abs(len(val) / total_size - 0.15) < 0.01
        assert abs(len(test) / total_size - 0.15) < 0.01

    def test_method_chaining(self, sample_data):
        """Test method chaining."""
        processor = FinancialDataProcessor(sample_data, price_col="close")

        # Chain multiple methods
        result = (
            processor.add_returns()
            .add_volatility()
            .add_moving_averages()
        )

        assert result is processor
        assert "log_return_1" in processor.data.columns
        assert "volatility_5" in processor.data.columns
        assert "sma_5" in processor.data.columns
