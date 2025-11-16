#!/usr/bin/env python3
"""
QuantScale: Distributed Training Framework for Financial Time Series

Basic usage example demonstrating the framework.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from quantscale.data.loaders import FinancialDataLoader, create_data_loaders
from quantscale.data.processors import FinancialDataProcessor
from quantscale.models.transformer import TemporalFusionTransformer
from quantscale.models.lstm import BidirectionalLSTM
from quantscale.training.trainer import QuantScaleTrainer
from quantscale.utils.metrics import calculate_all_metrics

import torch


def generate_sample_data(n_samples=10000, n_features=50):
    """Generate sample financial time series data."""
    print(
        f"Generating sample data with {n_samples} samples and {n_features} features..."
    )

    # Generate synthetic price data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

    # Create DataFrame
    data = {
        "close": prices,
        "volume": np.random.randint(1000, 10000, n_samples),
    }

    # Add random features
    for i in range(n_features - 2):
        data[f"feature_{i}"] = np.random.randn(n_samples)

    df = pd.DataFrame(data)

    # Create processor
    processor = FinancialDataProcessor(df, price_col="close")

    # Add technical indicators
    processor.add_returns(periods=[1, 5, 10])
    processor.add_volatility(windows=[5, 10, 20])
    processor.add_moving_averages(windows=[5, 10, 20])
    processor.add_momentum_indicators()

    # Handle missing values
    processor.handle_missing_values(method="drop")

    # Add target (next period return)
    processed_data = processor.get_data()
    processed_data["target"] = processed_data["log_return_1"].shift(-1)
    processed_data = processed_data.dropna()

    return processed_data


def main():
    """Main example function."""
    print("=" * 80)
    print("QuantScale: Basic Usage Example")
    print("=" * 80)

    # Generate sample data
    data = generate_sample_data(n_samples=5000, n_features=20)

    # Save to temporary files
    temp_dir = Path(tempfile.mkdtemp())
    train_path = temp_dir / "train.parquet"
    val_path = temp_dir / "val.parquet"
    test_path = temp_dir / "test.parquet"

    # Split data
    n = len(data)
    train_data = data.iloc[: int(n * 0.7)]
    val_data = data.iloc[int(n * 0.7) : int(n * 0.85)]
    test_data = data.iloc[int(n * 0.85) :]

    # Save data
    train_data.to_parquet(train_path, index=False)
    val_data.to_parquet(val_path, index=False)
    test_data.to_parquet(test_path, index=False)

    print(f"\nData split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # Create data loaders
    print("\nCreating data loaders...")
    loaders = create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        sequence_length=50,
        prediction_horizon=1,
        target="target",
        batch_size=32,
        num_workers=0,  # Use 0 for this example
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Get input dimension from first batch
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[2]
    print(f"Input dimension: {input_dim}")

    # Example 1: Train Temporal Fusion Transformer
    print("\n" + "=" * 80)
    print("Example 1: Training Temporal Fusion Transformer")
    print("=" * 80)

    model = TemporalFusionTransformer(
        input_dim=input_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    )

    trainer = QuantScaleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        gradient_clip=1.0,
        early_stopping_patience=5,
    )

    print("\nTraining Transformer model...")
    history = trainer.fit(epochs=10)

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # Example 2: Train Bidirectional LSTM
    print("\n" + "=" * 80)
    print("Example 2: Training Bidirectional LSTM")
    print("=" * 80)

    model = BidirectionalLSTM(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        use_attention=True,
    )

    trainer = QuantScaleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        gradient_clip=1.0,
        early_stopping_patience=5,
    )

    print("\nTraining LSTM model...")
    history = trainer.fit(epochs=10)

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
