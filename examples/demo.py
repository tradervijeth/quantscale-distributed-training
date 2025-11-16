#!/usr/bin/env python3
"""
QuantScale: Complete End-to-End Demo

This script demonstrates the full QuantScale workflow:
1. Generate synthetic financial data
2. Feature engineering
3. Train models (Transformer & LSTM)
4. Evaluate performance
5. Visualize results

No external data required - runs completely standalone!

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import argparse
import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from quantscale.data.loaders import create_data_loaders
from quantscale.data.processors import FinancialDataProcessor
from quantscale.models.lstm import BidirectionalLSTM
from quantscale.models.transformer import TemporalFusionTransformer
from quantscale.training.trainer import QuantScaleTrainer
from quantscale.utils.metrics import calculate_all_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_header(text: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic financial time series data.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic financial data
    """
    print_header("STEP 1: Generating Synthetic Financial Data")

    np.random.seed(seed)

    # Generate price with drift and volatility
    drift = 0.0001
    volatility = 0.02
    returns = np.random.normal(drift, volatility, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate related features
    data = pd.DataFrame(
        {
            "close": prices,
            "high": prices * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
            "low": prices * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
            "volume": np.random.lognormal(12, 1, n_samples),
            "open": prices + np.random.randn(n_samples) * 0.5,
        }
    )

    logger.info(f"Generated {len(data)} samples of synthetic data")
    logger.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    logger.info(f"Final price: ${data['close'].iloc[-1]:.2f}")

    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to raw data.

    Args:
        data: Raw financial data

    Returns:
        Processed data with engineered features
    """
    print_header("STEP 2: Feature Engineering")

    processor = FinancialDataProcessor(data, price_col="close", volume_col="volume")

    # Add technical indicators
    logger.info("Adding returns...")
    processor.add_returns(periods=[1, 5, 10, 20])

    logger.info("Adding volatility...")
    processor.add_volatility(windows=[5, 10, 20])

    logger.info("Adding moving averages...")
    processor.add_moving_averages(windows=[5, 10, 20, 50])

    logger.info("Adding momentum indicators...")
    processor.add_momentum_indicators()

    logger.info("Adding volume indicators...")
    processor.add_volume_indicators()

    # Handle missing values
    processor.handle_missing_values(method="drop")

    # Create target
    processed_data = processor.get_data()
    processed_data["target"] = processed_data["log_return_1"].shift(-1)
    processed_data = processed_data.dropna()

    logger.info(f"Original features: {len(data.columns)}")
    logger.info(f"Engineered features: {len(processed_data.columns)}")
    logger.info(f"Final dataset: {len(processed_data)} samples")

    return processed_data


def create_data_splits(data: pd.DataFrame, temp_dir: Path) -> dict:
    """
    Split data and save to temporary files.

    Args:
        data: Processed data
        temp_dir: Temporary directory for files

    Returns:
        Dictionary with data loader paths
    """
    print_header("STEP 3: Creating Train/Val/Test Splits")

    # Split data
    n = len(data)
    train_data = data.iloc[: int(n * 0.7)]
    val_data = data.iloc[int(n * 0.7) : int(n * 0.85)]
    test_data = data.iloc[int(n * 0.85) :]

    # Save to files
    train_path = temp_dir / "train.parquet"
    val_path = temp_dir / "val.parquet"
    test_path = temp_dir / "test.parquet"

    train_data.to_parquet(train_path, index=False)
    val_data.to_parquet(val_path, index=False)
    test_data.to_parquet(test_path, index=False)

    logger.info(f"Train: {len(train_data)} samples ({len(train_data)/n*100:.1f}%)")
    logger.info(f"Val:   {len(val_data)} samples ({len(val_data)/n*100:.1f}%)")
    logger.info(f"Test:  {len(test_data)} samples ({len(test_data)/n*100:.1f}%)")

    return {"train": train_path, "val": val_path, "test": test_path}


def train_model(
    model_name: str,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    epochs: int = 20,
) -> QuantScaleTrainer:
    """
    Train a model.

    Args:
        model_name: Name of the model
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs

    Returns:
        Trained trainer object
    """
    print_header(f"STEP 4: Training {model_name}")

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = QuantScaleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        gradient_clip=1.0,
        early_stopping_patience=5,
        checkpoint_dir=f"checkpoints/{model_name.lower().replace(' ', '_')}",
    )

    logger.info(f"Training for {epochs} epochs...")
    history = trainer.fit(epochs=epochs)

    logger.info(f"Training completed!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")

    return trainer


def evaluate_and_visualize(
    trainer: QuantScaleTrainer, test_loader, model_name: str
) -> dict:
    """
    Evaluate model and create visualizations.

    Args:
        trainer: Trained trainer
        test_loader: Test data loader
        model_name: Name of the model

    Returns:
        Dictionary with test metrics
    """
    print_header(f"STEP 5: Evaluating {model_name}")

    # Evaluate
    test_metrics = trainer.evaluate(test_loader)

    logger.info(f"\n{model_name} Test Results:")
    logger.info("-" * 50)
    for metric, value in test_metrics.items():
        logger.info(f"  {metric:.<40} {value:.6f}")

    # Get predictions for visualization
    trainer.model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch, target in test_loader:
            batch = batch.to(trainer.device)
            preds = trainer.model(batch).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(target.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    return {
        "metrics": test_metrics,
        "predictions": all_preds,
        "targets": all_targets,
    }


def plot_results(results_transformer: dict, results_lstm: dict) -> None:
    """
    Create comparison plots for both models.

    Args:
        results_transformer: Results from Transformer
        results_lstm: Results from LSTM
    """
    print_header("STEP 6: Visualizing Results")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("QuantScale Model Comparison", fontsize=16, fontweight="bold")

    # Plot 1: Transformer predictions vs actuals
    ax = axes[0, 0]
    ax.scatter(
        results_transformer["targets"],
        results_transformer["predictions"],
        alpha=0.3,
        label="Transformer",
    )
    lim = [
        min(
            results_transformer["targets"].min(),
            results_transformer["predictions"].min(),
        ),
        max(
            results_transformer["targets"].max(),
            results_transformer["predictions"].max(),
        ),
    ]
    ax.plot(lim, lim, "r--", alpha=0.5)
    ax.set_xlabel("Actual Returns")
    ax.set_ylabel("Predicted Returns")
    ax.set_title("Transformer: Predictions vs Actuals")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: LSTM predictions vs actuals
    ax = axes[0, 1]
    ax.scatter(
        results_lstm["targets"], results_lstm["predictions"], alpha=0.3, label="LSTM"
    )
    lim = [
        min(results_lstm["targets"].min(), results_lstm["predictions"].min()),
        max(results_lstm["targets"].max(), results_lstm["predictions"].max()),
    ]
    ax.plot(lim, lim, "r--", alpha=0.5)
    ax.set_xlabel("Actual Returns")
    ax.set_ylabel("Predicted Returns")
    ax.set_title("LSTM: Predictions vs Actuals")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Error distributions
    ax = axes[0, 2]
    errors_transformer = (
        results_transformer["predictions"] - results_transformer["targets"]
    )
    errors_lstm = results_lstm["predictions"] - results_lstm["targets"]
    ax.hist(errors_transformer, bins=50, alpha=0.5, label="Transformer", density=True)
    ax.hist(errors_lstm, bins=50, alpha=0.5, label="LSTM", density=True)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Cumulative returns (Transformer)
    ax = axes[1, 0]
    cum_actual = np.cumprod(1 + results_transformer["targets"])
    cum_pred = np.cumprod(1 + results_transformer["predictions"])
    ax.plot(cum_actual, label="Actual", linewidth=2)
    ax.plot(cum_pred, label="Predicted", linewidth=2, alpha=0.7)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Transformer: Cumulative Returns")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 5: Cumulative returns (LSTM)
    ax = axes[1, 1]
    cum_actual = np.cumprod(1 + results_lstm["targets"])
    cum_pred = np.cumprod(1 + results_lstm["predictions"])
    ax.plot(cum_actual, label="Actual", linewidth=2)
    ax.plot(cum_pred, label="Predicted", linewidth=2, alpha=0.7)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("LSTM: Cumulative Returns")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 6: Metrics comparison
    ax = axes[1, 2]
    metrics_to_plot = [
        "directional_accuracy",
        "sharpe_ratio",
        "test_loss",
    ]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    transformer_values = [results_transformer["metrics"][m] for m in metrics_to_plot]
    lstm_values = [results_lstm["metrics"][m] for m in metrics_to_plot]

    ax.bar(x - width / 2, transformer_values, width, label="Transformer", alpha=0.8)
    ax.bar(x + width / 2, lstm_values, width, label="LSTM", alpha=0.8)
    ax.set_ylabel("Value")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("quantscale_demo_results.png", dpi=150, bbox_inches="tight")
    logger.info("Results saved to: quantscale_demo_results.png")
    plt.show()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="QuantScale End-to-End Demo")
    parser.add_argument(
        "--n-samples", type=int, default=5000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    print_header("QuantScale: Complete End-to-End Demo")
    logger.info("Author: Vithushan Jeyapahan <finance@vijeth.com>")
    logger.info("Company: Vijeth Ltd")
    logger.info("License: Apache 2.0\n")

    # Step 1: Generate data
    data = generate_synthetic_data(n_samples=args.n_samples)

    # Step 2: Engineer features
    processed_data = engineer_features(data)

    # Step 3: Create data splits
    temp_dir = Path(tempfile.mkdtemp())
    paths = create_data_splits(processed_data, temp_dir)

    # Create data loaders
    loaders = create_data_loaders(
        train_path=paths["train"],
        val_path=paths["val"],
        test_path=paths["test"],
        sequence_length=50,
        prediction_horizon=1,
        target="target",
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Get input dimension
    sample_batch, _ = next(iter(loaders["train"]))
    input_dim = sample_batch.shape[2]
    logger.info(f"\nInput dimension: {input_dim}")

    # Step 4A: Train Transformer
    transformer = TemporalFusionTransformer(
        input_dim=input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1
    )

    trainer_transformer = train_model(
        "Temporal Fusion Transformer",
        transformer,
        loaders["train"],
        loaders["val"],
        epochs=args.epochs,
    )

    # Step 4B: Train LSTM
    lstm = BidirectionalLSTM(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        use_attention=True,
    )

    trainer_lstm = train_model(
        "Bidirectional LSTM",
        lstm,
        loaders["train"],
        loaders["val"],
        epochs=args.epochs,
    )

    # Step 5: Evaluate both models
    results_transformer = evaluate_and_visualize(
        trainer_transformer, loaders["test"], "Temporal Fusion Transformer"
    )

    results_lstm = evaluate_and_visualize(
        trainer_lstm, loaders["test"], "Bidirectional LSTM"
    )

    # Step 6: Visualize results
    if not args.no_plot:
        plot_results(results_transformer, results_lstm)

    print_header("Demo Completed Successfully!")
    logger.info("Thank you for trying QuantScale!")
    logger.info(
        "For more information, visit: https://github.com/tradervijeth/quantscale-distributed-training"
    )


if __name__ == "__main__":
    main()
