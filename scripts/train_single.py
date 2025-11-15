#!/usr/bin/env python3
"""
QuantScale: Distributed Training Framework for Financial Time Series

Single GPU training script.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from quantscale.data.loaders import FinancialDataLoader, create_data_loaders
from quantscale.models.transformer import TemporalFusionTransformer
from quantscale.models.lstm import BidirectionalLSTM
from quantscale.training.trainer import QuantScaleTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train financial time series model on single GPU"
    )

    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data (CSV/Parquet/HDF5)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=["transformer", "lstm"],
        default="transformer",
        help="Model architecture",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=50,
        help="Number of input features",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (transformer only)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping threshold",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )

    # Data loader arguments
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Sequence length for time series",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=1,
        help="Prediction horizon",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="Target column name",
    )

    # Other arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="Use MLflow for experiment tracking",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    logger.info("Starting single GPU training")
    logger.info(f"Arguments: {args}")

    # Create data loaders
    logger.info("Loading data...")
    loader_kwargs = {
        "sequence_length": args.sequence_length,
        "prediction_horizon": args.prediction_horizon,
        "target": args.target,
    }

    loaders = create_data_loaders(
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **loader_kwargs,
    )

    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    test_loader = loaders.get("test")

    # Create model
    logger.info(f"Creating {args.model} model...")
    if args.model == "transformer":
        model = TemporalFusionTransformer(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:  # lstm
        model = BidirectionalLSTM(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    # Create trainer
    trainer = QuantScaleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip=args.gradient_clip,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping_patience,
        use_mlflow=args.use_mlflow,
        device=args.device,
    )

    # Train
    logger.info("Starting training...")
    history = trainer.fit(epochs=args.epochs)

    # Evaluate on test set if available
    if test_loader is not None:
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        logger.info(f"Test metrics: {test_metrics}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
