#!/usr/bin/env python3
"""
QuantScale: Distributed Training Framework for Financial Time Series

Multi-GPU distributed training script using PyTorch DDP.

Run with: torchrun --nproc_per_node=4 scripts/train_distributed.py [args]

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import argparse
import logging
import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from quantscale.data.loaders import FinancialDataLoader
from quantscale.models.transformer import TemporalFusionTransformer
from quantscale.models.lstm import BidirectionalLSTM
from quantscale.training.distributed import (
    DistributedTrainer,
    setup_distributed,
    cleanup_distributed,
    create_distributed_dataloaders,
    is_main_process,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train financial time series model with distributed training (DDP)"
    )

    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data",
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
        help="Batch size per GPU",
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
        help="Weight decay",
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
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend",
    )

    return parser.parse_args()


def main():
    """Main distributed training function."""
    args = parse_args()

    # Initialize distributed training
    setup_distributed(backend=args.backend)

    if is_main_process():
        logger.info("Starting distributed training")
        logger.info(f"Arguments: {args}")
        logger.info(f"World size: {torch.distributed.get_world_size()}")

    try:
        # Create datasets
        if is_main_process():
            logger.info("Loading data...")

        loader_kwargs = {
            "sequence_length": args.sequence_length,
            "prediction_horizon": args.prediction_horizon,
            "target": args.target,
        }

        train_dataset = FinancialDataLoader(args.train_data, **loader_kwargs)

        val_dataset = None
        if args.val_data is not None:
            val_dataset = FinancialDataLoader(args.val_data, **loader_kwargs)

        # Create distributed data loaders
        loaders = create_distributed_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        train_loader = loaders["train"]
        val_loader = loaders.get("val")

        # Create model
        if is_main_process():
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
            verbose=True if is_main_process() else False,
        )

        # Create distributed trainer
        trainer = DistributedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip=args.gradient_clip,
            checkpoint_dir=args.checkpoint_dir,
            early_stopping_patience=args.early_stopping_patience,
        )

        # Train
        if is_main_process():
            logger.info("Starting training...")

        history = trainer.fit(epochs=args.epochs)

        if is_main_process():
            logger.info("Training completed!")

    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main()
