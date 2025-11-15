#!/usr/bin/env python3
"""
QuantScale: Distributed Training Framework for Financial Time Series

Distributed hyperparameter search using Ray Tune.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import argparse
import logging
from pathlib import Path

import ray
from ray import tune

from quantscale.data.loaders import FinancialDataLoader
from quantscale.models.transformer import TemporalFusionTransformer
from quantscale.models.lstm import BidirectionalLSTM
from quantscale.optimization.hyperparameter import HyperparameterSearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for financial time series models"
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
        required=True,
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

    # Search arguments
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum epochs per trial",
    )
    parser.add_argument(
        "--gpus-per-trial",
        type=float,
        default=1.0,
        help="GPUs to allocate per trial",
    )
    parser.add_argument(
        "--cpus-per-trial",
        type=int,
        default=2,
        help="CPUs to allocate per trial",
    )
    parser.add_argument(
        "--search-alg",
        type=str,
        choices=["random", "optuna", "hyperopt"],
        default="optuna",
        help="Search algorithm",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["asha", "pbt", "none"],
        default="asha",
        help="Scheduler algorithm",
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
        "--local-dir",
        type=str,
        default="./ray_results",
        help="Directory to save Ray Tune results",
    )

    return parser.parse_args()


def create_search_space(model_type: str):
    """
    Create hyperparameter search space.

    Args:
        model_type: Type of model ("transformer" or "lstm")

    Returns:
        Search space dictionary
    """
    if model_type == "transformer":
        return {
            "model_params": {
                "hidden_dim": tune.choice([128, 256, 512]),
                "num_heads": tune.choice([4, 8, 16]),
                "num_layers": tune.choice([2, 3, 4, 6]),
                "dropout": tune.uniform(0.1, 0.5),
            },
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 128]),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "gradient_clip": tune.choice([0.5, 1.0, 2.0]),
        }
    else:  # lstm
        return {
            "model_params": {
                "hidden_dim": tune.choice([64, 128, 256, 512]),
                "num_layers": tune.choice([1, 2, 3, 4]),
                "dropout": tune.uniform(0.1, 0.5),
                "use_attention": tune.choice([True, False]),
            },
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 128]),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "gradient_clip": tune.choice([0.5, 1.0, 2.0]),
        }


def main():
    """Main hyperparameter search function."""
    args = parse_args()

    logger.info("Starting hyperparameter search")
    logger.info(f"Arguments: {args}")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Load datasets
        logger.info("Loading data...")
        loader_kwargs = {
            "sequence_length": args.sequence_length,
            "prediction_horizon": args.prediction_horizon,
            "target": args.target,
        }

        train_dataset = FinancialDataLoader(args.train_data, **loader_kwargs)
        val_dataset = FinancialDataLoader(args.val_data, **loader_kwargs)

        logger.info(
            f"Loaded {len(train_dataset)} training samples, "
            f"{len(val_dataset)} validation samples"
        )

        # Select model class
        if args.model == "transformer":
            model_class = TemporalFusionTransformer
        else:
            model_class = BidirectionalLSTM

        # Create search space
        search_space = create_search_space(args.model)

        # Add input_dim to model params
        search_space["model_params"]["input_dim"] = args.input_dim

        logger.info(f"Search space: {search_space}")

        # Create hyperparameter search
        search = HyperparameterSearch(
            model_class=model_class,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=search_space,
            num_samples=args.num_samples,
            max_epochs=args.max_epochs,
            gpus_per_trial=args.gpus_per_trial,
            cpus_per_trial=args.cpus_per_trial,
            search_alg=args.search_alg,
            scheduler=args.scheduler if args.scheduler != "none" else None,
            local_dir=args.local_dir,
        )

        # Run search
        logger.info(f"Running {args.num_samples} trials...")
        best_config = search.run()

        # Save best config
        import json
        output_path = Path(args.local_dir) / "best_config.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(best_config, f, indent=2)

        logger.info(f"Best configuration saved to {output_path}")
        logger.info(f"Best hyperparameters: {best_config}")

    finally:
        # Shutdown Ray
        ray.shutdown()

    logger.info("Hyperparameter search completed!")


if __name__ == "__main__":
    main()
