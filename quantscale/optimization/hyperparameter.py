"""
QuantScale: Distributed Training Framework for Financial Time Series

Hyperparameter optimization using Ray Tune.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search.hyperopt import HyperOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from quantscale.training.trainer import QuantScaleTrainer

logger = logging.getLogger(__name__)


class HyperparameterSearch:
    """
    Hyperparameter optimization using Ray Tune.

    Supports various search algorithms (Random, Optuna, HyperOpt) and schedulers
    (ASHA, PBT) for efficient hyperparameter tuning.

    Args:
        model_class: Model class to instantiate
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Hyperparameter search space configuration
        metric: Metric to optimize (default: "val_loss")
        mode: Optimization mode ("min" or "max")
        num_samples: Number of trials to run
        max_epochs: Maximum epochs per trial
        gpus_per_trial: GPUs to allocate per trial
        cpus_per_trial: CPUs to allocate per trial
        search_alg: Search algorithm ("random", "optuna", "hyperopt")
        scheduler: Scheduler algorithm ("asha", "pbt", None)
        local_dir: Directory to save results

    Example:
        >>> from ray import tune
        >>> config = {
        ...     "hidden_dim": tune.choice([128, 256, 512]),
        ...     "num_layers": tune.choice([2, 3, 4]),
        ...     "dropout": tune.uniform(0.1, 0.5),
        ...     "learning_rate": tune.loguniform(1e-4, 1e-2),
        ... }
        >>> search = HyperparameterSearch(
        ...     model_class=TemporalFusionTransformer,
        ...     train_dataset=train_dataset,
        ...     val_dataset=val_dataset,
        ...     config=config,
        ...     num_samples=50
        ... )
        >>> best_config = search.run()
    """

    def __init__(
        self,
        model_class: type,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        config: Dict[str, Any],
        metric: str = "val_loss",
        mode: str = "min",
        num_samples: int = 50,
        max_epochs: int = 100,
        gpus_per_trial: float = 1.0,
        cpus_per_trial: int = 2,
        search_alg: str = "optuna",
        scheduler: str = "asha",
        local_dir: str = "./ray_results",
    ) -> None:
        """Initialize hyperparameter search."""
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray Tune is required for hyperparameter optimization. "
                "Install with: pip install ray[tune]"
            )

        self.model_class = model_class
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.metric = metric
        self.mode = mode
        self.num_samples = num_samples
        self.max_epochs = max_epochs
        self.gpus_per_trial = gpus_per_trial
        self.cpus_per_trial = cpus_per_trial
        self.search_alg = search_alg
        self.scheduler_name = scheduler
        self.local_dir = Path(local_dir)

        logger.info(
            f"Initialized hyperparameter search with {num_samples} trials, "
            f"search_alg={search_alg}, scheduler={scheduler}"
        )

    def _create_search_algorithm(self) -> Optional[Any]:
        """Create search algorithm based on configuration."""
        if self.search_alg == "random":
            return None  # Ray Tune default is random search

        elif self.search_alg == "optuna":
            return OptunaSearch(
                metric=self.metric,
                mode=self.mode,
            )

        elif self.search_alg == "hyperopt":
            return HyperOptSearch(
                metric=self.metric,
                mode=self.mode,
            )

        else:
            logger.warning(
                f"Unknown search algorithm: {self.search_alg}. Using random search."
            )
            return None

    def _create_scheduler(self) -> Optional[Any]:
        """Create scheduler based on configuration."""
        if self.scheduler_name == "asha":
            return ASHAScheduler(
                metric=self.metric,
                mode=self.mode,
                max_t=self.max_epochs,
                grace_period=10,
                reduction_factor=3,
            )

        elif self.scheduler_name == "pbt":
            return PopulationBasedTraining(
                metric=self.metric,
                mode=self.mode,
                perturbation_interval=5,
                hyperparam_mutations={
                    "learning_rate": tune.loguniform(1e-5, 1e-2),
                },
            )

        elif self.scheduler_name is None:
            return None

        else:
            logger.warning(
                f"Unknown scheduler: {self.scheduler_name}. Not using scheduler."
            )
            return None

    def _training_function(self, config: Dict[str, Any]) -> None:
        """
        Training function for a single trial.

        Args:
            config: Hyperparameter configuration for this trial
        """
        # Create model with hyperparameters from config
        model = self.model_class(**config.get("model_params", {}))

        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=True,
            num_workers=2,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            num_workers=2,
        )

        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 0.0),
        )

        # Create trainer
        trainer = QuantScaleTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            gradient_clip=config.get("gradient_clip", 1.0),
            early_stopping_patience=config.get("early_stopping_patience", 10),
        )

        # Training loop with Ray Tune reporting
        for epoch in range(1, self.max_epochs + 1):
            train_loss = trainer._train_epoch(epoch, log_interval=100)
            val_loss, val_metrics = trainer._validate_epoch(epoch)

            # Report metrics to Ray Tune
            tune.report(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                **val_metrics,
            )

    def run(self) -> Dict[str, Any]:
        """
        Run hyperparameter search.

        Returns:
            Best hyperparameter configuration
        """
        logger.info("Starting hyperparameter search")

        # Create search algorithm and scheduler
        search_alg = self._create_search_algorithm()
        scheduler = self._create_scheduler()

        # Configure reporter
        reporter = CLIReporter(
            metric_columns=[
                self.metric,
                "train_loss",
                "directional_accuracy",
                "sharpe_ratio",
            ],
            max_report_frequency=30,
        )

        # Run tuning
        analysis = tune.run(
            self._training_function,
            config=self.config,
            num_samples=self.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={
                "cpu": self.cpus_per_trial,
                "gpu": self.gpus_per_trial,
            },
            local_dir=str(self.local_dir),
            progress_reporter=reporter,
            verbose=1,
        )

        # Get best configuration
        best_config = analysis.get_best_config(
            metric=self.metric,
            mode=self.mode,
        )

        logger.info(f"Best hyperparameters: {best_config}")
        logger.info(f"Best {self.metric}: {analysis.best_result[self.metric]:.6f}")

        return best_config

    def get_best_checkpoint(self, analysis: Any) -> str:
        """
        Get path to best checkpoint.

        Args:
            analysis: Ray Tune analysis object

        Returns:
            Path to best checkpoint
        """
        best_trial = analysis.get_best_trial(
            metric=self.metric,
            mode=self.mode,
        )

        checkpoint_path = analysis.get_best_checkpoint(
            trial=best_trial,
            metric=self.metric,
            mode=self.mode,
        )

        return checkpoint_path


def run_hyperparameter_search(
    model_class: type,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    search_space: Dict[str, Any],
    num_samples: int = 50,
    max_epochs: int = 100,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience function to run hyperparameter search.

    Args:
        model_class: Model class to optimize
        train_dataset: Training dataset
        val_dataset: Validation dataset
        search_space: Hyperparameter search space
        num_samples: Number of trials
        max_epochs: Maximum epochs per trial
        **kwargs: Additional arguments passed to HyperparameterSearch

    Returns:
        Best hyperparameter configuration

    Example:
        >>> from ray import tune
        >>> search_space = {
        ...     "model_params": {
        ...         "hidden_dim": tune.choice([128, 256, 512]),
        ...         "num_layers": tune.choice([2, 3, 4]),
        ...     },
        ...     "learning_rate": tune.loguniform(1e-4, 1e-2),
        ...     "batch_size": tune.choice([32, 64, 128]),
        ... }
        >>> best_config = run_hyperparameter_search(
        ...     TemporalFusionTransformer,
        ...     train_dataset,
        ...     val_dataset,
        ...     search_space,
        ...     num_samples=50
        ... )
    """
    search = HyperparameterSearch(
        model_class=model_class,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=search_space,
        num_samples=num_samples,
        max_epochs=max_epochs,
        **kwargs,
    )

    best_config = search.run()
    return best_config
