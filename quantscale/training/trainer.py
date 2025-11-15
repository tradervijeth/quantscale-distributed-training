"""
QuantScale: Distributed Training Framework for Financial Time Series

Training loop implementation with checkpointing, early stopping, and metrics tracking.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from quantscale.utils.metrics import (
    sharpe_ratio,
    max_drawdown,
    directional_accuracy,
    calculate_all_metrics,
)

logger = logging.getLogger(__name__)


class QuantScaleTrainer:
    """
    Production-grade trainer for financial time series models.

    Features:
    - Training loop with validation
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - MLflow experiment tracking
    - Financial metrics (Sharpe, directional accuracy, max drawdown)

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: PyTorch optimizer (default: Adam)
        criterion: Loss function (default: MSELoss)
        device: Device to train on (default: auto-detect)
        scheduler: Learning rate scheduler (optional)
        gradient_clip: Max gradient norm for clipping (default: 1.0)
        checkpoint_dir: Directory to save checkpoints (default: "checkpoints")
        early_stopping_patience: Patience for early stopping (default: 10)
        use_mlflow: Whether to use MLflow tracking (default: False)

    Example:
        >>> model = TemporalFusionTransformer(input_dim=50, hidden_dim=256)
        >>> trainer = QuantScaleTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader
        ... )
        >>> trainer.fit(epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        scheduler: Optional[_LRScheduler] = None,
        gradient_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        early_stopping_patience: int = 10,
        use_mlflow: bool = False,
    ) -> None:
        """Initialize the trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer

        # Loss function
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

        self.scheduler = scheduler
        self.gradient_clip = gradient_clip

        # Checkpointing and early stopping
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # MLflow tracking
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        if use_mlflow and not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Install with: pip install mlflow")

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        logger.info(f"Initialized trainer with {sum(p.numel() for p in model.parameters())} parameters")

    def fit(
        self,
        epochs: int,
        log_interval: int = 10,
        save_best_only: bool = True,
    ) -> Dict[str, list]:
        """
        Train the model.

        Args:
            epochs: Number of epochs to train
            log_interval: Log metrics every N batches
            save_best_only: Only save checkpoints when validation loss improves

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")

        if self.use_mlflow:
            mlflow.log_params({
                "model": self.model.__class__.__name__,
                "optimizer": self.optimizer.__class__.__name__,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "batch_size": self.train_loader.batch_size,
                "epochs": epochs,
            })

        for epoch in range(1, epochs + 1):
            # Training phase
            train_loss = self._train_epoch(epoch, log_interval)
            self.history["train_loss"].append(train_loss)

            # Validation phase
            if self.val_loader is not None:
                val_loss, val_metrics = self._validate_epoch(epoch)
                self.history["val_loss"].append(val_loss)

                # Log to MLflow
                if self.use_mlflow:
                    mlflow.log_metrics(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            **{f"val_{k}": v for k, v in val_metrics.items()},
                        },
                        step=epoch,
                    )

                # Checkpointing and early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    if not save_best_only or (save_best_only and epoch > 1):
                        self._save_checkpoint(epoch, val_loss, is_best=True)
                        logger.info(f"New best model saved with val_loss: {val_loss:.6f}")
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch} epochs "
                        f"({self.early_stopping_patience} epochs without improvement)"
                    )
                    break

                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"train_loss: {train_loss:.6f}, "
                    f"val_loss: {val_loss:.6f}, "
                    f"sharpe: {val_metrics.get('sharpe_ratio', 0):.3f}, "
                    f"dir_acc: {val_metrics.get('directional_accuracy', 0):.3f}"
                )
            else:
                logger.info(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f}")

                if self.use_mlflow:
                    mlflow.log_metric("train_loss", train_loss, step=epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if self.val_loader else train_loss)
                else:
                    self.scheduler.step()

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

        logger.info("Training completed")
        return self.history

    def _train_epoch(self, epoch: int, log_interval: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
            log_interval: Logging interval

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
        )

        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            # Update progress bar
            if (batch_idx + 1) % log_interval == 0:
                progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def _validate_epoch(self, epoch: int) -> tuple:
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch} [Val]",
                leave=False,
            )

            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()

                # Store predictions and targets
                all_predictions.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        avg_loss = total_loss / len(self.val_loader)

        # Calculate financial metrics
        import numpy as np
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        metrics = {
            "directional_accuracy": directional_accuracy(predictions, targets),
            "sharpe_ratio": sharpe_ratio(targets) if len(targets) > 1 else 0.0,
            "max_drawdown": max_drawdown(targets) if len(targets) > 1 else 0.0,
        }

        return avg_loss, metrics

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                all_predictions.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        avg_loss = total_loss / len(test_loader)

        # Calculate comprehensive metrics
        import numpy as np
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        metrics = calculate_all_metrics(predictions, targets)
        metrics["test_loss"] = avg_loss

        logger.info(f"Test metrics: {metrics}")

        return metrics
