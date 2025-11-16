"""
QuantScale: Distributed Training Framework for Financial Time Series

Distributed training with PyTorch DDP (DistributedDataParallel).

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import logging
import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from quantscale.training.trainer import QuantScaleTrainer

logger = logging.getLogger(__name__)


def setup_distributed(backend: str = "nccl") -> None:
    """
    Initialize distributed training environment.

    Args:
        backend: Backend for distributed training ('nccl', 'gloo', 'mpi')

    Environment variables required:
        - RANK: Global rank of the process
        - LOCAL_RANK: Rank on the current node
        - WORLD_SIZE: Total number of processes
        - MASTER_ADDR: Address of rank 0 process
        - MASTER_PORT: Port of rank 0 process
    """
    if not dist.is_available():
        raise RuntimeError("Distributed training not available")

    # Get distributed training parameters from environment
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger.info(
        f"Initializing distributed training: rank={rank}, "
        f"local_rank={local_rank}, world_size={world_size}"
    )

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    # Set device for this process
    torch.cuda.set_device(local_rank)

    logger.info(f"Distributed training initialized successfully on rank {rank}")


def cleanup_distributed() -> None:
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")


def get_rank() -> int:
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def reduce_dict(input_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Reduce dictionary of metrics across all processes.

    Args:
        input_dict: Dictionary of metrics to reduce

    Returns:
        Reduced dictionary (averaged across processes)
    """
    if not dist.is_initialized():
        return input_dict

    world_size = get_world_size()
    if world_size == 1:
        return input_dict

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.tensor([input_dict[k] for k in keys], device="cuda")

        # All-reduce to sum across processes
        dist.all_reduce(values, op=dist.ReduceOp.SUM)

        # Average
        values /= world_size

        # Convert back to dict
        reduced_dict = {k: v.item() for k, v in zip(keys, values)}

    return reduced_dict


class DistributedTrainer(QuantScaleTrainer):
    """
    Distributed trainer for multi-GPU training with PyTorch DDP.

    Extends QuantScaleTrainer with distributed training capabilities.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        find_unused_parameters: Whether to find unused parameters in DDP (default: False)
        **kwargs: Additional arguments passed to QuantScaleTrainer

    Example:
        >>> # Run with torchrun:
        >>> # torchrun --nproc_per_node=4 train_script.py
        >>>
        >>> setup_distributed()
        >>> model = TemporalFusionTransformer(input_dim=50, hidden_dim=256)
        >>> trainer = DistributedTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader
        ... )
        >>> trainer.fit(epochs=100)
        >>> cleanup_distributed()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        find_unused_parameters: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize distributed trainer."""
        # Ensure distributed is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed training not initialized. Call setup_distributed() first."
            )

        # Set device to local rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")

        # Override device in kwargs
        kwargs["device"] = device

        # Initialize parent trainer
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            **kwargs,
        )

        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        )

        self.rank = get_rank()
        self.world_size = get_world_size()

        if is_main_process():
            logger.info(f"Initialized distributed trainer on {self.world_size} GPUs")

    def fit(
        self,
        epochs: int,
        log_interval: int = 10,
        save_best_only: bool = True,
    ) -> Dict[str, list]:
        """
        Train the model with distributed training.

        Args:
            epochs: Number of epochs to train
            log_interval: Log metrics every N batches
            save_best_only: Only save checkpoints when validation loss improves

        Returns:
            Training history dictionary
        """
        if is_main_process():
            logger.info(f"Starting distributed training for {epochs} epochs")

        # Ensure all processes are synchronized
        dist.barrier()

        # Call parent fit method
        history = super().fit(epochs, log_interval, save_best_only)

        # Synchronize at the end
        dist.barrier()

        return history

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """
        Save checkpoint (only on main process).

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        if is_main_process():
            # Unwrap DDP model before saving
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
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

        # Ensure all processes wait for checkpoint to be saved
        dist.barrier()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Map to correct device
        map_location = {"cuda:0": f"cuda:{self.rank}"}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Load into DDP module
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})

        if is_main_process():
            logger.info(f"Loaded checkpoint from {checkpoint_path}")

        # Synchronize after loading
        dist.barrier()


def create_distributed_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset] = None,
    test_dataset: Optional[torch.utils.data.Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create data loaders with distributed samplers.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size per GPU
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory

    Returns:
        Dictionary of data loaders
    """
    loaders = {}

    # Training loader with distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
    )

    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Validation loader
    if val_dataset is not None:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
        )

        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # Test loader
    if test_dataset is not None:
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
        )

        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if is_main_process():
        logger.info(
            f"Created distributed data loaders: "
            f"train={len(train_dataset)}, "
            f"val={len(val_dataset) if val_dataset else 0}, "
            f"test={len(test_dataset) if test_dataset else 0}"
        )

    return loaders


def synchronize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Synchronize metrics across all processes.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Averaged metrics across all processes
    """
    return reduce_dict(metrics)
