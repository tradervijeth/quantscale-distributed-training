"""
QuantScale: Distributed Training Framework for Financial Time Series

Data loading utilities for financial time series datasets.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class FinancialDataLoader(Dataset):
    """
    PyTorch Dataset for financial time series data with sliding window support.

    Supports multiple data formats (CSV, Parquet, HDF5) and handles missing data,
    normalization, and efficient caching for large datasets.

    Args:
        data_path: Path to the data file (CSV, Parquet, or HDF5)
        sequence_length: Length of input sequences for time series
        prediction_horizon: Number of steps ahead to predict
        features: List of feature column names. If None, all columns except target
        target: Name of target column
        stride: Stride for sliding window (default: 1)
        normalize: Whether to normalize features (default: True)
        cache_data: Whether to cache data in memory (default: True)

    Example:
        >>> loader = FinancialDataLoader(
        ...     "data/market_data.parquet",
        ...     sequence_length=100,
        ...     prediction_horizon=1,
        ...     target="returns"
        ... )
        >>> train_loader = DataLoader(loader, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        sequence_length: int,
        prediction_horizon: int = 1,
        features: Optional[list] = None,
        target: str = "target",
        stride: int = 1,
        normalize: bool = True,
        cache_data: bool = True,
    ) -> None:
        """Initialize the FinancialDataLoader."""
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target = target
        self.stride = stride
        self.normalize = normalize
        self.cache_data = cache_data

        # Validate inputs
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if prediction_horizon <= 0:
            raise ValueError(f"prediction_horizon must be positive, got {prediction_horizon}")

        logger.info(f"Loading data from {self.data_path}")

        # Load data
        self.data = self._load_data()

        # Handle missing values
        self._handle_missing_values()

        # Select features
        if features is None:
            self.features = [col for col in self.data.columns if col != target]
        else:
            self.features = features

        # Validate target exists
        if target not in self.data.columns:
            raise ValueError(f"Target column '{target}' not found in data")

        # Extract features and target
        self.X = self.data[self.features].values
        self.y = self.data[target].values

        # Normalization statistics
        self.mean_ = None
        self.std_ = None

        if self.normalize:
            self._normalize_data()

        # Calculate valid indices for sliding window
        self._calculate_indices()

        logger.info(
            f"Loaded {len(self)} samples with {len(self.features)} features, "
            f"sequence_length={sequence_length}, prediction_horizon={prediction_horizon}"
        )

    def _load_data(self) -> pd.DataFrame:
        """Load data from file based on extension."""
        suffix = self.data_path.suffix.lower()

        try:
            if suffix == ".csv":
                df = pd.read_csv(self.data_path)
            elif suffix == ".parquet":
                df = pd.read_parquet(self.data_path)
            elif suffix in [".h5", ".hdf5"]:
                df = pd.read_hdf(self.data_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {suffix}. "
                    "Supported formats: .csv, .parquet, .h5, .hdf5"
                )
            return df
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            raise

    def _handle_missing_values(self) -> None:
        """Handle missing values in the dataset."""
        missing_count = self.data.isnull().sum().sum()

        if missing_count > 0:
            logger.warning(
                f"Found {missing_count} missing values. "
                "Using forward fill followed by backward fill."
            )
            self.data = self.data.fillna(method="ffill").fillna(method="bfill")

            # Check if any missing values remain
            remaining_missing = self.data.isnull().sum().sum()
            if remaining_missing > 0:
                logger.warning(
                    f"{remaining_missing} missing values remain after filling. "
                    "Dropping rows with missing values."
                )
                self.data = self.data.dropna()

    def _normalize_data(self) -> None:
        """Normalize features to zero mean and unit variance."""
        self.mean_ = np.mean(self.X, axis=0)
        self.std_ = np.std(self.X, axis=0)

        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0

        self.X = (self.X - self.mean_) / self.std_
        logger.info("Features normalized to zero mean and unit variance")

    def _calculate_indices(self) -> None:
        """Calculate valid indices for sliding window."""
        max_idx = len(self.X) - self.sequence_length - self.prediction_horizon + 1
        self.indices = list(range(0, max_idx, self.stride))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (features, target) as PyTorch tensors
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.prediction_horizon - 1

        # Get sequence and target
        x = self.X[start_idx:end_idx]
        y = self.y[target_idx]

        # Convert to tensors
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor([y])

        return x_tensor, y_tensor

    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        """
        Get normalization parameters for inverse transformation.

        Returns:
            Dictionary with 'mean' and 'std' arrays
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Data has not been normalized")
        return {"mean": self.mean_, "std": self.std_}

    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        Args:
            normalized_data: Normalized data array

        Returns:
            Data in original scale
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Data has not been normalized")
        return normalized_data * self.std_ + self.mean_


def create_data_loaders(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    **loader_kwargs: Any,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        test_path: Path to test data (optional)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        **loader_kwargs: Additional arguments passed to FinancialDataLoader

    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders

    Example:
        >>> loaders = create_data_loaders(
        ...     "data/train.parquet",
        ...     val_path="data/val.parquet",
        ...     sequence_length=100,
        ...     batch_size=64
        ... )
    """
    loaders = {}

    # Training loader
    train_dataset = FinancialDataLoader(train_path, **loader_kwargs)
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    logger.info(f"Created training loader with {len(train_dataset)} samples")

    # Validation loader
    if val_path is not None:
        val_dataset = FinancialDataLoader(val_path, **loader_kwargs)
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info(f"Created validation loader with {len(val_dataset)} samples")

    # Test loader
    if test_path is not None:
        test_dataset = FinancialDataLoader(test_path, **loader_kwargs)
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info(f"Created test loader with {len(test_dataset)} samples")

    return loaders
