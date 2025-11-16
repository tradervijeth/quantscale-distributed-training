"""
QuantScale: Distributed Training Framework for Financial Time Series

Data processing and feature engineering for financial time series.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FinancialDataProcessor:
    """
    Feature engineering and data preprocessing for financial time series.

    Provides methods for:
    - Computing returns, volatility, and technical indicators
    - Data augmentation techniques
    - Train/validation/test splitting with temporal consistency

    Args:
        data: DataFrame with financial time series data
        price_col: Name of the price column (default: 'close')
        volume_col: Name of the volume column (default: 'volume')

    Example:
        >>> processor = FinancialDataProcessor(df, price_col='close')
        >>> processor.add_returns()
        >>> processor.add_volatility(window=20)
        >>> processor.add_technical_indicators()
        >>> train, val, test = processor.train_val_test_split()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "close",
        volume_col: str = "volume",
    ) -> None:
        """Initialize the FinancialDataProcessor."""
        self.data = data.copy()
        self.price_col = price_col
        self.volume_col = volume_col

        # Validate required columns
        if price_col not in self.data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")

        logger.info(f"Initialized processor with {len(self.data)} rows")

    def add_returns(
        self,
        periods: List[int] = [1, 5, 10, 20],
        log_returns: bool = True,
    ) -> "FinancialDataProcessor":
        """
        Add return features for multiple periods.

        Args:
            periods: List of periods for return calculation
            log_returns: Whether to use log returns (default: True)

        Returns:
            Self for method chaining
        """
        for period in periods:
            if log_returns:
                col_name = f"log_return_{period}"
                self.data[col_name] = np.log(
                    self.data[self.price_col] / self.data[self.price_col].shift(period)
                )
            else:
                col_name = f"return_{period}"
                self.data[col_name] = (
                    self.data[self.price_col].pct_change(periods=period)
                )

        logger.info(f"Added {'log ' if log_returns else ''}returns for periods: {periods}")
        return self

    def add_volatility(
        self,
        windows: List[int] = [5, 10, 20, 50],
        use_log_returns: bool = True,
    ) -> "FinancialDataProcessor":
        """
        Add rolling volatility features.

        Args:
            windows: List of window sizes for volatility calculation
            use_log_returns: Whether to compute volatility from log returns

        Returns:
            Self for method chaining
        """
        # Compute returns if not already present
        if use_log_returns and "log_return_1" not in self.data.columns:
            self.add_returns(periods=[1], log_returns=True)
            return_col = "log_return_1"
        elif not use_log_returns and "return_1" not in self.data.columns:
            self.add_returns(periods=[1], log_returns=False)
            return_col = "return_1"
        else:
            return_col = "log_return_1" if use_log_returns else "return_1"

        for window in windows:
            col_name = f"volatility_{window}"
            self.data[col_name] = (
                self.data[return_col].rolling(window=window).std()
            )

        logger.info(f"Added volatility for windows: {windows}")
        return self

    def add_moving_averages(
        self,
        windows: List[int] = [5, 10, 20, 50, 200],
    ) -> "FinancialDataProcessor":
        """
        Add simple and exponential moving averages.

        Args:
            windows: List of window sizes for moving averages

        Returns:
            Self for method chaining
        """
        for window in windows:
            # Simple moving average
            sma_col = f"sma_{window}"
            self.data[sma_col] = self.data[self.price_col].rolling(window=window).mean()

            # Exponential moving average
            ema_col = f"ema_{window}"
            self.data[ema_col] = self.data[self.price_col].ewm(span=window, adjust=False).mean()

        logger.info(f"Added moving averages for windows: {windows}")
        return self

    def add_momentum_indicators(self) -> "FinancialDataProcessor":
        """
        Add momentum-based technical indicators (RSI, MACD).

        Returns:
            Self for method chaining
        """
        # Relative Strength Index (RSI)
        delta = self.data[self.price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = self.data[self.price_col].ewm(span=12, adjust=False).mean()
        ema_26 = self.data[self.price_col].ewm(span=26, adjust=False).mean()
        self.data["macd"] = ema_12 - ema_26
        self.data["macd_signal"] = self.data["macd"].ewm(span=9, adjust=False).mean()
        self.data["macd_histogram"] = self.data["macd"] - self.data["macd_signal"]

        logger.info("Added momentum indicators (RSI, MACD)")
        return self

    def add_volume_indicators(self) -> "FinancialDataProcessor":
        """
        Add volume-based indicators.

        Returns:
            Self for method chaining
        """
        if self.volume_col not in self.data.columns:
            logger.warning(f"Volume column '{self.volume_col}' not found, skipping volume indicators")
            return self

        # Volume moving averages
        for window in [5, 10, 20]:
            col_name = f"volume_ma_{window}"
            self.data[col_name] = self.data[self.volume_col].rolling(window=window).mean()

        # Volume ratio
        self.data["volume_ratio"] = (
            self.data[self.volume_col] / self.data["volume_ma_20"]
        )

        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(self.data)):
            if self.data[self.price_col].iloc[i] > self.data[self.price_col].iloc[i - 1]:
                obv.append(obv[-1] + self.data[self.volume_col].iloc[i])
            elif self.data[self.price_col].iloc[i] < self.data[self.price_col].iloc[i - 1]:
                obv.append(obv[-1] - self.data[self.volume_col].iloc[i])
            else:
                obv.append(obv[-1])
        self.data["obv"] = obv

        logger.info("Added volume indicators")
        return self

    def add_technical_indicators(self) -> "FinancialDataProcessor":
        """
        Add a comprehensive set of technical indicators.

        Returns:
            Self for method chaining
        """
        self.add_returns()
        self.add_volatility()
        self.add_moving_averages()
        self.add_momentum_indicators()
        self.add_volume_indicators()

        logger.info("Added all technical indicators")
        return self

    def add_lagged_features(
        self,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 10],
    ) -> "FinancialDataProcessor":
        """
        Add lagged versions of specified columns.

        Args:
            columns: List of column names to create lags for
            lags: List of lag periods

        Returns:
            Self for method chaining
        """
        for col in columns:
            if col not in self.data.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue

            for lag in lags:
                lag_col = f"{col}_lag_{lag}"
                self.data[lag_col] = self.data[col].shift(lag)

        logger.info(f"Added lagged features for {len(columns)} columns with lags: {lags}")
        return self

    def add_rolling_statistics(
        self,
        columns: List[str],
        windows: List[int] = [5, 10, 20],
        statistics: List[str] = ["mean", "std", "min", "max"],
    ) -> "FinancialDataProcessor":
        """
        Add rolling statistics for specified columns.

        Args:
            columns: List of column names
            windows: List of window sizes
            statistics: List of statistics to compute

        Returns:
            Self for method chaining
        """
        for col in columns:
            if col not in self.data.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue

            for window in windows:
                for stat in statistics:
                    stat_col = f"{col}_rolling_{stat}_{window}"
                    if stat == "mean":
                        self.data[stat_col] = self.data[col].rolling(window).mean()
                    elif stat == "std":
                        self.data[stat_col] = self.data[col].rolling(window).std()
                    elif stat == "min":
                        self.data[stat_col] = self.data[col].rolling(window).min()
                    elif stat == "max":
                        self.data[stat_col] = self.data[col].rolling(window).max()

        logger.info(f"Added rolling statistics for {len(columns)} columns")
        return self

    def handle_missing_values(
        self,
        method: str = "drop",
        fill_value: Optional[float] = None,
    ) -> "FinancialDataProcessor":
        """
        Handle missing values in the dataset.

        Args:
            method: Method to handle missing values ('drop', 'ffill', 'bfill', 'fill')
            fill_value: Value to use when method='fill'

        Returns:
            Self for method chaining
        """
        missing_count = self.data.isnull().sum().sum()

        if missing_count == 0:
            logger.info("No missing values found")
            return self

        logger.info(f"Found {missing_count} missing values, using method: {method}")

        if method == "drop":
            self.data = self.data.dropna()
        elif method == "ffill":
            self.data = self.data.fillna(method="ffill")
        elif method == "bfill":
            self.data = self.data.fillna(method="bfill")
        elif method == "fill":
            if fill_value is None:
                raise ValueError("fill_value must be specified when method='fill'")
            self.data = self.data.fillna(fill_value)
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"After handling, {self.data.isnull().sum().sum()} missing values remain")
        return self

    def train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        For time series, maintains temporal order by default.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            shuffle: Whether to shuffle data (not recommended for time series)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        n = len(self.data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        if shuffle:
            # Shuffle while maintaining indices
            shuffled_data = self.data.sample(frac=1.0, random_state=42)
            train = shuffled_data.iloc[:train_size]
            val = shuffled_data.iloc[train_size:train_size + val_size]
            test = shuffled_data.iloc[train_size + val_size:]
        else:
            # Temporal split (recommended for time series)
            train = self.data.iloc[:train_size]
            val = self.data.iloc[train_size:train_size + val_size]
            test = self.data.iloc[train_size + val_size:]

        logger.info(
            f"Split data into train: {len(train)}, val: {len(val)}, test: {len(test)}"
        )

        return train, val, test

    def augment_data(
        self,
        noise_level: float = 0.01,
        num_augmentations: int = 1,
    ) -> "FinancialDataProcessor":
        """
        Augment data with random noise for regularization.

        Args:
            noise_level: Standard deviation of Gaussian noise
            num_augmentations: Number of augmented copies to create

        Returns:
            Self for method chaining
        """
        augmented_dfs = [self.data]

        for i in range(num_augmentations):
            augmented = self.data.copy()
            # Add Gaussian noise to numeric columns
            numeric_cols = augmented.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                noise = np.random.normal(0, noise_level * augmented[col].std(), len(augmented))
                augmented[col] = augmented[col] + noise

            augmented_dfs.append(augmented)

        self.data = pd.concat(augmented_dfs, ignore_index=True)
        logger.info(f"Augmented data with {num_augmentations} copies, new size: {len(self.data)}")

        return self

    def get_data(self) -> pd.DataFrame:
        """
        Get the processed DataFrame.

        Returns:
            Processed DataFrame
        """
        return self.data

    def save(self, path: str, format: str = "parquet") -> None:
        """
        Save processed data to file.

        Args:
            path: Output file path
            format: File format ('csv', 'parquet', 'hdf5')
        """
        if format == "csv":
            self.data.to_csv(path, index=False)
        elif format == "parquet":
            self.data.to_parquet(path, index=False)
        elif format == "hdf5":
            self.data.to_hdf(path, key="data", mode="w")
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved processed data to {path}")
