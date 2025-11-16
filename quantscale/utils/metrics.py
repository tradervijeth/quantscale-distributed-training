"""
QuantScale: Distributed Training Framework for Financial Time Series

Financial metrics for evaluating trading strategies and model performance.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import logging
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def sharpe_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Sharpe ratio of returns.

    The Sharpe ratio measures risk-adjusted returns by comparing excess returns
    to the volatility of those returns.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year for annualization (252 for daily)

    Returns:
        Sharpe ratio (annualized)

    Example:
        >>> returns = np.array([0.01, -0.02, 0.015, 0.03, -0.01])
        >>> sharpe = sharpe_ratio(returns, risk_free_rate=0.02)
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns).flatten()

    if len(returns) == 0:
        logger.warning("Empty returns array, returning 0.0")
        return 0.0

    # Convert annual risk-free rate to period rate
    period_rf = risk_free_rate / periods_per_year

    # Calculate excess returns
    excess_returns = returns - period_rf

    # Calculate mean and std of excess returns
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess == 0:
        logger.warning("Zero standard deviation, returning 0.0")
        return 0.0

    # Annualize
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)

    return float(sharpe)


def sortino_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Sortino ratio of returns.

    Similar to Sharpe ratio but only penalizes downside volatility.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year for annualization

    Returns:
        Sortino ratio (annualized)
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns).flatten()

    if len(returns) == 0:
        logger.warning("Empty returns array, returning 0.0")
        return 0.0

    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf

    mean_excess = np.mean(excess_returns)

    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        logger.warning("No negative returns, returning inf")
        return float("inf")

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0:
        logger.warning("Zero downside deviation, returning 0.0")
        return 0.0

    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)

    return float(sortino)


def max_drawdown(
    returns: Union[np.ndarray, torch.Tensor],
    cumulative: bool = False,
) -> float:
    """
    Calculate the maximum drawdown from peak.

    Maximum drawdown is the largest peak-to-trough decline in cumulative returns.

    Args:
        returns: Array of period returns or cumulative returns
        cumulative: Whether input is already cumulative returns

    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.25 for 25% drawdown)

    Example:
        >>> returns = np.array([0.01, -0.02, 0.015, -0.03, 0.02])
        >>> mdd = max_drawdown(returns)
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns).flatten()

    if len(returns) == 0:
        logger.warning("Empty returns array, returning 0.0")
        return 0.0

    # Calculate cumulative returns if needed
    if not cumulative:
        cumulative_returns = np.cumprod(1 + returns)
    else:
        cumulative_returns = returns

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns)

    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max

    # Get maximum drawdown (most negative)
    max_dd = np.min(drawdown)

    return float(abs(max_dd))


def calmar_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Calmar ratio (return / max drawdown).

    Args:
        returns: Array of period returns
        periods_per_year: Number of periods per year for annualization

    Returns:
        Calmar ratio
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns).flatten()

    if len(returns) == 0:
        return 0.0

    annualized_return = np.mean(returns) * periods_per_year
    mdd = max_drawdown(returns)

    if mdd == 0:
        logger.warning("Zero max drawdown, returning inf")
        return float("inf")

    return float(annualized_return / mdd)


def directional_accuracy(
    predictions: Union[np.ndarray, torch.Tensor],
    actuals: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.0,
) -> float:
    """
    Calculate directional accuracy (correct prediction of up/down movement).

    Args:
        predictions: Predicted values or returns
        actuals: Actual values or returns
        threshold: Threshold for considering a move significant (default: 0.0)

    Returns:
        Directional accuracy as a decimal between 0 and 1

    Example:
        >>> predictions = np.array([0.01, -0.02, 0.015, 0.03])
        >>> actuals = np.array([0.015, -0.01, 0.02, -0.005])
        >>> acc = directional_accuracy(predictions, actuals)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(actuals, torch.Tensor):
        actuals = actuals.detach().cpu().numpy()

    predictions = np.asarray(predictions).flatten()
    actuals = np.asarray(actuals).flatten()

    if len(predictions) != len(actuals):
        raise ValueError(
            f"predictions and actuals must have same length, "
            f"got {len(predictions)} and {len(actuals)}"
        )

    if len(predictions) == 0:
        logger.warning("Empty arrays, returning 0.0")
        return 0.0

    # Get signs
    pred_direction = np.sign(predictions - threshold)
    actual_direction = np.sign(actuals - threshold)

    # Calculate accuracy
    correct = np.sum(pred_direction == actual_direction)
    accuracy = correct / len(predictions)

    return float(accuracy)


def information_ratio(
    returns: Union[np.ndarray, torch.Tensor],
    benchmark_returns: Union[np.ndarray, torch.Tensor],
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Information Ratio (active return / tracking error).

    Measures risk-adjusted excess returns relative to a benchmark.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year for annualization

    Returns:
        Information ratio (annualized)

    Example:
        >>> strategy_returns = np.array([0.01, 0.02, -0.01, 0.03])
        >>> benchmark_returns = np.array([0.005, 0.015, -0.005, 0.02])
        >>> ir = information_ratio(strategy_returns, benchmark_returns)
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    if isinstance(benchmark_returns, torch.Tensor):
        benchmark_returns = benchmark_returns.detach().cpu().numpy()

    returns = np.asarray(returns).flatten()
    benchmark_returns = np.asarray(benchmark_returns).flatten()

    if len(returns) != len(benchmark_returns):
        raise ValueError("returns and benchmark_returns must have same length")

    if len(returns) == 0:
        logger.warning("Empty arrays, returning 0.0")
        return 0.0

    # Calculate active returns
    active_returns = returns - benchmark_returns

    # Calculate tracking error (std of active returns)
    tracking_error = np.std(active_returns, ddof=1)

    if tracking_error == 0:
        logger.warning("Zero tracking error, returning 0.0")
        return 0.0

    # Calculate information ratio
    mean_active = np.mean(active_returns)
    ir = (mean_active / tracking_error) * np.sqrt(periods_per_year)

    return float(ir)


def profit_factor(
    returns: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Array of returns

    Returns:
        Profit factor
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns).flatten()

    if len(returns) == 0:
        return 0.0

    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")
        return 0.0

    return float(gross_profit / gross_loss)


def win_rate(
    returns: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Array of returns

    Returns:
        Win rate as a decimal between 0 and 1
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = np.asarray(returns).flatten()

    if len(returns) == 0:
        return 0.0

    wins = np.sum(returns > 0)
    win_rate_val = wins / len(returns)

    return float(win_rate_val)


def calculate_all_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    actuals: Union[np.ndarray, torch.Tensor],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Calculate all financial metrics at once.

    Args:
        predictions: Predicted returns
        actuals: Actual returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Dictionary with all metrics

    Example:
        >>> metrics = calculate_all_metrics(predictions, actuals)
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(actuals, torch.Tensor):
        actuals = actuals.detach().cpu().numpy()

    predictions = np.asarray(predictions).flatten()
    actuals = np.asarray(actuals).flatten()

    metrics = {
        # Prediction accuracy
        "directional_accuracy": directional_accuracy(predictions, actuals),
        # Risk-adjusted metrics (on actual returns)
        "sharpe_ratio": sharpe_ratio(actuals, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(actuals, risk_free_rate, periods_per_year),
        "calmar_ratio": calmar_ratio(actuals, periods_per_year),
        # Drawdown
        "max_drawdown": max_drawdown(actuals),
        # Trading metrics
        "profit_factor": profit_factor(actuals),
        "win_rate": win_rate(actuals),
        # Basic statistics
        "mean_return": float(np.mean(actuals)),
        "std_return": float(np.std(actuals, ddof=1)),
        "total_return": float(np.prod(1 + actuals) - 1),
    }

    return metrics
