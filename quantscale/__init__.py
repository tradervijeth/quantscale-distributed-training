"""
QuantScale: Distributed Training Framework for Financial Time Series

A production-grade framework for distributed training of financial time series models
with support for multi-GPU training, hyperparameter optimization, and MLflow tracking.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

__version__ = "0.1.0"
__author__ = "Vithushan Jeyapahan"
__email__ = "finance@vijeth.com"
__license__ = "Apache License 2.0"
__copyright__ = "Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd"

# Import main components for easy access
from quantscale.training.trainer import QuantScaleTrainer

__all__ = [
    "QuantScaleTrainer",
    "__version__",
    "__author__",
    "__email__",
]
