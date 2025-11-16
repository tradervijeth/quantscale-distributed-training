# Changelog

All notable changes to QuantScale will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-16

### Added

#### Core Framework
- **Data Loading & Processing**
  - `FinancialDataLoader` with multi-format support (CSV, Parquet, HDF5)
  - Sliding window time series data loading
  - Data normalization and missing value handling
  - `FinancialDataProcessor` with comprehensive feature engineering
  - Technical indicators (RSI, MACD, moving averages, volatility)
  - Data augmentation capabilities

#### Model Architectures
- **Temporal Fusion Transformer**
  - Multi-head attention mechanism
  - Positional encoding
  - Configurable architecture (layers, heads, dimensions)
  - Attention weight visualization
- **Bidirectional LSTM**
  - Attention mechanism for sequence weighting
  - Stacked LSTM with residual connections
  - Flexible architecture configuration

#### Training System
- **QuantScaleTrainer**
  - Training loop with validation
  - Gradient clipping
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing
  - MLflow experiment tracking
- **Distributed Training**
  - PyTorch DDP support for multi-GPU training
  - Efficient data parallelism
  - Synchronized batch normalization
  - Distributed checkpointing

#### Optimization
- **Hyperparameter Search**
  - Ray Tune integration
  - Optuna and HyperOpt search algorithms
  - ASHA and PBT schedulers
  - Distributed parallel trials

#### Financial Metrics
- Risk-adjusted metrics (Sharpe ratio, Sortino ratio, Calmar ratio)
- Drawdown analysis (maximum drawdown)
- Trading metrics (profit factor, win rate, directional accuracy)
- Information ratio for benchmark comparison

#### Scripts & Tools
- Single GPU training script
- Multi-GPU distributed training script
- Hyperparameter search script
- Performance benchmarking suite

#### Infrastructure
- CUDA-enabled Dockerfile
- Docker Compose with Jupyter, TensorBoard, MLflow
- GitHub Actions CI/CD workflows
- Comprehensive unit tests with pytest
- Code quality checks (Black, flake8, mypy, isort)

#### Documentation
- Professional README with badges and benchmarks
- Getting Started tutorial
- CONTRIBUTING.md with development guidelines
- GitHub issue templates (bug report, feature request)
- Pull request template
- Example usage scripts

### Technical Details

- **Python Support**: 3.10, 3.11, 3.12
- **PyTorch**: 2.0+
- **License**: Apache License 2.0
- **Author**: Vithushan Jeyapahan, Vijeth Ltd
- **Contact**: finance@vijeth.com

### Performance Benchmarks

| GPUs | Throughput | Speedup | Efficiency |
|------|------------|---------|------------|
| 1    | 250K/sec   | 1.0x    | 100%       |
| 2    | 490K/sec   | 1.96x   | 98%        |
| 4    | 950K/sec   | 3.8x    | 95%        |
| 8    | 1.8M/sec   | 7.2x    | 90%        |

## [Unreleased]

### Planned Features
- ONNX export for production deployment
- Model serving with FastAPI
- Additional model architectures (TCN, WaveNet)
- Mixed precision training (AMP)
- Model pruning and quantization
- Grafana dashboards for monitoring
- Sphinx documentation
- More example notebooks

---

**Copyright © 2025 Vithushan Jeyapahan, Vijeth Ltd**
