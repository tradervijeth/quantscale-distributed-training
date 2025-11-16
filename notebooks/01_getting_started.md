# QuantScale Tutorial: Getting Started

**Author:** Vithushan Jeyapahan (finance@vijeth.com)
**Copyright:** © 2025 Vijeth Ltd
**License:** Apache 2.0

This tutorial provides a comprehensive introduction to the QuantScale distributed training framework for financial time series.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Evaluation](#evaluation)
5. [Distributed Training](#distributed-training)
6. [Hyperparameter Optimization](#hyperparameter-optimization)

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/tradervijeth/quantscale-distributed-training.git
cd quantscale-distributed-training

# Install in development mode
pip install -e .

# Install additional dependencies for notebooks
pip install jupyter matplotlib seaborn
```

## 2. Data Preparation

### Load and Preprocess Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantscale.data.processors import FinancialDataProcessor

# Generate sample data (replace with your own data)
np.random.seed(42)
n_samples = 10000

# Simulate stock prices
prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

data = pd.DataFrame({
    'close': prices,
    'high': prices + np.abs(np.random.randn(n_samples) * 0.5),
    'low': prices - np.abs(np.random.randn(n_samples) * 0.5),
    'volume': np.random.randint(100000, 1000000, n_samples),
})

# Display first few rows
print(data.head())
```

### Feature Engineering

```python
# Initialize processor
processor = FinancialDataProcessor(data, price_col='close', volume_col='volume')

# Add technical indicators
processor.add_returns(periods=[1, 5, 10, 20])
processor.add_volatility(windows=[5, 10, 20])
processor.add_moving_averages(windows=[5, 10, 20, 50])
processor.add_momentum_indicators()
processor.add_volume_indicators()

# Handle missing values
processor.handle_missing_values(method='drop')

# Get processed data
processed_data = processor.get_data()

print(f"Original features: {len(data.columns)}")
print(f"Processed features: {len(processed_data.columns)}")
print(f"\\nNew features: {list(processed_data.columns[len(data.columns):])}")
```

### Create Target Variable

```python
# Create target: next period return
processed_data['target'] = processed_data['log_return_1'].shift(-1)
processed_data = processed_data.dropna()

print(f"Final dataset shape: {processed_data.shape}")
```

### Split Data

```python
# Split into train/val/test
train, val, test = processor.train_val_test_split(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

print(f"Train: {len(train)} samples")
print(f"Val: {len(val)} samples")
print(f"Test: {len(test)} samples")

# Save to files
train.to_parquet('data/train.parquet')
val.to_parquet('data/val.parquet')
test.to_parquet('data/test.parquet')
```

## 3. Model Training

### Create Data Loaders

```python
from quantscale.data.loaders import create_data_loaders

loaders = create_data_loaders(
    train_path='data/train.parquet',
    val_path='data/val.parquet',
    test_path='data/test.parquet',
    sequence_length=50,
    prediction_horizon=1,
    target='target',
    batch_size=64,
    num_workers=4
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']

# Check a batch
sample_batch, sample_target = next(iter(train_loader))
print(f"Batch shape: {sample_batch.shape}")
print(f"Target shape: {sample_target.shape}")
```

### Train Temporal Fusion Transformer

```python
import torch
from quantscale.models.transformer import TemporalFusionTransformer
from quantscale.training.trainer import QuantScaleTrainer

# Get input dimension from data
input_dim = sample_batch.shape[2]

# Create model
model = TemporalFusionTransformer(
    input_dim=input_dim,
    hidden_dim=256,
    num_heads=8,
    num_layers=4,
    dropout=0.1
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create trainer
trainer = QuantScaleTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    gradient_clip=1.0,
    early_stopping_patience=10,
    checkpoint_dir='checkpoints/transformer'
)

# Train
history = trainer.fit(epochs=50)
```

### Visualize Training Progress

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['learning_rate'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 4. Evaluation

### Evaluate on Test Set

```python
# Evaluate
test_metrics = trainer.evaluate(test_loader)

print("\\nTest Set Results:")
print("=" * 50)
for metric, value in test_metrics.items():
    print(f"{metric:.<40} {value:.6f}")
```

### Analyze Predictions

```python
import numpy as np
from quantscale.utils.metrics import calculate_all_metrics

# Get predictions
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch, target in test_loader:
        batch = batch.to(trainer.device)
        preds = model(batch).cpu().numpy()
        all_preds.extend(preds.flatten())
        all_targets.extend(target.numpy().flatten())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# Visualize
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(all_targets, all_preds, alpha=0.3)
plt.plot([all_targets.min(), all_targets.max()],
         [all_targets.min(), all_targets.max()], 'r--')
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.title('Predictions vs Actuals')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.hist(all_preds - all_targets, bins=50, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True)

plt.subplot(2, 2, 3)
cumulative_actual = np.cumprod(1 + all_targets)
cumulative_pred = np.cumprod(1 + all_preds)
plt.plot(cumulative_actual, label='Actual')
plt.plot(cumulative_pred, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
metrics = calculate_all_metrics(all_preds, all_targets)
metric_names = list(metrics.keys())[:8]
metric_values = [metrics[k] for k in metric_names]
plt.barh(metric_names, metric_values)
plt.xlabel('Value')
plt.title('Performance Metrics')
plt.tight_layout()
plt.show()
```

## 5. Distributed Training

### Multi-GPU Training with PyTorch DDP

```python
# Save this as train_distributed.py and run with:
# torchrun --nproc_per_node=4 train_distributed.py

from quantscale.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    DistributedTrainer,
    create_distributed_dataloaders,
    is_main_process
)

# Initialize distributed training
setup_distributed(backend='nccl')

try:
    # Create datasets (same as before)
    from quantscale.data.loaders import FinancialDataLoader

    train_dataset = FinancialDataLoader(
        'data/train.parquet',
        sequence_length=50,
        target='target'
    )

    val_dataset = FinancialDataLoader(
        'data/val.parquet',
        sequence_length=50,
        target='target'
    )

    # Create distributed data loaders
    loaders = create_distributed_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=64,  # Per GPU
        num_workers=4
    )

    # Create model
    model = TemporalFusionTransformer(
        input_dim=input_dim,
        hidden_dim=512,  # Larger model for multi-GPU
        num_heads=16,
        num_layers=6
    )

    # Create distributed trainer
    trainer = DistributedTrainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val']
    )

    # Train
    if is_main_process():
        print("Starting distributed training on 4 GPUs...")

    history = trainer.fit(epochs=100)

finally:
    cleanup_distributed()
```

## 6. Hyperparameter Optimization

### Using Ray Tune

```python
from ray import tune
from quantscale.optimization.hyperparameter import HyperparameterSearch
from quantscale.models.transformer import TemporalFusionTransformer

# Define search space
search_space = {
    'model_params': {
        'input_dim': input_dim,
        'hidden_dim': tune.choice([128, 256, 512]),
        'num_heads': tune.choice([4, 8, 16]),
        'num_layers': tune.choice([2, 4, 6]),
        'dropout': tune.uniform(0.1, 0.5),
    },
    'learning_rate': tune.loguniform(1e-4, 1e-2),
    'batch_size': tune.choice([32, 64, 128]),
    'weight_decay': tune.loguniform(1e-6, 1e-3),
}

# Create search
search = HyperparameterSearch(
    model_class=TemporalFusionTransformer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=search_space,
    num_samples=50,
    max_epochs=50,
    gpus_per_trial=1.0,
    search_alg='optuna',
    scheduler='asha'
)

# Run search
best_config = search.run()

print("\\nBest hyperparameters:")
print(best_config)
```

## Next Steps

1. **Experiment with different models:** Try BidirectionalLSTM
2. **Add more features:** Engineer domain-specific features
3. **Advanced techniques:** Mixed precision training, model ensembles
4. **Production deployment:** Export to ONNX, serve with FastAPI

## Resources

- [Documentation](https://github.com/tradervijeth/quantscale-distributed-training)
- [Examples](../examples/)
- [API Reference](https://quantscale.readthedocs.io/)

---

**Questions or feedback?** Contact: finance@vijeth.com
