#!/usr/bin/env python3
"""
QuantScale: Distributed Training Framework for Financial Time Series

Performance benchmarking suite for measuring throughput and scaling efficiency.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quantscale.models.transformer import TemporalFusionTransformer
from quantscale.models.lstm import BidirectionalLSTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Benchmark runner for measuring model performance.

    Args:
        model: PyTorch model to benchmark
        device: Device to run benchmark on
        batch_sizes: List of batch sizes to test
        sequence_lengths: List of sequence lengths to test
        num_iterations: Number of iterations per configuration
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_sizes: List[int] = [32, 64, 128, 256],
        sequence_lengths: List[int] = [50, 100, 200],
        num_iterations: int = 100,
    ) -> None:
        """Initialize benchmark runner."""
        self.model = model.to(device)
        self.device = device
        self.batch_sizes = batch_sizes
        self.sequence_lengths = sequence_lengths
        self.num_iterations = num_iterations

        self.results: List[Dict[str, Any]] = []

    def benchmark_inference(
        self,
        batch_size: int,
        sequence_length: int,
        input_dim: int,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            input_dim: Input dimension

        Returns:
            Dictionary with benchmark results
        """
        logger.info(
            f"Benchmarking inference: batch_size={batch_size}, "
            f"seq_len={sequence_length}"
        )

        # Create dummy data
        dummy_input = torch.randn(
            batch_size, sequence_length, input_dim, device=self.device
        )

        # Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)

        # Benchmark
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(self.num_iterations):
                _ = self.model(dummy_input)

        torch.cuda.synchronize() if self.device.type == "cuda" else None
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / self.num_iterations
        throughput = batch_size / avg_time  # samples per second
        latency = avg_time * 1000  # milliseconds

        return {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "avg_time_ms": latency,
            "throughput": throughput,
            "total_samples": batch_size * self.num_iterations,
        }

    def benchmark_training(
        self,
        batch_size: int,
        sequence_length: int,
        input_dim: int,
    ) -> Dict[str, float]:
        """
        Benchmark training performance.

        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            input_dim: Input dimension

        Returns:
            Dictionary with benchmark results
        """
        logger.info(
            f"Benchmarking training: batch_size={batch_size}, "
            f"seq_len={sequence_length}"
        )

        # Create dummy data
        dummy_input = torch.randn(
            batch_size, sequence_length, input_dim, device=self.device
        )
        dummy_target = torch.randn(batch_size, 1, device=self.device)

        # Setup training
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Warmup
        for _ in range(10):
            optimizer.zero_grad()
            output = self.model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()

        # Benchmark
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        start_time = time.time()

        for _ in range(self.num_iterations):
            optimizer.zero_grad()
            output = self.model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize() if self.device.type == "cuda" else None
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / self.num_iterations
        throughput = batch_size / avg_time
        latency = avg_time * 1000

        return {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "avg_time_ms": latency,
            "throughput": throughput,
            "total_samples": batch_size * self.num_iterations,
        }

    def run_full_benchmark(
        self,
        input_dim: int = 50,
        mode: str = "inference",
    ) -> pd.DataFrame:
        """
        Run full benchmark across all configurations.

        Args:
            input_dim: Input dimension
            mode: Benchmark mode ('inference' or 'training')

        Returns:
            DataFrame with all results
        """
        logger.info(f"Running full {mode} benchmark...")

        results = []

        for batch_size in self.batch_sizes:
            for seq_length in self.sequence_lengths:
                try:
                    if mode == "inference":
                        result = self.benchmark_inference(
                            batch_size, seq_length, input_dim
                        )
                    else:
                        result = self.benchmark_training(
                            batch_size, seq_length, input_dim
                        )

                    result["mode"] = mode
                    results.append(result)

                except RuntimeError as e:
                    logger.warning(
                        f"Failed for batch_size={batch_size}, "
                        f"seq_len={seq_length}: {e}"
                    )

        return pd.DataFrame(results)


def benchmark_gpu_scaling(
    model_class: type,
    model_kwargs: Dict[str, Any],
    num_gpus: List[int] = [1, 2, 4],
    batch_size: int = 64,
    sequence_length: int = 100,
    input_dim: int = 50,
) -> pd.DataFrame:
    """
    Benchmark scaling efficiency across multiple GPUs.

    Args:
        model_class: Model class to benchmark
        model_kwargs: Model initialization arguments
        num_gpus: List of GPU counts to test
        batch_size: Batch size per GPU
        sequence_length: Sequence length
        input_dim: Input dimension

    Returns:
        DataFrame with scaling results
    """
    logger.info("Benchmarking GPU scaling efficiency...")

    results = []

    for n_gpus in num_gpus:
        if n_gpus > torch.cuda.device_count():
            logger.warning(f"Skipping {n_gpus} GPUs (only {torch.cuda.device_count()} available)")
            continue

        # Create model
        model = model_class(**model_kwargs)
        device = torch.device("cuda:0")

        # Run benchmark
        runner = BenchmarkRunner(
            model=model,
            device=device,
            batch_sizes=[batch_size * n_gpus],
            sequence_lengths=[sequence_length],
            num_iterations=50,
        )

        result = runner.benchmark_inference(
            batch_size * n_gpus, sequence_length, input_dim
        )

        result["num_gpus"] = n_gpus
        results.append(result)

    # Calculate scaling metrics
    df = pd.DataFrame(results)

    if len(df) > 0:
        baseline_throughput = df[df["num_gpus"] == 1]["throughput"].iloc[0]
        df["speedup"] = df["throughput"] / baseline_throughput
        df["efficiency"] = (df["speedup"] / df["num_gpus"]) * 100

    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark QuantScale models")

    parser.add_argument(
        "--model",
        type=str,
        choices=["transformer", "lstm"],
        default="transformer",
        help="Model to benchmark",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inference", "training", "both"],
        default="both",
        help="Benchmark mode",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=50,
        help="Input dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of iterations per configuration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--gpu-scaling",
        action="store_true",
        help="Run GPU scaling benchmark",
    )

    return parser.parse_args()


def main():
    """Main benchmark function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("QuantScale Performance Benchmark")
    logger.info("=" * 80)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")

    # Create model
    logger.info(f"Creating {args.model} model...")
    if args.model == "transformer":
        model = TemporalFusionTransformer(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_heads=8,
            num_layers=4,
        )
    else:
        model = BidirectionalLSTM(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=2,
        )

    # Create benchmark runner
    runner = BenchmarkRunner(
        model=model,
        device=device,
        num_iterations=args.num_iterations,
    )

    # Run benchmarks
    all_results = []

    if args.mode in ["inference", "both"]:
        inference_results = runner.run_full_benchmark(
            input_dim=args.input_dim, mode="inference"
        )
        all_results.append(inference_results)
        logger.info("\nInference Results:")
        logger.info(f"\n{inference_results.to_string()}")

    if args.mode in ["training", "both"]:
        training_results = runner.run_full_benchmark(
            input_dim=args.input_dim, mode="training"
        )
        all_results.append(training_results)
        logger.info("\nTraining Results:")
        logger.info(f"\n{training_results.to_string()}")

    # GPU scaling benchmark
    if args.gpu_scaling and device.type == "cuda":
        logger.info("\nRunning GPU scaling benchmark...")

        model_class = TemporalFusionTransformer if args.model == "transformer" else BidirectionalLSTM
        model_kwargs = {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
        }

        scaling_results = benchmark_gpu_scaling(
            model_class=model_class,
            model_kwargs=model_kwargs,
        )
        all_results.append(scaling_results)
        logger.info("\nGPU Scaling Results:")
        logger.info(f"\n{scaling_results.to_string()}")

    # Save results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(args.output, index=False)
        logger.info(f"\nResults saved to {args.output}")

    logger.info("\nBenchmark completed!")


if __name__ == "__main__":
    main()
