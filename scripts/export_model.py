#!/usr/bin/env python3
"""
QuantScale: Model Export Script

Export trained models to ONNX format for production deployment.

Copyright (c) 2025 Vithushan Jeyapahan, Vijeth Ltd
Licensed under the Apache License 2.0
Contact: finance@vijeth.com
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.onnx

from quantscale.models.lstm import BidirectionalLSTM
from quantscale.models.transformer import TemporalFusionTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple,
    opset_version: int = 14,
    dynamic_axes: bool = True,
) -> None:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Shape of dummy input (batch_size, seq_len, input_dim)
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic axes for batch size

    Example:
        >>> model = TemporalFusionTransformer(input_dim=50, hidden_dim=256)
        >>> export_to_onnx(model, "model.onnx", (1, 100, 50))
    """
    logger.info(f"Exporting model to ONNX format...")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Output path: {output_path}")

    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Define dynamic axes for variable batch size and sequence length
    if dynamic_axes:
        dynamic_axes_dict = {
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        }
    else:
        dynamic_axes_dict = None

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_dict,
        )
        logger.info(f"✓ Model exported successfully to {output_path}")

    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise


def verify_onnx_model(
    onnx_path: str,
    pytorch_model: torch.nn.Module,
    input_shape: tuple,
    tolerance: float = 1e-5,
) -> bool:
    """
    Verify ONNX model produces same outputs as PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        input_shape: Shape of test input
        tolerance: Numerical tolerance for comparison

    Returns:
        True if models match, False otherwise
    """
    logger.info("Verifying ONNX model...")

    try:
        import onnx
        import onnxruntime as ort

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model is valid")

        # Create test input
        test_input = torch.randn(*input_shape)

        # Get PyTorch output
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()

        # Get ONNX output
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        # Compare outputs
        max_diff = abs(pytorch_output - ort_output).max()
        logger.info(f"Maximum difference: {max_diff:.10f}")

        if max_diff < tolerance:
            logger.info("✓ ONNX model verified successfully!")
            return True
        else:
            logger.warning(f"⚠ Models differ by {max_diff} (tolerance: {tolerance})")
            return False

    except ImportError as e:
        logger.warning(f"Could not verify ONNX model: {e}")
        logger.warning("Install onnx and onnxruntime to enable verification")
        return False
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """
    Load model weights from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance

    Returns:
        Model with loaded weights
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info("✓ Checkpoint loaded successfully")
    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export QuantScale models to ONNX format"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path (default: <checkpoint>.onnx)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["transformer", "lstm"],
        required=True,
        help="Model architecture",
    )

    # Model configuration
    parser.add_argument(
        "--input-dim",
        type=int,
        required=True,
        help="Input dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        required=True,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (transformer only)",
    )

    # Export configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for dummy input",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Sequence length for dummy input",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no-dynamic-axes",
        action="store_true",
        help="Disable dynamic axes (fixed batch size)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX model after export",
    )

    return parser.parse_args()


def main():
    """Main export function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("QuantScale Model Export to ONNX")
    logger.info("=" * 80)

    # Determine output path
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        output_path = checkpoint_path.with_suffix(".onnx")
    else:
        output_path = Path(args.output)

    # Create model
    logger.info(f"Creating {args.model} model...")
    if args.model == "transformer":
        model = TemporalFusionTransformer(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
        )
    else:  # lstm
        model = BidirectionalLSTM(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )

    # Load checkpoint
    model = load_checkpoint(args.checkpoint, model)

    # Define input shape
    input_shape = (args.batch_size, args.sequence_length, args.input_dim)

    # Export to ONNX
    export_to_onnx(
        model=model,
        output_path=str(output_path),
        input_shape=input_shape,
        opset_version=args.opset_version,
        dynamic_axes=not args.no_dynamic_axes,
    )

    # Verify if requested
    if args.verify:
        verify_onnx_model(str(output_path), model, input_shape)

    # Print model info
    logger.info("\nModel Information:")
    logger.info(f"  Architecture: {args.model}")
    logger.info(f"  Input dimension: {args.input_dim}")
    logger.info(f"  Hidden dimension: {args.hidden_dim}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"\nONNX Model:")
    logger.info(f"  Path: {output_path}")
    logger.info(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    logger.info("\n✓ Export completed successfully!")
    logger.info("\nUsage in production:")
    logger.info(f"  import onnxruntime as ort")
    logger.info(f"  session = ort.InferenceSession('{output_path}')")
    logger.info(f"  output = session.run(None, {{'input': your_data}})")


if __name__ == "__main__":
    main()
