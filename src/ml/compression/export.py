"""
Model export utilities for deployment.

Provides ONNX export and optimization.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Export format."""
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TFLITE = "tflite"
    COREML = "coreml"


class ONNXOpset(int, Enum):
    """ONNX opset versions."""
    V11 = 11
    V12 = 12
    V13 = 13
    V14 = 14
    V15 = 15
    V16 = 16
    V17 = 17


@dataclass
class ExportConfig:
    """Export configuration."""
    format: ExportFormat = ExportFormat.ONNX
    opset_version: int = 14
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    input_names: List[str] = field(default_factory=lambda: ["input"])
    output_names: List[str] = field(default_factory=lambda: ["output"])
    optimize: bool = True
    simplify: bool = True
    fp16: bool = False
    int8: bool = False
    external_data: bool = False  # For large models
    check_model: bool = True


@dataclass
class ExportResult:
    """Result of model export."""
    export_path: str
    format: str
    original_size_mb: float
    exported_size_mb: float
    export_time_seconds: float
    is_valid: bool
    validation_message: str = ""
    optimizations_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "export_path": self.export_path,
            "format": self.format,
            "original_size_mb": round(self.original_size_mb, 2),
            "exported_size_mb": round(self.exported_size_mb, 2),
            "export_time_seconds": round(self.export_time_seconds, 2),
            "is_valid": self.is_valid,
            "validation_message": self.validation_message,
            "optimizations_applied": self.optimizations_applied,
        }


class ONNXExporter:
    """
    ONNX model exporter.

    Exports PyTorch models to ONNX format with optimization.
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize exporter.

        Args:
            config: Export configuration
        """
        self._config = config or ExportConfig()

    @property
    def config(self) -> ExportConfig:
        return self._config

    def export(
        self,
        model: Any,
        output_path: Union[str, Path],
        input_shape: Tuple[int, ...],
        input_dtype: str = "float32",
    ) -> ExportResult:
        """
        Export model to ONNX.

        Args:
            model: PyTorch model
            output_path: Output file path
            input_shape: Input tensor shape
            input_dtype: Input data type

        Returns:
            ExportResult
        """
        try:
            import torch
            import onnx
        except ImportError:
            raise ImportError("PyTorch and ONNX required: pip install torch onnx")

        start_time = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Measure original size
        original_size = self._get_model_size(model)

        # Create dummy input
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        torch_dtype = dtype_map.get(input_dtype, torch.float32)
        dummy_input = torch.randn(input_shape, dtype=torch_dtype)

        # Move model and input to same device
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)
        model.eval()

        # Setup dynamic axes
        dynamic_axes = self._config.dynamic_axes
        if dynamic_axes is None:
            # Default: batch dimension is dynamic
            dynamic_axes = {
                self._config.input_names[0]: {0: "batch_size"},
                self._config.output_names[0]: {0: "batch_size"},
            }

        # Export to ONNX
        optimizations = []
        temp_path = output_path.with_suffix(".temp.onnx")

        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(temp_path),
                input_names=self._config.input_names,
                output_names=self._config.output_names,
                dynamic_axes=dynamic_axes,
                opset_version=self._config.opset_version,
                do_constant_folding=True,
                export_params=True,
            )
            optimizations.append("constant_folding")

            # Load and validate
            onnx_model = onnx.load(str(temp_path))

            if self._config.check_model:
                try:
                    onnx.checker.check_model(onnx_model)
                except Exception as e:
                    logger.warning(f"ONNX validation warning: {e}")

            # Optimize
            if self._config.optimize:
                onnx_model = self._optimize_onnx(onnx_model)
                optimizations.append("graph_optimization")

            # Simplify
            if self._config.simplify:
                onnx_model = self._simplify_onnx(onnx_model)
                optimizations.append("simplification")

            # Convert to FP16 if requested
            if self._config.fp16:
                onnx_model = self._convert_fp16(onnx_model)
                optimizations.append("fp16_conversion")

            # Save final model
            if self._config.external_data:
                onnx.save_model(
                    onnx_model,
                    str(output_path),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"{output_path.stem}_data",
                )
            else:
                onnx.save(onnx_model, str(output_path))

            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()

            # Validate final model
            is_valid = True
            validation_message = "Export successful"

            if self._config.check_model:
                try:
                    final_model = onnx.load(str(output_path))
                    onnx.checker.check_model(final_model)
                except Exception as e:
                    is_valid = False
                    validation_message = str(e)

            export_time = time.time() - start_time
            exported_size = output_path.stat().st_size / (1024 * 1024)

            return ExportResult(
                export_path=str(output_path),
                format="onnx",
                original_size_mb=original_size,
                exported_size_mb=exported_size,
                export_time_seconds=export_time,
                is_valid=is_valid,
                validation_message=validation_message,
                optimizations_applied=optimizations,
            )

        except Exception as e:
            # Cleanup on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"ONNX export failed: {e}")

    def _optimize_onnx(self, model: Any) -> Any:
        """Apply ONNX graph optimizations."""
        try:
            from onnxoptimizer import optimize
            passes = [
                "eliminate_deadend",
                "eliminate_identity",
                "eliminate_nop_dropout",
                "eliminate_nop_pad",
                "eliminate_nop_transpose",
                "eliminate_unused_initializer",
                "fuse_add_bias_into_conv",
                "fuse_bn_into_conv",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes",
                "fuse_matmul_add_bias_into_gemm",
            ]
            return optimize(model, passes)
        except ImportError:
            logger.warning("onnxoptimizer not available, skipping optimization")
            return model

    def _simplify_onnx(self, model: Any) -> Any:
        """Simplify ONNX model."""
        try:
            import onnxsim
            model_simplified, check = onnxsim.simplify(model)
            if check:
                return model_simplified
            else:
                logger.warning("ONNX simplification check failed, using original")
                return model
        except ImportError:
            logger.warning("onnx-simplifier not available, skipping simplification")
            return model

    def _convert_fp16(self, model: Any) -> Any:
        """Convert model to FP16."""
        try:
            from onnxconverter_common import float16
            return float16.convert_float_to_float16(model, keep_io_types=True)
        except ImportError:
            logger.warning("onnxconverter-common not available, skipping FP16 conversion")
            return model

    def _get_model_size(self, model: Any) -> float:
        """Get model size in MB."""
        import torch
        import io

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.tell() / (1024 * 1024)

    def verify_export(
        self,
        original_model: Any,
        onnx_path: Union[str, Path],
        input_shape: Tuple[int, ...],
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> Tuple[bool, str]:
        """
        Verify ONNX export matches original model.

        Args:
            original_model: Original PyTorch model
            onnx_path: Path to ONNX model
            input_shape: Input shape for testing
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            (is_valid, message)
        """
        try:
            import torch
            import onnxruntime as ort
            import numpy as np
        except ImportError:
            return False, "onnxruntime required for verification"

        # Create test input
        test_input = torch.randn(input_shape)
        device = next(original_model.parameters()).device
        test_input_torch = test_input.to(device)

        # Run PyTorch model
        original_model.eval()
        with torch.no_grad():
            torch_output = original_model(test_input_torch).cpu().numpy()

        # Run ONNX model
        ort_session = ort.InferenceSession(str(onnx_path))
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        # Compare outputs
        try:
            np.testing.assert_allclose(torch_output, ort_output, rtol=rtol, atol=atol)
            return True, "Verification passed: outputs match within tolerance"
        except AssertionError as e:
            max_diff = np.max(np.abs(torch_output - ort_output))
            return False, f"Verification failed: max difference {max_diff:.6f}"


def export_to_onnx(
    model: Any,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...],
    opset_version: int = 14,
    optimize: bool = True,
) -> ExportResult:
    """
    Convenience function to export model to ONNX.

    Args:
        model: PyTorch model
        output_path: Output file path
        input_shape: Input tensor shape
        opset_version: ONNX opset version
        optimize: Apply optimizations

    Returns:
        ExportResult
    """
    config = ExportConfig(opset_version=opset_version, optimize=optimize)
    exporter = ONNXExporter(config)
    return exporter.export(model, output_path, input_shape)


def optimize_onnx(
    model_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    fp16: bool = False,
) -> ExportResult:
    """
    Optimize an existing ONNX model.

    Args:
        model_path: Path to ONNX model
        output_path: Output path (overwrites if None)
        fp16: Convert to FP16

    Returns:
        ExportResult
    """
    import onnx

    start_time = time.time()
    model_path = Path(model_path)
    output_path = Path(output_path) if output_path else model_path

    # Load model
    model = onnx.load(str(model_path))
    original_size = model_path.stat().st_size / (1024 * 1024)

    optimizations = []

    # Create temporary exporter for optimization methods
    config = ExportConfig(optimize=True, simplify=True, fp16=fp16)
    exporter = ONNXExporter(config)

    # Optimize
    model = exporter._optimize_onnx(model)
    optimizations.append("graph_optimization")

    model = exporter._simplify_onnx(model)
    optimizations.append("simplification")

    if fp16:
        model = exporter._convert_fp16(model)
        optimizations.append("fp16_conversion")

    # Save
    onnx.save(model, str(output_path))

    export_time = time.time() - start_time
    exported_size = output_path.stat().st_size / (1024 * 1024)

    return ExportResult(
        export_path=str(output_path),
        format="onnx",
        original_size_mb=original_size,
        exported_size_mb=exported_size,
        export_time_seconds=export_time,
        is_valid=True,
        validation_message="Optimization complete",
        optimizations_applied=optimizations,
    )
