"""
Model quantization for inference optimization.

Provides INT8/FP16 quantization with various calibration methods.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class QuantizationMode(str, Enum):
    """Quantization mode."""
    DYNAMIC = "dynamic"  # Dynamic quantization (weights only)
    STATIC = "static"  # Static quantization (weights + activations)
    QAT = "qat"  # Quantization-aware training
    FP16 = "fp16"  # Half precision


class CalibrationMethod(str, Enum):
    """Calibration method for static quantization."""
    MINMAX = "minmax"
    HISTOGRAM = "histogram"
    ENTROPY = "entropy"
    PERCENTILE = "percentile"


@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    mode: QuantizationMode = QuantizationMode.DYNAMIC
    dtype: str = "qint8"  # qint8, quint8, float16
    calibration_method: CalibrationMethod = CalibrationMethod.MINMAX
    calibration_samples: int = 100
    per_channel: bool = True
    symmetric: bool = True
    reduce_range: bool = False  # For x86 compatibility
    modules_to_quantize: Optional[List[str]] = None
    modules_to_exclude: Optional[List[str]] = None
    preserve_accuracy_threshold: float = 0.01  # Max accuracy drop


@dataclass
class QuantizationResult:
    """Result of quantization."""
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    original_latency_ms: float
    quantized_latency_ms: float
    speedup: float
    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None
    accuracy_drop: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_size_mb": round(self.original_size_mb, 2),
            "quantized_size_mb": round(self.quantized_size_mb, 2),
            "compression_ratio": round(self.compression_ratio, 2),
            "original_latency_ms": round(self.original_latency_ms, 2),
            "quantized_latency_ms": round(self.quantized_latency_ms, 2),
            "speedup": round(self.speedup, 2),
            "accuracy_before": self.accuracy_before,
            "accuracy_after": self.accuracy_after,
            "accuracy_drop": round(self.accuracy_drop, 4) if self.accuracy_drop else None,
        }


class Quantizer:
    """
    Model quantizer for inference optimization.

    Supports dynamic, static, and QAT quantization.
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize quantizer.

        Args:
            config: Quantization configuration
        """
        self._config = config or QuantizationConfig()
        self._calibration_data: List[Any] = []

    @property
    def config(self) -> QuantizationConfig:
        return self._config

    def quantize(
        self,
        model: Any,
        calibration_data: Optional[Iterator] = None,
        eval_fn: Optional[Callable] = None,
    ) -> Tuple[Any, QuantizationResult]:
        """
        Quantize a model.

        Args:
            model: PyTorch model to quantize
            calibration_data: Data for static quantization calibration
            eval_fn: Function to evaluate model accuracy

        Returns:
            (quantized_model, result)
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for quantization")

        # Measure original model
        original_size = self._get_model_size(model)
        original_latency = self._measure_latency(model)
        original_accuracy = eval_fn(model) if eval_fn else None

        # Select quantization method
        if self._config.mode == QuantizationMode.DYNAMIC:
            quantized_model = self._quantize_dynamic(model)
        elif self._config.mode == QuantizationMode.STATIC:
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            quantized_model = self._quantize_static(model, calibration_data)
        elif self._config.mode == QuantizationMode.QAT:
            quantized_model = self._prepare_qat(model)
        elif self._config.mode == QuantizationMode.FP16:
            quantized_model = self._quantize_fp16(model)
        else:
            raise ValueError(f"Unknown quantization mode: {self._config.mode}")

        # Measure quantized model
        quantized_size = self._get_model_size(quantized_model)
        quantized_latency = self._measure_latency(quantized_model)
        quantized_accuracy = eval_fn(quantized_model) if eval_fn else None

        # Calculate metrics
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        speedup = original_latency / quantized_latency if quantized_latency > 0 else 1.0
        accuracy_drop = None
        if original_accuracy is not None and quantized_accuracy is not None:
            accuracy_drop = original_accuracy - quantized_accuracy

        result = QuantizationResult(
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=compression_ratio,
            original_latency_ms=original_latency,
            quantized_latency_ms=quantized_latency,
            speedup=speedup,
            accuracy_before=original_accuracy,
            accuracy_after=quantized_accuracy,
            accuracy_drop=accuracy_drop,
        )

        logger.info(f"Quantization complete: {compression_ratio:.2f}x compression, {speedup:.2f}x speedup")
        return quantized_model, result

    def _quantize_dynamic(self, model: Any) -> Any:
        """Apply dynamic quantization."""
        import torch
        import torch.quantization as quant

        # Clone model
        model = copy.deepcopy(model)
        model.eval()

        # Get modules to quantize
        qconfig_spec = {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}

        if self._config.dtype == "float16":
            # Use FP16 instead
            return model.half()

        quantized_model = quant.quantize_dynamic(
            model,
            qconfig_spec,
            dtype=torch.qint8 if self._config.dtype == "qint8" else torch.quint8,
        )

        return quantized_model

    def _quantize_static(self, model: Any, calibration_data: Iterator) -> Any:
        """Apply static quantization."""
        import torch
        import torch.quantization as quant

        # Clone and prepare model
        model = copy.deepcopy(model)
        model.eval()

        # Set qconfig
        if self._config.per_channel:
            qconfig = quant.get_default_qconfig("fbgemm")
        else:
            qconfig = quant.get_default_qconfig("qnnpack")

        model.qconfig = qconfig

        # Fuse modules if applicable
        model = self._fuse_modules(model)

        # Prepare for calibration
        quant.prepare(model, inplace=True)

        # Run calibration
        logger.info("Running calibration...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if i >= self._config.calibration_samples:
                    break
                if isinstance(batch, (tuple, list)):
                    model(batch[0])
                else:
                    model(batch)

        # Convert to quantized model
        quant.convert(model, inplace=True)

        return model

    def _prepare_qat(self, model: Any) -> Any:
        """Prepare model for quantization-aware training."""
        import torch
        import torch.quantization as quant

        # Clone model
        model = copy.deepcopy(model)
        model.train()

        # Set qconfig for QAT
        model.qconfig = quant.get_default_qat_qconfig("fbgemm")

        # Fuse modules
        model = self._fuse_modules(model)

        # Prepare for QAT
        quant.prepare_qat(model, inplace=True)

        return model

    def _quantize_fp16(self, model: Any) -> Any:
        """Convert model to FP16."""
        import torch

        model = copy.deepcopy(model)
        model.half()
        return model

    def _fuse_modules(self, model: Any) -> Any:
        """Fuse modules for better quantization."""
        import torch
        import torch.quantization as quant

        # Find fuseable patterns
        patterns_to_fuse = []

        for name, module in model.named_modules():
            # Conv + BN + ReLU
            if hasattr(module, "conv") and hasattr(module, "bn") and hasattr(module, "relu"):
                patterns_to_fuse.append([f"{name}.conv", f"{name}.bn", f"{name}.relu"])
            # Linear + ReLU
            elif hasattr(module, "linear") and hasattr(module, "relu"):
                patterns_to_fuse.append([f"{name}.linear", f"{name}.relu"])

        if patterns_to_fuse:
            model = quant.fuse_modules(model, patterns_to_fuse)

        return model

    def _get_model_size(self, model: Any) -> float:
        """Get model size in MB."""
        import torch
        import io

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_bytes = buffer.tell()
        return size_bytes / (1024 * 1024)

    def _measure_latency(self, model: Any, num_runs: int = 100) -> float:
        """Measure inference latency in ms."""
        import torch

        model.eval()
        device = next(model.parameters()).device

        # Create dummy input
        dummy_input = torch.randn(1, 64).to(device)  # Adjust shape as needed

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                try:
                    model(dummy_input)
                except Exception:
                    break

        # Measure
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                try:
                    model(dummy_input)
                except Exception:
                    break

        total_time = (time.time() - start) * 1000  # ms
        return total_time / num_runs

    def save_quantized(self, model: Any, path: Union[str, Path]) -> None:
        """Save quantized model."""
        import torch

        path = Path(path)
        torch.save(model.state_dict(), path)
        logger.info(f"Saved quantized model to {path}")

    def load_quantized(self, model: Any, path: Union[str, Path]) -> Any:
        """Load quantized model weights."""
        import torch

        path = Path(path)
        model.load_state_dict(torch.load(path))
        return model


def quantize_model(
    model: Any,
    mode: QuantizationMode = QuantizationMode.DYNAMIC,
    calibration_data: Optional[Iterator] = None,
    eval_fn: Optional[Callable] = None,
) -> Tuple[Any, QuantizationResult]:
    """
    Convenience function to quantize a model.

    Args:
        model: Model to quantize
        mode: Quantization mode
        calibration_data: Calibration data for static quantization
        eval_fn: Evaluation function

    Returns:
        (quantized_model, result)
    """
    config = QuantizationConfig(mode=mode)
    quantizer = Quantizer(config)
    return quantizer.quantize(model, calibration_data, eval_fn)


def quantize_dynamic(model: Any) -> Any:
    """Apply dynamic quantization."""
    config = QuantizationConfig(mode=QuantizationMode.DYNAMIC)
    quantizer = Quantizer(config)
    quantized, _ = quantizer.quantize(model)
    return quantized


def quantize_static(model: Any, calibration_data: Iterator) -> Any:
    """Apply static quantization."""
    config = QuantizationConfig(mode=QuantizationMode.STATIC)
    quantizer = Quantizer(config)
    quantized, _ = quantizer.quantize(model, calibration_data)
    return quantized
