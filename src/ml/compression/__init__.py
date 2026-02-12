"""
Model Compression Module (M5).

Provides model optimization techniques:
- Quantization (INT8/FP16)
- Pruning (structured/unstructured)
- Knowledge distillation
- ONNX export
"""

from src.ml.compression.quantization import (
    Quantizer,
    QuantizationConfig,
    QuantizationMode,
    CalibrationMethod,
    quantize_model,
    quantize_dynamic,
    quantize_static,
)
from src.ml.compression.pruning import (
    Pruner,
    PruningConfig,
    PruningMethod,
    PruningSchedule,
    prune_model,
    prune_unstructured,
    prune_structured,
)
from src.ml.compression.distillation import (
    DistillationTrainer,
    DistillationConfig,
    DistillationLoss,
    distill_model,
)
from src.ml.compression.export import (
    ONNXExporter,
    ExportConfig,
    ExportFormat,
    export_to_onnx,
    optimize_onnx,
)

__all__ = [
    # Quantization
    "Quantizer",
    "QuantizationConfig",
    "QuantizationMode",
    "CalibrationMethod",
    "quantize_model",
    "quantize_dynamic",
    "quantize_static",
    # Pruning
    "Pruner",
    "PruningConfig",
    "PruningMethod",
    "PruningSchedule",
    "prune_model",
    "prune_unstructured",
    "prune_structured",
    # Distillation
    "DistillationTrainer",
    "DistillationConfig",
    "DistillationLoss",
    "distill_model",
    # Export
    "ONNXExporter",
    "ExportConfig",
    "ExportFormat",
    "export_to_onnx",
    "optimize_onnx",
]
