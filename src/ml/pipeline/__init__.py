"""
End-to-End ML Pipeline Module.

Provides complete ML workflow orchestration:
- Data loading and preprocessing
- Feature extraction
- Model training and evaluation
- Inference pipeline
"""

from src.ml.pipeline.stages import (
    PipelineStage,
    StageStatus,
    StageResult,
    DataLoadingStage,
    PreprocessingStage,
    FeatureExtractionStage,
    TrainingStage,
    EvaluationStage,
    InferenceStage,
)
from src.ml.pipeline.orchestrator import (
    Pipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStatus,
)
from src.ml.pipeline.builder import (
    PipelineBuilder,
    create_training_pipeline,
    create_inference_pipeline,
)

__all__ = [
    # Stages
    "PipelineStage",
    "StageStatus",
    "StageResult",
    "DataLoadingStage",
    "PreprocessingStage",
    "FeatureExtractionStage",
    "TrainingStage",
    "EvaluationStage",
    "InferenceStage",
    # Orchestrator
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStatus",
    # Builder
    "PipelineBuilder",
    "create_training_pipeline",
    "create_inference_pipeline",
]
