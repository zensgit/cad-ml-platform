"""
Pipeline builder for easy pipeline construction.

Provides fluent API and factory functions for common pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from src.ml.pipeline.orchestrator import Pipeline, PipelineConfig, PipelineResult
from src.ml.pipeline.stages import (
    PipelineStage,
    DataLoadingStage,
    PreprocessingStage,
    FeatureExtractionStage,
    TrainingStage,
    EvaluationStage,
    InferenceStage,
)

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Fluent builder for constructing ML pipelines.

    Provides a declarative API for building pipelines.
    """

    def __init__(self, name: str = "pipeline"):
        """
        Initialize pipeline builder.

        Args:
            name: Pipeline name
        """
        self._name = name
        self._description = ""
        self._stages: List[PipelineStage] = []
        self._config: Optional[PipelineConfig] = None
        self._context: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = {}

    def name(self, name: str) -> "PipelineBuilder":
        """Set pipeline name."""
        self._name = name
        return self

    def description(self, description: str) -> "PipelineBuilder":
        """Set pipeline description."""
        self._description = description
        return self

    def config(
        self,
        stop_on_error: bool = True,
        save_intermediate: bool = False,
        checkpoint_dir: Optional[str] = None,
        max_retries: int = 0,
        verbose: bool = True,
    ) -> "PipelineBuilder":
        """Configure pipeline behavior."""
        self._config = PipelineConfig(
            name=self._name,
            description=self._description,
            stop_on_error=stop_on_error,
            save_intermediate=save_intermediate,
            checkpoint_dir=checkpoint_dir,
            max_retries=max_retries,
            verbose=verbose,
        )
        return self

    def add_stage(self, stage: PipelineStage) -> "PipelineBuilder":
        """Add a custom stage."""
        self._stages.append(stage)
        return self

    def load_data(
        self,
        data_path: Optional[Union[str, Path]] = None,
        loader_fn: Optional[Callable] = None,
        **kwargs,
    ) -> "PipelineBuilder":
        """Add data loading stage."""
        stage = DataLoadingStage(
            data_path=data_path,
            loader_fn=loader_fn,
            **kwargs,
        )
        self._stages.append(stage)
        return self

    def preprocess(
        self,
        transforms: Optional[List[Callable]] = None,
        normalize: bool = True,
        **kwargs,
    ) -> "PipelineBuilder":
        """Add preprocessing stage."""
        stage = PreprocessingStage(
            transforms=transforms,
            normalize=normalize,
            **kwargs,
        )
        self._stages.append(stage)
        return self

    def extract_features(
        self,
        extractors: Optional[List[Callable]] = None,
        feature_dim: int = 128,
        **kwargs,
    ) -> "PipelineBuilder":
        """Add feature extraction stage."""
        stage = FeatureExtractionStage(
            feature_extractors=extractors,
            feature_dim=feature_dim,
            **kwargs,
        )
        self._stages.append(stage)
        return self

    def train(
        self,
        model: Optional[Any] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_split: float = 0.1,
        **kwargs,
    ) -> "PipelineBuilder":
        """Add training stage."""
        stage = TrainingStage(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_split,
            **kwargs,
        )
        self._stages.append(stage)
        return self

    def evaluate(
        self,
        metrics: Optional[List[Callable]] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> "PipelineBuilder":
        """Add evaluation stage."""
        stage = EvaluationStage(
            metrics_fn=metrics,
            threshold=threshold,
            **kwargs,
        )
        self._stages.append(stage)
        return self

    def infer(
        self,
        model: Optional[Any] = None,
        batch_size: int = 32,
        post_process_fn: Optional[Callable] = None,
        **kwargs,
    ) -> "PipelineBuilder":
        """Add inference stage."""
        stage = InferenceStage(
            model=model,
            batch_size=batch_size,
            post_process_fn=post_process_fn,
            **kwargs,
        )
        self._stages.append(stage)
        return self

    def with_context(self, **kwargs) -> "PipelineBuilder":
        """Set initial context values."""
        self._context.update(kwargs)
        return self

    def on_start(self, callback: Callable) -> "PipelineBuilder":
        """Register on_start callback."""
        self._callbacks.setdefault("on_start", []).append(callback)
        return self

    def on_stage_complete(self, callback: Callable) -> "PipelineBuilder":
        """Register on_stage_complete callback."""
        self._callbacks.setdefault("on_stage_complete", []).append(callback)
        return self

    def on_complete(self, callback: Callable) -> "PipelineBuilder":
        """Register on_complete callback."""
        self._callbacks.setdefault("on_complete", []).append(callback)
        return self

    def on_error(self, callback: Callable) -> "PipelineBuilder":
        """Register on_error callback."""
        self._callbacks.setdefault("on_error", []).append(callback)
        return self

    def build(self) -> Pipeline:
        """Build the pipeline."""
        if not self._config:
            self._config = PipelineConfig(
                name=self._name,
                description=self._description,
            )

        pipeline = Pipeline(self._config)

        # Add stages
        for stage in self._stages:
            pipeline.add_stage(stage)

        # Set context
        for key, value in self._context.items():
            pipeline.set_context(key, value)

        # Register callbacks
        for event, callbacks in self._callbacks.items():
            for callback in callbacks:
                pipeline.register_callback(event, callback)

        return pipeline

    def run(self, input_data: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Build and run the pipeline."""
        pipeline = self.build()
        return pipeline.run(input_data)


def create_training_pipeline(
    data_path: Union[str, Path],
    model: Any,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    transforms: Optional[List[Callable]] = None,
    feature_extractors: Optional[List[Callable]] = None,
    validation_split: float = 0.1,
    name: str = "training_pipeline",
) -> Pipeline:
    """
    Create a standard training pipeline.

    Args:
        data_path: Path to training data
        model: Model to train
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        transforms: Preprocessing transforms
        feature_extractors: Feature extraction functions
        validation_split: Validation split ratio
        name: Pipeline name

    Returns:
        Configured Pipeline
    """
    return (
        PipelineBuilder(name)
        .description("Standard training pipeline")
        .config(stop_on_error=True, verbose=True)
        .load_data(data_path=data_path)
        .preprocess(transforms=transforms, normalize=True)
        .extract_features(extractors=feature_extractors)
        .train(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_split,
        )
        .evaluate()
        .build()
    )


def create_inference_pipeline(
    model: Any,
    batch_size: int = 32,
    transforms: Optional[List[Callable]] = None,
    feature_extractors: Optional[List[Callable]] = None,
    post_process_fn: Optional[Callable] = None,
    name: str = "inference_pipeline",
) -> Pipeline:
    """
    Create a standard inference pipeline.

    Args:
        model: Trained model for inference
        batch_size: Batch size
        transforms: Preprocessing transforms
        feature_extractors: Feature extraction functions
        post_process_fn: Post-processing function
        name: Pipeline name

    Returns:
        Configured Pipeline
    """
    return (
        PipelineBuilder(name)
        .description("Standard inference pipeline")
        .config(stop_on_error=True, verbose=False)
        .preprocess(transforms=transforms, normalize=True)
        .extract_features(extractors=feature_extractors)
        .infer(
            model=model,
            batch_size=batch_size,
            post_process_fn=post_process_fn,
        )
        .build()
    )


def create_evaluation_pipeline(
    model: Any,
    data_path: Union[str, Path],
    metrics: Optional[List[Callable]] = None,
    transforms: Optional[List[Callable]] = None,
    feature_extractors: Optional[List[Callable]] = None,
    name: str = "evaluation_pipeline",
) -> Pipeline:
    """
    Create a model evaluation pipeline.

    Args:
        model: Model to evaluate
        data_path: Path to evaluation data
        metrics: List of metric functions
        transforms: Preprocessing transforms
        feature_extractors: Feature extraction functions
        name: Pipeline name

    Returns:
        Configured Pipeline
    """
    return (
        PipelineBuilder(name)
        .description("Model evaluation pipeline")
        .config(stop_on_error=True, verbose=True)
        .load_data(data_path=data_path)
        .preprocess(transforms=transforms, normalize=True)
        .extract_features(extractors=feature_extractors)
        .with_context(trained_model=model)
        .evaluate(metrics=metrics)
        .build()
    )


def create_cad_pipeline(
    model: Any,
    data_path: Optional[Union[str, Path]] = None,
    mode: str = "train",
    epochs: int = 10,
    batch_size: int = 32,
    name: str = "cad_pipeline",
) -> Pipeline:
    """
    Create a CAD-specific ML pipeline.

    Args:
        model: Model for training/inference
        data_path: Path to CAD data
        mode: Pipeline mode ('train', 'inference', 'evaluate')
        epochs: Training epochs (for train mode)
        batch_size: Batch size
        name: Pipeline name

    Returns:
        Configured Pipeline
    """
    builder = PipelineBuilder(name).description(f"CAD {mode} pipeline")

    if mode == "train":
        return (
            builder
            .config(stop_on_error=True, save_intermediate=True)
            .load_data(data_path=data_path)
            .preprocess(normalize=True)
            .extract_features()
            .train(model=model, epochs=epochs, batch_size=batch_size)
            .evaluate()
            .build()
        )
    elif mode == "inference":
        return (
            builder
            .config(stop_on_error=True, verbose=False)
            .preprocess(normalize=True)
            .extract_features()
            .infer(model=model, batch_size=batch_size)
            .build()
        )
    else:  # evaluate
        return (
            builder
            .config(stop_on_error=True)
            .load_data(data_path=data_path)
            .preprocess(normalize=True)
            .extract_features()
            .with_context(trained_model=model)
            .evaluate()
            .build()
        )
