"""
Tests for E2E Pipeline module.

Covers pipeline stages, orchestration, and builder.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Pipeline Stages Tests
# ============================================================================

class TestPipelineStages:
    """Tests for pipeline stages."""

    def test_stage_imports(self):
        """Test stage imports."""
        from src.ml.pipeline import (
            PipelineStage,
            DataLoadingStage,
            PreprocessingStage,
            FeatureExtractionStage,
            TrainingStage,
            EvaluationStage,
            InferenceStage,
        )
        assert PipelineStage is not None
        assert DataLoadingStage is not None

    def test_stage_status_enum(self):
        """Test StageStatus enum."""
        from src.ml.pipeline.stages import StageStatus

        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.RUNNING.value == "running"
        assert StageStatus.COMPLETED.value == "completed"
        assert StageStatus.FAILED.value == "failed"

    def test_stage_result_creation(self):
        """Test StageResult creation."""
        from src.ml.pipeline.stages import StageResult, StageStatus

        result = StageResult(
            stage_name="TestStage",
            status=StageStatus.COMPLETED,
            output={"data": [1, 2, 3]},
            execution_time=1.5,
        )
        assert result.success is True
        assert result.execution_time == 1.5

    def test_data_loading_stage_creation(self):
        """Test DataLoadingStage creation."""
        from src.ml.pipeline.stages import DataLoadingStage

        stage = DataLoadingStage(data_path="/path/to/data")
        assert stage.name == "DataLoading"

    def test_preprocessing_stage_creation(self):
        """Test PreprocessingStage creation."""
        from src.ml.pipeline.stages import PreprocessingStage

        stage = PreprocessingStage(normalize=True)
        assert stage.name == "Preprocessing"

    def test_feature_extraction_stage_creation(self):
        """Test FeatureExtractionStage creation."""
        from src.ml.pipeline.stages import FeatureExtractionStage

        stage = FeatureExtractionStage(feature_dim=128)
        assert stage.name == "FeatureExtraction"

    def test_training_stage_creation(self):
        """Test TrainingStage creation."""
        from src.ml.pipeline.stages import TrainingStage

        stage = TrainingStage(
            epochs=10,
            batch_size=32,
            learning_rate=1e-3,
        )
        assert stage.name == "Training"

    def test_evaluation_stage_creation(self):
        """Test EvaluationStage creation."""
        from src.ml.pipeline.stages import EvaluationStage

        stage = EvaluationStage(threshold=0.5)
        assert stage.name == "Evaluation"

    def test_inference_stage_creation(self):
        """Test InferenceStage creation."""
        from src.ml.pipeline.stages import InferenceStage

        stage = InferenceStage(batch_size=32)
        assert stage.name == "Inference"


# ============================================================================
# Pipeline Orchestrator Tests
# ============================================================================

class TestPipelineOrchestrator:
    """Tests for pipeline orchestrator."""

    def test_orchestrator_imports(self):
        """Test orchestrator imports."""
        from src.ml.pipeline import (
            Pipeline,
            PipelineConfig,
            PipelineResult,
            PipelineStatus,
        )
        assert Pipeline is not None
        assert PipelineConfig is not None

    def test_pipeline_status_enum(self):
        """Test PipelineStatus enum."""
        from src.ml.pipeline.orchestrator import PipelineStatus

        assert PipelineStatus.PENDING.value == "pending"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.FAILED.value == "failed"

    def test_pipeline_config_creation(self):
        """Test PipelineConfig creation."""
        from src.ml.pipeline.orchestrator import PipelineConfig

        config = PipelineConfig(
            name="test_pipeline",
            description="Test pipeline",
            stop_on_error=True,
            max_retries=3,
        )
        assert config.name == "test_pipeline"
        assert config.max_retries == 3

    def test_pipeline_result_creation(self):
        """Test PipelineResult creation."""
        from src.ml.pipeline.orchestrator import PipelineResult, PipelineStatus

        result = PipelineResult(
            pipeline_name="test",
            status=PipelineStatus.COMPLETED,
            total_time=10.5,
        )
        assert result.success is True
        assert result.total_time == 10.5

    def test_pipeline_creation(self):
        """Test Pipeline creation."""
        from src.ml.pipeline.orchestrator import Pipeline, PipelineConfig, PipelineStatus

        config = PipelineConfig(name="test")
        pipeline = Pipeline(config)
        assert pipeline.status == PipelineStatus.PENDING

    def test_pipeline_add_stage(self):
        """Test adding stages to pipeline."""
        from src.ml.pipeline.orchestrator import Pipeline
        from src.ml.pipeline.stages import PreprocessingStage

        pipeline = Pipeline()
        stage = PreprocessingStage()
        pipeline.add_stage(stage)

        assert len(pipeline.stages) == 1

    def test_pipeline_context(self):
        """Test pipeline context management."""
        from src.ml.pipeline.orchestrator import Pipeline

        pipeline = Pipeline()
        pipeline.set_context("model", "test_model")
        pipeline.set_context("device", "cuda")

        assert pipeline.context["model"] == "test_model"
        assert pipeline.context["device"] == "cuda"

    def test_pipeline_callback_registration(self):
        """Test callback registration."""
        from src.ml.pipeline.orchestrator import Pipeline

        pipeline = Pipeline()
        callback = Mock()

        pipeline.register_callback("on_start", callback)
        pipeline.register_callback("on_complete", callback)

        assert len(pipeline._callbacks["on_start"]) == 1
        assert len(pipeline._callbacks["on_complete"]) == 1


# ============================================================================
# Pipeline Builder Tests
# ============================================================================

class TestPipelineBuilder:
    """Tests for pipeline builder."""

    def test_builder_imports(self):
        """Test builder imports."""
        from src.ml.pipeline import (
            PipelineBuilder,
            create_training_pipeline,
            create_inference_pipeline,
        )
        assert PipelineBuilder is not None
        assert create_training_pipeline is not None

    def test_pipeline_builder_creation(self):
        """Test PipelineBuilder creation."""
        from src.ml.pipeline.builder import PipelineBuilder

        builder = PipelineBuilder("test_pipeline")
        assert builder._name == "test_pipeline"

    def test_builder_fluent_api(self):
        """Test builder fluent API."""
        from src.ml.pipeline.builder import PipelineBuilder

        builder = (
            PipelineBuilder("test")
            .name("my_pipeline")
            .description("Test pipeline")
            .config(stop_on_error=True, max_retries=2)
        )
        assert builder._name == "my_pipeline"
        assert builder._description == "Test pipeline"

    def test_builder_add_stages(self):
        """Test adding stages via builder."""
        from src.ml.pipeline.builder import PipelineBuilder

        builder = (
            PipelineBuilder("test")
            .preprocess(normalize=True)
            .extract_features(feature_dim=128)
        )
        assert len(builder._stages) == 2

    def test_builder_build(self):
        """Test building pipeline."""
        from src.ml.pipeline.builder import PipelineBuilder

        pipeline = (
            PipelineBuilder("test")
            .preprocess()
            .build()
        )
        assert pipeline is not None
        assert len(pipeline.stages) == 1

    def test_builder_with_context(self):
        """Test builder context setting."""
        from src.ml.pipeline.builder import PipelineBuilder

        pipeline = (
            PipelineBuilder("test")
            .with_context(model="test_model", device="cuda")
            .preprocess()
            .build()
        )
        assert pipeline.context["model"] == "test_model"
        assert pipeline.context["device"] == "cuda"

    def test_builder_callbacks(self):
        """Test builder callback registration."""
        from src.ml.pipeline.builder import PipelineBuilder

        callback = Mock()
        pipeline = (
            PipelineBuilder("test")
            .on_start(callback)
            .on_complete(callback)
            .preprocess()
            .build()
        )
        assert len(pipeline._callbacks["on_start"]) == 1
        assert len(pipeline._callbacks["on_complete"]) == 1


# ============================================================================
# Factory Functions Tests
# ============================================================================

class TestPipelineFactories:
    """Tests for pipeline factory functions."""

    def test_create_training_pipeline_signature(self):
        """Test create_training_pipeline function signature."""
        from src.ml.pipeline.builder import create_training_pipeline
        import inspect

        sig = inspect.signature(create_training_pipeline)
        params = list(sig.parameters.keys())

        assert "data_path" in params
        assert "model" in params
        assert "epochs" in params

    def test_create_inference_pipeline_signature(self):
        """Test create_inference_pipeline function signature."""
        from src.ml.pipeline.builder import create_inference_pipeline
        import inspect

        sig = inspect.signature(create_inference_pipeline)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "batch_size" in params


# ============================================================================
# Integration Tests
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for pipeline module."""

    def test_simple_pipeline_execution(self):
        """Test simple pipeline execution."""
        from src.ml.pipeline.orchestrator import Pipeline
        from src.ml.pipeline.stages import PreprocessingStage

        pipeline = Pipeline()
        pipeline.add_stage(PreprocessingStage(transforms=[], normalize=False))

        # Run with mock data
        result = pipeline.run({"data": {"key": "value"}})

        assert result is not None
        assert len(result.stage_results) == 1

    def test_multi_stage_pipeline(self):
        """Test multi-stage pipeline."""
        from src.ml.pipeline.builder import PipelineBuilder

        pipeline = (
            PipelineBuilder("multi_stage")
            .preprocess(normalize=False)
            .extract_features(extractors=[])
            .build()
        )

        result = pipeline.run({"data": {"key": "value"}})
        assert len(result.stage_results) == 2


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
