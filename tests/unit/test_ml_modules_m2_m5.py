"""
Tests for ML modules M2-M5.

Covers:
- M2: Hyperparameter tuning
- M3: Experiment tracking
- M4: Data augmentation
- M5: Model compression
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# M2: Hyperparameter Tuning Tests
# ============================================================================

class TestM2HyperparameterTuning:
    """Tests for M2 hyperparameter tuning module."""

    def test_search_space_imports(self):
        """Test search space imports."""
        from src.ml.tuning import (
            SearchSpace,
            IntParam,
            FloatParam,
            CategoricalParam,
        )
        assert SearchSpace is not None
        assert IntParam is not None

    def test_int_param_creation(self):
        """Test IntParam creation."""
        from src.ml.tuning.search_space import IntParam

        param = IntParam(name="batch_size", low=16, high=128, step=16)
        assert param.name == "batch_size"
        assert param.low == 16
        assert param.high == 128

    def test_float_param_creation(self):
        """Test FloatParam creation."""
        from src.ml.tuning.search_space import FloatParam

        param = FloatParam(name="learning_rate", low=1e-5, high=1e-2, log=True)
        assert param.name == "learning_rate"
        assert param.log is True

    def test_categorical_param_creation(self):
        """Test CategoricalParam creation."""
        from src.ml.tuning.search_space import CategoricalParam

        param = CategoricalParam(name="optimizer", choices=["adam", "sgd", "adamw"])
        assert param.name == "optimizer"
        assert "adam" in param.choices

    def test_search_space_add_params(self):
        """Test adding parameters to search space."""
        from src.ml.tuning.search_space import SearchSpace

        space = SearchSpace()
        space.add_int("epochs", 10, 100)
        space.add_float("lr", 1e-5, 1e-2, log=True)

        assert len(space.params) == 2

    def test_optimization_config(self):
        """Test optimization config creation."""
        from src.ml.tuning.optimizer import OptimizationConfig

        config = OptimizationConfig(
            n_trials=50,
            timeout=3600,
            direction="minimize",
        )
        assert config.n_trials == 50
        assert config.direction == "minimize"

    def test_tuning_context_creation(self):
        """Test TuningContext creation."""
        from src.ml.tuning.integration import TuningContext
        from src.ml.tuning.search_space import SearchSpace
        from src.ml.tuning.optimizer import HyperOptimizer, OptimizationConfig

        space = SearchSpace()
        space.add_int("epochs", 10, 100)
        config = OptimizationConfig(n_trials=10)
        optimizer = HyperOptimizer(space, config)

        ctx = TuningContext(
            search_space=space,
            optimizer=optimizer,
        )
        assert ctx.search_space is space


# ============================================================================
# M3: Experiment Tracking Tests
# ============================================================================

class TestM3ExperimentTracking:
    """Tests for M3 experiment tracking module."""

    def test_experiment_imports(self):
        """Test experiment tracking imports."""
        from src.ml.experiment import (
            ExperimentTracker,
            Run,
        )
        from src.ml.experiment.tracker import TrackerConfig
        assert ExperimentTracker is not None
        assert Run is not None

    def test_tracker_config(self):
        """Test TrackerConfig creation."""
        from src.ml.experiment.tracker import TrackerConfig

        config = TrackerConfig(
            base_path="./mlruns",
            auto_log_git=True,
        )
        assert config.base_path == "./mlruns"

    def test_metric_logging(self):
        """Test metric logging."""
        from src.ml.experiment import ExperimentTracker
        from src.ml.experiment.tracker import TrackerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrackerConfig(base_path=tmpdir)
            tracker = ExperimentTracker(config)

            run = tracker.start_run(experiment_name="test_exp", run_name="test_run")
            tracker.log_metric("accuracy", 0.95)
            assert "accuracy" in run.metrics

    def test_tracker_initialization(self):
        """Test ExperimentTracker initialization."""
        from src.ml.experiment import ExperimentTracker
        from src.ml.experiment.tracker import TrackerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrackerConfig(base_path=tmpdir)
            tracker = ExperimentTracker(config)
            assert tracker is not None

    def test_run_lifecycle(self):
        """Test run start and end."""
        from src.ml.experiment import ExperimentTracker
        from src.ml.experiment.tracker import TrackerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrackerConfig(base_path=tmpdir)
            tracker = ExperimentTracker(config)

            run = tracker.start_run(experiment_name="lifecycle_test", run_name="test_run")
            assert run is not None
            assert run.status == "running"

            tracker.end_run()


# ============================================================================
# M4: Data Augmentation Tests
# ============================================================================

class TestM4DataAugmentation:
    """Tests for M4 data augmentation module."""

    def test_augmentation_imports(self):
        """Test augmentation imports."""
        from src.ml.augmentation import (
            RandomRotation,
            RandomScale,
            RandomFlip,
            Compose,
            AugmentationPipeline,
        )
        assert RandomRotation is not None
        assert Compose is not None

    def test_random_rotation_creation(self):
        """Test RandomRotation creation."""
        from src.ml.augmentation.geometric import RandomRotation

        aug = RandomRotation(angle_range=(-15, 15), p=0.5)
        assert aug.p == 0.5

    def test_random_scale_creation(self):
        """Test RandomScale creation."""
        from src.ml.augmentation.geometric import RandomScale

        aug = RandomScale(scale_range=(0.9, 1.1), p=0.5)
        assert aug.p == 0.5

    def test_compose_augmentations(self):
        """Test composing augmentations."""
        from src.ml.augmentation.pipeline import Compose
        from src.ml.augmentation.geometric import RandomRotation, RandomScale

        pipeline = Compose([
            RandomRotation(angle_range=(-10, 10), p=1.0),
            RandomScale(scale_range=(0.95, 1.05), p=1.0),
        ])
        assert len(pipeline.augmentations) == 2

    def test_augmentation_pipeline_creation(self):
        """Test AugmentationPipeline creation."""
        from src.ml.augmentation.pipeline import AugmentationPipeline, AugmentationConfig

        config = AugmentationConfig(enabled=True, intensity="medium")
        pipeline = AugmentationPipeline(config)
        assert pipeline.enabled is True

    def test_cad_augmentation_layer_shuffle(self):
        """Test CAD-specific LayerShuffle."""
        from src.ml.augmentation.cad import LayerShuffle

        aug = LayerShuffle(shuffle_rate=0.1, p=0.5)
        assert aug.shuffle_rate == 0.1

    def test_cad_augmentation_entity_dropout(self):
        """Test CAD-specific EntityDropout."""
        from src.ml.augmentation.cad import EntityDropout

        aug = EntityDropout(dropout_rate=0.1, preserve_types=["LINE"], p=0.5)
        assert aug.dropout_rate == 0.1
        assert "LINE" in aug.preserve_types

    def test_graph_node_dropout(self):
        """Test graph NodeDropout."""
        from src.ml.augmentation.graph import NodeDropout

        aug = NodeDropout(dropout_rate=0.1, p=0.5)
        assert aug.dropout_rate == 0.1


# ============================================================================
# M5: Model Compression Tests
# ============================================================================

class TestM5ModelCompression:
    """Tests for M5 model compression module."""

    def test_compression_imports(self):
        """Test compression imports."""
        from src.ml.compression import (
            Quantizer,
            QuantizationConfig,
            Pruner,
            PruningConfig,
            DistillationTrainer,
            ONNXExporter,
        )
        assert Quantizer is not None
        assert Pruner is not None

    def test_quantization_config(self):
        """Test QuantizationConfig creation."""
        from src.ml.compression.quantization import QuantizationConfig, QuantizationMode

        config = QuantizationConfig(
            mode=QuantizationMode.DYNAMIC,
            dtype="qint8",
        )
        assert config.mode == QuantizationMode.DYNAMIC
        assert config.dtype == "qint8"

    def test_quantizer_creation(self):
        """Test Quantizer creation."""
        from src.ml.compression.quantization import Quantizer, QuantizationConfig

        config = QuantizationConfig()
        quantizer = Quantizer(config)
        assert quantizer.config == config

    def test_pruning_config(self):
        """Test PruningConfig creation."""
        from src.ml.compression.pruning import PruningConfig, PruningMethod

        config = PruningConfig(
            method=PruningMethod.MAGNITUDE,
            sparsity=0.5,
            structured=False,
        )
        assert config.method == PruningMethod.MAGNITUDE
        assert config.sparsity == 0.5

    def test_pruner_creation(self):
        """Test Pruner creation."""
        from src.ml.compression.pruning import Pruner, PruningConfig

        config = PruningConfig(sparsity=0.3)
        pruner = Pruner(config)
        assert pruner.config.sparsity == 0.3

    def test_distillation_config(self):
        """Test DistillationConfig creation."""
        from src.ml.compression.distillation import DistillationConfig, DistillationLoss

        config = DistillationConfig(
            temperature=4.0,
            alpha=0.5,
            loss_type=DistillationLoss.KL_DIVERGENCE,
        )
        assert config.temperature == 4.0
        assert config.alpha == 0.5

    def test_distillation_trainer_creation(self):
        """Test DistillationTrainer creation."""
        from src.ml.compression.distillation import DistillationTrainer, DistillationConfig

        config = DistillationConfig()
        trainer = DistillationTrainer(config)
        assert trainer.config == config

    def test_export_config(self):
        """Test ExportConfig creation."""
        from src.ml.compression.export import ExportConfig, ExportFormat

        config = ExportConfig(
            format=ExportFormat.ONNX,
            opset_version=14,
            optimize=True,
        )
        assert config.format == ExportFormat.ONNX
        assert config.opset_version == 14

    def test_onnx_exporter_creation(self):
        """Test ONNXExporter creation."""
        from src.ml.compression.export import ONNXExporter, ExportConfig

        config = ExportConfig()
        exporter = ONNXExporter(config)
        assert exporter.config == config


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
