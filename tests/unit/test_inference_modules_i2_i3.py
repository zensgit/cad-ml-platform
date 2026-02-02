"""
Tests for Inference modules I2-I3.

Covers:
- I2: Batch processing optimization
- I3: REST/gRPC API, versioning, A/B testing
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# I2: Batch Processing Optimization Tests
# ============================================================================

class TestI2BatchProcessing:
    """Tests for I2 batch processing module."""

    def test_gpu_imports(self):
        """Test GPU module imports."""
        from src.ml.serving import (
            GPUManager,
            GPUConfig,
            GPUInfo,
            MixedPrecisionInference,
        )
        assert GPUManager is not None
        assert GPUConfig is not None

    def test_gpu_config_creation(self):
        """Test GPUConfig creation."""
        from src.ml.serving.gpu import GPUConfig

        config = GPUConfig(
            device_ids=[0, 1],
            memory_fraction=0.8,
            enable_mixed_precision=True,
        )
        assert config.device_ids == [0, 1]
        assert config.memory_fraction == 0.8

    def test_async_queue_imports(self):
        """Test async queue imports."""
        from src.ml.serving import (
            AsyncInferenceQueue,
            QueueConfig,
            QueueStats,
            BatchAccumulator,
        )
        assert AsyncInferenceQueue is not None
        assert QueueConfig is not None

    def test_queue_config_creation(self):
        """Test QueueConfig creation."""
        from src.ml.serving.async_queue import QueueConfig

        config = QueueConfig(
            max_queue_size=1000,
            batch_wait_ms=100,
            max_concurrent=32,
        )
        assert config.max_queue_size == 1000
        assert config.batch_wait_ms == 100

    def test_batch_optimizer_imports(self):
        """Test batch optimizer imports."""
        from src.ml.serving import (
            BatchOptimizer,
            BatchOptimizerConfig,
            BatchStrategy,
            BatchPadder,
        )
        assert BatchOptimizer is not None
        assert BatchStrategy is not None

    def test_batch_optimizer_config(self):
        """Test BatchOptimizerConfig creation."""
        from src.ml.serving.batch_optimizer import BatchOptimizerConfig, BatchStrategy

        config = BatchOptimizerConfig(
            strategy=BatchStrategy.ADAPTIVE,
            min_batch_size=1,
            max_batch_size=64,
        )
        assert config.strategy == BatchStrategy.ADAPTIVE
        assert config.max_batch_size == 64


# ============================================================================
# I3: REST API Tests
# ============================================================================

class TestI3RestAPI:
    """Tests for I3 REST API module."""

    def test_rest_api_imports(self):
        """Test REST API imports."""
        from src.ml.serving import (
            InferenceRESTAPI,
            APIConfig,
            InferenceAPIRequest,
            InferenceAPIResponse,
            create_inference_api,
        )
        assert InferenceRESTAPI is not None
        assert APIConfig is not None

    def test_api_config_creation(self):
        """Test APIConfig creation."""
        from src.ml.serving.rest_api import APIConfig

        config = APIConfig(
            host="0.0.0.0",
            port=8000,
            workers=4,
            enable_cors=True,
        )
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 4

    def test_inference_request_creation(self):
        """Test InferenceAPIRequest creation."""
        from src.ml.serving.rest_api import InferenceAPIRequest

        request = InferenceAPIRequest(
            model_name="test_model",
            model_version="1.0",
            inputs={"data": [1, 2, 3]},
        )
        assert request.model_name == "test_model"
        assert request.model_version == "1.0"

    def test_inference_response_creation(self):
        """Test InferenceAPIResponse creation."""
        from src.ml.serving.rest_api import InferenceAPIResponse

        response = InferenceAPIResponse(
            request_id="abc123",
            model_name="test_model",
            model_version="1.0",
            predictions=[{"class": "A", "score": 0.9}],
            latency_ms=10.5,
        )
        assert response.request_id == "abc123"
        assert response.latency_ms == 10.5

    def test_create_inference_api(self):
        """Test create_inference_api function."""
        from src.ml.serving.rest_api import create_inference_api, APIConfig

        config = APIConfig(port=8080)
        api = create_inference_api(config)
        assert api is not None


# ============================================================================
# I3: gRPC Service Tests
# ============================================================================

class TestI3GRPCService:
    """Tests for I3 gRPC service module."""

    def test_grpc_imports(self):
        """Test gRPC imports."""
        from src.ml.serving import (
            InferenceGRPCService,
            GRPCConfig,
            GRPCRequest,
            GRPCResponse,
            create_grpc_service,
        )
        assert InferenceGRPCService is not None
        assert GRPCConfig is not None

    def test_grpc_config_creation(self):
        """Test GRPCConfig creation."""
        from src.ml.serving.grpc_service import GRPCConfig

        config = GRPCConfig(
            host="0.0.0.0",
            port=50051,
            max_workers=10,
            compression="gzip",
        )
        assert config.port == 50051
        assert config.compression == "gzip"

    def test_grpc_request_creation(self):
        """Test GRPCRequest creation."""
        from src.ml.serving.grpc_service import GRPCRequest

        request = GRPCRequest(
            model_name="test_model",
            inputs=b"test_data",
            input_format="tensor",
        )
        assert request.model_name == "test_model"
        assert request.input_format == "tensor"

    def test_grpc_service_creation(self):
        """Test gRPC service creation."""
        from src.ml.serving.grpc_service import create_grpc_service, GRPCConfig

        config = GRPCConfig()
        service = create_grpc_service(config)
        assert service is not None

    def test_grpc_proto_generation(self):
        """Test proto file generation."""
        from src.ml.serving.grpc_service import InferenceGRPCService

        service = InferenceGRPCService()
        proto = service.generate_proto()
        assert "InferenceService" in proto
        assert "PredictRequest" in proto
        assert "PredictResponse" in proto


# ============================================================================
# I3: Version Manager Tests
# ============================================================================

class TestI3VersionManager:
    """Tests for I3 version manager module."""

    def test_version_manager_imports(self):
        """Test version manager imports."""
        from src.ml.serving import (
            ModelVersionManager,
            VersionManagerConfig,
            ModelVersion,
            RegisteredModel,
            ModelStage,
        )
        assert ModelVersionManager is not None
        assert ModelStage is not None

    def test_model_stage_enum(self):
        """Test ModelStage enum."""
        from src.ml.serving.version_manager import ModelStage

        assert ModelStage.DEVELOPMENT.value == "development"
        assert ModelStage.STAGING.value == "staging"
        assert ModelStage.PRODUCTION.value == "production"

    def test_model_version_creation(self):
        """Test ModelVersion creation."""
        from src.ml.serving.version_manager import ModelVersion, ModelStage

        version = ModelVersion(
            model_name="test_model",
            version="1.0.0",
            stage=ModelStage.DEVELOPMENT,
            metrics={"accuracy": 0.95},
        )
        assert version.full_name == "test_model:1.0.0"
        assert version.metrics["accuracy"] == 0.95

    def test_version_manager_creation(self):
        """Test ModelVersionManager creation."""
        from src.ml.serving.version_manager import ModelVersionManager, VersionManagerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = VersionManagerConfig(
                registry_path=f"{tmpdir}/registry",
                artifact_store_path=f"{tmpdir}/artifacts",
            )
            manager = ModelVersionManager(config)
            assert manager is not None

    def test_register_model(self):
        """Test model registration."""
        from src.ml.serving.version_manager import ModelVersionManager, VersionManagerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = VersionManagerConfig(
                registry_path=f"{tmpdir}/registry",
                artifact_store_path=f"{tmpdir}/artifacts",
            )
            manager = ModelVersionManager(config)

            model = manager.register_model("test_model", "Test description")
            assert model.name == "test_model"

    def test_register_version(self):
        """Test version registration."""
        from src.ml.serving.version_manager import ModelVersionManager, VersionManagerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = VersionManagerConfig(
                registry_path=f"{tmpdir}/registry",
                artifact_store_path=f"{tmpdir}/artifacts",
            )
            manager = ModelVersionManager(config)

            version = manager.register_version(
                model_name="test_model",
                version="1.0.0",
                metrics={"accuracy": 0.95},
            )
            assert version.version == "1.0.0"


# ============================================================================
# I3: A/B Testing Tests
# ============================================================================

class TestI3ABTesting:
    """Tests for I3 A/B testing module."""

    def test_ab_testing_imports(self):
        """Test A/B testing imports."""
        from src.ml.serving import (
            ABTestManager,
            ABTestConfig,
            Experiment,
            Variant,
            ExperimentStatus,
            TrafficSplitStrategy,
        )
        assert ABTestManager is not None
        assert Experiment is not None

    def test_experiment_status_enum(self):
        """Test ExperimentStatus enum."""
        from src.ml.serving.ab_testing import ExperimentStatus

        assert ExperimentStatus.DRAFT.value == "draft"
        assert ExperimentStatus.RUNNING.value == "running"
        assert ExperimentStatus.COMPLETED.value == "completed"

    def test_variant_creation(self):
        """Test Variant creation."""
        from src.ml.serving.ab_testing import Variant

        variant = Variant(
            name="control",
            model_name="model_a",
            model_version="1.0",
            weight=0.5,
        )
        assert variant.name == "control"
        assert variant.weight == 0.5

    def test_variant_record_request(self):
        """Test recording requests in variant."""
        from src.ml.serving.ab_testing import Variant

        variant = Variant(
            name="control",
            model_name="model_a",
            model_version="1.0",
        )
        variant.record_request(success=True, latency_ms=10.0)
        variant.record_request(success=True, latency_ms=15.0)
        variant.record_request(success=False, latency_ms=20.0)

        assert variant.request_count == 3
        assert variant.success_count == 2
        assert variant.error_count == 1

    def test_ab_manager_creation(self):
        """Test ABTestManager creation."""
        from src.ml.serving.ab_testing import ABTestManager, ABTestConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ABTestConfig(storage_path=tmpdir)
            manager = ABTestManager(config)
            assert manager is not None

    def test_create_experiment(self):
        """Test experiment creation."""
        from src.ml.serving.ab_testing import ABTestManager, ABTestConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ABTestConfig(storage_path=tmpdir)
            manager = ABTestManager(config)

            exp = manager.create_experiment(
                name="test_experiment",
                variants=[
                    {"name": "control", "model_name": "model_a", "model_version": "1.0"},
                    {"name": "treatment", "model_name": "model_b", "model_version": "1.0"},
                ],
            )
            assert exp.name == "test_experiment"
            assert len(exp.variants) == 2


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
