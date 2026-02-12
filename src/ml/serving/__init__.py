"""
Model Serving Module (I1 + I2 + I3).

Provides high-performance model inference serving:
- Multi-model parallel serving
- Dynamic model loading/unloading
- Request load balancing
- Batch inference optimization
- GPU-optimized inference (I2)
- Async inference queue (I2)
- REST/gRPC API (I3)
- Model versioning (I3)
- A/B testing (I3)
"""

from __future__ import annotations

from src.ml.serving.server import (
    ModelServer,
    ServerConfig,
    get_model_server,
)
from src.ml.serving.worker import (
    ModelWorker,
    WorkerConfig,
    WorkerStats,
)
from src.ml.serving.router import (
    RequestRouter,
    RoutingStrategy,
    RouteConfig,
)
from src.ml.serving.batch import (
    DynamicBatcher,
    BatchConfig,
    BatchRequest,
    BatchResult,
)
from src.ml.serving.health import (
    HealthChecker,
    HealthStatus,
    ModelHealth,
)
from src.ml.serving.request import (
    InferenceRequest,
    InferenceResponse,
    Prediction,
)

# I2 - Inference Optimization
from src.ml.serving.gpu import (
    GPUManager,
    GPUConfig,
    GPUInfo,
    MixedPrecisionInference,
    get_gpu_manager,
    get_best_device,
)
from src.ml.serving.async_queue import (
    AsyncInferenceQueue,
    QueueConfig,
    QueueStats,
    QueueState,
    BatchAccumulator,
)
from src.ml.serving.batch_optimizer import (
    BatchOptimizer,
    BatchOptimizerConfig,
    BatchStrategy,
    BatchMetrics,
    BatchPadder,
    BatchInferenceRunner,
)

# I3 - REST API
from src.ml.serving.rest_api import (
    InferenceRESTAPI,
    APIConfig,
    InferenceAPIRequest,
    InferenceAPIResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    ModelInfo,
    create_inference_api,
)

# I3 - gRPC Service
from src.ml.serving.grpc_service import (
    InferenceGRPCService,
    GRPCConfig,
    GRPCRequest,
    GRPCResponse,
    StreamingRequest,
    StreamingResponse,
    create_grpc_service,
)

# I3 - Model Version Management
from src.ml.serving.version_manager import (
    ModelVersionManager,
    VersionManagerConfig,
    ModelVersion,
    RegisteredModel,
    ModelStage,
    VersionStatus,
    get_version_manager,
)

# I3 - A/B Testing
from src.ml.serving.ab_testing import (
    ABTestManager,
    ABTestConfig,
    Experiment,
    Variant,
    ExperimentStatus,
    TrafficSplitStrategy,
    WinnerCriterion,
    get_ab_test_manager,
)

__all__ = [
    # Server
    "ModelServer",
    "ServerConfig",
    "get_model_server",
    # Worker
    "ModelWorker",
    "WorkerConfig",
    "WorkerStats",
    # Router
    "RequestRouter",
    "RoutingStrategy",
    "RouteConfig",
    # Batch
    "DynamicBatcher",
    "BatchConfig",
    "BatchRequest",
    "BatchResult",
    # Health
    "HealthChecker",
    "HealthStatus",
    "ModelHealth",
    # Request/Response
    "InferenceRequest",
    "InferenceResponse",
    "Prediction",
    # I2 - GPU
    "GPUManager",
    "GPUConfig",
    "GPUInfo",
    "MixedPrecisionInference",
    "get_gpu_manager",
    "get_best_device",
    # I2 - Async Queue
    "AsyncInferenceQueue",
    "QueueConfig",
    "QueueStats",
    "QueueState",
    "BatchAccumulator",
    # I2 - Batch Optimizer
    "BatchOptimizer",
    "BatchOptimizerConfig",
    "BatchStrategy",
    "BatchMetrics",
    "BatchPadder",
    "BatchInferenceRunner",
    # I3 - REST API
    "InferenceRESTAPI",
    "APIConfig",
    "InferenceAPIRequest",
    "InferenceAPIResponse",
    "BatchInferenceRequest",
    "BatchInferenceResponse",
    "ModelInfo",
    "create_inference_api",
    # I3 - gRPC Service
    "InferenceGRPCService",
    "GRPCConfig",
    "GRPCRequest",
    "GRPCResponse",
    "StreamingRequest",
    "StreamingResponse",
    "create_grpc_service",
    # I3 - Model Version Management
    "ModelVersionManager",
    "VersionManagerConfig",
    "ModelVersion",
    "RegisteredModel",
    "ModelStage",
    "VersionStatus",
    "get_version_manager",
    # I3 - A/B Testing
    "ABTestManager",
    "ABTestConfig",
    "Experiment",
    "Variant",
    "ExperimentStatus",
    "TrafficSplitStrategy",
    "WinnerCriterion",
    "get_ab_test_manager",
]
