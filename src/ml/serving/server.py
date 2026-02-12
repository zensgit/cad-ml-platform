"""
Model Server - Main interface for model serving.

Provides:
- Unified model serving interface
- Multi-model management
- Request routing and batching
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.ml.serving.worker import ModelWorker, WorkerConfig, WorkerStats
from src.ml.serving.router import RequestRouter, RouteConfig, RoutingStrategy
from src.ml.serving.batch import DynamicBatcher, BatchConfig, BatchRequest, BatchResult
from src.ml.serving.health import HealthChecker, HealthStatus, ModelHealth
from src.ml.serving.request import (
    InferenceRequest,
    InferenceResponse,
    Prediction,
    RequestPriority,
)

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for model server."""
    max_models: int = 10
    default_device: str = "cpu"
    enable_batching: bool = True
    batch_config: Optional[BatchConfig] = None
    route_config: Optional[RouteConfig] = None
    enable_health_checks: bool = True
    health_check_interval: float = 30.0
    warmup_on_load: bool = True


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    version: str
    path: str
    device: str
    is_loaded: bool
    stats: WorkerStats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "stats": self.stats.to_dict(),
        }


class ModelServer:
    """
    High-level model serving interface.

    Provides:
    - Model loading and management
    - Request routing
    - Batch inference
    - Health monitoring
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        """
        Initialize model server.

        Args:
            config: Server configuration
        """
        self._config = config or ServerConfig()
        self._workers: Dict[str, ModelWorker] = {}
        self._router = RequestRouter(self._config.route_config)
        self._batcher = DynamicBatcher(self._config.batch_config) if self._config.enable_batching else None
        self._health_checker = HealthChecker(
            check_interval=self._config.health_check_interval
        ) if self._config.enable_health_checks else None

        self._start_time = time.time()
        self._total_requests = 0

    @property
    def loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return [name for name, worker in self._workers.items() if worker.is_loaded]

    @property
    def model_count(self) -> int:
        """Get number of loaded models."""
        return len(self._workers)

    def load_model(
        self,
        model_path: str,
        model_name: str,
        version: str = "",
        device: Optional[str] = None,
        predict_fn: Optional[Callable] = None,
    ) -> ModelInfo:
        """
        Load a model into the server.

        Args:
            model_path: Path to model file
            model_name: Name for the model
            version: Model version
            device: Device to load on
            predict_fn: Custom prediction function

        Returns:
            ModelInfo
        """
        if len(self._workers) >= self._config.max_models:
            raise RuntimeError(f"Maximum models ({self._config.max_models}) already loaded")

        device = device or self._config.default_device
        worker_config = WorkerConfig(device=device)

        worker = ModelWorker(
            model_path=model_path,
            model_name=model_name,
            config=worker_config,
            predict_fn=predict_fn,
        )

        # Load and optionally warmup
        load_time = worker.load()

        if self._config.warmup_on_load:
            worker.warmup()

        self._workers[model_name] = worker

        # Register with router
        worker_id = f"{model_name}_0"
        self._router.register_worker(worker_id, model_name)

        # Register with health checker
        if self._health_checker:
            self._health_checker.register_model(
                model_name,
                lambda: worker.is_loaded,
            )

        # Register batch handler
        if self._batcher:
            self._batcher.register_handler(
                model_name,
                lambda batch: self._process_batch(batch),
            )

        logger.info(f"Loaded model {model_name} (version={version}, device={device})")

        return ModelInfo(
            name=model_name,
            version=version,
            path=model_path,
            device=device,
            is_loaded=True,
            stats=worker.stats,
        )

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from the server.

        Args:
            model_name: Name of model to unload

        Returns:
            True if unloaded
        """
        worker = self._workers.get(model_name)
        if worker is None:
            return False

        worker.unload()
        del self._workers[model_name]

        # Unregister from router
        self._router.unregister_worker(f"{model_name}_0")

        # Unregister from health checker
        if self._health_checker:
            self._health_checker.unregister_model(model_name)

        logger.info(f"Unloaded model {model_name}")
        return True

    def predict(
        self,
        model_name: str,
        inputs: List[Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 30.0,
    ) -> InferenceResponse:
        """
        Make predictions using a model.

        Args:
            model_name: Name of model to use
            inputs: Input data
            priority: Request priority
            timeout: Request timeout

        Returns:
            InferenceResponse
        """
        request = InferenceRequest(
            model_name=model_name,
            inputs=inputs,
            priority=priority,
            timeout=timeout,
        )

        return self.predict_request(request)

    def predict_request(self, request: InferenceRequest) -> InferenceResponse:
        """
        Process an inference request.

        Args:
            request: Inference request

        Returns:
            InferenceResponse
        """
        self._total_requests += 1
        start_time = time.time()

        # Get worker
        worker = self._workers.get(request.model_name)
        if worker is None:
            return InferenceResponse.error_response(
                request.request_id,
                request.model_name,
                f"Model '{request.model_name}' not loaded",
            )

        # Route request
        worker_id = self._router.route(request)
        if worker_id is None:
            return InferenceResponse.error_response(
                request.request_id,
                request.model_name,
                "No available workers",
            )

        try:
            # Execute prediction
            response = worker.predict(request)

            # Record health metrics
            latency = time.time() - start_time
            if self._health_checker:
                self._health_checker.record_request(
                    request.model_name,
                    response.success,
                    latency * 1000,
                )

            # Release worker
            self._router.release(worker_id, latency)

            return response

        except Exception as e:
            logger.error(f"Prediction error: {e}")

            if self._health_checker:
                self._health_checker.record_request(
                    request.model_name,
                    False,
                    (time.time() - start_time) * 1000,
                )

            return InferenceResponse.error_response(
                request.request_id,
                request.model_name,
                str(e),
            )

    async def predict_async(
        self,
        model_name: str,
        inputs: List[Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 30.0,
    ) -> InferenceResponse:
        """
        Async prediction with batching support.

        Args:
            model_name: Name of model
            inputs: Input data
            priority: Request priority
            timeout: Request timeout

        Returns:
            InferenceResponse
        """
        request = InferenceRequest(
            model_name=model_name,
            inputs=inputs,
            priority=priority,
            timeout=timeout,
        )

        # Use batching if enabled
        if self._batcher:
            self._batcher.submit(request)
            batch = self._batcher.get_batch(model_name, timeout=0.1)
            if batch:
                responses = self._batcher.process_batch(batch)
                # Find our response
                for resp in responses:
                    if resp.request_id == request.request_id:
                        return resp

        # Fallback to direct prediction
        return self.predict_request(request)

    def predict_batch(
        self,
        model_name: str,
        inputs_list: List[List[Any]],
    ) -> List[InferenceResponse]:
        """
        Make batch predictions.

        Args:
            model_name: Model name
            inputs_list: List of input batches

        Returns:
            List of InferenceResponse
        """
        responses = []
        for inputs in inputs_list:
            response = self.predict(model_name, inputs)
            responses.append(response)
        return responses

    def _process_batch(self, batch: BatchRequest) -> BatchResult:
        """Process a batch of requests."""
        worker = self._workers.get(batch.model_name)
        if worker is None:
            return BatchResult(
                batch_id=batch.batch_id,
                predictions=[],
                boundaries=[],
                latency_ms=0,
                model_name=batch.model_name,
            )

        # Flatten inputs
        all_inputs, boundaries = batch.get_all_inputs()

        start_time = time.time()

        # Run batch inference
        request = InferenceRequest(
            model_name=batch.model_name,
            inputs=all_inputs,
            batch_id=batch.batch_id,
        )

        response = worker.predict(request)
        latency = (time.time() - start_time) * 1000

        return BatchResult(
            batch_id=batch.batch_id,
            predictions=response.predictions,
            boundaries=boundaries,
            latency_ms=latency,
            model_name=batch.model_name,
        )

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get information about a loaded model.

        Args:
            model_name: Model name

        Returns:
            ModelInfo or None
        """
        worker = self._workers.get(model_name)
        if worker is None:
            return None

        return ModelInfo(
            name=worker.model_name,
            version=worker.stats.model_version,
            path=str(worker._model_path),
            device=worker._config.device,
            is_loaded=worker.is_loaded,
            stats=worker.stats,
        )

    def list_models(self) -> List[ModelInfo]:
        """
        List all loaded models.

        Returns:
            List of ModelInfo
        """
        return [
            self.get_model_info(name)
            for name in self._workers.keys()
        ]

    def get_health(self) -> Dict[str, Any]:
        """
        Get server health status.

        Returns:
            Health status dict
        """
        if self._health_checker:
            return self._health_checker.get_all_health()

        return {
            "status": "healthy" if self._workers else "no_models",
            "models_loaded": len(self._workers),
            "total_requests": self._total_requests,
            "uptime_seconds": time.time() - self._start_time,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.

        Returns:
            Stats dict
        """
        model_stats = {}
        for name, worker in self._workers.items():
            model_stats[name] = worker.stats.to_dict()

        return {
            "uptime_seconds": time.time() - self._start_time,
            "total_requests": self._total_requests,
            "models_loaded": len(self._workers),
            "router": self._router.get_stats(),
            "batcher": self._batcher.get_stats() if self._batcher else None,
            "models": model_stats,
        }

    async def start(self) -> None:
        """Start server background tasks."""
        if self._health_checker:
            await self._health_checker.start_background_checks()
        logger.info("Model server started")

    async def stop(self) -> None:
        """Stop server and cleanup."""
        if self._health_checker:
            await self._health_checker.stop_background_checks()

        # Unload all models
        for model_name in list(self._workers.keys()):
            self.unload_model(model_name)

        logger.info("Model server stopped")


# Global server instance
_default_server: Optional[ModelServer] = None


def get_model_server() -> ModelServer:
    """Get default model server instance."""
    global _default_server
    if _default_server is None:
        _default_server = ModelServer()
    return _default_server


def set_model_server(server: ModelServer) -> None:
    """Set default model server instance."""
    global _default_server
    _default_server = server
