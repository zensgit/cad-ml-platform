"""
REST API endpoint implementation for model inference.

Provides FastAPI-based REST endpoints for:
- Single and batch inference
- Model management
- Health checks
- Metrics exposure
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ResponseFormat(str, Enum):
    """API response format."""
    JSON = "json"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"


@dataclass
class APIConfig:
    """REST API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    request_timeout: float = 30.0
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_docs: bool = True
    enable_metrics: bool = True
    api_prefix: str = "/api/v1"
    auth_enabled: bool = False
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60


@dataclass
class InferenceAPIRequest:
    """Inference request model."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    model_version: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout: Optional[float] = None
    return_probabilities: bool = False
    return_features: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "priority": self.priority,
            "timeout": self.timeout,
            "return_probabilities": self.return_probabilities,
            "return_features": self.return_features,
            "metadata": self.metadata,
        }


@dataclass
class InferenceAPIResponse:
    """Inference response model."""
    request_id: str
    model_name: str
    model_version: str
    predictions: List[Dict[str, Any]]
    probabilities: Optional[List[List[float]]] = None
    features: Optional[List[List[float]]] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "predictions": self.predictions,
            "probabilities": self.probabilities,
            "features": self.features,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class BatchInferenceRequest:
    """Batch inference request model."""
    requests: List[InferenceAPIRequest]
    parallel: bool = True
    max_batch_size: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests": [r.to_dict() for r in self.requests],
            "parallel": self.parallel,
            "max_batch_size": self.max_batch_size,
        }


@dataclass
class BatchInferenceResponse:
    """Batch inference response model."""
    responses: List[InferenceAPIResponse]
    total_latency_ms: float = 0.0
    successful: int = 0
    failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "responses": [r.to_dict() for r in self.responses],
            "total_latency_ms": self.total_latency_ms,
            "successful": self.successful,
            "failed": self.failed,
        }


@dataclass
class ModelInfo:
    """Model information response."""
    name: str
    version: str
    status: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    loaded_at: Optional[datetime] = None
    request_count: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "metadata": self.metadata,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "request_count": self.request_count,
            "avg_latency_ms": self.avg_latency_ms,
        }


class InferenceRESTAPI:
    """
    REST API implementation for model inference.

    Provides a FastAPI-based REST interface for model serving.
    """

    def __init__(
        self,
        config: Optional[APIConfig] = None,
        model_server: Optional[Any] = None,
    ):
        """
        Initialize REST API.

        Args:
            config: API configuration
            model_server: Model server instance
        """
        self._config = config or APIConfig()
        self._model_server = model_server
        self._app = None
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0.0,
        }
        self._request_history: List[Dict[str, Any]] = []

    def create_app(self):
        """Create FastAPI application."""
        try:
            from fastapi import FastAPI, HTTPException, Request, Response
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import JSONResponse
        except ImportError:
            raise ImportError("FastAPI required: pip install fastapi")

        app = FastAPI(
            title="CAD-ML Inference API",
            description="REST API for CAD model inference",
            version="1.0.0",
            docs_url="/docs" if self._config.enable_docs else None,
            redoc_url="/redoc" if self._config.enable_docs else None,
        )

        # CORS middleware
        if self._config.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self._config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Register routes
        self._register_routes(app)

        self._app = app
        return app

    def _register_routes(self, app):
        """Register API routes."""
        from fastapi import HTTPException, Request
        from fastapi.responses import JSONResponse

        prefix = self._config.api_prefix

        @app.get(f"{prefix}/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_server": self._model_server is not None,
            }

        @app.get(f"{prefix}/ready")
        async def readiness_check():
            """Readiness check endpoint."""
            if self._model_server is None:
                raise HTTPException(status_code=503, detail="Model server not initialized")
            return {"status": "ready"}

        @app.post(f"{prefix}/predict")
        async def predict(request: Request):
            """Single inference endpoint."""
            start_time = time.time()

            try:
                data = await request.json()
                req = InferenceAPIRequest(**data)

                # Perform inference
                result = await self._perform_inference(req)

                self._metrics["total_requests"] += 1
                self._metrics["successful_requests"] += 1

                return result.to_dict()

            except Exception as e:
                self._metrics["total_requests"] += 1
                self._metrics["failed_requests"] += 1
                logger.error(f"Inference error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post(f"{prefix}/predict/batch")
        async def predict_batch(request: Request):
            """Batch inference endpoint."""
            start_time = time.time()

            try:
                data = await request.json()
                requests_data = data.get("requests", [])
                parallel = data.get("parallel", True)

                requests = [InferenceAPIRequest(**r) for r in requests_data]

                # Perform batch inference
                if parallel:
                    tasks = [self._perform_inference(r) for r in requests]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    results = []
                    for r in requests:
                        try:
                            result = await self._perform_inference(r)
                            results.append(result)
                        except Exception as e:
                            results.append(e)

                # Process results
                responses = []
                successful = 0
                failed = 0

                for result in results:
                    if isinstance(result, Exception):
                        failed += 1
                    else:
                        responses.append(result)
                        successful += 1

                total_latency = (time.time() - start_time) * 1000

                batch_response = BatchInferenceResponse(
                    responses=responses,
                    total_latency_ms=total_latency,
                    successful=successful,
                    failed=failed,
                )

                return batch_response.to_dict()

            except Exception as e:
                logger.error(f"Batch inference error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get(f"{prefix}/models")
        async def list_models():
            """List available models."""
            if self._model_server is None:
                return {"models": []}

            try:
                models = self._model_server.list_models()
                return {"models": [m.to_dict() if hasattr(m, "to_dict") else m for m in models]}
            except Exception as e:
                logger.error(f"List models error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get(f"{prefix}/models/{{model_name}}")
        async def get_model(model_name: str):
            """Get model information."""
            if self._model_server is None:
                raise HTTPException(status_code=404, detail="Model server not initialized")

            try:
                model_info = self._model_server.get_model_info(model_name)
                if model_info is None:
                    raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
                return model_info.to_dict() if hasattr(model_info, "to_dict") else model_info
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get model error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post(f"{prefix}/models/{{model_name}}/load")
        async def load_model(model_name: str, request: Request):
            """Load a model."""
            if self._model_server is None:
                raise HTTPException(status_code=503, detail="Model server not initialized")

            try:
                data = await request.json()
                version = data.get("version")
                path = data.get("path")

                result = await self._model_server.load_model(model_name, version=version, path=path)
                return {"status": "loaded", "model_name": model_name}
            except Exception as e:
                logger.error(f"Load model error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post(f"{prefix}/models/{{model_name}}/unload")
        async def unload_model(model_name: str):
            """Unload a model."""
            if self._model_server is None:
                raise HTTPException(status_code=503, detail="Model server not initialized")

            try:
                await self._model_server.unload_model(model_name)
                return {"status": "unloaded", "model_name": model_name}
            except Exception as e:
                logger.error(f"Unload model error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get(f"{prefix}/metrics")
        async def get_metrics():
            """Get API metrics."""
            avg_latency = 0.0
            if self._metrics["successful_requests"] > 0:
                avg_latency = self._metrics["total_latency_ms"] / self._metrics["successful_requests"]

            return {
                **self._metrics,
                "avg_latency_ms": avg_latency,
            }

    async def _perform_inference(self, request: InferenceAPIRequest) -> InferenceAPIResponse:
        """Perform inference for a single request."""
        start_time = time.time()

        if self._model_server is None:
            # Mock response for testing
            latency = (time.time() - start_time) * 1000
            return InferenceAPIResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                model_version=request.model_version or "1.0.0",
                predictions=[{"class": "unknown", "confidence": 0.0}],
                latency_ms=latency,
            )

        # Use model server for inference
        result = await self._model_server.infer(
            model_name=request.model_name,
            inputs=request.inputs,
            parameters=request.parameters,
            version=request.model_version,
        )

        latency = (time.time() - start_time) * 1000
        self._metrics["total_latency_ms"] += latency

        return InferenceAPIResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            model_version=result.get("version", "1.0.0"),
            predictions=result.get("predictions", []),
            probabilities=result.get("probabilities") if request.return_probabilities else None,
            features=result.get("features") if request.return_features else None,
            latency_ms=latency,
            metadata=result.get("metadata", {}),
        )

    def run(self, **kwargs):
        """Run the API server."""
        try:
            import uvicorn
        except ImportError:
            raise ImportError("Uvicorn required: pip install uvicorn")

        if self._app is None:
            self.create_app()

        uvicorn.run(
            self._app,
            host=kwargs.get("host", self._config.host),
            port=kwargs.get("port", self._config.port),
            workers=kwargs.get("workers", self._config.workers),
        )

    @property
    def app(self):
        """Get the FastAPI app instance."""
        if self._app is None:
            self.create_app()
        return self._app


def create_inference_api(
    config: Optional[APIConfig] = None,
    model_server: Optional[Any] = None,
) -> InferenceRESTAPI:
    """
    Create an inference REST API instance.

    Args:
        config: API configuration
        model_server: Model server instance

    Returns:
        InferenceRESTAPI instance
    """
    return InferenceRESTAPI(config, model_server)
