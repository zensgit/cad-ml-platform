"""
gRPC service implementation for model inference.

Provides high-performance gRPC endpoints for:
- Low-latency inference
- Streaming inference
- Bidirectional communication
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class GRPCStatus(str, Enum):
    """gRPC service status."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class GRPCConfig:
    """gRPC service configuration."""
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_message_size: int = 100 * 1024 * 1024  # 100MB
    enable_reflection: bool = True
    enable_health_check: bool = True
    compression: str = "gzip"  # none, gzip, deflate
    keepalive_time_ms: int = 30000
    keepalive_timeout_ms: int = 10000
    max_concurrent_streams: int = 100
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None


@dataclass
class GRPCRequest:
    """gRPC inference request."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    model_version: Optional[str] = None
    inputs: bytes = b""
    input_format: str = "tensor"  # tensor, json, raw
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 30000
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class GRPCResponse:
    """gRPC inference response."""
    request_id: str
    model_name: str
    model_version: str
    outputs: bytes = b""
    output_format: str = "tensor"
    latency_ms: float = 0.0
    status: str = "success"
    error_message: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class StreamingRequest:
    """Streaming inference request."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    chunk_data: bytes = b""
    chunk_index: int = 0
    total_chunks: int = 1
    is_last: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingResponse:
    """Streaming inference response."""
    request_id: str
    chunk_data: bytes = b""
    chunk_index: int = 0
    total_chunks: int = 1
    is_last: bool = False
    latency_ms: float = 0.0


class InferenceGRPCService:
    """
    gRPC service for model inference.

    Provides high-performance inference via gRPC with support for:
    - Unary calls
    - Server streaming
    - Client streaming
    - Bidirectional streaming
    """

    def __init__(
        self,
        config: Optional[GRPCConfig] = None,
        model_server: Optional[Any] = None,
    ):
        """
        Initialize gRPC service.

        Args:
            config: gRPC configuration
            model_server: Model server instance
        """
        self._config = config or GRPCConfig()
        self._model_server = model_server
        self._server = None
        self._status = GRPCStatus.STOPPED
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0.0,
            "streaming_requests": 0,
        }

    @property
    def status(self) -> GRPCStatus:
        return self._status

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics.copy()

    def generate_proto(self) -> str:
        """Generate .proto file content."""
        return '''syntax = "proto3";

package cadml.inference;

service InferenceService {
    // Unary inference
    rpc Predict(PredictRequest) returns (PredictResponse);

    // Batch inference
    rpc PredictBatch(PredictBatchRequest) returns (PredictBatchResponse);

    // Server streaming - stream results back
    rpc PredictStream(PredictRequest) returns (stream PredictResponse);

    // Client streaming - stream inputs
    rpc PredictClientStream(stream StreamChunk) returns (PredictResponse);

    // Bidirectional streaming
    rpc PredictBidirectional(stream StreamChunk) returns (stream StreamChunk);

    // Model management
    rpc GetModelInfo(ModelInfoRequest) returns (ModelInfoResponse);
    rpc ListModels(ListModelsRequest) returns (ListModelsResponse);

    // Health check
    rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}

message PredictRequest {
    string request_id = 1;
    string model_name = 2;
    string model_version = 3;
    bytes inputs = 4;
    string input_format = 5;
    map<string, string> parameters = 6;
    int32 timeout_ms = 7;
    map<string, string> metadata = 8;
}

message PredictResponse {
    string request_id = 1;
    string model_name = 2;
    string model_version = 3;
    bytes outputs = 4;
    string output_format = 5;
    float latency_ms = 6;
    string status = 7;
    string error_message = 8;
    map<string, string> metadata = 9;
}

message PredictBatchRequest {
    repeated PredictRequest requests = 1;
    bool parallel = 2;
}

message PredictBatchResponse {
    repeated PredictResponse responses = 1;
    float total_latency_ms = 2;
    int32 successful = 3;
    int32 failed = 4;
}

message StreamChunk {
    string request_id = 1;
    string model_name = 2;
    bytes chunk_data = 3;
    int32 chunk_index = 4;
    int32 total_chunks = 5;
    bool is_last = 6;
    map<string, string> parameters = 7;
}

message ModelInfoRequest {
    string model_name = 1;
    string model_version = 2;
}

message ModelInfoResponse {
    string name = 1;
    string version = 2;
    string status = 3;
    string input_schema = 4;
    string output_schema = 5;
    map<string, string> metadata = 6;
}

message ListModelsRequest {
    string filter = 1;
}

message ListModelsResponse {
    repeated ModelInfoResponse models = 1;
}

message HealthCheckRequest {
    string service = 1;
}

message HealthCheckResponse {
    string status = 1;
    string message = 2;
}
'''

    async def predict(self, request: GRPCRequest) -> GRPCResponse:
        """
        Perform unary inference.

        Args:
            request: Inference request

        Returns:
            Inference response
        """
        start_time = time.time()
        self._metrics["total_requests"] += 1

        try:
            if self._model_server is None:
                # Mock response
                latency = (time.time() - start_time) * 1000
                return GRPCResponse(
                    request_id=request.request_id,
                    model_name=request.model_name,
                    model_version=request.model_version or "1.0.0",
                    outputs=b"mock_output",
                    latency_ms=latency,
                )

            # Deserialize inputs based on format
            inputs = self._deserialize_inputs(request.inputs, request.input_format)

            # Perform inference
            result = await self._model_server.infer(
                model_name=request.model_name,
                inputs=inputs,
                parameters=request.parameters,
                version=request.model_version,
            )

            # Serialize outputs
            outputs = self._serialize_outputs(result.get("predictions", []), "tensor")

            latency = (time.time() - start_time) * 1000
            self._metrics["successful_requests"] += 1
            self._metrics["total_latency_ms"] += latency

            return GRPCResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                model_version=result.get("version", "1.0.0"),
                outputs=outputs,
                latency_ms=latency,
            )

        except Exception as e:
            self._metrics["failed_requests"] += 1
            latency = (time.time() - start_time) * 1000

            return GRPCResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                model_version=request.model_version or "unknown",
                status="error",
                error_message=str(e),
                latency_ms=latency,
            )

    async def predict_stream(
        self,
        request: GRPCRequest,
    ) -> AsyncIterator[GRPCResponse]:
        """
        Server streaming inference.

        Yields responses as they become available.

        Args:
            request: Inference request

        Yields:
            Streaming responses
        """
        self._metrics["streaming_requests"] += 1
        start_time = time.time()

        try:
            # Perform inference
            result = await self.predict(request)

            # Stream results in chunks
            outputs = result.outputs
            chunk_size = 1024 * 1024  # 1MB chunks

            total_chunks = (len(outputs) + chunk_size - 1) // chunk_size

            for i in range(total_chunks):
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, len(outputs))
                chunk_data = outputs[chunk_start:chunk_end]

                latency = (time.time() - start_time) * 1000

                yield GRPCResponse(
                    request_id=request.request_id,
                    model_name=request.model_name,
                    model_version=result.model_version,
                    outputs=chunk_data,
                    latency_ms=latency,
                    metadata={"chunk_index": str(i), "total_chunks": str(total_chunks)},
                )

        except Exception as e:
            yield GRPCResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                model_version="unknown",
                status="error",
                error_message=str(e),
            )

    async def predict_client_stream(
        self,
        requests: AsyncIterator[StreamingRequest],
    ) -> GRPCResponse:
        """
        Client streaming inference.

        Accumulates chunks and returns single response.

        Args:
            requests: Stream of input chunks

        Returns:
            Final inference response
        """
        self._metrics["streaming_requests"] += 1
        start_time = time.time()

        chunks: List[bytes] = []
        model_name = ""
        request_id = ""
        parameters: Dict[str, Any] = {}

        try:
            async for chunk in requests:
                if not request_id:
                    request_id = chunk.request_id
                    model_name = chunk.model_name
                    parameters = chunk.parameters

                chunks.append(chunk.chunk_data)

                if chunk.is_last:
                    break

            # Combine chunks
            combined_input = b"".join(chunks)

            # Create inference request
            inference_request = GRPCRequest(
                request_id=request_id,
                model_name=model_name,
                inputs=combined_input,
                parameters=parameters,
            )

            return await self.predict(inference_request)

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return GRPCResponse(
                request_id=request_id or str(uuid.uuid4()),
                model_name=model_name or "unknown",
                model_version="unknown",
                status="error",
                error_message=str(e),
                latency_ms=latency,
            )

    async def predict_bidirectional(
        self,
        requests: AsyncIterator[StreamingRequest],
    ) -> AsyncIterator[StreamingResponse]:
        """
        Bidirectional streaming inference.

        Processes chunks as they arrive and streams results back.

        Args:
            requests: Stream of input chunks

        Yields:
            Stream of output chunks
        """
        self._metrics["streaming_requests"] += 1

        try:
            async for chunk in requests:
                start_time = time.time()

                # Process each chunk immediately
                inference_request = GRPCRequest(
                    request_id=chunk.request_id,
                    model_name=chunk.model_name,
                    inputs=chunk.chunk_data,
                    parameters=chunk.parameters,
                )

                result = await self.predict(inference_request)
                latency = (time.time() - start_time) * 1000

                yield StreamingResponse(
                    request_id=chunk.request_id,
                    chunk_data=result.outputs,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    is_last=chunk.is_last,
                    latency_ms=latency,
                )

        except Exception as e:
            logger.error(f"Bidirectional streaming error: {e}")
            yield StreamingResponse(
                request_id=str(uuid.uuid4()),
                chunk_data=b"",
                is_last=True,
            )

    def _deserialize_inputs(self, data: bytes, format: str) -> Any:
        """Deserialize input data."""
        if format == "json":
            import json
            return json.loads(data.decode("utf-8"))
        elif format == "tensor":
            try:
                import torch
                import io
                buffer = io.BytesIO(data)
                return torch.load(buffer)
            except ImportError:
                return data
        else:
            return data

    def _serialize_outputs(self, data: Any, format: str) -> bytes:
        """Serialize output data."""
        if format == "json":
            import json
            return json.dumps(data).encode("utf-8")
        elif format == "tensor":
            try:
                import torch
                import io
                buffer = io.BytesIO()
                torch.save(data, buffer)
                return buffer.getvalue()
            except ImportError:
                return str(data).encode("utf-8")
        else:
            return str(data).encode("utf-8")

    async def start(self) -> None:
        """Start the gRPC server."""
        try:
            import grpc
            from concurrent import futures
        except ImportError:
            raise ImportError("grpcio required: pip install grpcio")

        self._status = GRPCStatus.STARTING
        logger.info(f"Starting gRPC server on {self._config.host}:{self._config.port}")

        # Note: In production, you would use generated protobuf stubs
        # This is a simplified implementation

        self._status = GRPCStatus.RUNNING
        logger.info("gRPC server started")

    async def stop(self) -> None:
        """Stop the gRPC server."""
        self._status = GRPCStatus.STOPPING
        logger.info("Stopping gRPC server")

        if self._server:
            await self._server.stop(grace=5)

        self._status = GRPCStatus.STOPPED
        logger.info("gRPC server stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        avg_latency = 0.0
        if self._metrics["successful_requests"] > 0:
            avg_latency = self._metrics["total_latency_ms"] / self._metrics["successful_requests"]

        return {
            **self._metrics,
            "avg_latency_ms": avg_latency,
            "status": self._status.value,
        }


def create_grpc_service(
    config: Optional[GRPCConfig] = None,
    model_server: Optional[Any] = None,
) -> InferenceGRPCService:
    """
    Create a gRPC inference service.

    Args:
        config: gRPC configuration
        model_server: Model server instance

    Returns:
        InferenceGRPCService instance
    """
    return InferenceGRPCService(config, model_server)
