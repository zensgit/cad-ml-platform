"""gRPC Server implementation for CAD ML Platform.

Features:
- Protobuf-based service definitions
- Bidirectional streaming
- Health checking
- Reflection service
- Graceful shutdown
"""

from __future__ import annotations

import asyncio
import logging
from concurrent import futures
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Try to import grpcio - gracefully degrade if not available
try:
    import grpc
    from grpc import aio
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc
    from grpc_reflection.v1alpha import reflection

    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    grpc = None
    aio = None


@dataclass
class GrpcServerConfig:
    """gRPC server configuration."""

    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_message_length: int = 4 * 1024 * 1024  # 4MB
    enable_reflection: bool = True
    enable_health_check: bool = True


class GrpcServiceBase:
    """Base class for gRPC services."""

    def get_service_name(self) -> str:
        """Return the service name for registration."""
        raise NotImplementedError

    def add_to_server(self, server: Any) -> None:
        """Add this service to a gRPC server."""
        raise NotImplementedError


class PredictionService(GrpcServiceBase):
    """gRPC service for predictions (mock implementation).

    In production, this would use actual protobuf-generated code.
    """

    def __init__(self, model_service: Optional[Any] = None):
        self.model_service = model_service

    def get_service_name(self) -> str:
        return "cad_ml.PredictionService"

    def add_to_server(self, server: Any) -> None:
        # In production, use generated pb2_grpc.add_...
        logger.info(f"Registering {self.get_service_name()}")

    async def Predict(self, request: Any, context: Any) -> Dict[str, Any]:
        """Handle prediction request."""
        # Mock implementation
        return {
            "prediction": "class_a",
            "confidence": 0.95,
            "model_id": "default",
        }

    async def PredictStream(
        self, request_iterator: Any, context: Any
    ) -> Iterator[Dict[str, Any]]:
        """Handle streaming predictions."""
        async for request in request_iterator:
            yield {
                "prediction": "class_a",
                "confidence": 0.95,
            }


class AnalysisService(GrpcServiceBase):
    """gRPC service for CAD analysis."""

    def __init__(self, analyzer: Optional[Any] = None):
        self.analyzer = analyzer

    def get_service_name(self) -> str:
        return "cad_ml.AnalysisService"

    def add_to_server(self, server: Any) -> None:
        logger.info(f"Registering {self.get_service_name()}")

    async def AnalyzeDrawing(self, request: Any, context: Any) -> Dict[str, Any]:
        """Analyze a CAD drawing."""
        return {
            "status": "analyzed",
            "features_found": 10,
            "confidence": 0.92,
        }

    async def ExtractFeatures(self, request: Any, context: Any) -> Dict[str, Any]:
        """Extract features from a drawing."""
        return {
            "features": [
                {"type": "hole", "count": 5},
                {"type": "slot", "count": 3},
            ],
        }


class GrpcServer:
    """Async gRPC server for CAD ML Platform."""

    def __init__(self, config: Optional[GrpcServerConfig] = None):
        self.config = config or GrpcServerConfig()
        self._server: Optional[Any] = None
        self._services: List[GrpcServiceBase] = []
        self._running = False
        self._health_servicer: Optional[Any] = None

    def add_service(self, service: GrpcServiceBase) -> None:
        """Add a service to the server."""
        self._services.append(service)
        logger.info(f"Added gRPC service: {service.get_service_name()}")

    async def start(self) -> bool:
        """Start the gRPC server."""
        if not GRPC_AVAILABLE:
            logger.warning("gRPC not available - server not started")
            return False

        if self._running:
            return True

        try:
            # Create server
            self._server = aio.server(
                futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
                options=[
                    ("grpc.max_send_message_length", self.config.max_message_length),
                    ("grpc.max_receive_message_length", self.config.max_message_length),
                ],
            )

            # Register services
            service_names = []
            for service in self._services:
                service.add_to_server(self._server)
                service_names.append(service.get_service_name())

            # Health check
            if self.config.enable_health_check:
                self._health_servicer = health.HealthServicer()
                health_pb2_grpc.add_HealthServicer_to_server(
                    self._health_servicer, self._server
                )

                # Set all services as serving
                for name in service_names:
                    self._health_servicer.set(name, health_pb2.HealthCheckResponse.SERVING)
                self._health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

            # Reflection
            if self.config.enable_reflection:
                reflection.enable_server_reflection(
                    service_names + [reflection.SERVICE_NAME],
                    self._server,
                )

            # Bind and start
            address = f"{self.config.host}:{self.config.port}"
            self._server.add_insecure_port(address)
            await self._server.start()

            self._running = True
            logger.info(f"gRPC server started on {address}")
            return True

        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")
            return False

    async def stop(self, grace_period: float = 5.0) -> None:
        """Stop the gRPC server gracefully."""
        if not self._running or not self._server:
            return

        # Mark services as not serving
        if self._health_servicer:
            for service in self._services:
                self._health_servicer.set(
                    service.get_service_name(),
                    health_pb2.HealthCheckResponse.NOT_SERVING,
                )
            self._health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)

        # Graceful shutdown
        await self._server.stop(grace_period)
        self._running = False
        logger.info("gRPC server stopped")

    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self._server:
            await self._server.wait_for_termination()

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


# Synchronous gRPC server for compatibility
class GrpcServerSync:
    """Synchronous gRPC server."""

    def __init__(self, config: Optional[GrpcServerConfig] = None):
        self.config = config or GrpcServerConfig()
        self._server: Optional[Any] = None
        self._services: List[GrpcServiceBase] = []
        self._running = False

    def add_service(self, service: GrpcServiceBase) -> None:
        """Add a service."""
        self._services.append(service)

    def start(self) -> bool:
        """Start the server."""
        if not GRPC_AVAILABLE:
            logger.warning("gRPC not available")
            return False

        try:
            self._server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
                options=[
                    ("grpc.max_send_message_length", self.config.max_message_length),
                    ("grpc.max_receive_message_length", self.config.max_message_length),
                ],
            )

            for service in self._services:
                service.add_to_server(self._server)

            address = f"{self.config.host}:{self.config.port}"
            self._server.add_insecure_port(address)
            self._server.start()

            self._running = True
            logger.info(f"gRPC server (sync) started on {address}")
            return True

        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")
            return False

    def stop(self, grace_period: float = 5.0) -> None:
        """Stop the server."""
        if self._server:
            self._server.stop(grace_period)
            self._running = False
            logger.info("gRPC server stopped")

    def wait_for_termination(self) -> None:
        """Wait for termination."""
        if self._server:
            self._server.wait_for_termination()


# Global server instance
_grpc_server: Optional[GrpcServer] = None


def get_grpc_server() -> GrpcServer:
    """Get the global gRPC server instance."""
    global _grpc_server
    if _grpc_server is None:
        _grpc_server = GrpcServer()
    return _grpc_server


def is_grpc_available() -> bool:
    """Check if gRPC is available."""
    return GRPC_AVAILABLE
