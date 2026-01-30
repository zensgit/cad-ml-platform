"""Tests for gRPC server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGrpcServerConfig:
    """Tests for GrpcServerConfig."""

    def test_default_values(self):
        """Test default config values."""
        from src.api.grpc.server import GrpcServerConfig

        config = GrpcServerConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 50051
        assert config.max_workers == 10
        assert config.enable_reflection is True
        assert config.enable_health_check is True

    def test_custom_values(self):
        """Test custom config values."""
        from src.api.grpc.server import GrpcServerConfig

        config = GrpcServerConfig(
            host="127.0.0.1",
            port=9000,
            max_workers=20,
        )

        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.max_workers == 20


class TestGrpcServiceBase:
    """Tests for GrpcServiceBase."""

    def test_get_service_name_not_implemented(self):
        """Test base class raises NotImplementedError."""
        from src.api.grpc.server import GrpcServiceBase

        service = GrpcServiceBase()

        with pytest.raises(NotImplementedError):
            service.get_service_name()

    def test_add_to_server_not_implemented(self):
        """Test base class raises NotImplementedError."""
        from src.api.grpc.server import GrpcServiceBase

        service = GrpcServiceBase()

        with pytest.raises(NotImplementedError):
            service.add_to_server(MagicMock())


class TestPredictionService:
    """Tests for PredictionService."""

    def test_get_service_name(self):
        """Test service name."""
        from src.api.grpc.server import PredictionService

        service = PredictionService()
        assert service.get_service_name() == "cad_ml.PredictionService"

    @pytest.mark.asyncio
    async def test_predict(self):
        """Test predict method."""
        from src.api.grpc.server import PredictionService

        service = PredictionService()
        result = await service.Predict(MagicMock(), MagicMock())

        assert "prediction" in result
        assert "confidence" in result
        assert result["confidence"] > 0


class TestAnalysisService:
    """Tests for AnalysisService."""

    def test_get_service_name(self):
        """Test service name."""
        from src.api.grpc.server import AnalysisService

        service = AnalysisService()
        assert service.get_service_name() == "cad_ml.AnalysisService"

    @pytest.mark.asyncio
    async def test_analyze_drawing(self):
        """Test analyze drawing method."""
        from src.api.grpc.server import AnalysisService

        service = AnalysisService()
        result = await service.AnalyzeDrawing(MagicMock(), MagicMock())

        assert result["status"] == "analyzed"
        assert "features_found" in result

    @pytest.mark.asyncio
    async def test_extract_features(self):
        """Test extract features method."""
        from src.api.grpc.server import AnalysisService

        service = AnalysisService()
        result = await service.ExtractFeatures(MagicMock(), MagicMock())

        assert "features" in result
        assert len(result["features"]) > 0


class TestGrpcServer:
    """Tests for GrpcServer."""

    def test_init_with_config(self):
        """Test server initialization with config."""
        from src.api.grpc.server import GrpcServer, GrpcServerConfig

        config = GrpcServerConfig(port=9000)
        server = GrpcServer(config=config)

        assert server.config.port == 9000
        assert server._running is False

    def test_add_service(self):
        """Test adding a service."""
        from src.api.grpc.server import GrpcServer, PredictionService

        server = GrpcServer()
        service = PredictionService()

        server.add_service(service)

        assert len(server._services) == 1
        assert server._services[0] == service

    @pytest.mark.asyncio
    async def test_start_without_grpc(self):
        """Test start when gRPC not available."""
        from src.api.grpc.server import GrpcServer

        with patch("src.api.grpc.server.GRPC_AVAILABLE", False):
            server = GrpcServer()
            result = await server.start()

        assert result is False

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Test start when already running returns True immediately."""
        from src.api.grpc.server import GrpcServer

        with patch("src.api.grpc.server.GRPC_AVAILABLE", True):
            server = GrpcServer()
            server._running = True

            result = await server.start()

        assert result is True

    def test_is_running(self):
        """Test is_running method."""
        from src.api.grpc.server import GrpcServer

        server = GrpcServer()
        assert server.is_running() is False

        server._running = True
        assert server.is_running() is True

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """Test stop when not running."""
        from src.api.grpc.server import GrpcServer

        server = GrpcServer()
        # Should not raise
        await server.stop()


class TestGrpcServerSync:
    """Tests for synchronous GrpcServer."""

    def test_init(self):
        """Test initialization."""
        from src.api.grpc.server import GrpcServerSync

        server = GrpcServerSync()
        assert server._running is False

    def test_add_service(self):
        """Test adding a service."""
        from src.api.grpc.server import GrpcServerSync, PredictionService

        server = GrpcServerSync()
        service = PredictionService()

        server.add_service(service)

        assert len(server._services) == 1

    def test_start_without_grpc(self):
        """Test start when gRPC not available."""
        from src.api.grpc.server import GrpcServerSync

        with patch("src.api.grpc.server.GRPC_AVAILABLE", False):
            server = GrpcServerSync()
            result = server.start()

        assert result is False

    def test_stop_not_running(self):
        """Test stop when not running."""
        from src.api.grpc.server import GrpcServerSync

        server = GrpcServerSync()
        # Should not raise
        server.stop()


class TestGlobalFunctions:
    """Tests for global functions."""

    def test_get_grpc_server_singleton(self):
        """Test get_grpc_server returns singleton."""
        from src.api.grpc import server as grpc_module

        # Reset global
        grpc_module._grpc_server = None

        srv1 = grpc_module.get_grpc_server()
        srv2 = grpc_module.get_grpc_server()

        assert srv1 is srv2

        # Cleanup
        grpc_module._grpc_server = None

    def test_is_grpc_available(self):
        """Test is_grpc_available function."""
        from src.api.grpc.server import is_grpc_available

        # Just check it returns a boolean
        result = is_grpc_available()
        assert isinstance(result, bool)
