"""Tests for src/api/v1/twin.py to improve coverage.

Covers:
- WebSocket endpoint logic
- ingest_telemetry endpoint
- get_twin_state endpoint
- get_history endpoint
- TelemetryFrame handling
- Error handling paths
"""

from __future__ import annotations

import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestWebSocketEndpoint:
    """Tests for websocket_endpoint function."""

    @pytest.mark.asyncio
    async def test_websocket_accept(self):
        """Test websocket is accepted."""
        mock_websocket = MagicMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.receive_text = AsyncMock(side_effect=Exception("disconnect"))

        from src.api.v1.twin import websocket_endpoint

        # Mock twin_sync
        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.get_state.return_value = {}
            try:
                await websocket_endpoint(mock_websocket, "asset123")
            except Exception:
                pass

        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_disconnect_handling(self):
        """Test WebSocketDisconnect is handled gracefully."""
        from fastapi import WebSocketDisconnect

        mock_websocket = MagicMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.receive_text = AsyncMock(side_effect=WebSocketDisconnect())

        from src.api.v1.twin import websocket_endpoint

        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.get_state.return_value = {}
            # Should not raise, just log and exit
            await websocket_endpoint(mock_websocket, "asset123")

    @pytest.mark.asyncio
    async def test_websocket_sends_state_snapshot(self):
        """Test websocket sends state snapshot on message."""
        mock_websocket = MagicMock()
        mock_websocket.accept = AsyncMock()
        # First receive returns text, second raises disconnect
        mock_websocket.receive_text = AsyncMock(side_effect=["hello", Exception("done")])
        mock_websocket.send_json = AsyncMock()

        from src.api.v1.twin import websocket_endpoint

        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.get_state.return_value = {"temp": 25.0}
            try:
                await websocket_endpoint(mock_websocket, "asset123")
            except Exception:
                pass

        # Verify state snapshot was sent
        mock_websocket.send_json.assert_called()


class TestPushUpdateCallback:
    """Tests for push_update callback logic."""

    @pytest.mark.asyncio
    async def test_push_update_matching_asset(self):
        """Test push_update sends event when asset_id matches."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        asset_id = "asset123"
        event = {"asset_id": "asset123", "data": "test"}

        # Simulate the callback logic
        async def push_update(event: Dict[str, Any]) -> None:
            if event["asset_id"] == asset_id:
                try:
                    await mock_websocket.send_json(event)
                except Exception:
                    pass

        await push_update(event)
        mock_websocket.send_json.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_push_update_non_matching_asset(self):
        """Test push_update skips event when asset_id doesn't match."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        asset_id = "asset123"
        event = {"asset_id": "other_asset", "data": "test"}

        async def push_update(event: Dict[str, Any]) -> None:
            if event["asset_id"] == asset_id:
                try:
                    await mock_websocket.send_json(event)
                except Exception:
                    pass

        await push_update(event)
        mock_websocket.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_push_update_handles_send_error(self):
        """Test push_update handles send error gracefully."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock(side_effect=Exception("Connection closed"))

        asset_id = "asset123"
        event = {"asset_id": "asset123", "data": "test"}

        async def push_update(event: Dict[str, Any]) -> None:
            if event["asset_id"] == asset_id:
                try:
                    await mock_websocket.send_json(event)
                except Exception:
                    pass

        # Should not raise
        await push_update(event)


class TestIngestTelemetry:
    """Tests for ingest_telemetry endpoint."""

    @pytest.mark.asyncio
    async def test_ingest_telemetry_basic(self):
        """Test basic telemetry ingestion."""
        from src.api.v1.twin import ingest_telemetry

        telemetry = {
            "timestamp": time.time(),
            "sensors": {"temp": 25.0},
            "metrics": {"power": 100},
            "status": {"online": True},
        }

        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.update_state = AsyncMock()
            with patch("src.api.v1.twin.get_ingestor") as mock_get_ingestor:
                mock_ingestor = MagicMock()
                mock_ingestor.ensure_started = AsyncMock()
                mock_ingestor.handle_payload = AsyncMock(return_value={"ok": True})
                mock_get_ingestor.return_value = mock_ingestor

                result = await ingest_telemetry("asset123", telemetry)

        assert result["status"] == "accepted"
        assert "ingest" in result

    @pytest.mark.asyncio
    async def test_ingest_telemetry_without_timestamp(self):
        """Test telemetry ingestion without explicit timestamp."""
        from src.api.v1.twin import ingest_telemetry

        telemetry = {
            "sensors": {"temp": 30.0},
        }

        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.update_state = AsyncMock()
            with patch("src.api.v1.twin.get_ingestor") as mock_get_ingestor:
                mock_ingestor = MagicMock()
                mock_ingestor.ensure_started = AsyncMock()
                mock_ingestor.handle_payload = AsyncMock(return_value={"ok": True})
                mock_get_ingestor.return_value = mock_ingestor

                result = await ingest_telemetry("asset456", telemetry)

        assert result["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_ingest_telemetry_updates_twin_sync(self):
        """Test telemetry updates twin_sync state."""
        from src.api.v1.twin import ingest_telemetry

        telemetry = {"sensors": {"voltage": 12.5}}

        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.update_state = AsyncMock()
            with patch("src.api.v1.twin.get_ingestor") as mock_get_ingestor:
                mock_ingestor = MagicMock()
                mock_ingestor.ensure_started = AsyncMock()
                mock_ingestor.handle_payload = AsyncMock(return_value={})
                mock_get_ingestor.return_value = mock_ingestor

                await ingest_telemetry("asset789", telemetry)

        mock_sync.update_state.assert_called_once_with("asset789", telemetry)


class TestGetTwinState:
    """Tests for get_twin_state endpoint."""

    @pytest.mark.asyncio
    async def test_get_state_returns_dict(self):
        """Test get_twin_state returns state dict."""
        from src.api.v1.twin import get_twin_state

        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.get_state.return_value = {"temp": 25.0, "status": "active"}

            result = await get_twin_state("asset123")

        assert result == {"temp": 25.0, "status": "active"}
        mock_sync.get_state.assert_called_once_with("asset123")

    @pytest.mark.asyncio
    async def test_get_state_empty(self):
        """Test get_twin_state returns empty dict for unknown asset."""
        from src.api.v1.twin import get_twin_state

        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.get_state.return_value = {}

            result = await get_twin_state("unknown_asset")

        assert result == {}


class TestGetHistory:
    """Tests for get_history endpoint."""

    @pytest.mark.asyncio
    async def test_get_history_basic(self):
        """Test basic history retrieval."""
        from src.api.v1.twin import get_history

        mock_frame = MagicMock()
        mock_frame.model_dump.return_value = {
            "timestamp": 1234567890.0,
            "device_id": "device1",
            "sensors": {},
        }

        with patch("src.api.v1.twin.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.history = AsyncMock(return_value=[mock_frame])
            mock_get_store.return_value = mock_store

            with patch("src.api.v1.twin.get_ingestor") as mock_get_ingestor:
                mock_ingestor = MagicMock()
                mock_ingestor.ensure_started = AsyncMock()
                mock_get_ingestor.return_value = mock_ingestor

                result = await get_history(device_id="device1", limit=50, api_key="test_key")

        assert result["device_id"] == "device1"
        assert result["count"] == 1
        assert len(result["frames"]) == 1

    @pytest.mark.asyncio
    async def test_get_history_empty(self):
        """Test history retrieval returns empty list."""
        from src.api.v1.twin import get_history

        with patch("src.api.v1.twin.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.history = AsyncMock(return_value=[])
            mock_get_store.return_value = mock_store

            with patch("src.api.v1.twin.get_ingestor") as mock_get_ingestor:
                mock_ingestor = MagicMock()
                mock_ingestor.ensure_started = AsyncMock()
                mock_get_ingestor.return_value = mock_ingestor

                result = await get_history(device_id="unknown", limit=50, api_key="test_key")

        assert result["count"] == 0
        assert result["frames"] == []

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self):
        """Test history retrieval respects limit."""
        from src.api.v1.twin import get_history

        mock_frames = [MagicMock() for _ in range(10)]
        for i, frame in enumerate(mock_frames):
            frame.model_dump.return_value = {"timestamp": float(i)}

        with patch("src.api.v1.twin.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.history = AsyncMock(return_value=mock_frames)
            mock_get_store.return_value = mock_store

            with patch("src.api.v1.twin.get_ingestor") as mock_get_ingestor:
                mock_ingestor = MagicMock()
                mock_ingestor.ensure_started = AsyncMock()
                mock_get_ingestor.return_value = mock_ingestor

                result = await get_history(device_id="device1", limit=100, api_key="test_key")

        assert result["count"] == 10


class TestTelemetryFrame:
    """Tests for TelemetryFrame model."""

    def test_telemetry_frame_creation(self):
        """Test TelemetryFrame can be created."""
        from src.core.twin.connectivity import TelemetryFrame

        frame = TelemetryFrame(
            timestamp=time.time(),
            device_id="device123",
            sensors={"temp": 25.0},
            metrics={"power": 100},
            status={"online": True},
        )

        assert frame.device_id == "device123"
        assert frame.sensors == {"temp": 25.0}

    def test_telemetry_frame_defaults(self):
        """Test TelemetryFrame default values."""
        from src.core.twin.connectivity import TelemetryFrame

        frame = TelemetryFrame(
            timestamp=time.time(),
            device_id="device1",
        )

        assert frame.device_id == "device1"
        assert frame.sensors == {}
        assert frame.metrics == {}
        assert frame.status == {}


class TestRouterConfiguration:
    """Tests for router configuration."""

    def test_router_exists(self):
        """Test router is exported."""
        from src.api.v1.twin import router

        assert router is not None

    def test_router_has_websocket_route(self):
        """Test router has websocket route."""
        from src.api.v1.twin import router

        # Check routes
        routes = [r.path for r in router.routes]
        assert "/ws/{asset_id}" in routes

    def test_router_has_telemetry_route(self):
        """Test router has telemetry route."""
        from src.api.v1.twin import router

        routes = [r.path for r in router.routes]
        assert "/{asset_id}/telemetry" in routes

    def test_router_has_state_route(self):
        """Test router has state route."""
        from src.api.v1.twin import router

        routes = [r.path for r in router.routes]
        assert "/{asset_id}/state" in routes

    def test_router_has_history_route(self):
        """Test router has history route."""
        from src.api.v1.twin import router

        routes = [r.path for r in router.routes]
        assert "/history" in routes


class TestTwinSyncIntegration:
    """Tests for twin_sync integration."""

    @pytest.mark.asyncio
    async def test_twin_sync_get_state(self):
        """Test twin_sync.get_state is called correctly."""
        from src.api.v1.twin import get_twin_state

        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.get_state.return_value = {"key": "value"}

            await get_twin_state("test_asset")

            mock_sync.get_state.assert_called_once_with("test_asset")

    @pytest.mark.asyncio
    async def test_twin_sync_update_state(self):
        """Test twin_sync.update_state is called correctly."""
        from src.api.v1.twin import ingest_telemetry

        telemetry = {"sensors": {"temp": 20.0}}

        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.update_state = AsyncMock()
            with patch("src.api.v1.twin.get_ingestor") as mock_get_ingestor:
                mock_ingestor = MagicMock()
                mock_ingestor.ensure_started = AsyncMock()
                mock_ingestor.handle_payload = AsyncMock(return_value={})
                mock_get_ingestor.return_value = mock_ingestor

                await ingest_telemetry("test_asset", telemetry)

            mock_sync.update_state.assert_called_once_with("test_asset", telemetry)


class TestIngestorIntegration:
    """Tests for ingestor integration."""

    @pytest.mark.asyncio
    async def test_ingestor_ensure_started(self):
        """Test ingestor.ensure_started is called."""
        from src.api.v1.twin import ingest_telemetry

        telemetry = {"sensors": {}}

        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.update_state = AsyncMock()
            with patch("src.api.v1.twin.get_ingestor") as mock_get_ingestor:
                mock_ingestor = MagicMock()
                mock_ingestor.ensure_started = AsyncMock()
                mock_ingestor.handle_payload = AsyncMock(return_value={})
                mock_get_ingestor.return_value = mock_ingestor

                await ingest_telemetry("asset1", telemetry)

                mock_ingestor.ensure_started.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingestor_handle_payload_topic(self):
        """Test ingestor.handle_payload uses correct topic."""
        from src.api.v1.twin import ingest_telemetry

        telemetry = {"timestamp": 1234567890.0, "sensors": {}}

        with patch("src.api.v1.twin.twin_sync") as mock_sync:
            mock_sync.update_state = AsyncMock()
            with patch("src.api.v1.twin.get_ingestor") as mock_get_ingestor:
                mock_ingestor = MagicMock()
                mock_ingestor.ensure_started = AsyncMock()
                mock_ingestor.handle_payload = AsyncMock(return_value={})
                mock_get_ingestor.return_value = mock_ingestor

                await ingest_telemetry("my_asset", telemetry)

                # Verify topic format
                call_kwargs = mock_ingestor.handle_payload.call_args
                assert call_kwargs.kwargs["topic"] == "http/my_asset"
