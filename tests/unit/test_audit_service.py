"""Tests for audit logging service."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestAuditAction:
    """Tests for AuditAction enum."""

    def test_action_values(self):
        """Test action enum values."""
        from src.core.audit.service import AuditAction

        assert AuditAction.LOGIN.value == "auth.login"
        assert AuditAction.API_CALL.value == "api.call"
        assert AuditAction.CREATE.value == "data.create"
        assert AuditAction.SECURITY_ALERT.value == "security.alert"


class TestAuditLevel:
    """Tests for AuditLevel enum."""

    def test_level_values(self):
        """Test level enum values."""
        from src.core.audit.service import AuditLevel

        assert AuditLevel.DEBUG.value == "debug"
        assert AuditLevel.INFO.value == "info"
        assert AuditLevel.WARNING.value == "warning"
        assert AuditLevel.ERROR.value == "error"
        assert AuditLevel.CRITICAL.value == "critical"


class TestAuditActor:
    """Tests for AuditActor dataclass."""

    def test_default_values(self):
        """Test default actor values."""
        from src.core.audit.service import AuditActor

        actor = AuditActor(id="user-1")

        assert actor.id == "user-1"
        assert actor.type == "user"
        assert actor.name is None
        assert actor.tenant_id is None

    def test_custom_values(self):
        """Test custom actor values."""
        from src.core.audit.service import AuditActor

        actor = AuditActor(
            id="user-1",
            type="admin",
            name="John Doe",
            tenant_id="tenant-1",
            ip_address="192.168.1.1",
        )

        assert actor.type == "admin"
        assert actor.name == "John Doe"
        assert actor.tenant_id == "tenant-1"
        assert actor.ip_address == "192.168.1.1"


class TestAuditResource:
    """Tests for AuditResource dataclass."""

    def test_default_values(self):
        """Test default resource values."""
        from src.core.audit.service import AuditResource

        resource = AuditResource(type="document", id="doc-1")

        assert resource.type == "document"
        assert resource.id == "doc-1"
        assert resource.name is None
        assert resource.attributes == {}

    def test_with_attributes(self):
        """Test resource with attributes."""
        from src.core.audit.service import AuditResource

        resource = AuditResource(
            type="api_endpoint",
            id="/api/v1/users",
            name="Get Users",
            attributes={"method": "GET"},
        )

        assert resource.attributes["method"] == "GET"


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_to_dict(self):
        """Test event to dictionary conversion."""
        from src.core.audit.service import AuditActor, AuditEvent

        actor = AuditActor(id="user-1", type="user")
        event = AuditEvent(
            id="event-1",
            timestamp=1000.0,
            action="api.call",
            level="info",
            actor=actor,
            resource=None,
            details={"path": "/api/test"},
            outcome="success",
        )

        data = event.to_dict()

        assert data["id"] == "event-1"
        assert data["action"] == "api.call"
        assert data["actor"]["id"] == "user-1"
        assert data["details"]["path"] == "/api/test"
        assert "timestamp_iso" in data

    def test_to_json(self):
        """Test event to JSON conversion."""
        from src.core.audit.service import AuditActor, AuditEvent

        actor = AuditActor(id="user-1")
        event = AuditEvent(
            id="event-1",
            timestamp=1000.0,
            action="api.call",
            level="info",
            actor=actor,
            resource=None,
            details={},
            outcome="success",
        )

        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["id"] == "event-1"


class TestMemoryAuditStorage:
    """Tests for MemoryAuditStorage."""

    @pytest.mark.asyncio
    async def test_write_and_query(self):
        """Test writing and querying events."""
        from src.core.audit.service import AuditActor, AuditEvent, MemoryAuditStorage

        storage = MemoryAuditStorage()

        actor = AuditActor(id="user-1")
        event = AuditEvent(
            id="event-1",
            timestamp=time.time(),
            action="api.call",
            level="info",
            actor=actor,
            resource=None,
            details={},
            outcome="success",
        )

        await storage.write(event)
        results = await storage.query()

        assert len(results) == 1
        assert results[0].id == "event-1"

    @pytest.mark.asyncio
    async def test_query_with_filters(self):
        """Test querying with filters."""
        from src.core.audit.service import AuditActor, AuditEvent, MemoryAuditStorage

        storage = MemoryAuditStorage()

        # Write events for different users
        for i in range(5):
            actor = AuditActor(id=f"user-{i % 2}")
            event = AuditEvent(
                id=f"event-{i}",
                timestamp=time.time(),
                action="api.call",
                level="info",
                actor=actor,
                resource=None,
                details={},
                outcome="success",
            )
            await storage.write(event)

        # Query for specific user
        results = await storage.query(actor_id="user-0")

        assert len(results) == 3  # user-0, user-0, user-0 (indices 0, 2, 4)

    @pytest.mark.asyncio
    async def test_max_events_limit(self):
        """Test max events limit is enforced."""
        from src.core.audit.service import AuditActor, AuditEvent, MemoryAuditStorage

        storage = MemoryAuditStorage(max_events=5)

        # Write more than max events
        for i in range(10):
            actor = AuditActor(id="user-1")
            event = AuditEvent(
                id=f"event-{i}",
                timestamp=time.time(),
                action="api.call",
                level="info",
                actor=actor,
                resource=None,
                details={},
                outcome="success",
            )
            await storage.write(event)

        results = await storage.query(limit=100)
        assert len(results) == 5


class TestFileAuditStorage:
    """Tests for FileAuditStorage."""

    @pytest.mark.asyncio
    async def test_write_creates_file(self):
        """Test writing creates log file."""
        from src.core.audit.service import AuditActor, AuditEvent, FileAuditStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileAuditStorage(log_dir=tmpdir)

            actor = AuditActor(id="user-1")
            event = AuditEvent(
                id="event-1",
                timestamp=time.time(),
                action="api.call",
                level="info",
                actor=actor,
                resource=None,
                details={},
                outcome="success",
            )

            await storage.write(event)

            # Check file was created
            log_files = list(Path(tmpdir).glob("audit_*.jsonl"))
            assert len(log_files) == 1

    @pytest.mark.asyncio
    async def test_write_and_query(self):
        """Test writing and querying from file."""
        from src.core.audit.service import AuditActor, AuditEvent, FileAuditStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileAuditStorage(log_dir=tmpdir)

            actor = AuditActor(id="user-1", tenant_id="tenant-1")
            event = AuditEvent(
                id="event-1",
                timestamp=time.time(),
                action="api.call",
                level="info",
                actor=actor,
                resource=None,
                details={"test": "data"},
                outcome="success",
            )

            await storage.write(event)
            results = await storage.query()

            assert len(results) == 1
            assert results[0].id == "event-1"
            assert results[0].actor.id == "user-1"


class TestAuditLogger:
    """Tests for AuditLogger."""

    @pytest.mark.asyncio
    async def test_log_creates_event(self):
        """Test logging creates an event."""
        from src.core.audit.service import (
            AuditAction,
            AuditActor,
            AuditLogger,
            MemoryAuditStorage,
        )

        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)

        actor = AuditActor(id="user-1")
        event_id = await logger.log(
            action=AuditAction.API_CALL,
            actor=actor,
            outcome="success",
        )

        assert event_id is not None

        # Flush buffer
        await logger._flush()

        # Query to verify
        results = await storage.query()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_log_with_resource(self):
        """Test logging with resource."""
        from src.core.audit.service import (
            AuditAction,
            AuditActor,
            AuditLogger,
            AuditResource,
            MemoryAuditStorage,
        )

        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)

        actor = AuditActor(id="user-1")
        resource = AuditResource(type="document", id="doc-1")

        await logger.log(
            action=AuditAction.READ,
            actor=actor,
            resource=resource,
            outcome="success",
        )

        await logger._flush()

        results = await storage.query()
        assert results[0].resource is not None
        assert results[0].resource.type == "document"

    @pytest.mark.asyncio
    async def test_event_handler(self):
        """Test event handlers are called."""
        from src.core.audit.service import (
            AuditAction,
            AuditActor,
            AuditLogger,
            MemoryAuditStorage,
        )

        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)

        received_events = []

        def handler(event):
            received_events.append(event)

        logger.add_handler(handler)

        actor = AuditActor(id="user-1")
        await logger.log(
            action=AuditAction.LOGIN,
            actor=actor,
            outcome="success",
        )

        assert len(received_events) == 1
        assert received_events[0].action == "auth.login"

    @pytest.mark.asyncio
    async def test_batch_flush(self):
        """Test batch flushing."""
        from src.core.audit.service import (
            AuditAction,
            AuditActor,
            AuditLogger,
            MemoryAuditStorage,
        )

        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage, batch_size=5)

        actor = AuditActor(id="user-1")

        # Log 5 events (should trigger flush)
        for _ in range(5):
            await logger.log(
                action=AuditAction.API_CALL,
                actor=actor,
                outcome="success",
            )

        # Give time for async flush
        await asyncio.sleep(0.1)

        results = await storage.query()
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting metrics."""
        from src.core.audit.service import (
            AuditAction,
            AuditActor,
            AuditLogger,
            MemoryAuditStorage,
        )

        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)

        actor = AuditActor(id="user-1")
        await logger.log(
            action=AuditAction.API_CALL,
            actor=actor,
            outcome="success",
        )

        metrics = logger.get_metrics()

        assert metrics["events_logged"] == 1
        assert metrics["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test start and stop."""
        from src.core.audit.service import AuditLogger, MemoryAuditStorage

        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)

        await logger.start()
        assert logger._running is True

        await logger.stop()
        assert logger._running is False

    @pytest.mark.asyncio
    async def test_query(self):
        """Test query method."""
        from datetime import datetime

        from src.core.audit.service import (
            AuditAction,
            AuditActor,
            AuditLogger,
            MemoryAuditStorage,
        )

        storage = MemoryAuditStorage()
        logger = AuditLogger(storage=storage)

        actor = AuditActor(id="user-1")
        await logger.log(
            action=AuditAction.API_CALL,
            actor=actor,
            outcome="success",
        )

        await logger._flush()

        results = await logger.query(action=AuditAction.API_CALL)
        assert len(results) == 1


class TestGetAuditLogger:
    """Tests for get_audit_logger function."""

    @pytest.mark.asyncio
    async def test_returns_singleton(self):
        """Test returns singleton instance."""
        from src.core.audit import service as audit_module

        # Reset global
        audit_module._audit_logger = None

        logger1 = audit_module.get_audit_logger()
        logger2 = audit_module.get_audit_logger()

        assert logger1 is logger2

        # Cleanup
        audit_module._audit_logger = None
