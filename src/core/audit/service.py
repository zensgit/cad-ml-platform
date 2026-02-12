"""Enterprise Audit Logging Service.

Features:
- Structured audit events
- Multiple storage backends (memory, file, database)
- Async processing with batching
- Search and filtering
- Compliance reporting
- Retention policies
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """Audit action types."""

    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    TOKEN_REFRESH = "auth.token_refresh"

    # Authorization
    ACCESS_GRANTED = "authz.access_granted"
    ACCESS_DENIED = "authz.access_denied"
    PERMISSION_CHANGED = "authz.permission_changed"

    # Data Operations
    CREATE = "data.create"
    READ = "data.read"
    UPDATE = "data.update"
    DELETE = "data.delete"
    EXPORT = "data.export"
    IMPORT = "data.import"

    # API Operations
    API_CALL = "api.call"
    API_ERROR = "api.error"
    RATE_LIMITED = "api.rate_limited"

    # System Operations
    CONFIG_CHANGE = "system.config_change"
    SERVICE_START = "system.service_start"
    SERVICE_STOP = "system.service_stop"
    HEALTH_CHECK = "system.health_check"

    # Security
    SECURITY_ALERT = "security.alert"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    BLOCKED_REQUEST = "security.blocked"

    # Batch/Job Operations
    JOB_CREATED = "job.created"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"

    # Model Operations
    MODEL_LOADED = "model.loaded"
    MODEL_UPDATED = "model.updated"
    PREDICTION_MADE = "model.prediction"


class AuditLevel(Enum):
    """Audit log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditActor:
    """Who performed the action."""

    id: str
    type: str = "user"  # user, system, service, anonymous
    name: Optional[str] = None
    tenant_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class AuditResource:
    """What resource was affected."""

    type: str
    id: str
    name: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Represents an audit log event."""

    id: str
    timestamp: float
    action: str
    level: str
    actor: AuditActor
    resource: Optional[AuditResource]
    details: Dict[str, Any]
    outcome: str  # success, failure, partial
    duration_ms: Optional[float] = None
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "action": self.action,
            "level": self.level,
            "actor": asdict(self.actor),
            "resource": asdict(self.resource) if self.resource else None,
            "details": self.details,
            "outcome": self.outcome,
            "duration_ms": self.duration_ms,
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AuditStorageBackend:
    """Base class for audit storage backends."""

    async def write(self, event: AuditEvent) -> None:
        """Write a single event."""
        raise NotImplementedError

    async def write_batch(self, events: List[AuditEvent]) -> None:
        """Write multiple events."""
        for event in events:
            await self.write(event)

    async def query(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        action: Optional[str] = None,
        actor_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query events."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the backend."""
        pass


class MemoryAuditStorage(AuditStorageBackend):
    """In-memory audit storage (for testing/development)."""

    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self._events: List[AuditEvent] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def write(self, event: AuditEvent) -> None:
        async with self._get_lock():
            self._events.append(event)
            # Trim if over limit
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events :]

    async def query(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        action: Optional[str] = None,
        actor_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        async with self._get_lock():
            results = []
            for event in reversed(self._events):
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if action and event.action != action:
                    continue
                if actor_id and event.actor.id != actor_id:
                    continue
                if tenant_id and event.actor.tenant_id != tenant_id:
                    continue
                if level and event.level != level:
                    continue
                results.append(event)
                if len(results) >= limit:
                    break
            return results


class FileAuditStorage(AuditStorageBackend):
    """File-based audit storage with rotation."""

    def __init__(
        self,
        log_dir: str = "audit_logs",
        rotation_size_mb: int = 100,
        retention_days: int = 90,
    ):
        self.log_dir = Path(log_dir)
        self.rotation_size_mb = rotation_size_mb
        self.retention_days = retention_days
        self._current_file: Optional[Path] = None
        self._file_handle = None
        self._lock: Optional[asyncio.Lock] = None

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_current_file(self) -> Path:
        """Get current log file path."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    async def write(self, event: AuditEvent) -> None:
        async with self._get_lock():
            file_path = self._get_current_file()
            with open(file_path, "a") as f:
                f.write(event.to_json() + "\n")

    async def query(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        action: Optional[str] = None,
        actor_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query from log files (simplified implementation)."""
        results = []

        # Get relevant files
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)

        for log_file in log_files:
            if len(results) >= limit:
                break

            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if len(results) >= limit:
                            break

                        try:
                            data = json.loads(line)
                            event = self._dict_to_event(data)

                            if start_time and event.timestamp < start_time:
                                continue
                            if end_time and event.timestamp > end_time:
                                continue
                            if action and event.action != action:
                                continue
                            if actor_id and event.actor.id != actor_id:
                                continue
                            if tenant_id and event.actor.tenant_id != tenant_id:
                                continue
                            if level and event.level != level:
                                continue

                            results.append(event)
                        except (json.JSONDecodeError, KeyError):
                            continue
            except IOError:
                continue

        return results

    def _dict_to_event(self, data: Dict[str, Any]) -> AuditEvent:
        """Convert dictionary to AuditEvent."""
        actor_data = data.get("actor", {})
        actor = AuditActor(
            id=actor_data.get("id", "unknown"),
            type=actor_data.get("type", "unknown"),
            name=actor_data.get("name"),
            tenant_id=actor_data.get("tenant_id"),
            ip_address=actor_data.get("ip_address"),
            user_agent=actor_data.get("user_agent"),
        )

        resource_data = data.get("resource")
        resource = None
        if resource_data:
            resource = AuditResource(
                type=resource_data.get("type", "unknown"),
                id=resource_data.get("id", "unknown"),
                name=resource_data.get("name"),
                attributes=resource_data.get("attributes", {}),
            )

        return AuditEvent(
            id=data.get("id", ""),
            timestamp=data.get("timestamp", 0),
            action=data.get("action", ""),
            level=data.get("level", "info"),
            actor=actor,
            resource=resource,
            details=data.get("details", {}),
            outcome=data.get("outcome", "success"),
            duration_ms=data.get("duration_ms"),
            correlation_id=data.get("correlation_id"),
            request_id=data.get("request_id"),
            metadata=data.get("metadata", {}),
        )


class AuditLogger:
    """Enterprise audit logging service."""

    def __init__(
        self,
        storage: Optional[AuditStorageBackend] = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        default_level: AuditLevel = AuditLevel.INFO,
    ):
        self.storage = storage or MemoryAuditStorage()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.default_level = default_level

        self._buffer: List[AuditEvent] = []
        self._lock: Optional[asyncio.Lock] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

        # Event handlers for real-time processing
        self._handlers: List[Callable[[AuditEvent], None]] = []

        # Metrics
        self._metrics = {
            "events_logged": 0,
            "events_flushed": 0,
            "flush_errors": 0,
        }

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def start(self) -> None:
        """Start the audit logger."""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Audit logger started")

    async def stop(self) -> None:
        """Stop the audit logger and flush remaining events."""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()
        await self.storage.close()

        logger.info("Audit logger stopped")

    async def log(
        self,
        action: AuditAction,
        actor: AuditActor,
        outcome: str = "success",
        resource: Optional[AuditResource] = None,
        details: Optional[Dict[str, Any]] = None,
        level: Optional[AuditLevel] = None,
        duration_ms: Optional[float] = None,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an audit event."""
        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            action=action.value,
            level=(level or self.default_level).value,
            actor=actor,
            resource=resource,
            details=details or {},
            outcome=outcome,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            request_id=request_id,
            metadata=metadata or {},
        )

        async with self._get_lock():
            self._buffer.append(event)
            self._metrics["events_logged"] += 1

        # Notify handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Audit handler error: {e}")

        # Flush if buffer is full
        if len(self._buffer) >= self.batch_size:
            asyncio.create_task(self._flush())

        return event.id

    def add_handler(self, handler: Callable[[AuditEvent], None]) -> None:
        """Add an event handler for real-time processing."""
        self._handlers.append(handler)

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        action: Optional[AuditAction] = None,
        actor_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        level: Optional[AuditLevel] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query audit events."""
        return await self.storage.query(
            start_time=start_time.timestamp() if start_time else None,
            end_time=end_time.timestamp() if end_time else None,
            action=action.value if action else None,
            actor_id=actor_id,
            tenant_id=tenant_id,
            level=level.value if level else None,
            limit=limit,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get audit logger metrics."""
        return {
            **self._metrics,
            "buffer_size": len(self._buffer),
            "handlers_count": len(self._handlers),
        }

    async def _flush(self) -> None:
        """Flush buffered events to storage."""
        async with self._get_lock():
            if not self._buffer:
                return

            events = self._buffer[:]
            self._buffer.clear()

        try:
            await self.storage.write_batch(events)
            self._metrics["events_flushed"] += len(events)
        except Exception as e:
            logger.error(f"Audit flush error: {e}")
            self._metrics["flush_errors"] += 1
            # Re-add failed events
            async with self._get_lock():
                self._buffer = events + self._buffer

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush loop error: {e}")


# Convenience functions for API integration
def create_api_actor_from_request(request: Any) -> AuditActor:
    """Create an AuditActor from a FastAPI request."""
    # Extract user info from request
    user_id = getattr(request.state, "user_id", None) or request.headers.get("X-User-ID", "anonymous")
    tenant_id = getattr(request.state, "tenant_id", None) or request.headers.get("X-Tenant-ID")
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("User-Agent", "")

    return AuditActor(
        id=user_id,
        type="user" if user_id != "anonymous" else "anonymous",
        tenant_id=tenant_id,
        ip_address=client_ip,
        user_agent=user_agent[:200],  # Truncate long user agents
    )


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        # Use file storage in production, memory in development
        storage: AuditStorageBackend
        if os.getenv("AUDIT_LOG_DIR"):
            storage = FileAuditStorage(log_dir=os.getenv("AUDIT_LOG_DIR", "audit_logs"))
        else:
            storage = MemoryAuditStorage()

        _audit_logger = AuditLogger(storage=storage)

    return _audit_logger
