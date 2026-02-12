"""Audit Event Logger.

Provides structured audit logging with multiple backends.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_LOGIN_FAILED = "auth.login_failed"
    AUTH_MFA_ENABLED = "auth.mfa_enabled"
    AUTH_PASSWORD_CHANGED = "auth.password_changed"
    AUTH_API_KEY_CREATED = "auth.api_key_created"
    AUTH_API_KEY_REVOKED = "auth.api_key_revoked"

    # Authorization
    AUTHZ_PERMISSION_GRANTED = "authz.permission_granted"
    AUTHZ_PERMISSION_DENIED = "authz.permission_denied"
    AUTHZ_ROLE_ASSIGNED = "authz.role_assigned"
    AUTHZ_ROLE_REMOVED = "authz.role_removed"

    # Data access
    DATA_READ = "data.read"
    DATA_CREATE = "data.create"
    DATA_UPDATE = "data.update"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"

    # Document operations
    DOC_UPLOADED = "document.uploaded"
    DOC_DOWNLOADED = "document.downloaded"
    DOC_PROCESSED = "document.processed"
    DOC_DELETED = "document.deleted"
    DOC_SHARED = "document.shared"

    # Model operations
    MODEL_TRAINED = "model.trained"
    MODEL_DEPLOYED = "model.deployed"
    MODEL_PREDICTION = "model.prediction"
    MODEL_DELETED = "model.deleted"
    MODEL_VERSIONED = "model.versioned"

    # System events
    SYS_CONFIG_CHANGED = "system.config_changed"
    SYS_SERVICE_STARTED = "system.service_started"
    SYS_SERVICE_STOPPED = "system.service_stopped"
    SYS_ERROR = "system.error"
    SYS_SECURITY_ALERT = "system.security_alert"

    # Admin operations
    ADMIN_USER_CREATED = "admin.user_created"
    ADMIN_USER_DELETED = "admin.user_deleted"
    ADMIN_USER_MODIFIED = "admin.user_modified"
    ADMIN_TENANT_CREATED = "admin.tenant_created"
    ADMIN_TENANT_MODIFIED = "admin.tenant_modified"

    # Compliance
    COMPLIANCE_CONSENT_GIVEN = "compliance.consent_given"
    COMPLIANCE_CONSENT_REVOKED = "compliance.consent_revoked"
    COMPLIANCE_DATA_REQUEST = "compliance.data_request"
    COMPLIANCE_DATA_DELETION = "compliance.data_deletion"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditContext:
    """Context information for audit events."""

    user_id: Optional[str] = None
    username: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    api_key_id: Optional[str] = None
    service_name: Optional[str] = None


@dataclass
class AuditEvent:
    """Structured audit event."""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    severity: AuditSeverity
    context: AuditContext
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"  # success, failure, error
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Integrity
    checksum: Optional[str] = None

    def compute_checksum(self, secret: str) -> str:
        """Compute integrity checksum."""
        data = f"{self.event_id}:{self.event_type}:{self.timestamp.isoformat()}:{self.action}"
        return hashlib.sha256(f"{data}:{secret}".encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "context": asdict(self.context),
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "outcome": self.outcome,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditBackend:
    """Base class for audit backends."""

    async def write(self, event: AuditEvent) -> None:
        """Write an audit event."""
        raise NotImplementedError

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query audit events."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the backend."""
        pass


class MemoryAuditBackend(AuditBackend):
    """In-memory audit backend for testing."""

    def __init__(self, max_events: int = 10000):
        self._events: List[AuditEvent] = []
        self._max_events = max_events
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def write(self, event: AuditEvent) -> None:
        async with self._get_lock():
            self._events.append(event)
            # Trim if needed
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        async with self._get_lock():
            results = []
            for event in reversed(self._events):
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if event_types and event.event_type not in event_types:
                    continue
                if user_id and event.context.user_id != user_id:
                    continue
                if tenant_id and event.context.tenant_id != tenant_id:
                    continue
                results.append(event)
                if len(results) >= limit:
                    break
            return results


class FileAuditBackend(AuditBackend):
    """File-based audit backend."""

    def __init__(self, log_dir: str, rotate_size_mb: int = 100):
        self.log_dir = log_dir
        self.rotate_size_mb = rotate_size_mb
        self._current_file: Optional[str] = None
        self._file_handle: Optional[Any] = None
        self._lock: Optional[asyncio.Lock] = None

        os.makedirs(log_dir, exist_ok=True)

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_current_file(self) -> str:
        """Get current log file path."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"audit-{date_str}.jsonl")

    async def write(self, event: AuditEvent) -> None:
        async with self._get_lock():
            file_path = self._get_current_file()

            # Check if we need to rotate
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if size_mb >= self.rotate_size_mb:
                    # Add timestamp suffix
                    timestamp = datetime.utcnow().strftime("%H%M%S")
                    new_path = file_path.replace(".jsonl", f"-{timestamp}.jsonl")
                    os.rename(file_path, new_path)

            # Write event
            with open(file_path, "a") as f:
                f.write(event.to_json() + "\n")

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        # Simple implementation - reads all files
        results = []
        for filename in sorted(os.listdir(self.log_dir), reverse=True):
            if not filename.endswith(".jsonl"):
                continue

            file_path = os.path.join(self.log_dir, filename)
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        event = self._dict_to_event(data)

                        # Apply filters
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue
                        if event_types and event.event_type not in event_types:
                            continue
                        if user_id and event.context.user_id != user_id:
                            continue
                        if tenant_id and event.context.tenant_id != tenant_id:
                            continue

                        results.append(event)
                        if len(results) >= limit:
                            return results
                    except Exception:
                        continue

        return results

    def _dict_to_event(self, data: Dict[str, Any]) -> AuditEvent:
        """Convert dictionary to AuditEvent."""
        context = AuditContext(**data.get("context", {}))
        return AuditEvent(
            event_id=data["event_id"],
            event_type=AuditEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            severity=AuditSeverity(data["severity"]),
            context=context,
            action=data["action"],
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            details=data.get("details", {}),
            outcome=data.get("outcome", "success"),
            duration_ms=data.get("duration_ms"),
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum"),
        )


class AuditLogger:
    """Main audit logger class."""

    def __init__(
        self,
        backends: Optional[List[AuditBackend]] = None,
        integrity_secret: Optional[str] = None,
        default_service: str = "cad-ml-platform",
    ):
        self._backends = backends or [MemoryAuditBackend()]
        self._integrity_secret = integrity_secret or os.getenv("AUDIT_SECRET", "default-secret")
        self._default_service = default_service
        self._context_stack: List[AuditContext] = []

    def push_context(self, context: AuditContext) -> None:
        """Push context onto stack."""
        self._context_stack.append(context)

    def pop_context(self) -> Optional[AuditContext]:
        """Pop context from stack."""
        if self._context_stack:
            return self._context_stack.pop()
        return None

    def get_current_context(self) -> AuditContext:
        """Get current context."""
        if self._context_stack:
            return self._context_stack[-1]
        return AuditContext(service_name=self._default_service)

    async def log(
        self,
        event_type: AuditEventType,
        action: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        outcome: str = "success",
        duration_ms: Optional[int] = None,
        context_override: Optional[AuditContext] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log an audit event.

        Args:
            event_type: Type of event
            action: Human-readable action description
            severity: Event severity
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            details: Additional event details
            outcome: Event outcome (success/failure/error)
            duration_ms: Operation duration
            context_override: Override current context
            metadata: Additional metadata

        Returns:
            Created AuditEvent
        """
        context = context_override or self.get_current_context()

        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            severity=severity,
            context=context,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            outcome=outcome,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        # Compute integrity checksum
        event.checksum = event.compute_checksum(self._integrity_secret)

        # Write to all backends
        for backend in self._backends:
            try:
                await backend.write(event)
            except Exception as e:
                logger.error(f"Failed to write audit event to backend: {e}")

        # Also log to standard logger
        log_level = getattr(logging, severity.value.upper(), logging.INFO)
        logger.log(log_level, f"AUDIT: {event_type.value} - {action}", extra={"audit_event": event.to_dict()})

        return event

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query audit events from primary backend."""
        if self._backends:
            return await self._backends[0].query(
                start_time=start_time,
                end_time=end_time,
                event_types=event_types,
                user_id=user_id,
                tenant_id=tenant_id,
                limit=limit,
            )
        return []

    async def close(self) -> None:
        """Close all backends."""
        for backend in self._backends:
            await backend.close()


# Global audit logger
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def set_audit_logger(audit_logger: AuditLogger) -> None:
    """Set global audit logger."""
    global _audit_logger
    _audit_logger = audit_logger


def audit_log(
    event_type: AuditEventType,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    severity: AuditSeverity = AuditSeverity.INFO,
) -> Callable[[F], F]:
    """Decorator for automatic audit logging.

    Args:
        event_type: Type of audit event
        action: Action description (defaults to function name)
        resource_type: Type of resource
        severity: Event severity

    Example:
        @audit_log(AuditEventType.DATA_READ, resource_type="document")
        async def get_document(doc_id: str):
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            outcome = "success"
            error_detail = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                outcome = "error"
                error_detail = str(e)
                raise
            finally:
                duration_ms = int((time.time() - start_time) * 1000)

                # Try to extract resource_id from kwargs
                resource_id = kwargs.get("id") or kwargs.get("doc_id") or kwargs.get("resource_id")

                details = {}
                if error_detail:
                    details["error"] = error_detail

                audit_logger = get_audit_logger()
                await audit_logger.log(
                    event_type=event_type,
                    action=action or func.__name__,
                    severity=severity if outcome == "success" else AuditSeverity.ERROR,
                    resource_type=resource_type,
                    resource_id=str(resource_id) if resource_id else None,
                    details=details,
                    outcome=outcome,
                    duration_ms=duration_ms,
                )

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            outcome = "success"
            error_detail = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                outcome = "error"
                error_detail = str(e)
                raise
            finally:
                duration_ms = int((time.time() - start_time) * 1000)

                resource_id = kwargs.get("id") or kwargs.get("doc_id") or kwargs.get("resource_id")

                details = {}
                if error_detail:
                    details["error"] = error_detail

                # Run async log in sync context
                loop = asyncio.new_event_loop()
                try:
                    audit_logger = get_audit_logger()
                    loop.run_until_complete(
                        audit_logger.log(
                            event_type=event_type,
                            action=action or func.__name__,
                            severity=severity if outcome == "success" else AuditSeverity.ERROR,
                            resource_type=resource_type,
                            resource_id=str(resource_id) if resource_id else None,
                            details=details,
                            outcome=outcome,
                            duration_ms=duration_ms,
                        )
                    )
                finally:
                    loop.close()

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class AuditMiddleware:
    """FastAPI middleware for request audit logging."""

    def __init__(
        self,
        app: Any,
        audit_logger: Optional[AuditLogger] = None,
        exclude_paths: Optional[set] = None,
    ):
        self.app = app
        self.audit_logger = audit_logger or get_audit_logger()
        self.exclude_paths = exclude_paths or {"/health", "/metrics", "/favicon.ico"}

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.exclude_paths:
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = 200

        # Capture response status
        original_send = send

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
            await original_send(message)

        # Extract context
        headers = dict(scope.get("headers", []))
        client = scope.get("client", (None, None))

        context = AuditContext(
            user_id=headers.get(b"x-user-id", b"").decode() or None,
            tenant_id=headers.get(b"x-tenant-id", b"").decode() or None,
            request_id=headers.get(b"x-request-id", b"").decode() or None,
            ip_address=client[0],
            user_agent=headers.get(b"user-agent", b"").decode() or None,
            api_key_id=headers.get(b"x-api-key-id", b"").decode() or None,
        )

        self.audit_logger.push_context(context)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self.audit_logger.pop_context()

            duration_ms = int((time.time() - start_time) * 1000)
            method = scope.get("method", "GET")

            # Determine event type based on method
            if method == "GET":
                event_type = AuditEventType.DATA_READ
            elif method == "POST":
                event_type = AuditEventType.DATA_CREATE
            elif method in ("PUT", "PATCH"):
                event_type = AuditEventType.DATA_UPDATE
            elif method == "DELETE":
                event_type = AuditEventType.DATA_DELETE
            else:
                event_type = AuditEventType.DATA_READ

            outcome = "success" if status_code < 400 else "failure" if status_code < 500 else "error"
            severity = AuditSeverity.INFO if outcome == "success" else AuditSeverity.WARNING

            await self.audit_logger.log(
                event_type=event_type,
                action=f"{method} {path}",
                severity=severity,
                resource_type="api",
                resource_id=path,
                details={"status_code": status_code, "method": method},
                outcome=outcome,
                duration_ms=duration_ms,
                context_override=context,
            )
