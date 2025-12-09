"""Audit logging module for Vision Provider system.

This module provides comprehensive audit trail capabilities including:
- Request/response logging
- User activity tracking
- Compliance audit trails
- Data retention policies
- Audit log export
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from .base import VisionDescription, VisionProvider


class AuditEventType(Enum):
    """Types of audit events."""

    REQUEST_STARTED = "request_started"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"
    PROVIDER_SELECTED = "provider_selected"
    PROVIDER_FALLBACK = "provider_fallback"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    VALIDATION_FAILED = "validation_failed"
    CONFIG_CHANGED = "config_changed"
    ACCESS_DENIED = "access_denied"
    DATA_EXPORTED = "data_exported"
    DATA_DELETED = "data_deleted"


class AuditLevel(Enum):
    """Audit logging levels."""

    MINIMAL = "minimal"  # Only critical events
    STANDARD = "standard"  # Normal operational events
    DETAILED = "detailed"  # Include request/response data
    DEBUG = "debug"  # Maximum detail for troubleshooting


@dataclass
class AuditEntry:
    """A single audit log entry."""

    entry_id: str
    timestamp: datetime
    event_type: AuditEventType
    provider_name: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    client_ip: Optional[str] = None
    action: str = ""
    resource: str = ""
    outcome: str = ""
    duration_ms: Optional[float] = None
    request_hash: Optional[str] = None
    response_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "provider_name": self.provider_name,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "client_ip": self.client_ip,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "duration_ms": self.duration_ms,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "metadata": self.metadata,
            "details": self.details,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class AuditContext:
    """Context for audit logging."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    client_ip: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""

    name: str
    retention_days: int = 90
    archive_after_days: int = 30
    compress_archived: bool = True
    delete_on_expiry: bool = True
    exempt_event_types: List[AuditEventType] = field(default_factory=list)


@dataclass
class AuditConfig:
    """Audit logging configuration."""

    level: AuditLevel = AuditLevel.STANDARD
    retention_policy: RetentionPolicy = field(
        default_factory=lambda: RetentionPolicy(name="default")
    )
    hash_sensitive_data: bool = True
    log_request_data: bool = False
    log_response_data: bool = False
    max_entries_in_memory: int = 10000
    flush_interval_seconds: int = 60
    enabled_event_types: Optional[List[AuditEventType]] = None
    disabled_event_types: List[AuditEventType] = field(default_factory=list)


class AuditStorage(ABC):
    """Abstract base for audit log storage."""

    @abstractmethod
    def store(self, entry: AuditEntry) -> None:
        """Store an audit entry."""
        pass

    @abstractmethod
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query audit entries."""
        pass

    @abstractmethod
    def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> int:
        """Count matching entries."""
        pass

    @abstractmethod
    def delete_before(self, before: datetime) -> int:
        """Delete entries before a timestamp."""
        pass


class InMemoryAuditStorage(AuditStorage):
    """In-memory audit storage implementation."""

    def __init__(self, max_entries: int = 10000) -> None:
        """Initialize in-memory storage."""
        self._entries: List[AuditEntry] = []
        self._max_entries = max_entries

    def store(self, entry: AuditEntry) -> None:
        """Store an audit entry."""
        self._entries.append(entry)

        # Trim if exceeding max
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query audit entries."""
        results = self._entries

        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        if event_types:
            results = [e for e in results if e.event_type in event_types]
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if request_id:
            results = [e for e in results if e.request_id == request_id]

        return results[offset:offset + limit]

    def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> int:
        """Count matching entries."""
        results = self._entries

        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        if event_types:
            results = [e for e in results if e.event_type in event_types]

        return len(results)

    def delete_before(self, before: datetime) -> int:
        """Delete entries before a timestamp."""
        original_count = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= before]
        return original_count - len(self._entries)

    def get_all(self) -> List[AuditEntry]:
        """Get all entries."""
        return list(self._entries)

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()


class AuditLogger:
    """Main audit logging system."""

    def __init__(
        self,
        config: Optional[AuditConfig] = None,
        storage: Optional[AuditStorage] = None,
    ) -> None:
        """Initialize the audit logger.

        Args:
            config: Audit configuration
            storage: Storage backend
        """
        self._config = config or AuditConfig()
        self._storage = storage or InMemoryAuditStorage(
            max_entries=self._config.max_entries_in_memory
        )
        self._entry_counter = 0
        self._listeners: List[Callable[[AuditEntry], None]] = []

    def log(
        self,
        event_type: AuditEventType,
        action: str = "",
        resource: str = "",
        outcome: str = "",
        context: Optional[AuditContext] = None,
        provider_name: Optional[str] = None,
        duration_ms: Optional[float] = None,
        request_data: Optional[bytes] = None,
        response_data: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log an audit event.

        Args:
            event_type: Type of audit event
            action: Action performed
            resource: Resource affected
            outcome: Outcome of the action
            context: Audit context
            provider_name: Name of provider used
            duration_ms: Operation duration
            request_data: Raw request data (will be hashed if configured)
            response_data: Response data (will be hashed if configured)
            details: Additional details
            metadata: Extra metadata

        Returns:
            The created audit entry
        """
        # Check if event type is enabled
        if not self._should_log(event_type):
            return None

        self._entry_counter += 1
        entry_id = f"audit_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._entry_counter}"

        ctx = context or AuditContext()

        # Hash sensitive data if configured
        request_hash = None
        response_hash = None

        if request_data and self._config.hash_sensitive_data:
            request_hash = hashlib.sha256(request_data).hexdigest()[:16]

        if response_data and self._config.hash_sensitive_data:
            response_hash = hashlib.sha256(response_data.encode()).hexdigest()[:16]

        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=datetime.now(),
            event_type=event_type,
            provider_name=provider_name,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            request_id=ctx.request_id,
            client_ip=ctx.client_ip,
            action=action,
            resource=resource,
            outcome=outcome,
            duration_ms=duration_ms,
            request_hash=request_hash,
            response_hash=response_hash,
            metadata={**ctx.metadata, **(metadata or {})},
            details=details or {},
        )

        # Store the entry
        self._storage.store(entry)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(entry)
            except Exception:
                pass  # Don't let listener errors affect audit logging

        return entry

    def _should_log(self, event_type: AuditEventType) -> bool:
        """Check if an event type should be logged."""
        if event_type in self._config.disabled_event_types:
            return False

        if self._config.enabled_event_types is not None:
            return event_type in self._config.enabled_event_types

        # Filter by audit level
        if self._config.level == AuditLevel.MINIMAL:
            critical_events = {
                AuditEventType.REQUEST_FAILED,
                AuditEventType.ACCESS_DENIED,
                AuditEventType.VALIDATION_FAILED,
                AuditEventType.CONFIG_CHANGED,
                AuditEventType.DATA_DELETED,
            }
            return event_type in critical_events

        return True

    def add_listener(self, listener: Callable[[AuditEntry], None]) -> None:
        """Add an audit event listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[AuditEntry], None]) -> None:
        """Remove an audit event listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query audit entries."""
        return self._storage.query(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            user_id=user_id,
            request_id=request_id,
            limit=limit,
            offset=offset,
        )

    def count(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> int:
        """Count matching entries."""
        return self._storage.count(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
        )

    def apply_retention_policy(self) -> int:
        """Apply the retention policy and delete old entries."""
        policy = self._config.retention_policy
        cutoff = datetime.now() - timedelta(days=policy.retention_days)

        # Don't delete exempt event types
        # For simplicity, we just delete all old entries
        # A real implementation would be more sophisticated
        return self._storage.delete_before(cutoff)

    def export(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json",
    ) -> str:
        """Export audit entries.

        Args:
            start_time: Start of export range
            end_time: End of export range
            format: Export format (json, csv)

        Returns:
            Exported data as string
        """
        entries = self._storage.query(
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        if format == "json":
            return json.dumps([e.to_dict() for e in entries], indent=2)
        elif format == "csv":
            if not entries:
                return ""

            headers = [
                "entry_id", "timestamp", "event_type", "provider_name",
                "user_id", "action", "resource", "outcome", "duration_ms"
            ]
            lines = [",".join(headers)]

            for entry in entries:
                row = [
                    entry.entry_id,
                    entry.timestamp.isoformat(),
                    entry.event_type.value,
                    entry.provider_name or "",
                    entry.user_id or "",
                    entry.action,
                    entry.resource,
                    entry.outcome,
                    str(entry.duration_ms or ""),
                ]
                lines.append(",".join(f'"{v}"' for v in row))

            return "\n".join(lines)

        return json.dumps([e.to_dict() for e in entries])


class AuditingVisionProvider(VisionProvider):
    """Vision provider wrapper with audit logging."""

    def __init__(
        self,
        provider: VisionProvider,
        audit_logger: AuditLogger,
        context_provider: Optional[Callable[[], AuditContext]] = None,
    ) -> None:
        """Initialize the auditing provider.

        Args:
            provider: The underlying provider
            audit_logger: The audit logger
            context_provider: Optional function to get audit context
        """
        self._provider = provider
        self._logger = audit_logger
        self._context_provider = context_provider

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"auditing_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with audit logging.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        import time
        import uuid

        request_id = str(uuid.uuid4())
        context = self._context_provider() if self._context_provider else AuditContext()
        context.request_id = request_id

        # Log request started
        self._logger.log(
            event_type=AuditEventType.REQUEST_STARTED,
            action="analyze_image",
            resource="image",
            outcome="started",
            context=context,
            provider_name=self._provider.provider_name,
            request_data=image_data,
            details={
                "image_size": len(image_data),
                "include_description": include_description,
            },
        )

        start_time = time.time()

        try:
            result = await self._provider.analyze_image(image_data, include_description)
            duration_ms = (time.time() - start_time) * 1000

            # Log request completed
            self._logger.log(
                event_type=AuditEventType.REQUEST_COMPLETED,
                action="analyze_image",
                resource="image",
                outcome="success",
                context=context,
                provider_name=self._provider.provider_name,
                duration_ms=duration_ms,
                response_data=result.summary,
                details={
                    "confidence": result.confidence,
                    "details_count": len(result.details) if result.details else 0,
                },
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Log request failed
            self._logger.log(
                event_type=AuditEventType.REQUEST_FAILED,
                action="analyze_image",
                resource="image",
                outcome="failure",
                context=context,
                provider_name=self._provider.provider_name,
                duration_ms=duration_ms,
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )

            raise

    def get_audit_logger(self) -> AuditLogger:
        """Get the audit logger."""
        return self._logger


def create_auditing_provider(
    provider: VisionProvider,
    config: Optional[AuditConfig] = None,
    storage: Optional[AuditStorage] = None,
) -> tuple:
    """Create an auditing provider wrapper.

    Args:
        provider: The underlying provider
        config: Audit configuration
        storage: Storage backend

    Returns:
        Tuple of (auditing_provider, audit_logger)
    """
    logger = AuditLogger(config=config, storage=storage)
    auditing_provider = AuditingVisionProvider(
        provider=provider,
        audit_logger=logger,
    )
    return auditing_provider, logger


# Singleton audit logger
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(
    config: Optional[AuditConfig] = None,
    storage: Optional[AuditStorage] = None,
) -> AuditLogger:
    """Get or create the global audit logger singleton."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(config=config, storage=storage)
    return _audit_logger
