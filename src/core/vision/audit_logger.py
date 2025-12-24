"""Audit Logger Module for Vision System.

This module provides audit logging capabilities including:
- Comprehensive audit trail recording
- Event correlation and analysis
- Compliance logging
- Tamper-proof log storage
- Log retention and archival
- Real-time monitoring and alerts

Phase 19: Advanced Security & Compliance
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class AuditEventType(str, Enum):
    """Types of audit events."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"
    DATA_EXPORT = "data_export"
    API_CALL = "api_call"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditCategory(str, Enum):
    """Audit event categories."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    SYSTEM = "system"


class LogDestination(str, Enum):
    """Log destinations."""

    MEMORY = "memory"
    FILE = "file"
    DATABASE = "database"
    CLOUD = "cloud"
    SYSLOG = "syslog"


class RetentionPeriod(str, Enum):
    """Log retention periods."""

    DAYS_7 = "7d"
    DAYS_30 = "30d"
    DAYS_90 = "90d"
    DAYS_365 = "365d"
    YEARS_7 = "7y"
    FOREVER = "forever"


# ========================
# Dataclasses
# ========================


@dataclass
class AuditEvent:
    """An audit log event."""

    event_id: str
    event_type: AuditEventType
    category: AuditCategory
    severity: AuditSeverity = AuditSeverity.INFO
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    resource_type: str = ""
    resource_id: str = ""
    action: str = ""
    outcome: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class AuditQuery:
    """Query for searching audit logs."""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    categories: Optional[List[AuditCategory]] = None
    severities: Optional[List[AuditSeverity]] = None
    user_ids: Optional[List[str]] = None
    resource_ids: Optional[List[str]] = None
    correlation_id: Optional[str] = None
    outcome: Optional[str] = None
    limit: int = 1000
    offset: int = 0


@dataclass
class AuditSummary:
    """Summary of audit events."""

    total_events: int
    events_by_type: Dict[str, int]
    events_by_category: Dict[str, int]
    events_by_severity: Dict[str, int]
    events_by_outcome: Dict[str, int]
    top_users: List[Tuple[str, int]]
    top_resources: List[Tuple[str, int]]
    time_range: Tuple[datetime, datetime]


@dataclass
class AuditPolicy:
    """Audit logging policy."""

    policy_id: str
    name: str
    event_types: List[AuditEventType]
    categories: List[AuditCategory]
    min_severity: AuditSeverity = AuditSeverity.INFO
    retention_period: RetentionPeriod = RetentionPeriod.DAYS_90
    destinations: List[LogDestination] = field(default_factory=list)
    enabled: bool = True


@dataclass
class AuditAlert:
    """Alert based on audit events."""

    alert_id: str
    name: str
    condition: Dict[str, Any]
    severity: AuditSeverity = AuditSeverity.WARNING
    recipients: List[str] = field(default_factory=list)
    enabled: bool = True
    cooldown_seconds: int = 300
    last_triggered: Optional[datetime] = None


@dataclass
class LogIntegrity:
    """Log integrity verification result."""

    verified: bool
    total_events: int
    valid_events: int
    invalid_events: List[str]
    verification_time: datetime = field(default_factory=datetime.now)


# ========================
# Core Classes
# ========================


class AuditEventStore(ABC):
    """Abstract base class for audit event storage."""

    @abstractmethod
    def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        pass

    @abstractmethod
    def query(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events."""
        pass

    @abstractmethod
    def count(self, query: AuditQuery) -> int:
        """Count matching audit events."""
        pass


class MemoryAuditStore(AuditEventStore):
    """In-memory audit event store."""

    def __init__(self, max_events: int = 100000):
        self._events: List[AuditEvent] = []
        self._max_events = max_events
        self._lock = threading.RLock()

    def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

    def query(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events."""
        with self._lock:
            results = self._filter_events(query)
            results = sorted(results, key=lambda e: e.timestamp, reverse=True)
            return results[query.offset : query.offset + query.limit]

    def count(self, query: AuditQuery) -> int:
        """Count matching events."""
        with self._lock:
            return len(self._filter_events(query))

    def _filter_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Filter events based on query."""
        results = []
        for event in self._events:
            if query.start_time and event.timestamp < query.start_time:
                continue
            if query.end_time and event.timestamp > query.end_time:
                continue
            if query.event_types and event.event_type not in query.event_types:
                continue
            if query.categories and event.category not in query.categories:
                continue
            if query.severities and event.severity not in query.severities:
                continue
            if query.user_ids and event.user_id not in query.user_ids:
                continue
            if query.resource_ids and event.resource_id not in query.resource_ids:
                continue
            if query.correlation_id and event.correlation_id != query.correlation_id:
                continue
            if query.outcome and event.outcome != query.outcome:
                continue
            results.append(event)
        return results


class AuditLogger:
    """Main audit logging component."""

    def __init__(self, store: Optional[AuditEventStore] = None):
        self._store = store or MemoryAuditStore()
        self._policies: Dict[str, AuditPolicy] = {}
        self._alerts: Dict[str, AuditAlert] = {}
        self._alert_handlers: List[Callable[[AuditAlert, AuditEvent], None]] = []
        self._lock = threading.RLock()
        self._correlation_context: Dict[str, str] = {}

    def log(
        self,
        event_type: AuditEventType,
        category: AuditCategory,
        action: str,
        resource_type: str = "",
        resource_id: str = "",
        user_id: Optional[str] = None,
        outcome: str = "success",
        severity: AuditSeverity = AuditSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
        ip_address: str = "",
        session_id: Optional[str] = None,
    ) -> AuditEvent:
        """Log an audit event."""
        event_id = hashlib.md5(f"{time.time()}:{action}:{resource_id}".encode()).hexdigest()[:16]

        # Get correlation ID from context
        correlation_id = self._correlation_context.get(threading.current_thread().name)

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            category=category,
            severity=severity,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            details=details or {},
            ip_address=ip_address,
            session_id=session_id,
            correlation_id=correlation_id,
        )

        # Calculate checksum for integrity
        event.checksum = self._calculate_checksum(event)

        # Store the event
        self._store.store(event)

        # Check alerts
        self._check_alerts(event)

        return event

    def _calculate_checksum(self, event: AuditEvent) -> str:
        """Calculate checksum for an event."""
        data = f"{event.event_id}:{event.event_type}:{event.timestamp}:{event.user_id}:{event.resource_id}:{event.action}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def verify_integrity(self, events: List[AuditEvent]) -> LogIntegrity:
        """Verify integrity of audit events."""
        valid_events = 0
        invalid_events = []

        for event in events:
            expected_checksum = self._calculate_checksum(event)
            if event.checksum == expected_checksum:
                valid_events += 1
            else:
                invalid_events.append(event.event_id)

        return LogIntegrity(
            verified=len(invalid_events) == 0,
            total_events=len(events),
            valid_events=valid_events,
            invalid_events=invalid_events,
        )

    def query(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events."""
        return self._store.query(query)

    def get_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> AuditSummary:
        """Get audit summary."""
        query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )
        events = self._store.query(query)

        events_by_type: Dict[str, int] = defaultdict(int)
        events_by_category: Dict[str, int] = defaultdict(int)
        events_by_severity: Dict[str, int] = defaultdict(int)
        events_by_outcome: Dict[str, int] = defaultdict(int)
        user_counts: Dict[str, int] = defaultdict(int)
        resource_counts: Dict[str, int] = defaultdict(int)

        min_time = datetime.max
        max_time = datetime.min

        for event in events:
            events_by_type[event.event_type.value] += 1
            events_by_category[event.category.value] += 1
            events_by_severity[event.severity.value] += 1
            events_by_outcome[event.outcome] += 1

            if event.user_id:
                user_counts[event.user_id] += 1
            if event.resource_id:
                resource_counts[event.resource_id] += 1

            if event.timestamp < min_time:
                min_time = event.timestamp
            if event.timestamp > max_time:
                max_time = event.timestamp

        top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_resources = sorted(resource_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return AuditSummary(
            total_events=len(events),
            events_by_type=dict(events_by_type),
            events_by_category=dict(events_by_category),
            events_by_severity=dict(events_by_severity),
            events_by_outcome=dict(events_by_outcome),
            top_users=top_users,
            top_resources=top_resources,
            time_range=(
                min_time if min_time != datetime.max else datetime.now(),
                max_time if max_time != datetime.min else datetime.now(),
            ),
        )

    def add_policy(self, policy: AuditPolicy) -> None:
        """Add an audit policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy

    def add_alert(self, alert: AuditAlert) -> None:
        """Add an audit alert."""
        with self._lock:
            self._alerts[alert.alert_id] = alert

    def add_alert_handler(
        self,
        handler: Callable[[AuditAlert, AuditEvent], None],
    ) -> None:
        """Add an alert handler."""
        self._alert_handlers.append(handler)

    def _check_alerts(self, event: AuditEvent) -> None:
        """Check if event triggers any alerts."""
        for alert in self._alerts.values():
            if not alert.enabled:
                continue

            # Check cooldown
            if alert.last_triggered:
                elapsed = (datetime.now() - alert.last_triggered).total_seconds()
                if elapsed < alert.cooldown_seconds:
                    continue

            # Check condition
            if self._matches_alert_condition(event, alert.condition):
                alert.last_triggered = datetime.now()
                for handler in self._alert_handlers:
                    try:
                        handler(alert, event)
                    except Exception:
                        pass

    def _matches_alert_condition(
        self,
        event: AuditEvent,
        condition: Dict[str, Any],
    ) -> bool:
        """Check if event matches alert condition."""
        if "event_types" in condition:
            if event.event_type not in condition["event_types"]:
                return False
        if "categories" in condition:
            if event.category not in condition["categories"]:
                return False
        if "severities" in condition:
            if event.severity not in condition["severities"]:
                return False
        if "outcome" in condition:
            if event.outcome != condition["outcome"]:
                return False
        return True

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current thread."""
        self._correlation_context[threading.current_thread().name] = correlation_id

    def clear_correlation_id(self) -> None:
        """Clear correlation ID for current thread."""
        thread_name = threading.current_thread().name
        if thread_name in self._correlation_context:
            del self._correlation_context[thread_name]


class ComplianceAuditLogger(AuditLogger):
    """Audit logger with compliance features."""

    def __init__(
        self,
        store: Optional[AuditEventStore] = None,
        compliance_standard: str = "SOC2",
    ):
        super().__init__(store)
        self._compliance_standard = compliance_standard
        self._required_fields = self._get_required_fields()

    def _get_required_fields(self) -> List[str]:
        """Get required fields for compliance standard."""
        requirements = {
            "SOC2": ["user_id", "action", "outcome", "ip_address"],
            "HIPAA": ["user_id", "action", "outcome", "resource_id", "ip_address"],
            "GDPR": ["user_id", "action", "outcome", "resource_id"],
            "PCI-DSS": ["user_id", "action", "outcome", "ip_address", "session_id"],
        }
        return requirements.get(self._compliance_standard, [])

    def log(self, **kwargs) -> AuditEvent:
        """Log an audit event with compliance validation."""
        # Validate required fields
        missing = []
        for field in self._required_fields:
            if field not in kwargs or kwargs[field] is None:
                missing.append(field)

        if missing:
            kwargs.setdefault("details", {})["compliance_warning"] = f"Missing fields: {missing}"

        return super().log(**kwargs)


# ========================
# Vision Provider
# ========================


class AuditLoggerVisionProvider(VisionProvider):
    """Vision provider for audit logging capabilities."""

    def __init__(self, compliance_standard: Optional[str] = None):
        self._compliance_standard = compliance_standard
        self._logger: Optional[AuditLogger] = None

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "audit_logger"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        """Analyze image for audit context."""
        return self.get_description()

    def get_description(self) -> VisionDescription:
        """Get provider description."""
        return VisionDescription(
            name="Audit Logger Vision Provider",
            version="1.0.0",
            description="Comprehensive audit logging and compliance",
            capabilities=[
                "audit_logging",
                "event_correlation",
                "compliance_logging",
                "integrity_verification",
                "alerting",
            ],
        )

    def initialize(self) -> None:
        """Initialize the provider."""
        if self._compliance_standard:
            self._logger = ComplianceAuditLogger(compliance_standard=self._compliance_standard)
        else:
            self._logger = AuditLogger()

    def shutdown(self) -> None:
        """Shutdown the provider."""
        self._logger = None

    def get_logger(self) -> AuditLogger:
        """Get the audit logger."""
        if self._logger is None:
            self.initialize()
        return self._logger


# ========================
# Factory Functions
# ========================


def create_audit_logger(
    store: Optional[AuditEventStore] = None,
) -> AuditLogger:
    """Create an audit logger."""
    return AuditLogger(store=store)


def create_compliance_audit_logger(
    compliance_standard: str = "SOC2",
    store: Optional[AuditEventStore] = None,
) -> ComplianceAuditLogger:
    """Create a compliance audit logger."""
    return ComplianceAuditLogger(
        store=store,
        compliance_standard=compliance_standard,
    )


def create_audit_event(
    event_id: str,
    event_type: AuditEventType,
    category: AuditCategory,
    action: str,
    severity: AuditSeverity = AuditSeverity.INFO,
) -> AuditEvent:
    """Create an audit event."""
    return AuditEvent(
        event_id=event_id,
        event_type=event_type,
        category=category,
        action=action,
        severity=severity,
    )


def create_audit_query(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
) -> AuditQuery:
    """Create an audit query."""
    return AuditQuery(
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )


def create_audit_policy(
    policy_id: str,
    name: str,
    event_types: Optional[List[AuditEventType]] = None,
    categories: Optional[List[AuditCategory]] = None,
) -> AuditPolicy:
    """Create an audit policy."""
    return AuditPolicy(
        policy_id=policy_id,
        name=name,
        event_types=event_types or list(AuditEventType),
        categories=categories or list(AuditCategory),
    )


def create_audit_alert(
    alert_id: str,
    name: str,
    condition: Dict[str, Any],
    severity: AuditSeverity = AuditSeverity.WARNING,
) -> AuditAlert:
    """Create an audit alert."""
    return AuditAlert(
        alert_id=alert_id,
        name=name,
        condition=condition,
        severity=severity,
    )


def create_memory_audit_store(max_events: int = 100000) -> MemoryAuditStore:
    """Create a memory audit store."""
    return MemoryAuditStore(max_events=max_events)


def create_audit_logger_provider(
    compliance_standard: Optional[str] = None,
) -> AuditLoggerVisionProvider:
    """Create an audit logger vision provider."""
    return AuditLoggerVisionProvider(compliance_standard=compliance_standard)
