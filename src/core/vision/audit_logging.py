"""Audit logging and compliance tracking for Vision Provider system.

This module provides audit features including:
- Audit event logging
- Compliance tracking
- Access logging
- Change tracking
- Audit trail management
"""

import hashlib
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .base import VisionDescription, VisionProvider


class AuditEventType(Enum):
    """Audit event types."""

    # Access events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"

    # Data events
    DATA_READ = "data_read"
    DATA_CREATED = "data_created"
    DATA_UPDATED = "data_updated"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"

    # System events
    CONFIG_CHANGED = "config_changed"
    PERMISSION_CHANGED = "permission_changed"
    KEY_ROTATED = "key_rotated"
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"

    # Security events
    SECURITY_ALERT = "security_alert"
    INTRUSION_DETECTED = "intrusion_detected"
    ANOMALY_DETECTED = "anomaly_detected"

    # Vision events
    IMAGE_ANALYZED = "image_analyzed"
    ANALYSIS_FAILED = "analysis_failed"
    PROVIDER_CHANGED = "provider_changed"


class AuditSeverity(Enum):
    """Audit severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """Compliance frameworks."""

    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    CCPA = "ccpa"


@dataclass
class AuditActor:
    """Audit actor (who performed the action)."""

    actor_id: str
    actor_type: str = "user"  # user, system, service
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "roles": list(self.roles),
            "metadata": dict(self.metadata),
        }


@dataclass
class AuditResource:
    """Audit resource (what was affected)."""

    resource_id: str
    resource_type: str
    resource_name: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "resource_name": self.resource_name,
            "parent_id": self.parent_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class AuditEvent:
    """Audit event."""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime = field(default_factory=datetime.now)
    actor: Optional[AuditActor] = None
    resource: Optional[AuditResource] = None
    action: str = ""
    outcome: str = "success"  # success, failure, error
    severity: AuditSeverity = AuditSeverity.INFO
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    compliance_tags: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor.to_dict() if self.actor else None,
            "resource": self.resource.to_dict() if self.resource else None,
            "action": self.action,
            "outcome": self.outcome,
            "severity": self.severity.value,
            "message": self.message,
            "details": dict(self.details),
            "before_state": self.before_state,
            "after_state": self.after_state,
            "compliance_tags": list(self.compliance_tags),
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def get_hash(self) -> str:
        """Get event hash for integrity verification."""
        data = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()


class AuditStore(ABC):
    """Abstract audit store."""

    @abstractmethod
    def append(self, event: AuditEvent) -> None:
        """Append event to store."""
        pass

    @abstractmethod
    def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        actor_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get events."""
        pass

    @abstractmethod
    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get single event."""
        pass


class InMemoryAuditStore(AuditStore):
    """In-memory audit store."""

    def __init__(self, max_events: int = 10000) -> None:
        """Initialize store."""
        self._events: List[AuditEvent] = []
        self._by_id: Dict[str, AuditEvent] = {}
        self._max_events = max_events
        self._lock = threading.Lock()

    def append(self, event: AuditEvent) -> None:
        """Append event to store."""
        with self._lock:
            self._events.append(event)
            self._by_id[event.event_id] = event

            # Trim if necessary
            if len(self._events) > self._max_events:
                removed = self._events.pop(0)
                del self._by_id[removed.event_id]

    def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        actor_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get events."""
        with self._lock:
            events = list(self._events)

        # Apply filters
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if actor_id:
            events = [e for e in events if e.actor and e.actor.actor_id == actor_id]
        if resource_id:
            events = [e for e in events if e.resource and e.resource.resource_id == resource_id]

        # Sort by timestamp descending and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get single event."""
        with self._lock:
            return self._by_id.get(event_id)

    def count(self) -> int:
        """Get event count."""
        with self._lock:
            return len(self._events)

    def clear(self) -> None:
        """Clear all events."""
        with self._lock:
            self._events.clear()
            self._by_id.clear()


class AuditLogger:
    """Audit logger."""

    def __init__(
        self,
        store: Optional[AuditStore] = None,
        default_actor: Optional[AuditActor] = None,
    ) -> None:
        """Initialize logger."""
        self._store = store or InMemoryAuditStore()
        self._default_actor = default_actor
        self._correlation_id: Optional[str] = None
        self._hooks: List[Callable[[AuditEvent], None]] = []

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for subsequent events."""
        self._correlation_id = correlation_id

    def clear_correlation_id(self) -> None:
        """Clear correlation ID."""
        self._correlation_id = None

    def add_hook(self, hook: Callable[[AuditEvent], None]) -> None:
        """Add event hook."""
        self._hooks.append(hook)

    def log(
        self,
        event_type: AuditEventType,
        action: str,
        message: str = "",
        actor: Optional[AuditActor] = None,
        resource: Optional[AuditResource] = None,
        outcome: str = "success",
        severity: AuditSeverity = AuditSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        compliance_tags: Optional[List[str]] = None,
        parent_event_id: Optional[str] = None,
    ) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            actor=actor or self._default_actor,
            resource=resource,
            action=action,
            outcome=outcome,
            severity=severity,
            message=message,
            details=details or {},
            before_state=before_state,
            after_state=after_state,
            compliance_tags=compliance_tags or [],
            correlation_id=self._correlation_id,
            parent_event_id=parent_event_id,
        )

        self._store.append(event)

        # Call hooks
        for hook in self._hooks:
            try:
                hook(event)
            except Exception:
                pass  # Don't let hooks break logging

        return event

    def log_access(
        self,
        actor: AuditActor,
        resource: AuditResource,
        granted: bool,
        reason: str = "",
    ) -> AuditEvent:
        """Log access event."""
        return self.log(
            event_type=AuditEventType.ACCESS_GRANTED if granted else AuditEventType.ACCESS_DENIED,
            action="access_check",
            actor=actor,
            resource=resource,
            outcome="success" if granted else "denied",
            severity=AuditSeverity.INFO if granted else AuditSeverity.WARNING,
            message=reason,
        )

    def log_data_change(
        self,
        event_type: AuditEventType,
        resource: AuditResource,
        before_state: Optional[Dict[str, Any]],
        after_state: Optional[Dict[str, Any]],
        actor: Optional[AuditActor] = None,
    ) -> AuditEvent:
        """Log data change event."""
        return self.log(
            event_type=event_type,
            action="data_change",
            actor=actor,
            resource=resource,
            before_state=before_state,
            after_state=after_state,
            compliance_tags=["data_change"],
        )

    def log_security_event(
        self,
        event_type: AuditEventType,
        message: str,
        severity: AuditSeverity = AuditSeverity.WARNING,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log security event."""
        return self.log(
            event_type=event_type,
            action="security_event",
            message=message,
            severity=severity,
            details=details or {},
            compliance_tags=["security"],
        )

    def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        actor_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get events."""
        return self._store.get_events(
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            actor_id=actor_id,
            resource_id=resource_id,
            limit=limit,
        )

    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get single event."""
        return self._store.get_event(event_id)


@dataclass
class ComplianceRequirement:
    """Compliance requirement."""

    requirement_id: str
    framework: ComplianceFramework
    name: str
    description: str
    required_events: List[AuditEventType] = field(default_factory=list)
    retention_days: int = 365
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceTracker:
    """Compliance tracker."""

    def __init__(self, audit_logger: AuditLogger) -> None:
        """Initialize tracker."""
        self._audit_logger = audit_logger
        self._requirements: Dict[str, ComplianceRequirement] = {}
        self._violations: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_requirement(self, requirement: ComplianceRequirement) -> None:
        """Add compliance requirement."""
        with self._lock:
            self._requirements[requirement.requirement_id] = requirement

    def remove_requirement(self, requirement_id: str) -> None:
        """Remove compliance requirement."""
        with self._lock:
            self._requirements.pop(requirement_id, None)

    def check_compliance(
        self,
        framework: Optional[ComplianceFramework] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Check compliance status."""
        with self._lock:
            requirements = list(self._requirements.values())
            if framework:
                requirements = [r for r in requirements if r.framework == framework]

        results = {
            "compliant": True,
            "checked_at": datetime.now().isoformat(),
            "requirements": [],
            "violations": [],
        }

        for req in requirements:
            req_result = self._check_requirement(req, start_time, end_time)
            results["requirements"].append(req_result)
            if not req_result["compliant"]:
                results["compliant"] = False
                results["violations"].append(req_result)

        return results

    def _check_requirement(
        self,
        requirement: ComplianceRequirement,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> Dict[str, Any]:
        """Check single requirement."""
        result = {
            "requirement_id": requirement.requirement_id,
            "framework": requirement.framework.value,
            "name": requirement.name,
            "compliant": True,
            "details": {},
        }

        # Check if required events are being logged
        for event_type in requirement.required_events:
            events = self._audit_logger.get_events(
                start_time=start_time,
                end_time=end_time,
                event_type=event_type,
                limit=1,
            )
            if not events:
                result["compliant"] = False
                result["details"][event_type.value] = "Missing required audit events"

        return result

    def record_violation(
        self,
        requirement_id: str,
        description: str,
        severity: AuditSeverity = AuditSeverity.WARNING,
    ) -> None:
        """Record compliance violation."""
        violation = {
            "violation_id": str(uuid.uuid4()),
            "requirement_id": requirement_id,
            "description": description,
            "severity": severity.value,
            "timestamp": datetime.now().isoformat(),
        }

        with self._lock:
            self._violations.append(violation)

        # Log to audit
        self._audit_logger.log(
            event_type=AuditEventType.SECURITY_ALERT,
            action="compliance_violation",
            message=description,
            severity=severity,
            details=violation,
            compliance_tags=["violation", requirement_id],
        )

    def get_violations(
        self,
        requirement_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get violations."""
        with self._lock:
            violations = list(self._violations)

        if requirement_id:
            violations = [v for v in violations if v["requirement_id"] == requirement_id]

        return violations[:limit]

    def list_requirements(
        self,
        framework: Optional[ComplianceFramework] = None,
    ) -> List[ComplianceRequirement]:
        """List requirements."""
        with self._lock:
            requirements = list(self._requirements.values())
            if framework:
                requirements = [r for r in requirements if r.framework == framework]
            return requirements


class AuditTrail:
    """Audit trail for tracking changes."""

    def __init__(self, audit_logger: AuditLogger) -> None:
        """Initialize audit trail."""
        self._audit_logger = audit_logger
        self._tracked_resources: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def track_resource(
        self,
        resource_id: str,
        resource_type: str,
        initial_state: Dict[str, Any],
    ) -> None:
        """Start tracking a resource."""
        with self._lock:
            self._tracked_resources[resource_id] = {
                "resource_type": resource_type,
                "state": initial_state,
                "tracked_at": datetime.now(),
            }

    def update_resource(
        self,
        resource_id: str,
        new_state: Dict[str, Any],
        actor: Optional[AuditActor] = None,
    ) -> Optional[AuditEvent]:
        """Update tracked resource and log change."""
        with self._lock:
            if resource_id not in self._tracked_resources:
                return None

            tracked = self._tracked_resources[resource_id]
            before_state = tracked["state"]

            # Find changes
            changes = self._find_changes(before_state, new_state)
            if not changes:
                return None

            # Update state
            tracked["state"] = new_state

        # Log change
        resource = AuditResource(
            resource_id=resource_id,
            resource_type=tracked["resource_type"],
        )

        return self._audit_logger.log_data_change(
            event_type=AuditEventType.DATA_UPDATED,
            resource=resource,
            before_state=before_state,
            after_state=new_state,
            actor=actor,
        )

    def _find_changes(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Find changes between two states."""
        changes = {}

        all_keys = set(before.keys()) | set(after.keys())
        for key in all_keys:
            before_val = before.get(key)
            after_val = after.get(key)
            if before_val != after_val:
                changes[key] = {"before": before_val, "after": after_val}

        return changes

    def get_history(
        self,
        resource_id: str,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get change history for a resource."""
        return self._audit_logger.get_events(
            resource_id=resource_id,
            limit=limit,
        )

    def untrack_resource(self, resource_id: str) -> None:
        """Stop tracking a resource."""
        with self._lock:
            self._tracked_resources.pop(resource_id, None)


class AuditedVisionProvider(VisionProvider):
    """Vision provider with audit logging."""

    def __init__(
        self,
        provider: VisionProvider,
        audit_logger: Optional[AuditLogger] = None,
        actor: Optional[AuditActor] = None,
    ) -> None:
        """Initialize provider."""
        self._provider = provider
        self._audit_logger = audit_logger or AuditLogger()
        self._actor = actor or AuditActor(actor_id="system", actor_type="service")

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"audited_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with audit logging."""
        request_id = str(uuid.uuid4())

        # Create resource
        resource = AuditResource(
            resource_id=request_id,
            resource_type="image_analysis",
            metadata={"size": len(image_data)},
        )

        # Log request
        self._audit_logger.log(
            event_type=AuditEventType.DATA_READ,
            action="analyze_image_start",
            actor=self._actor,
            resource=resource,
            message=f"Starting image analysis: {request_id}",
            details={
                "image_size": len(image_data),
                "include_description": include_description,
            },
        )

        try:
            # Call underlying provider
            result = await self._provider.analyze_image(image_data, include_description)

            # Log success
            self._audit_logger.log(
                event_type=AuditEventType.IMAGE_ANALYZED,
                action="analyze_image_complete",
                actor=self._actor,
                resource=resource,
                outcome="success",
                message=f"Image analysis completed: {request_id}",
                details={
                    "confidence": result.confidence,
                    "details_count": len(result.details),
                },
            )

            return result

        except Exception as e:
            # Log failure
            self._audit_logger.log(
                event_type=AuditEventType.ANALYSIS_FAILED,
                action="analyze_image_error",
                actor=self._actor,
                resource=resource,
                outcome="error",
                severity=AuditSeverity.ERROR,
                message=f"Image analysis failed: {str(e)}",
                details={"error": str(e)},
            )
            raise

    def get_audit_logger(self) -> AuditLogger:
        """Get audit logger."""
        return self._audit_logger


def create_audit_logger(
    store: Optional[AuditStore] = None,
    default_actor: Optional[AuditActor] = None,
) -> AuditLogger:
    """Create audit logger.

    Args:
        store: Optional audit store
        default_actor: Optional default actor

    Returns:
        Audit logger
    """
    return AuditLogger(store, default_actor)


def create_audited_provider(
    provider: VisionProvider,
    audit_logger: Optional[AuditLogger] = None,
) -> AuditedVisionProvider:
    """Create audited vision provider.

    Args:
        provider: Provider to wrap
        audit_logger: Optional audit logger

    Returns:
        Audited provider
    """
    return AuditedVisionProvider(provider, audit_logger)


def create_compliance_tracker(
    audit_logger: AuditLogger,
) -> ComplianceTracker:
    """Create compliance tracker.

    Args:
        audit_logger: Audit logger

    Returns:
        Compliance tracker
    """
    return ComplianceTracker(audit_logger)
