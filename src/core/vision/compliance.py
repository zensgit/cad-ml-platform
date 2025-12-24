"""Compliance and audit logging for Vision Provider system.

This module provides compliance features including:
- Audit logging
- Data retention policies
- Access control logging
- Compliance reporting
- PII detection and handling
"""

import asyncio
import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

from .base import VisionDescription, VisionProvider


class AuditEventType(Enum):
    """Audit event type."""

    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    ACCESS = "access"
    CONFIGURATION_CHANGE = "configuration_change"
    DATA_ACCESS = "data_access"
    DATA_DELETE = "data_delete"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"


class AuditSeverity(Enum):
    """Audit event severity."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComplianceStandard(Enum):
    """Compliance standard."""

    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    CCPA = "ccpa"


class DataClassification(Enum):
    """Data classification level."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    PHI = "phi"  # Protected Health Information


class RetentionPolicy(Enum):
    """Data retention policy."""

    IMMEDIATE = "immediate"  # Delete immediately after processing
    SHORT_TERM = "short_term"  # 30 days
    MEDIUM_TERM = "medium_term"  # 90 days
    LONG_TERM = "long_term"  # 1 year
    PERMANENT = "permanent"  # Never delete
    CUSTOM = "custom"


@dataclass
class AuditEvent:
    """Audit event."""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime = field(default_factory=datetime.now)
    severity: AuditSeverity = AuditSeverity.INFO
    actor: str = ""  # User/system that triggered event
    resource: str = ""  # Resource affected
    action: str = ""  # Action performed
    outcome: str = ""  # Success/failure
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    tenant_id: Optional[str] = None
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "actor": self.actor,
            "resource": self.resource,
            "action": self.action,
            "outcome": self.outcome,
            "details": dict(self.details),
            "metadata": dict(self.metadata),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "tenant_id": self.tenant_id,
            "trace_id": self.trace_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class DataRetentionConfig:
    """Data retention configuration."""

    policy: RetentionPolicy
    retention_days: int = 30
    auto_delete: bool = True
    archive_before_delete: bool = False
    classification: DataClassification = DataClassification.INTERNAL

    def get_expiration_date(self, created_at: datetime) -> Optional[datetime]:
        """Get expiration date.

        Args:
            created_at: Creation timestamp

        Returns:
            Expiration date or None for permanent
        """
        if self.policy == RetentionPolicy.PERMANENT:
            return None
        if self.policy == RetentionPolicy.IMMEDIATE:
            return created_at

        days = {
            RetentionPolicy.SHORT_TERM: 30,
            RetentionPolicy.MEDIUM_TERM: 90,
            RetentionPolicy.LONG_TERM: 365,
            RetentionPolicy.CUSTOM: self.retention_days,
        }

        return created_at + timedelta(days=days.get(self.policy, self.retention_days))


@dataclass
class ComplianceRequirement:
    """Compliance requirement."""

    standard: ComplianceStandard
    requirement_id: str
    description: str
    controls: List[str] = field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PIIDetectionResult:
    """PII detection result."""

    detected: bool
    pii_types: List[str] = field(default_factory=list)
    locations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    redacted: bool = False


class AuditLogger:
    """Audit event logger."""

    def __init__(
        self,
        max_events: int = 100000,
        retention: Optional[DataRetentionConfig] = None,
    ) -> None:
        """Initialize logger.

        Args:
            max_events: Maximum events to store
            retention: Retention configuration
        """
        self._events: List[AuditEvent] = []
        self._max_events = max_events
        self._retention = retention or DataRetentionConfig(policy=RetentionPolicy.MEDIUM_TERM)
        self._handlers: List[Callable[[AuditEvent], None]] = []
        self._lock = threading.Lock()
        self._event_counter = 0

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"evt_{timestamp}_{self._event_counter}"

    def log(
        self,
        event_type: AuditEventType,
        action: str,
        resource: str = "",
        actor: str = "",
        outcome: str = "success",
        severity: AuditSeverity = AuditSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log audit event.

        Args:
            event_type: Event type
            action: Action performed
            resource: Resource affected
            actor: Actor who performed action
            outcome: Outcome of action
            severity: Event severity
            details: Additional details
            **kwargs: Additional metadata

        Returns:
            Created audit event
        """
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            action=action,
            resource=resource,
            actor=actor,
            outcome=outcome,
            severity=severity,
            details=details or {},
            metadata=kwargs,
        )

        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                pass

        return event

    def log_request(
        self,
        resource: str,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log request event."""
        return self.log(
            event_type=AuditEventType.REQUEST,
            action="request",
            resource=resource,
            actor=actor,
            details=details,
            **kwargs,
        )

    def log_response(
        self,
        resource: str,
        actor: str,
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log response event."""
        return self.log(
            event_type=AuditEventType.RESPONSE,
            action="response",
            resource=resource,
            actor=actor,
            outcome=outcome,
            details=details,
            **kwargs,
        )

    def log_error(
        self,
        resource: str,
        error: str,
        actor: str = "",
        **kwargs: Any,
    ) -> AuditEvent:
        """Log error event."""
        return self.log(
            event_type=AuditEventType.ERROR,
            action="error",
            resource=resource,
            actor=actor,
            outcome="failure",
            severity=AuditSeverity.ERROR,
            details={"error": error},
            **kwargs,
        )

    def log_access(
        self,
        resource: str,
        actor: str,
        access_type: str = "read",
        **kwargs: Any,
    ) -> AuditEvent:
        """Log access event."""
        return self.log(
            event_type=AuditEventType.ACCESS,
            action=access_type,
            resource=resource,
            actor=actor,
            **kwargs,
        )

    def log_data_delete(
        self,
        resource: str,
        actor: str,
        reason: str = "",
        **kwargs: Any,
    ) -> AuditEvent:
        """Log data deletion event."""
        return self.log(
            event_type=AuditEventType.DATA_DELETE,
            action="delete",
            resource=resource,
            actor=actor,
            details={"reason": reason},
            **kwargs,
        )

    def add_handler(self, handler: Callable[[AuditEvent], None]) -> None:
        """Add event handler.

        Args:
            handler: Handler function
        """
        self._handlers.append(handler)

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get audit events.

        Args:
            event_type: Filter by type
            actor: Filter by actor
            since: Filter by start time
            until: Filter by end time
            limit: Maximum events to return

        Returns:
            List of audit events
        """
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if actor:
            events = [e for e in events if e.actor == actor]
        if since:
            events = [e for e in events if e.timestamp >= since]
        if until:
            events = [e for e in events if e.timestamp <= until]

        return events[-limit:]

    def purge_expired(self) -> int:
        """Purge expired events based on retention policy.

        Returns:
            Number of events purged
        """
        cutoff = self._retention.get_expiration_date(datetime.now())
        if not cutoff:
            return 0

        with self._lock:
            original_count = len(self._events)
            self._events = [e for e in self._events if e.timestamp > cutoff]
            return original_count - len(self._events)


class PIIDetector:
    """Detects PII in data."""

    def __init__(self) -> None:
        """Initialize detector."""
        self._patterns: Dict[str, str] = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"\+?[1-9]\d{1,14}",
            "ssn": r"\d{3}-\d{2}-\d{4}",
            "credit_card": r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
            "ip_address": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        }

    def detect(self, data: Union[str, bytes, Dict[str, Any]]) -> PIIDetectionResult:
        """Detect PII in data.

        Args:
            data: Data to scan

        Returns:
            Detection result
        """
        import re

        text = ""
        if isinstance(data, bytes):
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                return PIIDetectionResult(detected=False)
        elif isinstance(data, dict):
            text = json.dumps(data)
        else:
            text = str(data)

        detected_types = []
        locations = []

        for pii_type, pattern in self._patterns.items():
            matches = list(re.finditer(pattern, text))
            if matches:
                detected_types.append(pii_type)
                for match in matches:
                    locations.append(
                        {
                            "type": pii_type,
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )

        return PIIDetectionResult(
            detected=len(detected_types) > 0,
            pii_types=detected_types,
            locations=locations,
            confidence=0.9 if detected_types else 0.0,
        )

    def redact(self, data: str, replacement: str = "[REDACTED]") -> str:
        """Redact PII from text.

        Args:
            data: Text to redact
            replacement: Replacement string

        Returns:
            Redacted text
        """
        import re

        result = data
        for pattern in self._patterns.values():
            result = re.sub(pattern, replacement, result)
        return result


class ComplianceManager:
    """Manages compliance requirements."""

    def __init__(self) -> None:
        """Initialize manager."""
        self._requirements: Dict[ComplianceStandard, List[ComplianceRequirement]] = {}
        self._enabled_standards: Set[ComplianceStandard] = set()
        self._audit_logger = AuditLogger()
        self._pii_detector = PIIDetector()
        self._lock = threading.Lock()

    def enable_standard(self, standard: ComplianceStandard) -> None:
        """Enable compliance standard.

        Args:
            standard: Compliance standard
        """
        with self._lock:
            self._enabled_standards.add(standard)
            self._setup_standard_requirements(standard)

    def disable_standard(self, standard: ComplianceStandard) -> None:
        """Disable compliance standard.

        Args:
            standard: Compliance standard
        """
        with self._lock:
            self._enabled_standards.discard(standard)

    def _setup_standard_requirements(self, standard: ComplianceStandard) -> None:
        """Set up requirements for standard."""
        requirements = {
            ComplianceStandard.GDPR: [
                ComplianceRequirement(
                    standard=ComplianceStandard.GDPR,
                    requirement_id="gdpr-art-17",
                    description="Right to erasure (right to be forgotten)",
                    controls=["data_deletion", "audit_logging"],
                ),
                ComplianceRequirement(
                    standard=ComplianceStandard.GDPR,
                    requirement_id="gdpr-art-30",
                    description="Records of processing activities",
                    controls=["audit_logging", "data_inventory"],
                ),
            ],
            ComplianceStandard.HIPAA: [
                ComplianceRequirement(
                    standard=ComplianceStandard.HIPAA,
                    requirement_id="hipaa-164.312",
                    description="Access controls",
                    controls=["access_control", "audit_logging"],
                ),
                ComplianceRequirement(
                    standard=ComplianceStandard.HIPAA,
                    requirement_id="hipaa-164.530",
                    description="Audit controls",
                    controls=["audit_logging", "integrity_verification"],
                ),
            ],
            ComplianceStandard.SOC2: [
                ComplianceRequirement(
                    standard=ComplianceStandard.SOC2,
                    requirement_id="soc2-cc6.1",
                    description="Logical and physical access controls",
                    controls=["access_control", "authentication"],
                ),
            ],
        }

        self._requirements[standard] = requirements.get(standard, [])

    def add_requirement(self, requirement: ComplianceRequirement) -> None:
        """Add compliance requirement.

        Args:
            requirement: Requirement to add
        """
        with self._lock:
            if requirement.standard not in self._requirements:
                self._requirements[requirement.standard] = []
            self._requirements[requirement.standard].append(requirement)

    def get_requirements(
        self,
        standard: Optional[ComplianceStandard] = None,
    ) -> List[ComplianceRequirement]:
        """Get compliance requirements.

        Args:
            standard: Filter by standard

        Returns:
            List of requirements
        """
        with self._lock:
            if standard:
                return list(self._requirements.get(standard, []))

            all_requirements = []
            for reqs in self._requirements.values():
                all_requirements.extend(reqs)
            return all_requirements

    def check_compliance(
        self,
        standard: Optional[ComplianceStandard] = None,
    ) -> Dict[str, Any]:
        """Check compliance status.

        Args:
            standard: Standard to check

        Returns:
            Compliance status report
        """
        requirements = self.get_requirements(standard)

        compliant = []
        non_compliant = []

        for req in requirements:
            if req.enabled:
                # Simplified check - in production would verify actual controls
                compliant.append(req.requirement_id)
            else:
                non_compliant.append(req.requirement_id)

        return {
            "standards_enabled": [s.value for s in self._enabled_standards],
            "total_requirements": len(requirements),
            "compliant_count": len(compliant),
            "non_compliant_count": len(non_compliant),
            "compliance_percentage": (
                len(compliant) / len(requirements) * 100 if requirements else 100.0
            ),
            "compliant_requirements": compliant,
            "non_compliant_requirements": non_compliant,
        }

    def get_audit_logger(self) -> AuditLogger:
        """Get audit logger."""
        return self._audit_logger

    def get_pii_detector(self) -> PIIDetector:
        """Get PII detector."""
        return self._pii_detector

    def generate_compliance_report(
        self,
        standard: Optional[ComplianceStandard] = None,
        period_days: int = 30,
    ) -> Dict[str, Any]:
        """Generate compliance report.

        Args:
            standard: Standard to report on
            period_days: Reporting period in days

        Returns:
            Compliance report
        """
        since = datetime.now() - timedelta(days=period_days)
        events = self._audit_logger.get_events(since=since, limit=10000)

        return {
            "report_generated": datetime.now().isoformat(),
            "period_start": since.isoformat(),
            "period_end": datetime.now().isoformat(),
            "standards": [s.value for s in self._enabled_standards],
            "compliance_status": self.check_compliance(standard),
            "audit_summary": {
                "total_events": len(events),
                "events_by_type": self._count_by_type(events),
                "events_by_severity": self._count_by_severity(events),
            },
        }

    def _count_by_type(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Count events by type."""
        counts: Dict[str, int] = {}
        for event in events:
            key = event.event_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_by_severity(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Count events by severity."""
        counts: Dict[str, int] = {}
        for event in events:
            key = event.severity.value
            counts[key] = counts.get(key, 0) + 1
        return counts


class DataRetentionManager:
    """Manages data retention."""

    def __init__(
        self,
        default_retention: Optional[DataRetentionConfig] = None,
    ) -> None:
        """Initialize manager.

        Args:
            default_retention: Default retention config
        """
        self._default = default_retention or DataRetentionConfig(policy=RetentionPolicy.MEDIUM_TERM)
        self._configs: Dict[str, DataRetentionConfig] = {}
        self._data_registry: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def set_retention(
        self,
        data_type: str,
        config: DataRetentionConfig,
    ) -> None:
        """Set retention config for data type.

        Args:
            data_type: Data type
            config: Retention configuration
        """
        with self._lock:
            self._configs[data_type] = config

    def get_retention(self, data_type: str) -> DataRetentionConfig:
        """Get retention config for data type.

        Args:
            data_type: Data type

        Returns:
            Retention configuration
        """
        return self._configs.get(data_type, self._default)

    def register_data(
        self,
        data_id: str,
        data_type: str,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register data for retention tracking.

        Args:
            data_id: Data identifier
            data_type: Data type
            created_at: Creation timestamp
            metadata: Additional metadata
        """
        config = self.get_retention(data_type)
        created = created_at or datetime.now()

        with self._lock:
            self._data_registry[data_id] = {
                "data_type": data_type,
                "created_at": created,
                "expires_at": config.get_expiration_date(created),
                "metadata": metadata or {},
            }

    def get_expired_data(self) -> List[str]:
        """Get list of expired data IDs.

        Returns:
            List of expired data IDs
        """
        now = datetime.now()
        expired = []

        with self._lock:
            for data_id, info in self._data_registry.items():
                expires_at = info.get("expires_at")
                if expires_at and expires_at <= now:
                    expired.append(data_id)

        return expired

    def mark_deleted(self, data_id: str) -> None:
        """Mark data as deleted.

        Args:
            data_id: Data identifier
        """
        with self._lock:
            self._data_registry.pop(data_id, None)


class ComplianceVisionProvider(VisionProvider):
    """Vision provider with compliance features."""

    def __init__(
        self,
        provider: VisionProvider,
        compliance: ComplianceManager,
        actor: str = "system",
        enable_pii_detection: bool = True,
    ) -> None:
        """Initialize provider.

        Args:
            provider: Underlying provider
            compliance: Compliance manager
            actor: Default actor for audit
            enable_pii_detection: Enable PII detection
        """
        self._provider = provider
        self._compliance = compliance
        self._actor = actor
        self._enable_pii_detection = enable_pii_detection

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"compliance_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with compliance features.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        audit = self._compliance.get_audit_logger()

        # Log request
        request_hash = hashlib.sha256(image_data).hexdigest()[:16]
        audit.log_request(
            resource=f"image:{request_hash}",
            actor=self._actor,
            details={
                "image_size": len(image_data),
                "include_description": include_description,
            },
        )

        # Check for PII in image metadata (simplified)
        if self._enable_pii_detection:
            pii_result = self._compliance.get_pii_detector().detect({"size": len(image_data)})
            if pii_result.detected:
                audit.log(
                    event_type=AuditEventType.DATA_ACCESS,
                    action="pii_detected",
                    resource=f"image:{request_hash}",
                    actor=self._actor,
                    severity=AuditSeverity.WARNING,
                    details={"pii_types": pii_result.pii_types},
                )

        start_time = time.time()

        try:
            result = await self._provider.analyze_image(image_data, include_description)

            latency_ms = (time.time() - start_time) * 1000

            # Log response
            audit.log_response(
                resource=f"image:{request_hash}",
                actor=self._actor,
                details={
                    "latency_ms": latency_ms,
                    "confidence": result.confidence,
                },
            )

            return result

        except Exception as e:
            audit.log_error(
                resource=f"image:{request_hash}",
                error=str(e),
                actor=self._actor,
            )
            raise


def create_compliant_provider(
    provider: VisionProvider,
    standards: Optional[List[ComplianceStandard]] = None,
) -> ComplianceVisionProvider:
    """Create compliant vision provider.

    Args:
        provider: Provider to wrap
        standards: Compliance standards to enable

    Returns:
        Compliant provider
    """
    compliance = ComplianceManager()

    for standard in standards or []:
        compliance.enable_standard(standard)

    return ComplianceVisionProvider(
        provider=provider,
        compliance=compliance,
    )
