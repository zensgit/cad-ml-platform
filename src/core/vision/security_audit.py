"""
Security Audit Module for Vision Provider System.

Provides comprehensive security logging, threat detection, anomaly detection,
and security event management for vision analysis operations.
"""

import asyncio
import hashlib
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base import VisionDescription, VisionProvider

# ============================================================================
# Enums and Types
# ============================================================================


class SecurityEventType(Enum):
    """Types of security events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ACCESS_DENIED = "access_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    ANOMALY_DETECTED = "anomaly_detected"
    THREAT_DETECTED = "threat_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"
    ENCRYPTION_EVENT = "encryption_event"
    KEY_OPERATION = "key_operation"
    SYSTEM_EVENT = "system_event"


class SeverityLevel(Enum):
    """Security event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ThreatLevel(Enum):
    """Threat level classifications."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Security alert status."""

    NEW = "new"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class SecurityEvent:
    """A security event."""

    event_id: str
    event_type: SecurityEventType
    severity: SeverityLevel
    source: str
    message: str
    user_id: Optional[str] = None
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "source": self.source,
            "message": self.message,
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "details": self.details,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SecurityAlert:
    """A security alert."""

    alert_id: str
    title: str
    description: str
    severity: SeverityLevel
    threat_level: ThreatLevel
    status: AlertStatus = AlertStatus.NEW
    related_events: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ThreatIndicator:
    """A threat indicator."""

    indicator_id: str
    indicator_type: str
    value: str
    threat_level: ThreatLevel
    description: str = ""
    source: str = "internal"
    active: bool = True
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AuditPolicy:
    """An audit policy."""

    policy_id: str
    name: str
    event_types: List[SecurityEventType]
    severity_threshold: SeverityLevel
    retention_days: int = 90
    alert_enabled: bool = True
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityMetrics:
    """Security metrics summary."""

    total_events: int = 0
    events_by_type: Dict[SecurityEventType, int] = field(default_factory=dict)
    events_by_severity: Dict[SeverityLevel, int] = field(default_factory=dict)
    alerts_count: int = 0
    threats_detected: int = 0
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# Threat Detection
# ============================================================================


class ThreatDetector(ABC):
    """Abstract base for threat detection."""

    @abstractmethod
    def detect(self, event: SecurityEvent) -> Optional[ThreatIndicator]:
        """Detect threats in a security event."""
        pass


class BruteForceDetector(ThreatDetector):
    """Detects brute force attacks."""

    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: int = 300,
    ) -> None:
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        self._attempts: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = threading.Lock()

    def detect(self, event: SecurityEvent) -> Optional[ThreatIndicator]:
        """Detect brute force attempts."""
        if event.event_type != SecurityEventType.AUTHENTICATION:
            return None

        if event.details.get("success", True):
            return None

        key = event.user_id or event.ip_address
        if not key:
            return None

        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self._window_seconds)

        with self._lock:
            # Clean old attempts
            self._attempts[key] = [t for t in self._attempts[key] if t > window_start]

            # Record this attempt
            self._attempts[key].append(now)

            # Check threshold
            if len(self._attempts[key]) >= self._max_attempts:
                return ThreatIndicator(
                    indicator_id=str(uuid.uuid4()),
                    indicator_type="brute_force",
                    value=key,
                    threat_level=ThreatLevel.HIGH,
                    description=f"Brute force attack detected: {len(self._attempts[key])} failed attempts",
                )

        return None


class AnomalyDetector(ThreatDetector):
    """Detects anomalous behavior patterns."""

    def __init__(self) -> None:
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._recent_events: Dict[str, List[SecurityEvent]] = defaultdict(list)
        self._lock = threading.Lock()

    def detect(self, event: SecurityEvent) -> Optional[ThreatIndicator]:
        """Detect anomalous events."""
        key = event.user_id or "system"

        with self._lock:
            # Track recent events
            self._recent_events[key].append(event)
            # Keep only last 100 events per key
            self._recent_events[key] = self._recent_events[key][-100:]

            # Check for unusual patterns
            recent = self._recent_events[key]

            # Unusual access time
            hour = event.timestamp.hour
            if hour < 6 or hour > 22:  # Outside business hours
                if event.event_type == SecurityEventType.DATA_ACCESS:
                    return ThreatIndicator(
                        indicator_id=str(uuid.uuid4()),
                        indicator_type="anomaly_time",
                        value=f"{key}:{hour}",
                        threat_level=ThreatLevel.MEDIUM,
                        description=f"Unusual access time: {hour}:00",
                    )

            # High frequency
            if len(recent) >= 50:
                time_span = (recent[-1].timestamp - recent[0].timestamp).total_seconds()
                if time_span > 0:
                    rate = len(recent) / time_span * 60  # events per minute
                    if rate > 10:
                        return ThreatIndicator(
                            indicator_id=str(uuid.uuid4()),
                            indicator_type="anomaly_frequency",
                            value=f"{key}:{rate:.2f}",
                            threat_level=ThreatLevel.MEDIUM,
                            description=f"High event frequency: {rate:.2f} events/min",
                        )

        return None


class IPReputationDetector(ThreatDetector):
    """Detects threats based on IP reputation."""

    def __init__(self) -> None:
        self._blocked_ips: Set[str] = set()
        self._suspicious_ips: Set[str] = set()
        self._lock = threading.Lock()

    def add_blocked_ip(self, ip: str) -> None:
        """Add a blocked IP."""
        with self._lock:
            self._blocked_ips.add(ip)

    def add_suspicious_ip(self, ip: str) -> None:
        """Add a suspicious IP."""
        with self._lock:
            self._suspicious_ips.add(ip)

    def detect(self, event: SecurityEvent) -> Optional[ThreatIndicator]:
        """Detect threats based on IP."""
        if not event.ip_address:
            return None

        with self._lock:
            if event.ip_address in self._blocked_ips:
                return ThreatIndicator(
                    indicator_id=str(uuid.uuid4()),
                    indicator_type="blocked_ip",
                    value=event.ip_address,
                    threat_level=ThreatLevel.CRITICAL,
                    description=f"Access from blocked IP: {event.ip_address}",
                )

            if event.ip_address in self._suspicious_ips:
                return ThreatIndicator(
                    indicator_id=str(uuid.uuid4()),
                    indicator_type="suspicious_ip",
                    value=event.ip_address,
                    threat_level=ThreatLevel.HIGH,
                    description=f"Access from suspicious IP: {event.ip_address}",
                )

        return None


# ============================================================================
# Audit Logger
# ============================================================================


class AuditLogger:
    """Security audit logger."""

    def __init__(self, max_events: int = 100000) -> None:
        self._events: List[SecurityEvent] = []
        self._max_events = max_events
        self._handlers: List[Callable[[SecurityEvent], None]] = []
        self._lock = threading.Lock()

    def log(self, event: SecurityEvent) -> None:
        """Log a security event."""
        with self._lock:
            self._events.append(event)

            # Trim if exceeds max
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

        # Notify handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                pass  # Don't let handler errors affect logging

    def add_handler(self, handler: Callable[[SecurityEvent], None]) -> None:
        """Add an event handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[SecurityEvent], None]) -> bool:
        """Remove an event handler."""
        try:
            self._handlers.remove(handler)
            return True
        except ValueError:
            return False

    def query(
        self,
        event_type: Optional[SecurityEventType] = None,
        severity: Optional[SeverityLevel] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SecurityEvent]:
        """Query security events."""
        with self._lock:
            results = self._events.copy()

        if event_type:
            results = [e for e in results if e.event_type == event_type]

        if severity:
            results = [e for e in results if e.severity == severity]

        if user_id:
            results = [e for e in results if e.user_id == user_id]

        if start_time:
            results = [e for e in results if e.timestamp >= start_time]

        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        return results[-limit:]

    def get_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> SecurityMetrics:
        """Get security metrics."""
        events = self.query(start_time=start_time, end_time=end_time, limit=10000)

        metrics = SecurityMetrics()
        metrics.total_events = len(events)
        metrics.period_start = start_time or datetime.min
        metrics.period_end = end_time or datetime.utcnow()

        for event in events:
            if event.event_type not in metrics.events_by_type:
                metrics.events_by_type[event.event_type] = 0
            metrics.events_by_type[event.event_type] += 1

            if event.severity not in metrics.events_by_severity:
                metrics.events_by_severity[event.severity] = 0
            metrics.events_by_severity[event.severity] += 1

            if event.event_type == SecurityEventType.THREAT_DETECTED:
                metrics.threats_detected += 1

        return metrics

    def export_events(
        self,
        format: str = "json",
        limit: int = 1000,
    ) -> str:
        """Export events to a format."""
        events = self.query(limit=limit)

        if format == "json":
            return json.dumps([e.to_dict() for e in events], indent=2)

        # CSV format
        lines = ["event_id,event_type,severity,source,message,timestamp"]
        for e in events:
            lines.append(
                f"{e.event_id},{e.event_type.value},{e.severity.value},"
                f"{e.source},{e.message},{e.timestamp.isoformat()}"
            )
        return "\n".join(lines)


# ============================================================================
# Alert Manager
# ============================================================================


class AlertManager:
    """Manages security alerts."""

    def __init__(self) -> None:
        self._alerts: Dict[str, SecurityAlert] = {}
        self._handlers: List[Callable[[SecurityAlert], None]] = []
        self._lock = threading.Lock()

    def create_alert(
        self,
        title: str,
        description: str,
        severity: SeverityLevel,
        threat_level: ThreatLevel,
        related_events: Optional[List[str]] = None,
    ) -> SecurityAlert:
        """Create a new alert."""
        alert = SecurityAlert(
            alert_id=str(uuid.uuid4()),
            title=title,
            description=description,
            severity=severity,
            threat_level=threat_level,
            related_events=related_events or [],
        )

        with self._lock:
            self._alerts[alert.alert_id] = alert

        # Notify handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass

        return alert

    def get_alert(self, alert_id: str) -> Optional[SecurityAlert]:
        """Get an alert by ID."""
        return self._alerts.get(alert_id)

    def update_alert(
        self,
        alert_id: str,
        status: Optional[AlertStatus] = None,
        assigned_to: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> Optional[SecurityAlert]:
        """Update an alert."""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return None

            if status:
                alert.status = status
            if assigned_to:
                alert.assigned_to = assigned_to
            if resolution:
                alert.resolution = resolution
            alert.updated_at = datetime.utcnow()

            return alert

    def list_alerts(
        self,
        status: Optional[AlertStatus] = None,
        threat_level: Optional[ThreatLevel] = None,
        limit: int = 100,
    ) -> List[SecurityAlert]:
        """List alerts."""
        alerts = list(self._alerts.values())

        if status:
            alerts = [a for a in alerts if a.status == status]

        if threat_level:
            alerts = [a for a in alerts if a.threat_level == threat_level]

        alerts.sort(key=lambda a: a.created_at, reverse=True)
        return alerts[:limit]

    def add_handler(self, handler: Callable[[SecurityAlert], None]) -> None:
        """Add an alert handler."""
        self._handlers.append(handler)


# ============================================================================
# Security Audit Manager
# ============================================================================


class SecurityAuditManager:
    """Comprehensive security audit manager."""

    def __init__(self) -> None:
        self._logger = AuditLogger()
        self._alert_manager = AlertManager()
        self._detectors: List[ThreatDetector] = [
            BruteForceDetector(),
            AnomalyDetector(),
            IPReputationDetector(),
        ]
        self._policies: Dict[str, AuditPolicy] = {}
        self._threat_indicators: Dict[str, ThreatIndicator] = {}
        self._lock = threading.Lock()

    @property
    def logger(self) -> AuditLogger:
        """Get the audit logger."""
        return self._logger

    @property
    def alert_manager(self) -> AlertManager:
        """Get the alert manager."""
        return self._alert_manager

    def add_detector(self, detector: ThreatDetector) -> None:
        """Add a threat detector."""
        self._detectors.append(detector)

    def add_policy(self, policy: AuditPolicy) -> None:
        """Add an audit policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy

    def log_event(
        self,
        event_type: SecurityEventType,
        severity: SeverityLevel,
        source: str,
        message: str,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> SecurityEvent:
        """Log a security event."""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            source=source,
            message=message,
            user_id=user_id,
            resource_id=resource_id,
            ip_address=ip_address,
            details=details or {},
        )

        # Log the event
        self._logger.log(event)

        # Run threat detection
        self._detect_threats(event)

        # Check policies
        self._check_policies(event)

        return event

    def _detect_threats(self, event: SecurityEvent) -> None:
        """Run threat detection on event."""
        for detector in self._detectors:
            indicator = detector.detect(event)
            if indicator:
                with self._lock:
                    self._threat_indicators[indicator.indicator_id] = indicator

                # Create alert for threats
                self._alert_manager.create_alert(
                    title=f"Threat Detected: {indicator.indicator_type}",
                    description=indicator.description,
                    severity=SeverityLevel.WARNING
                    if indicator.threat_level == ThreatLevel.MEDIUM
                    else SeverityLevel.CRITICAL,
                    threat_level=indicator.threat_level,
                    related_events=[event.event_id],
                )

                # Log threat event
                self._logger.log(
                    SecurityEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=SecurityEventType.THREAT_DETECTED,
                        severity=SeverityLevel.WARNING,
                        source="threat_detector",
                        message=indicator.description,
                        details={"indicator_id": indicator.indicator_id},
                    )
                )

    def _check_policies(self, event: SecurityEvent) -> None:
        """Check event against audit policies."""
        for policy in self._policies.values():
            if not policy.enabled:
                continue

            if event.event_type not in policy.event_types:
                continue

            severity_order = list(SeverityLevel)
            if severity_order.index(event.severity) < severity_order.index(
                policy.severity_threshold
            ):
                continue

            if policy.alert_enabled:
                self._alert_manager.create_alert(
                    title=f"Policy Alert: {policy.name}",
                    description=f"Event matched policy: {event.message}",
                    severity=event.severity,
                    threat_level=ThreatLevel.LOW,
                    related_events=[event.event_id],
                )

    def get_threat_indicators(
        self,
        active_only: bool = True,
    ) -> List[ThreatIndicator]:
        """Get threat indicators."""
        with self._lock:
            indicators = list(self._threat_indicators.values())

        if active_only:
            now = datetime.utcnow()
            indicators = [
                i for i in indicators if i.active and (not i.expires_at or i.expires_at > now)
            ]

        return indicators


# ============================================================================
# Security Audit Vision Provider
# ============================================================================


class SecurityAuditVisionProvider(VisionProvider):
    """Vision provider with security auditing."""

    def __init__(
        self,
        provider: VisionProvider,
        audit_manager: SecurityAuditManager,
        user_id: Optional[str] = None,
    ) -> None:
        self._provider = provider
        self._audit_manager = audit_manager
        self._user_id = user_id

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"audited_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        **kwargs: Any,
    ) -> VisionDescription:
        """Analyze image with security auditing."""
        # Log access event
        self._audit_manager.log_event(
            event_type=SecurityEventType.DATA_ACCESS,
            severity=SeverityLevel.INFO,
            source=self.provider_name,
            message="Image analysis requested",
            user_id=self._user_id or kwargs.get("user_id"),
            resource_id=kwargs.get("resource_id"),
            ip_address=kwargs.get("ip_address"),
            details={
                "image_size": len(image_data),
                "include_description": include_description,
            },
        )

        try:
            result = await self._provider.analyze_image(image_data, include_description)

            # Log success
            self._audit_manager.log_event(
                event_type=SecurityEventType.DATA_ACCESS,
                severity=SeverityLevel.INFO,
                source=self.provider_name,
                message="Image analysis completed successfully",
                user_id=self._user_id or kwargs.get("user_id"),
                details={"confidence": result.confidence},
            )

            return result

        except Exception as e:
            # Log failure
            self._audit_manager.log_event(
                event_type=SecurityEventType.ANOMALY_DETECTED,
                severity=SeverityLevel.ERROR,
                source=self.provider_name,
                message=f"Image analysis failed: {str(e)}",
                user_id=self._user_id or kwargs.get("user_id"),
                details={"error": str(e)},
            )
            raise

    def get_audit_manager(self) -> SecurityAuditManager:
        """Get the audit manager."""
        return self._audit_manager


# ============================================================================
# Factory Functions
# ============================================================================


def create_audit_logger(max_events: int = 100000) -> AuditLogger:
    """Create an audit logger."""
    return AuditLogger(max_events)


def create_alert_manager() -> AlertManager:
    """Create an alert manager."""
    return AlertManager()


def create_security_audit_manager() -> SecurityAuditManager:
    """Create a security audit manager."""
    return SecurityAuditManager()


def create_brute_force_detector(
    max_attempts: int = 5,
    window_seconds: int = 300,
) -> BruteForceDetector:
    """Create a brute force detector."""
    return BruteForceDetector(max_attempts, window_seconds)


def create_anomaly_detector() -> AnomalyDetector:
    """Create an anomaly detector."""
    return AnomalyDetector()


def create_ip_reputation_detector() -> IPReputationDetector:
    """Create an IP reputation detector."""
    return IPReputationDetector()


def create_audit_provider(
    provider: VisionProvider,
    audit_manager: Optional[SecurityAuditManager] = None,
    user_id: Optional[str] = None,
) -> SecurityAuditVisionProvider:
    """Create a security audit vision provider."""
    if audit_manager is None:
        audit_manager = create_security_audit_manager()
    return SecurityAuditVisionProvider(provider, audit_manager, user_id)


def create_security_event(
    event_type: SecurityEventType,
    severity: SeverityLevel,
    source: str,
    message: str,
) -> SecurityEvent:
    """Create a security event."""
    return SecurityEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        severity=severity,
        source=source,
        message=message,
    )


def create_audit_policy(
    name: str,
    event_types: List[SecurityEventType],
    severity_threshold: SeverityLevel = SeverityLevel.WARNING,
    retention_days: int = 90,
) -> AuditPolicy:
    """Create an audit policy."""
    return AuditPolicy(
        policy_id=str(uuid.uuid4()),
        name=name,
        event_types=event_types,
        severity_threshold=severity_threshold,
        retention_days=retention_days,
    )
