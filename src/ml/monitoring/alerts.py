"""
Alerting system for model monitoring.

Generates and manages alerts based on monitoring data.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """An alert instance."""
    alert_id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    source: str
    created_at: float
    updated_at: float
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
            "acknowledged_by": self.acknowledged_by,
            "metadata": self.metadata,
            "tags": self.tags,
        }


@dataclass
class AlertRule:
    """Rule for generating alerts."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: float = 300.0  # 5 minutes
    tags: List[str] = field(default_factory=list)
    enabled: bool = True

    _last_fired: float = 0.0

    def evaluate(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Evaluate rule against context.

        Returns alert message if triggered, None otherwise.
        """
        if not self.enabled:
            return None

        # Check cooldown
        if time.time() - self._last_fired < self.cooldown_seconds:
            return None

        try:
            if self.condition(context):
                self._last_fired = time.time()
                return self.message_template.format(**context)
        except Exception as e:
            logger.error(f"Error evaluating alert rule {self.name}: {e}")

        return None


class AlertChannel(ABC):
    """Abstract base class for alert notification channels."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        pass


class LogChannel(AlertChannel):
    """Send alerts to logging system."""

    def __init__(self, logger_name: str = "alerts"):
        self._logger = logging.getLogger(logger_name)

    def send(self, alert: Alert) -> bool:
        """Log the alert."""
        level_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }
        level = level_map.get(alert.severity, logging.WARNING)
        self._logger.log(level, f"[{alert.name}] {alert.message}", extra=alert.metadata)
        return True


class WebhookChannel(AlertChannel):
    """Send alerts via webhook."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {}

    def send(self, alert: Alert) -> bool:
        """Send alert via HTTP POST."""
        try:
            import urllib.request
            import json

            data = json.dumps(alert.to_dict()).encode('utf-8')
            headers = {"Content-Type": "application/json", **self.headers}

            req = urllib.request.Request(self.url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class CallbackChannel(AlertChannel):
    """Send alerts via callback function."""

    def __init__(self, callback: Callable[[Alert], None]):
        self.callback = callback

    def send(self, alert: Alert) -> bool:
        """Call the callback with the alert."""
        try:
            self.callback(alert)
            return True
        except Exception as e:
            logger.error(f"Alert callback failed: {e}")
            return False


class AlertManager:
    """
    Central alert management.

    Handles alert lifecycle, rules, and notifications.
    """

    def __init__(
        self,
        max_active_alerts: int = 1000,
        retention_hours: float = 24.0,
    ):
        self.max_active_alerts = max_active_alerts
        self.retention_hours = retention_hours

        self._alerts: Dict[str, Alert] = {}
        self._rules: List[AlertRule] = []
        self._channels: List[AlertChannel] = []
        self._lock = threading.Lock()
        self._alert_counter = 0

        # Default log channel
        self._channels.append(LogChannel())

        # Initialize default rules
        self._init_default_rules()

    def _init_default_rules(self) -> None:
        """Initialize default monitoring rules."""
        # High latency alert
        self.add_rule(AlertRule(
            name="high_latency",
            condition=lambda ctx: ctx.get("latency_p99", 0) > 5.0,
            severity=AlertSeverity.WARNING,
            message_template="High latency detected: P99={latency_p99:.2f}s",
            cooldown_seconds=300,
            tags=["performance"],
        ))

        # Error rate alert
        self.add_rule(AlertRule(
            name="high_error_rate",
            condition=lambda ctx: ctx.get("error_rate", 0) > 0.05,
            severity=AlertSeverity.ERROR,
            message_template="High error rate: {error_rate:.1%}",
            cooldown_seconds=300,
            tags=["reliability"],
        ))

        # Low confidence alert
        self.add_rule(AlertRule(
            name="low_confidence",
            condition=lambda ctx: ctx.get("mean_confidence", 1.0) < 0.5,
            severity=AlertSeverity.WARNING,
            message_template="Low average confidence: {mean_confidence:.2f}",
            cooldown_seconds=600,
            tags=["model"],
        ))

        # Drift detected alert
        self.add_rule(AlertRule(
            name="drift_detected",
            condition=lambda ctx: ctx.get("drift_detected", False),
            severity=AlertSeverity.WARNING,
            message_template="Data drift detected: {drift_type}",
            cooldown_seconds=3600,
            tags=["drift"],
        ))

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self._lock:
            self._rules.append(rule)

    def add_channel(self, channel: AlertChannel) -> None:
        """Add a notification channel."""
        with self._lock:
            self._channels.append(channel)

    def evaluate_rules(self, context: Dict[str, Any]) -> List[Alert]:
        """Evaluate all rules against context."""
        new_alerts = []

        with self._lock:
            for rule in self._rules:
                message = rule.evaluate(context)
                if message:
                    alert = self._create_alert(
                        name=rule.name,
                        severity=rule.severity,
                        message=message,
                        source="rule:" + rule.name,
                        tags=rule.tags,
                        metadata=context,
                    )
                    new_alerts.append(alert)

                    # Notify channels
                    for channel in self._channels:
                        try:
                            channel.send(alert)
                        except Exception as e:
                            logger.error(f"Failed to send alert to channel: {e}")

        return new_alerts

    def _create_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        source: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create a new alert."""
        self._alert_counter += 1
        now = time.time()

        alert = Alert(
            alert_id=f"alert-{self._alert_counter}",
            name=name,
            severity=severity,
            status=AlertStatus.ACTIVE,
            message=message,
            source=source,
            created_at=now,
            updated_at=now,
            tags=tags or [],
            metadata=metadata or {},
        )

        self._alerts[alert.alert_id] = alert
        self._cleanup_old_alerts()

        logger.info(f"Alert created: {alert.alert_id} - {alert.name}")
        return alert

    def fire_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        source: str = "manual",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Manually fire an alert."""
        with self._lock:
            alert = self._create_alert(name, severity, message, source, tags, metadata)

            for channel in self._channels:
                try:
                    channel.send(alert)
                except Exception as e:
                    logger.error(f"Failed to send alert: {e}")

            return alert

    def acknowledge(self, alert_id: str, by: str = "system") -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                return False

            alert = self._alerts[alert_id]
            if alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                alert.acknowledged_by = by
                alert.updated_at = time.time()
                return True
            return False

    def resolve(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                return False

            alert = self._alerts[alert_id]
            if alert.status in (AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED):
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                alert.updated_at = time.time()
                return True
            return False

    def suppress(self, alert_id: str) -> bool:
        """Suppress an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                return False

            alert = self._alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.updated_at = time.time()
            return True

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return [
                a for a in self._alerts.values()
                if a.status == AlertStatus.ACTIVE
            ]

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        with self._lock:
            return self._alerts.get(alert_id)

    def get_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alerts with optional filtering."""
        with self._lock:
            alerts = list(self._alerts.values())

            if status:
                alerts = [a for a in alerts if a.status == status]
            if severity:
                alerts = [a for a in alerts if a.severity == severity]

            # Sort by creation time, newest first
            alerts.sort(key=lambda a: a.created_at, reverse=True)
            return alerts[:limit]

    def _cleanup_old_alerts(self) -> None:
        """Remove old resolved alerts."""
        cutoff = time.time() - (self.retention_hours * 3600)

        to_remove = [
            alert_id for alert_id, alert in self._alerts.items()
            if alert.status == AlertStatus.RESOLVED and alert.resolved_at and alert.resolved_at < cutoff
        ]

        for alert_id in to_remove:
            del self._alerts[alert_id]

        # Also enforce max alerts limit
        if len(self._alerts) > self.max_active_alerts:
            # Remove oldest resolved alerts first
            resolved = sorted(
                [a for a in self._alerts.values() if a.status == AlertStatus.RESOLVED],
                key=lambda a: a.created_at
            )
            for alert in resolved[:len(self._alerts) - self.max_active_alerts]:
                del self._alerts[alert.alert_id]

    def get_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        with self._lock:
            by_status = {}
            by_severity = {}

            for alert in self._alerts.values():
                by_status[alert.status.value] = by_status.get(alert.status.value, 0) + 1
                by_severity[alert.severity.value] = by_severity.get(alert.severity.value, 0) + 1

            return {
                "total": len(self._alerts),
                "by_status": by_status,
                "by_severity": by_severity,
                "active_count": by_status.get("active", 0),
            }


# Global alert manager
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
