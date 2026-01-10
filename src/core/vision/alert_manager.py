"""Alert Manager Module for Vision System.

This module provides alerting capabilities including:
- Alert rule definition and evaluation
- Multi-channel notifications (email, Slack, webhook, PagerDuty)
- Alert escalation policies
- Alert grouping and deduplication
- Silence and inhibition rules
- Alert history and analytics

Phase 17: Advanced Observability & Monitoring
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


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertState(str, Enum):
    """Alert states."""

    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"
    INHIBITED = "inhibited"


class NotificationChannel(str, Enum):
    """Notification channel types."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SMS = "sms"
    TEAMS = "teams"
    OPSGENIE = "opsgenie"
    CUSTOM = "custom"


class ComparisonOperator(str, Enum):
    """Comparison operators for alert rules."""

    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"


class EscalationAction(str, Enum):
    """Escalation actions."""

    NOTIFY = "notify"
    ESCALATE = "escalate"
    AUTO_RESOLVE = "auto_resolve"
    RUNBOOK = "runbook"


# ========================
# Data Classes
# ========================


@dataclass
class AlertCondition:
    """Condition for triggering an alert."""

    metric: str
    operator: ComparisonOperator
    threshold: float
    duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Definition of an alert rule."""

    rule_id: str
    name: str
    conditions: List[AlertCondition]
    severity: AlertSeverity = AlertSeverity.MEDIUM
    description: str = ""
    runbook_url: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    notify_channels: List[str] = field(default_factory=list)
    evaluation_interval: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    for_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    enabled: bool = True


@dataclass
class Alert:
    """An active or historical alert."""

    alert_id: str
    rule_id: str
    name: str
    severity: AlertSeverity
    state: AlertState
    message: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    last_evaluated: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    value: Optional[float] = None
    fingerprint: str = ""
    notified_channels: List[str] = field(default_factory=list)


@dataclass
class NotificationConfig:
    """Configuration for a notification channel."""

    channel_id: str
    channel_type: NotificationChannel
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=list)
    labels_filter: Dict[str, str] = field(default_factory=dict)


@dataclass
class EscalationStep:
    """A step in an escalation policy."""

    step_id: str
    action: EscalationAction
    delay: timedelta
    channels: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Expression to check


@dataclass
class EscalationPolicy:
    """Escalation policy for alerts."""

    policy_id: str
    name: str
    steps: List[EscalationStep]
    enabled: bool = True
    repeat_interval: Optional[timedelta] = None


@dataclass
class SilenceRule:
    """Rule for silencing alerts."""

    silence_id: str
    matchers: Dict[str, str]
    starts_at: datetime
    ends_at: datetime
    created_by: str
    comment: str = ""


@dataclass
class InhibitionRule:
    """Rule for inhibiting alerts based on other alerts."""

    inhibition_id: str
    source_matchers: Dict[str, str]  # Alert that inhibits
    target_matchers: Dict[str, str]  # Alert being inhibited
    equal_labels: List[str] = field(default_factory=list)


@dataclass
class NotificationResult:
    """Result of a notification attempt."""

    channel_id: str
    success: bool
    message: str = ""
    sent_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


# ========================
# Notification Handlers
# ========================


class NotificationHandler(ABC):
    """Abstract base class for notification handlers."""

    @abstractmethod
    async def send(self, alert: Alert, config: NotificationConfig) -> NotificationResult:
        """Send notification for an alert."""
        pass


class EmailNotificationHandler(NotificationHandler):
    """Email notification handler."""

    async def send(self, alert: Alert, config: NotificationConfig) -> NotificationResult:
        """Send email notification."""
        # Simulated email sending
        recipients = config.config.get("recipients", [])

        return NotificationResult(
            channel_id=config.channel_id,
            success=True,
            message=f"Email sent to {len(recipients)} recipients",
        )


class SlackNotificationHandler(NotificationHandler):
    """Slack notification handler."""

    async def send(self, alert: Alert, config: NotificationConfig) -> NotificationResult:
        """Send Slack notification."""
        webhook_url = config.config.get("webhook_url", "")
        channel = config.config.get("channel", "#alerts")

        # Simulated Slack message
        return NotificationResult(
            channel_id=config.channel_id, success=True, message=f"Slack message sent to {channel}"
        )


class WebhookNotificationHandler(NotificationHandler):
    """Webhook notification handler."""

    async def send(self, alert: Alert, config: NotificationConfig) -> NotificationResult:
        """Send webhook notification."""
        url = config.config.get("url", "")

        # Simulated webhook call
        return NotificationResult(
            channel_id=config.channel_id, success=True, message=f"Webhook sent to {url}"
        )


class PagerDutyNotificationHandler(NotificationHandler):
    """PagerDuty notification handler."""

    async def send(self, alert: Alert, config: NotificationConfig) -> NotificationResult:
        """Send PagerDuty notification."""
        service_key = config.config.get("service_key", "")

        # Simulated PagerDuty event
        return NotificationResult(
            channel_id=config.channel_id, success=True, message="PagerDuty incident created"
        )


# ========================
# Alert Evaluator
# ========================


class AlertEvaluator:
    """Evaluates alert conditions."""

    def __init__(self, metric_getter: Callable[[str, Dict[str, str]], Optional[float]]):
        self._metric_getter = metric_getter
        self._pending_alerts: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def evaluate_rule(self, rule: AlertRule) -> Tuple[bool, Optional[float]]:
        """Evaluate an alert rule."""
        for condition in rule.conditions:
            value = self._metric_getter(condition.metric, condition.labels)

            if value is None:
                continue

            if not self._check_condition(value, condition):
                return False, value

        return True, value if rule.conditions else None

    def _check_condition(self, value: float, condition: AlertCondition) -> bool:
        """Check if a single condition is met."""
        threshold = condition.threshold
        operator = condition.operator

        if operator == ComparisonOperator.GREATER_THAN:
            return value > threshold
        elif operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
            return value >= threshold
        elif operator == ComparisonOperator.LESS_THAN:
            return value < threshold
        elif operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
            return value <= threshold
        elif operator == ComparisonOperator.EQUAL:
            return value == threshold
        elif operator == ComparisonOperator.NOT_EQUAL:
            return value != threshold

        return False

    def check_pending_duration(self, rule: AlertRule) -> bool:
        """Check if alert has been pending long enough."""
        with self._lock:
            pending_since = self._pending_alerts.get(rule.rule_id)

            if pending_since is None:
                return False

            return datetime.now() - pending_since >= rule.for_duration

    def mark_pending(self, rule: AlertRule) -> None:
        """Mark an alert as pending."""
        with self._lock:
            if rule.rule_id not in self._pending_alerts:
                self._pending_alerts[rule.rule_id] = datetime.now()

    def clear_pending(self, rule: AlertRule) -> None:
        """Clear pending status for a rule."""
        with self._lock:
            self._pending_alerts.pop(rule.rule_id, None)


# ========================
# Alert Manager
# ========================


class AlertManager:
    """Main alert manager coordinating all alerting operations."""

    def __init__(
        self, metric_getter: Optional[Callable[[str, Dict[str, str]], Optional[float]]] = None
    ):
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._channels: Dict[str, NotificationConfig] = {}
        self._policies: Dict[str, EscalationPolicy] = {}
        self._silences: Dict[str, SilenceRule] = {}
        self._inhibitions: List[InhibitionRule] = []

        self._handlers: Dict[NotificationChannel, NotificationHandler] = {
            NotificationChannel.EMAIL: EmailNotificationHandler(),
            NotificationChannel.SLACK: SlackNotificationHandler(),
            NotificationChannel.WEBHOOK: WebhookNotificationHandler(),
            NotificationChannel.PAGERDUTY: PagerDutyNotificationHandler(),
        }

        self._metric_getter = metric_getter or (lambda m, l: None)
        self._evaluator = AlertEvaluator(self._metric_getter)
        self._lock = threading.Lock()

    # Rule Management

    def register_rule(self, rule: AlertRule) -> bool:
        """Register an alert rule."""
        with self._lock:
            if rule.rule_id in self._rules:
                return False
            self._rules[rule.rule_id] = rule
            return True

    def update_rule(self, rule: AlertRule) -> bool:
        """Update an existing rule."""
        with self._lock:
            if rule.rule_id not in self._rules:
                return False
            self._rules[rule.rule_id] = rule
            return True

    def delete_rule(self, rule_id: str) -> bool:
        """Delete an alert rule."""
        with self._lock:
            if rule_id not in self._rules:
                return False
            del self._rules[rule_id]
            return True

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule."""
        with self._lock:
            return self._rules.get(rule_id)

    def list_rules(self) -> List[AlertRule]:
        """List all rules."""
        with self._lock:
            return list(self._rules.values())

    # Channel Management

    def register_channel(self, config: NotificationConfig) -> bool:
        """Register a notification channel."""
        with self._lock:
            if config.channel_id in self._channels:
                return False
            self._channels[config.channel_id] = config
            return True

    def update_channel(self, config: NotificationConfig) -> bool:
        """Update a notification channel."""
        with self._lock:
            if config.channel_id not in self._channels:
                return False
            self._channels[config.channel_id] = config
            return True

    def delete_channel(self, channel_id: str) -> bool:
        """Delete a notification channel."""
        with self._lock:
            if channel_id not in self._channels:
                return False
            del self._channels[channel_id]
            return True

    def get_channel(self, channel_id: str) -> Optional[NotificationConfig]:
        """Get a notification channel."""
        with self._lock:
            return self._channels.get(channel_id)

    def list_channels(self) -> List[NotificationConfig]:
        """List all channels."""
        with self._lock:
            return list(self._channels.values())

    # Escalation Policy Management

    def register_policy(self, policy: EscalationPolicy) -> bool:
        """Register an escalation policy."""
        with self._lock:
            if policy.policy_id in self._policies:
                return False
            self._policies[policy.policy_id] = policy
            return True

    def get_policy(self, policy_id: str) -> Optional[EscalationPolicy]:
        """Get an escalation policy."""
        with self._lock:
            return self._policies.get(policy_id)

    # Silence Management

    def create_silence(self, silence: SilenceRule) -> bool:
        """Create a silence rule."""
        with self._lock:
            if silence.silence_id in self._silences:
                return False
            self._silences[silence.silence_id] = silence
            return True

    def delete_silence(self, silence_id: str) -> bool:
        """Delete a silence rule."""
        with self._lock:
            if silence_id not in self._silences:
                return False
            del self._silences[silence_id]
            return True

    def list_silences(self) -> List[SilenceRule]:
        """List all silence rules."""
        with self._lock:
            return list(self._silences.values())

    def is_silenced(self, alert: Alert) -> bool:
        """Check if an alert is silenced."""
        now = datetime.now()

        with self._lock:
            for silence in self._silences.values():
                if silence.starts_at <= now <= silence.ends_at:
                    if self._matches_labels(alert.labels, silence.matchers):
                        return True
            return False

    # Inhibition Management

    def add_inhibition(self, rule: InhibitionRule) -> None:
        """Add an inhibition rule."""
        with self._lock:
            self._inhibitions.append(rule)

    def is_inhibited(self, alert: Alert) -> bool:
        """Check if an alert is inhibited."""
        with self._lock:
            for inhibition in self._inhibitions:
                # Check if target matches this alert
                if not self._matches_labels(alert.labels, inhibition.target_matchers):
                    continue

                # Check if there's a firing source alert
                for existing_alert in self._alerts.values():
                    if existing_alert.state != AlertState.FIRING:
                        continue

                    if not self._matches_labels(existing_alert.labels, inhibition.source_matchers):
                        continue

                    # Check equal labels
                    if all(
                        alert.labels.get(label) == existing_alert.labels.get(label)
                        for label in inhibition.equal_labels
                    ):
                        return True

            return False

    def _matches_labels(self, alert_labels: Dict[str, str], matchers: Dict[str, str]) -> bool:
        """Check if alert labels match matchers."""
        for key, value in matchers.items():
            if alert_labels.get(key) != value:
                return False
        return True

    # Alert Evaluation

    async def evaluate_rules(self) -> List[Alert]:
        """Evaluate all enabled rules."""
        new_alerts = []

        with self._lock:
            rules = [r for r in self._rules.values() if r.enabled]

        for rule in rules:
            alert = await self._evaluate_single_rule(rule)
            if alert:
                new_alerts.append(alert)

        return new_alerts

    async def _evaluate_single_rule(self, rule: AlertRule) -> Optional[Alert]:
        """Evaluate a single rule."""
        is_firing, value = self._evaluator.evaluate_rule(rule)

        with self._lock:
            existing_alert = self._alerts.get(rule.rule_id)

        if is_firing:
            self._evaluator.mark_pending(rule)

            if self._evaluator.check_pending_duration(rule):
                if existing_alert and existing_alert.state == AlertState.FIRING:
                    # Update existing alert
                    existing_alert.last_evaluated = datetime.now()
                    existing_alert.value = value
                    return None
                else:
                    # Create new alert
                    alert = self._create_alert(rule, value)

                    # Check silencing and inhibition
                    if self.is_silenced(alert):
                        alert.state = AlertState.SILENCED
                    elif self.is_inhibited(alert):
                        alert.state = AlertState.INHIBITED

                    with self._lock:
                        self._alerts[rule.rule_id] = alert

                    # Send notifications if firing
                    if alert.state == AlertState.FIRING:
                        await self._send_notifications(alert, rule)

                    return alert
        else:
            self._evaluator.clear_pending(rule)

            if existing_alert and existing_alert.state == AlertState.FIRING:
                # Resolve the alert
                existing_alert.state = AlertState.RESOLVED
                existing_alert.ended_at = datetime.now()

                with self._lock:
                    self._alert_history.append(existing_alert)
                    del self._alerts[rule.rule_id]

                # Send resolution notification
                await self._send_notifications(existing_alert, rule)

                return existing_alert

        return None

    def _create_alert(self, rule: AlertRule, value: Optional[float]) -> Alert:
        """Create an alert from a rule."""
        fingerprint = hashlib.sha256(
            f"{rule.rule_id}:{json.dumps(rule.labels, sort_keys=True)}".encode()
        ).hexdigest()

        return Alert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            rule_id=rule.rule_id,
            name=rule.name,
            severity=rule.severity,
            state=AlertState.FIRING,
            message=rule.description,
            started_at=datetime.now(),
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy(),
            value=value,
            fingerprint=fingerprint,
        )

    async def _send_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications for an alert."""
        for channel_id in rule.notify_channels:
            channel = self.get_channel(channel_id)
            if not channel or not channel.enabled:
                continue

            # Check severity filter
            if channel.severity_filter and alert.severity not in channel.severity_filter:
                continue

            # Check labels filter
            if not self._matches_labels(alert.labels, channel.labels_filter):
                continue

            handler = self._handlers.get(channel.channel_type)
            if handler:
                try:
                    result = await handler.send(alert, channel)
                    if result.success:
                        alert.notified_channels.append(channel_id)
                except Exception:
                    pass

    # Alert Queries

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        with self._lock:
            for alert in self._alerts.values():
                if alert.alert_id == alert_id:
                    return alert
            for alert in self._alert_history:
                if alert.alert_id == alert_id:
                    return alert
            return None

    def list_alerts(
        self, state: Optional[AlertState] = None, severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """List alerts with optional filters."""
        with self._lock:
            alerts = list(self._alerts.values())

        if state:
            alerts = [a for a in alerts if a.state == state]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def get_alert_history(
        self,
        rule_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            history = self._alert_history[:]

        if rule_id:
            history = [a for a in history if a.rule_id == rule_id]
        if start_time:
            history = [a for a in history if a.started_at >= start_time]
        if end_time:
            history = [a for a in history if a.started_at <= end_time]

        return history

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts.values():
                if alert.alert_id == alert_id:
                    alert.annotations["acknowledged_by"] = acknowledged_by
                    alert.annotations["acknowledged_at"] = datetime.now().isoformat()
                    return True
            return False


# ========================
# Alert Manager Provider
# ========================


class AlertManagerVisionProvider(VisionProvider):
    """Vision provider with alert management integration."""

    def __init__(self, base_provider: VisionProvider, alert_manager: Optional[AlertManager] = None):
        self._base_provider = base_provider
        self._alert_manager = alert_manager or AlertManager()
        self._error_count = 0
        self._request_count = 0

    @property
    def provider_name(self) -> str:
        return f"alert_manager_{self._base_provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, context: Optional[Dict[str, Any]] = None
    ) -> VisionDescription:
        """Analyze image with alert monitoring."""
        self._request_count += 1

        try:
            result = await self._base_provider.analyze_image(image_data, context)

            # Reset error count on success
            self._error_count = 0

            return result

        except Exception as e:
            self._error_count += 1

            # Evaluate rules after error
            await self._alert_manager.evaluate_rules()

            raise

    def get_alert_manager(self) -> AlertManager:
        """Get the alert manager."""
        return self._alert_manager


# ========================
# Factory Functions
# ========================


def create_alert_manager(
    metric_getter: Optional[Callable[[str, Dict[str, str]], Optional[float]]] = None
) -> AlertManager:
    """Create a new alert manager."""
    return AlertManager(metric_getter)


def create_alert_rule(
    rule_id: str,
    name: str,
    metric: str,
    operator: ComparisonOperator,
    threshold: float,
    severity: AlertSeverity = AlertSeverity.MEDIUM,
    notify_channels: Optional[List[str]] = None,
) -> AlertRule:
    """Create an alert rule with a single condition."""
    return AlertRule(
        rule_id=rule_id,
        name=name,
        conditions=[AlertCondition(metric=metric, operator=operator, threshold=threshold)],
        severity=severity,
        notify_channels=notify_channels or [],
    )


def create_notification_channel(
    channel_id: str, channel_type: NotificationChannel, name: str, **config: Any
) -> NotificationConfig:
    """Create a notification channel."""
    return NotificationConfig(
        channel_id=channel_id, channel_type=channel_type, name=name, config=config
    )


def create_silence(
    silence_id: str,
    matchers: Dict[str, str],
    duration: timedelta,
    created_by: str,
    comment: str = "",
) -> SilenceRule:
    """Create a silence rule."""
    now = datetime.now()
    return SilenceRule(
        silence_id=silence_id,
        matchers=matchers,
        starts_at=now,
        ends_at=now + duration,
        created_by=created_by,
        comment=comment,
    )


def create_escalation_policy(
    policy_id: str, name: str, steps: List[EscalationStep]
) -> EscalationPolicy:
    """Create an escalation policy."""
    return EscalationPolicy(policy_id=policy_id, name=name, steps=steps)


def create_alert_manager_provider(
    base_provider: VisionProvider, alert_manager: Optional[AlertManager] = None
) -> AlertManagerVisionProvider:
    """Create an alert manager vision provider."""
    return AlertManagerVisionProvider(base_provider, alert_manager)
