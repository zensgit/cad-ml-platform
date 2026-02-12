"""
Model monitoring modules.

Provides:
- Real-time metrics collection
- Drift detection
- Alerting system
"""

from src.ml.monitoring.metrics import (
    MetricType,
    MetricValue,
    MetricSummary,
    Counter,
    Gauge,
    Histogram,
    SlidingWindowMetric,
    MetricsCollector,
    get_metrics_collector,
)

from src.ml.monitoring.drift import (
    DriftType,
    DriftSeverity,
    DriftResult,
    DriftDetector,
    KSTestDetector,
    PSIDetector,
    PageHinkleyDetector,
    DriftMonitor,
)

from src.ml.monitoring.alerts import (
    AlertSeverity,
    AlertStatus,
    Alert,
    AlertRule,
    AlertChannel,
    LogChannel,
    WebhookChannel,
    CallbackChannel,
    AlertManager,
    get_alert_manager,
)

__all__ = [
    # Metrics
    "MetricType",
    "MetricValue",
    "MetricSummary",
    "Counter",
    "Gauge",
    "Histogram",
    "SlidingWindowMetric",
    "MetricsCollector",
    "get_metrics_collector",
    # Drift
    "DriftType",
    "DriftSeverity",
    "DriftResult",
    "DriftDetector",
    "KSTestDetector",
    "PSIDetector",
    "PageHinkleyDetector",
    "DriftMonitor",
    # Alerts
    "AlertSeverity",
    "AlertStatus",
    "Alert",
    "AlertRule",
    "AlertChannel",
    "LogChannel",
    "WebhookChannel",
    "CallbackChannel",
    "AlertManager",
    "get_alert_manager",
]
