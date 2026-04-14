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

from src.ml.monitoring.anomaly_detector import (
    AnomalyResult,
    MetricsAnomalyDetector,
)

from src.ml.monitoring.auto_remediation import (
    AutoRemediation,
    RemediationResult,
    REMEDIATION_RULES,
)

from src.ml.monitoring.prediction_monitor import (
    PredictionRecord,
    PredictionMonitor,
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
    # Anomaly detection
    "AnomalyResult",
    "MetricsAnomalyDetector",
    # Auto-remediation
    "AutoRemediation",
    "RemediationResult",
    "REMEDIATION_RULES",
    # B5.3 Prediction monitor
    "PredictionRecord",
    "PredictionMonitor",
]
