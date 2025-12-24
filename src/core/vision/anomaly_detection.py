"""
Anomaly Detection Module - Phase 12.

Provides anomaly detection capabilities including statistical methods,
threshold monitoring, pattern recognition, and alerting.
"""

import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from .base import VisionDescription, VisionProvider

# ============================================================================
# Enums
# ============================================================================


class AnomalyType(Enum):
    """Types of anomalies."""

    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    SEASONAL = "seasonal"


class DetectionMethod(Enum):
    """Anomaly detection methods."""

    ZSCORE = "zscore"
    IQR = "iqr"
    MAD = "mad"
    THRESHOLD = "threshold"
    PERCENTILE = "percentile"
    MOVING_AVERAGE = "moving_average"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DataPoint:
    """A single data point for analysis."""

    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    """A detected anomaly."""

    anomaly_id: str
    anomaly_type: AnomalyType
    value: float
    expected_value: float
    deviation: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0
    method: DetectionMethod = DetectionMethod.ZSCORE
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def deviation_percentage(self) -> float:
        """Calculate deviation as percentage."""
        if self.expected_value == 0:
            return 100.0 if self.value != 0 else 0.0
        return abs(self.value - self.expected_value) / abs(self.expected_value) * 100


@dataclass
class Alert:
    """An alert triggered by anomaly detection."""

    alert_id: str
    anomaly: Anomaly
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    message: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Threshold:
    """A threshold configuration."""

    name: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def check(self, value: float) -> bool:
        """Check if value exceeds threshold."""
        if not self.enabled:
            return False
        if self.lower_bound is not None and value < self.lower_bound:
            return True
        if self.upper_bound is not None and value > self.upper_bound:
            return True
        return False


@dataclass
class DetectionConfig:
    """Configuration for anomaly detection."""

    method: DetectionMethod = DetectionMethod.ZSCORE
    sensitivity: float = 2.0  # Z-score threshold or IQR multiplier
    min_samples: int = 10
    window_size: int = 100
    ignore_nulls: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Result of anomaly detection."""

    data_points: int
    anomalies_found: int
    anomalies: List[Anomaly]
    statistics: Dict[str, float]
    detection_time_ms: float


# ============================================================================
# Anomaly Detector Interface
# ============================================================================


class AnomalyDetector(ABC):
    """Abstract anomaly detector."""

    @abstractmethod
    def detect(self, data: List[DataPoint]) -> List[Anomaly]:
        """Detect anomalies in data."""
        pass

    @abstractmethod
    def detect_point(self, point: DataPoint, history: List[DataPoint]) -> Optional[Anomaly]:
        """Detect if a single point is anomalous."""
        pass


class ZScoreDetector(AnomalyDetector):
    """Z-score based anomaly detector."""

    def __init__(self, threshold: float = 2.0) -> None:
        self._threshold = threshold

    def detect(self, data: List[DataPoint]) -> List[Anomaly]:
        """Detect anomalies using Z-score."""
        if len(data) < 3:
            return []

        values = [p.value for p in data]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0

        if stdev == 0:
            return []

        anomalies: List[Anomaly] = []
        for i, point in enumerate(data):
            z_score = abs(point.value - mean) / stdev
            if z_score > self._threshold:
                anomalies.append(
                    Anomaly(
                        anomaly_id=f"zscore_{i}",
                        anomaly_type=AnomalyType.POINT,
                        value=point.value,
                        expected_value=mean,
                        deviation=z_score,
                        timestamp=point.timestamp,
                        confidence=min(z_score / self._threshold, 1.0),
                        method=DetectionMethod.ZSCORE,
                    )
                )

        return anomalies

    def detect_point(self, point: DataPoint, history: List[DataPoint]) -> Optional[Anomaly]:
        """Detect if a single point is anomalous."""
        if len(history) < 3:
            return None

        values = [p.value for p in history]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0

        if stdev == 0:
            return None

        z_score = abs(point.value - mean) / stdev
        if z_score > self._threshold:
            return Anomaly(
                anomaly_id=f"zscore_point",
                anomaly_type=AnomalyType.POINT,
                value=point.value,
                expected_value=mean,
                deviation=z_score,
                timestamp=point.timestamp,
                confidence=min(z_score / self._threshold, 1.0),
                method=DetectionMethod.ZSCORE,
            )

        return None


class IQRDetector(AnomalyDetector):
    """IQR (Interquartile Range) based anomaly detector."""

    def __init__(self, multiplier: float = 1.5) -> None:
        self._multiplier = multiplier

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def detect(self, data: List[DataPoint]) -> List[Anomaly]:
        """Detect anomalies using IQR."""
        if len(data) < 4:
            return []

        values = [p.value for p in data]
        q1 = self._percentile(values, 25)
        q3 = self._percentile(values, 75)
        iqr = q3 - q1

        lower_fence = q1 - self._multiplier * iqr
        upper_fence = q3 + self._multiplier * iqr
        median = statistics.median(values)

        anomalies: List[Anomaly] = []
        for i, point in enumerate(data):
            if point.value < lower_fence or point.value > upper_fence:
                deviation = (
                    max(
                        abs(point.value - lower_fence) if point.value < lower_fence else 0,
                        abs(point.value - upper_fence) if point.value > upper_fence else 0,
                    )
                    / iqr
                    if iqr > 0
                    else 0
                )

                anomalies.append(
                    Anomaly(
                        anomaly_id=f"iqr_{i}",
                        anomaly_type=AnomalyType.POINT,
                        value=point.value,
                        expected_value=median,
                        deviation=deviation,
                        timestamp=point.timestamp,
                        confidence=min(deviation, 1.0),
                        method=DetectionMethod.IQR,
                    )
                )

        return anomalies

    def detect_point(self, point: DataPoint, history: List[DataPoint]) -> Optional[Anomaly]:
        """Detect if a single point is anomalous."""
        if len(history) < 4:
            return None

        values = [p.value for p in history]
        q1 = self._percentile(values, 25)
        q3 = self._percentile(values, 75)
        iqr = q3 - q1

        lower_fence = q1 - self._multiplier * iqr
        upper_fence = q3 + self._multiplier * iqr
        median = statistics.median(values)

        if point.value < lower_fence or point.value > upper_fence:
            deviation = (
                max(
                    abs(point.value - lower_fence) if point.value < lower_fence else 0,
                    abs(point.value - upper_fence) if point.value > upper_fence else 0,
                )
                / iqr
                if iqr > 0
                else 0
            )

            return Anomaly(
                anomaly_id="iqr_point",
                anomaly_type=AnomalyType.POINT,
                value=point.value,
                expected_value=median,
                deviation=deviation,
                timestamp=point.timestamp,
                confidence=min(deviation, 1.0),
                method=DetectionMethod.IQR,
            )

        return None


class ThresholdDetector(AnomalyDetector):
    """Threshold-based anomaly detector."""

    def __init__(self, thresholds: List[Threshold]) -> None:
        self._thresholds = thresholds

    def detect(self, data: List[DataPoint]) -> List[Anomaly]:
        """Detect anomalies using thresholds."""
        anomalies: List[Anomaly] = []

        for i, point in enumerate(data):
            for threshold in self._thresholds:
                if threshold.check(point.value):
                    expected = (
                        threshold.lower_bound
                        if threshold.lower_bound is not None and point.value < threshold.lower_bound
                        else threshold.upper_bound
                    )
                    anomalies.append(
                        Anomaly(
                            anomaly_id=f"threshold_{threshold.name}_{i}",
                            anomaly_type=AnomalyType.POINT,
                            value=point.value,
                            expected_value=expected or 0,
                            deviation=abs(point.value - (expected or 0)),
                            timestamp=point.timestamp,
                            confidence=1.0,
                            method=DetectionMethod.THRESHOLD,
                            metadata={"threshold_name": threshold.name},
                        )
                    )
                    break  # Only one threshold violation per point

        return anomalies

    def detect_point(self, point: DataPoint, history: List[DataPoint]) -> Optional[Anomaly]:
        """Detect if a single point violates thresholds."""
        for threshold in self._thresholds:
            if threshold.check(point.value):
                expected = (
                    threshold.lower_bound
                    if threshold.lower_bound is not None and point.value < threshold.lower_bound
                    else threshold.upper_bound
                )
                return Anomaly(
                    anomaly_id=f"threshold_{threshold.name}",
                    anomaly_type=AnomalyType.POINT,
                    value=point.value,
                    expected_value=expected or 0,
                    deviation=abs(point.value - (expected or 0)),
                    timestamp=point.timestamp,
                    confidence=1.0,
                    method=DetectionMethod.THRESHOLD,
                    metadata={"threshold_name": threshold.name},
                )
        return None


class MovingAverageDetector(AnomalyDetector):
    """Moving average based anomaly detector."""

    def __init__(self, window_size: int = 10, threshold: float = 2.0) -> None:
        self._window_size = window_size
        self._threshold = threshold

    def detect(self, data: List[DataPoint]) -> List[Anomaly]:
        """Detect anomalies using moving average."""
        if len(data) < self._window_size:
            return []

        anomalies: List[Anomaly] = []
        values = [p.value for p in data]

        for i in range(self._window_size, len(data)):
            window = values[i - self._window_size : i]
            ma = statistics.mean(window)
            std = statistics.stdev(window) if len(window) > 1 else 0

            if std > 0:
                deviation = abs(data[i].value - ma) / std
                if deviation > self._threshold:
                    anomalies.append(
                        Anomaly(
                            anomaly_id=f"ma_{i}",
                            anomaly_type=AnomalyType.CONTEXTUAL,
                            value=data[i].value,
                            expected_value=ma,
                            deviation=deviation,
                            timestamp=data[i].timestamp,
                            confidence=min(deviation / self._threshold, 1.0),
                            method=DetectionMethod.MOVING_AVERAGE,
                        )
                    )

        return anomalies

    def detect_point(self, point: DataPoint, history: List[DataPoint]) -> Optional[Anomaly]:
        """Detect if a single point is anomalous against moving average."""
        if len(history) < self._window_size:
            return None

        window = [p.value for p in history[-self._window_size :]]
        ma = statistics.mean(window)
        std = statistics.stdev(window) if len(window) > 1 else 0

        if std > 0:
            deviation = abs(point.value - ma) / std
            if deviation > self._threshold:
                return Anomaly(
                    anomaly_id="ma_point",
                    anomaly_type=AnomalyType.CONTEXTUAL,
                    value=point.value,
                    expected_value=ma,
                    deviation=deviation,
                    timestamp=point.timestamp,
                    confidence=min(deviation / self._threshold, 1.0),
                    method=DetectionMethod.MOVING_AVERAGE,
                )

        return None


# ============================================================================
# Alert Manager
# ============================================================================


class AlertManager:
    """Manages alerts from anomaly detection."""

    def __init__(self) -> None:
        self._alerts: Dict[str, Alert] = {}
        self._handlers: List[Callable[[Alert], None]] = []

    def create_alert(
        self,
        anomaly: Anomaly,
        severity: AlertSeverity,
        message: str = "",
    ) -> Alert:
        """Create an alert from an anomaly."""
        alert_id = f"alert_{anomaly.anomaly_id}"
        alert = Alert(
            alert_id=alert_id,
            anomaly=anomaly,
            severity=severity,
            message=message
            or f"Anomaly detected: {anomaly.value} (expected: {anomaly.expected_value})",
        )
        self._alerts[alert_id] = alert

        # Notify handlers
        for handler in self._handlers:
            handler(alert)

        return alert

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
            self._alerts[alert_id].acknowledged_at = datetime.utcnow()
            return True
        return False

    def resolve(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.RESOLVED
            self._alerts[alert_id].resolved_at = datetime.utcnow()
            return True
        return False

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts."""
        alerts = [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register an alert handler."""
        self._handlers.append(handler)


# ============================================================================
# Anomaly Detection Engine
# ============================================================================


class AnomalyDetectionEngine:
    """Main anomaly detection engine."""

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        self._config = config or DetectionConfig()
        self._alert_manager = alert_manager or AlertManager()
        self._detectors: Dict[DetectionMethod, AnomalyDetector] = {}
        self._data_buffer: List[DataPoint] = []

        # Register default detectors
        self._detectors[DetectionMethod.ZSCORE] = ZScoreDetector(self._config.sensitivity)
        self._detectors[DetectionMethod.IQR] = IQRDetector(self._config.sensitivity)
        self._detectors[DetectionMethod.MOVING_AVERAGE] = MovingAverageDetector(
            self._config.window_size, self._config.sensitivity
        )

    def register_detector(self, method: DetectionMethod, detector: AnomalyDetector) -> None:
        """Register a custom detector."""
        self._detectors[method] = detector

    def add_data_point(self, point: DataPoint) -> Optional[Anomaly]:
        """Add a data point and check for anomalies."""
        self._data_buffer.append(point)

        # Keep buffer at configured size
        if len(self._data_buffer) > self._config.window_size:
            self._data_buffer = self._data_buffer[-self._config.window_size :]

        # Check for anomalies
        detector = self._detectors.get(self._config.method)
        if detector and len(self._data_buffer) >= self._config.min_samples:
            anomaly = detector.detect_point(point, self._data_buffer[:-1])
            if anomaly:
                # Create alert
                severity = self._get_severity(anomaly.confidence)
                self._alert_manager.create_alert(anomaly, severity)
                return anomaly

        return None

    def detect(self, data: List[DataPoint]) -> DetectionResult:
        """Run full anomaly detection on data."""
        start_time = datetime.utcnow()

        detector = self._detectors.get(self._config.method)
        if not detector:
            return DetectionResult(
                data_points=len(data),
                anomalies_found=0,
                anomalies=[],
                statistics={},
                detection_time_ms=0,
            )

        anomalies = detector.detect(data)

        # Calculate statistics
        values = [p.value for p in data]
        stats = {
            "mean": statistics.mean(values) if values else 0,
            "median": statistics.median(values) if values else 0,
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
        }

        detection_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return DetectionResult(
            data_points=len(data),
            anomalies_found=len(anomalies),
            anomalies=anomalies,
            statistics=stats,
            detection_time_ms=detection_time,
        )

    def _get_severity(self, confidence: float) -> AlertSeverity:
        """Determine alert severity based on confidence."""
        if confidence >= 0.9:
            return AlertSeverity.CRITICAL
        elif confidence >= 0.7:
            return AlertSeverity.WARNING
        return AlertSeverity.INFO

    def get_alert_manager(self) -> AlertManager:
        """Get the alert manager."""
        return self._alert_manager


# ============================================================================
# Anomaly Detection Vision Provider
# ============================================================================


class AnomalyDetectionVisionProvider(VisionProvider):
    """Vision provider with anomaly detection."""

    def __init__(
        self,
        provider: VisionProvider,
        engine: AnomalyDetectionEngine,
        metric_name: str = "confidence",
    ) -> None:
        self._provider = provider
        self._engine = engine
        self._metric_name = metric_name
        self._request_count = 0

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"anomaly_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VisionDescription:
        """Analyze image with anomaly detection."""
        result = await self._provider.analyze_image(image_data, prompt, **kwargs)

        # Track confidence as a metric
        point = DataPoint(
            value=result.confidence,
            labels={"metric": self._metric_name},
        )
        anomaly = self._engine.add_data_point(point)

        if anomaly:
            # Add anomaly warning to details
            result = VisionDescription(
                summary=result.summary,
                details=result.details
                + [f"Warning: Anomalous confidence detected ({anomaly.deviation:.2f} deviation)"],
                confidence=result.confidence,
            )

        self._request_count += 1
        return result

    def get_engine(self) -> AnomalyDetectionEngine:
        """Get the anomaly detection engine."""
        return self._engine


# ============================================================================
# Factory Functions
# ============================================================================


def create_anomaly_engine(
    config: Optional[DetectionConfig] = None,
    alert_manager: Optional[AlertManager] = None,
) -> AnomalyDetectionEngine:
    """Create an anomaly detection engine."""
    return AnomalyDetectionEngine(config, alert_manager)


def create_anomaly_provider(
    provider: VisionProvider,
    engine: Optional[AnomalyDetectionEngine] = None,
    metric_name: str = "confidence",
) -> AnomalyDetectionVisionProvider:
    """Create an anomaly detection vision provider."""
    return AnomalyDetectionVisionProvider(
        provider=provider,
        engine=engine or create_anomaly_engine(),
        metric_name=metric_name,
    )


def create_zscore_detector(threshold: float = 2.0) -> ZScoreDetector:
    """Create a Z-score detector."""
    return ZScoreDetector(threshold)


def create_iqr_detector(multiplier: float = 1.5) -> IQRDetector:
    """Create an IQR detector."""
    return IQRDetector(multiplier)


def create_threshold_detector(thresholds: List[Threshold]) -> ThresholdDetector:
    """Create a threshold detector."""
    return ThresholdDetector(thresholds)


def create_moving_average_detector(
    window_size: int = 10,
    threshold: float = 2.0,
) -> MovingAverageDetector:
    """Create a moving average detector."""
    return MovingAverageDetector(window_size, threshold)


def create_alert_manager() -> AlertManager:
    """Create an alert manager."""
    return AlertManager()


def create_threshold(
    name: str,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    severity: AlertSeverity = AlertSeverity.WARNING,
) -> Threshold:
    """Create a threshold."""
    return Threshold(
        name=name,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        severity=severity,
    )
