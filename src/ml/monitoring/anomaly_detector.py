"""
Metrics anomaly detection using Isolation Forest.

Trains per-metric anomaly models and classifies incoming values
as normal or anomalous with a severity grade.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful sklearn import -- fall back to a no-op stub so the module can be
# imported even when scikit-learn is not installed.
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn is not installed; anomaly detection will be disabled. "
        "Install scikit-learn to enable IsolationForest-based detection."
    )

# ---------------------------------------------------------------------------
# YAML loader -- optional dependency
# ---------------------------------------------------------------------------
try:
    import yaml  # type: ignore[import-untyped]

    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Persistence -- joblib preferred, pickle as fallback
# ---------------------------------------------------------------------------
try:
    import joblib  # type: ignore[import-untyped]

    _JOBLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JOBLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default configuration (used when the YAML file is absent)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: Dict[str, Any] = {
    "contamination": 0.05,
    "n_estimators": 100,
    "random_state": 42,
    "severity_thresholds": {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "critical": 0.9,
    },
    "rate_limit_window_seconds": 3600,
    "monitored_metrics": [
        "classification_accuracy",
        "cache_hit_rate",
        "p95_latency_seconds",
        "rejection_rate",
        "drift_score",
    ],
}


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------
@dataclass
class AnomalyResult:
    """Result produced by the anomaly detector for a single metric."""

    is_anomaly: bool
    anomaly_score: float  # 0-1, higher = more anomalous
    severity: str  # NONE / LOW / MEDIUM / HIGH / CRITICAL
    metric_name: str
    current_value: float
    threshold: float  # decision boundary used
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "severity": self.severity,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "details": self.details,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Internal model wrapper (scaler + isolation forest)
# ---------------------------------------------------------------------------
@dataclass
class _MetricModel:
    """Holds the trained scaler and isolation-forest for one metric."""

    scaler: Any  # StandardScaler
    forest: Any  # IsolationForest
    trained_at: float = field(default_factory=time.time)
    sample_count: int = 0


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------
class MetricsAnomalyDetector:
    """IsolationForest-based per-metric anomaly detector.

    Usage::

        detector = MetricsAnomalyDetector()
        detector.fit("latency", historical_latency_array)
        result = detector.detect("latency", 42.0)
    """

    def __init__(self, config_path: str = "config/anomaly_detection.yaml") -> None:
        self._config = self._load_config(config_path)
        self._models: Dict[str, _MetricModel] = {}
        logger.info(
            "MetricsAnomalyDetector initialised (sklearn=%s, config=%s)",
            _SKLEARN_AVAILABLE,
            config_path,
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML, falling back to defaults."""
        path = Path(config_path)
        if path.is_file() and _YAML_AVAILABLE:
            try:
                with open(path, "r") as fh:
                    loaded = yaml.safe_load(fh) or {}
                merged = {**_DEFAULT_CONFIG, **loaded}
                logger.debug("Loaded anomaly-detection config from %s", path)
                return merged
            except Exception:
                logger.warning(
                    "Failed to load config from %s; using defaults", path, exc_info=True
                )
        return dict(_DEFAULT_CONFIG)

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, metric_name: str, historical_data: np.ndarray) -> None:
        """Train an Isolation Forest model for *metric_name*.

        Parameters
        ----------
        metric_name:
            Identifier for the metric (e.g. ``"p95_latency_seconds"``).
        historical_data:
            1-D array of historical observations.
        """
        if not _SKLEARN_AVAILABLE:
            logger.warning(
                "sklearn unavailable -- skipping fit for metric %s", metric_name
            )
            return

        historical_data = np.asarray(historical_data, dtype=np.float64).ravel()
        if historical_data.size < 2:
            logger.warning(
                "Insufficient data to train model for %s (got %d samples)",
                metric_name,
                historical_data.size,
            )
            return

        scaler = StandardScaler()
        X = scaler.fit_transform(historical_data.reshape(-1, 1))

        forest = IsolationForest(
            contamination=self._config["contamination"],
            n_estimators=self._config["n_estimators"],
            random_state=self._config["random_state"],
        )
        forest.fit(X)

        self._models[metric_name] = _MetricModel(
            scaler=scaler,
            forest=forest,
            trained_at=time.time(),
            sample_count=int(historical_data.size),
        )
        logger.info(
            "Trained anomaly model for %s (%d samples)", metric_name, historical_data.size
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, metric_name: str, current_value: float) -> AnomalyResult:
        """Classify *current_value* as normal or anomalous for *metric_name*.

        If no model has been trained for the metric (or sklearn is absent),
        a safe ``is_anomaly=False`` result is returned.
        """
        if not _SKLEARN_AVAILABLE:
            return self._safe_result(
                metric_name, current_value, reason="sklearn_unavailable"
            )

        model = self._models.get(metric_name)
        if model is None:
            return self._safe_result(
                metric_name, current_value, reason="no_model_trained"
            )

        X = model.scaler.transform(np.array([[current_value]]))
        prediction = int(model.forest.predict(X)[0])  # 1 = normal, -1 = anomaly
        raw_score = float(model.forest.decision_function(X)[0])

        # Convert raw_score to a 0-1 anomaly score.  IsolationForest
        # decision_function returns positive for normal, negative for anomalous.
        # We map it so that higher = more anomalous:
        #   anomaly_score = 1 / (1 + exp(raw_score))   (logistic)
        anomaly_score = float(1.0 / (1.0 + np.exp(np.clip(raw_score, -500, 500))))
        is_anomaly = bool(prediction == -1)
        severity = self._score_to_severity(anomaly_score)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 6),
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold=float(-model.forest.offset_),
            details={
                "raw_score": float(raw_score),
                "prediction": int(prediction),
            },
        )

    def detect_batch(
        self, metrics: Dict[str, float]
    ) -> Dict[str, AnomalyResult]:
        """Run detection for multiple metrics at once."""
        return {name: self.detect(name, value) for name, value in metrics.items()}

    # ------------------------------------------------------------------
    # Severity mapping
    # ------------------------------------------------------------------
    def _score_to_severity(self, score: float) -> str:
        """Map a 0-1 anomaly score to a severity label."""
        thresholds = self._config.get("severity_thresholds", {})
        if score >= thresholds.get("critical", 0.9):
            return "CRITICAL"
        if score >= thresholds.get("high", 0.7):
            return "HIGH"
        if score >= thresholds.get("medium", 0.5):
            return "MEDIUM"
        if score >= thresholds.get("low", 0.3):
            return "LOW"
        return "NONE"

    # ------------------------------------------------------------------
    # Status / persistence
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Return a summary of the detector state."""
        return {
            "sklearn_available": _SKLEARN_AVAILABLE,
            "models_trained": len(self._models),
            "metrics_tracked": sorted(self._models.keys()),
            "config": {
                "contamination": self._config["contamination"],
                "n_estimators": self._config["n_estimators"],
            },
        }

    def save_models(self, path: str) -> None:
        """Persist all trained models to *path*."""
        if not self._models:
            logger.warning("No models to save")
            return

        payload = {
            name: {
                "scaler": m.scaler,
                "forest": m.forest,
                "trained_at": m.trained_at,
                "sample_count": m.sample_count,
            }
            for name, m in self._models.items()
        }

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if _JOBLIB_AVAILABLE:
            joblib.dump(payload, dest)
        else:
            import pickle

            with open(dest, "wb") as fh:
                pickle.dump(payload, fh)

        logger.info("Saved %d model(s) to %s", len(payload), dest)

    def load_models(self, path: str) -> None:
        """Load previously saved models from *path*."""
        src = Path(path)
        if not src.is_file():
            logger.warning("Model file not found: %s", src)
            return

        if _JOBLIB_AVAILABLE:
            payload = joblib.load(src)
        else:
            import pickle

            with open(src, "rb") as fh:
                payload = pickle.load(fh)  # noqa: S301

        for name, data in payload.items():
            self._models[name] = _MetricModel(
                scaler=data["scaler"],
                forest=data["forest"],
                trained_at=data.get("trained_at", 0.0),
                sample_count=data.get("sample_count", 0),
            )

        logger.info("Loaded %d model(s) from %s", len(payload), src)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_result(
        metric_name: str,
        current_value: float,
        reason: str = "unknown",
    ) -> AnomalyResult:
        """Return a conservative non-anomalous result."""
        return AnomalyResult(
            is_anomaly=False,
            anomaly_score=0.0,
            severity="NONE",
            metric_name=metric_name,
            current_value=current_value,
            threshold=0.0,
            details={"reason": reason},
        )
