"""
Drift detection for model monitoring.

Detects distribution shifts in input data and predictions.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Types of drift."""
    DATA_DRIFT = "data_drift"  # Input distribution change
    CONCEPT_DRIFT = "concept_drift"  # Relationship change
    PREDICTION_DRIFT = "prediction_drift"  # Output distribution change
    PERFORMANCE_DRIFT = "performance_drift"  # Accuracy degradation


class DriftSeverity(str, Enum):
    """Severity levels for drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of drift detection."""
    drift_type: DriftType
    detected: bool
    severity: DriftSeverity
    score: float  # 0-1, higher = more drift
    p_value: Optional[float] = None
    threshold: float = 0.05
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_type": self.drift_type.value,
            "detected": self.detected,
            "severity": self.severity.value,
            "score": self.score,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class DriftDetector(ABC):
    """Abstract base class for drift detectors."""

    @abstractmethod
    def fit(self, reference_data: np.ndarray) -> None:
        """Fit detector on reference data."""
        pass

    @abstractmethod
    def detect(self, current_data: np.ndarray) -> DriftResult:
        """Detect drift in current data."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass


class KSTestDetector(DriftDetector):
    """
    Kolmogorov-Smirnov test for drift detection.

    Compares distributions using the KS statistic.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        drift_type: DriftType = DriftType.DATA_DRIFT,
    ):
        self.threshold = threshold
        self.drift_type = drift_type
        self._reference_data: Optional[np.ndarray] = None

    def fit(self, reference_data: np.ndarray) -> None:
        """Store reference data."""
        self._reference_data = np.array(reference_data).flatten()
        logger.debug(f"KS detector fitted with {len(self._reference_data)} samples")

    def detect(self, current_data: np.ndarray) -> DriftResult:
        """Detect drift using KS test."""
        if self._reference_data is None:
            raise ValueError("Detector not fitted")

        current = np.array(current_data).flatten()

        try:
            from scipy import stats
            statistic, p_value = stats.ks_2samp(self._reference_data, current)
        except ImportError:
            # Fallback: simple implementation
            statistic, p_value = self._ks_2samp_simple(self._reference_data, current)

        detected = p_value < self.threshold
        severity = self._score_to_severity(statistic)

        return DriftResult(
            drift_type=self.drift_type,
            detected=detected,
            severity=severity,
            score=float(statistic),
            p_value=float(p_value),
            threshold=self.threshold,
            details={
                "reference_size": len(self._reference_data),
                "current_size": len(current),
                "reference_mean": float(np.mean(self._reference_data)),
                "current_mean": float(np.mean(current)),
            },
        )

    def _ks_2samp_simple(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """Simple KS test implementation."""
        n1, n2 = len(a), len(b)
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)

        all_values = np.concatenate([a_sorted, b_sorted])
        all_values = np.unique(all_values)

        cdf_a = np.searchsorted(a_sorted, all_values, side='right') / n1
        cdf_b = np.searchsorted(b_sorted, all_values, side='right') / n2

        d = np.max(np.abs(cdf_a - cdf_b))

        # Approximate p-value
        n = n1 * n2 / (n1 + n2)
        p_value = 2 * np.exp(-2 * n * d ** 2)
        p_value = min(1.0, max(0.0, p_value))

        return float(d), float(p_value)

    def _score_to_severity(self, score: float) -> DriftSeverity:
        """Convert drift score to severity."""
        if score < 0.1:
            return DriftSeverity.NONE
        elif score < 0.2:
            return DriftSeverity.LOW
        elif score < 0.4:
            return DriftSeverity.MEDIUM
        elif score < 0.6:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def reset(self) -> None:
        """Reset detector."""
        self._reference_data = None


class PSIDetector(DriftDetector):
    """
    Population Stability Index (PSI) detector.

    Measures distribution shift using binned comparison.
    """

    def __init__(
        self,
        n_bins: int = 10,
        threshold: float = 0.2,
        drift_type: DriftType = DriftType.DATA_DRIFT,
    ):
        self.n_bins = n_bins
        self.threshold = threshold
        self.drift_type = drift_type
        self._bin_edges: Optional[np.ndarray] = None
        self._reference_proportions: Optional[np.ndarray] = None

    def fit(self, reference_data: np.ndarray) -> None:
        """Compute reference bins."""
        data = np.array(reference_data).flatten()

        # Create bins using quantiles
        self._bin_edges = np.percentile(
            data,
            np.linspace(0, 100, self.n_bins + 1)
        )

        # Compute reference proportions
        hist, _ = np.histogram(data, bins=self._bin_edges)
        self._reference_proportions = hist / len(data)

        # Avoid zeros
        self._reference_proportions = np.clip(self._reference_proportions, 1e-10, 1.0)

        logger.debug(f"PSI detector fitted with {len(data)} samples, {self.n_bins} bins")

    def detect(self, current_data: np.ndarray) -> DriftResult:
        """Detect drift using PSI."""
        if self._bin_edges is None or self._reference_proportions is None:
            raise ValueError("Detector not fitted")

        current = np.array(current_data).flatten()

        # Compute current proportions
        hist, _ = np.histogram(current, bins=self._bin_edges)
        current_proportions = hist / len(current)
        current_proportions = np.clip(current_proportions, 1e-10, 1.0)

        # Compute PSI
        psi = np.sum(
            (current_proportions - self._reference_proportions) *
            np.log(current_proportions / self._reference_proportions)
        )

        detected = psi > self.threshold
        severity = self._psi_to_severity(psi)

        return DriftResult(
            drift_type=self.drift_type,
            detected=detected,
            severity=severity,
            score=float(psi),
            p_value=None,
            threshold=self.threshold,
            details={
                "n_bins": self.n_bins,
                "reference_proportions": self._reference_proportions.tolist(),
                "current_proportions": current_proportions.tolist(),
            },
        )

    def _psi_to_severity(self, psi: float) -> DriftSeverity:
        """Convert PSI to severity."""
        if psi < 0.1:
            return DriftSeverity.NONE
        elif psi < 0.2:
            return DriftSeverity.LOW
        elif psi < 0.25:
            return DriftSeverity.MEDIUM
        elif psi < 0.5:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def reset(self) -> None:
        """Reset detector."""
        self._bin_edges = None
        self._reference_proportions = None


class PageHinkleyDetector(DriftDetector):
    """
    Page-Hinkley test for online drift detection.

    Detects changes in the mean of a sequence.
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        drift_type: DriftType = DriftType.PERFORMANCE_DRIFT,
    ):
        self.delta = delta
        self.threshold = threshold
        self.drift_type = drift_type

        self._sum: float = 0.0
        self._min_sum: float = 0.0
        self._count: int = 0
        self._mean: float = 0.0

    def fit(self, reference_data: np.ndarray) -> None:
        """Initialize with reference mean."""
        data = np.array(reference_data).flatten()
        self._mean = float(np.mean(data))
        self.reset()
        logger.debug(f"Page-Hinkley detector fitted with mean={self._mean:.4f}")

    def detect(self, current_data: np.ndarray) -> DriftResult:
        """Update and detect drift."""
        for value in np.array(current_data).flatten():
            self._update(value)

        ph_value = self._sum - self._min_sum
        detected = ph_value > self.threshold
        score = ph_value / self.threshold if self.threshold > 0 else 0
        severity = self._score_to_severity(score)

        return DriftResult(
            drift_type=self.drift_type,
            detected=detected,
            severity=severity,
            score=min(1.0, score),
            p_value=None,
            threshold=self.threshold,
            details={
                "ph_value": ph_value,
                "sum": self._sum,
                "min_sum": self._min_sum,
                "count": self._count,
            },
        )

    def _update(self, value: float) -> None:
        """Update with single value."""
        self._count += 1
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)

    def _score_to_severity(self, score: float) -> DriftSeverity:
        """Convert score to severity."""
        if score < 0.5:
            return DriftSeverity.NONE
        elif score < 0.75:
            return DriftSeverity.LOW
        elif score < 1.0:
            return DriftSeverity.MEDIUM
        elif score < 2.0:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def reset(self) -> None:
        """Reset detector state."""
        self._sum = 0.0
        self._min_sum = 0.0
        self._count = 0


class DriftMonitor:
    """
    Comprehensive drift monitoring.

    Combines multiple detectors for robust drift detection.
    """

    def __init__(
        self,
        window_size: int = 1000,
        reference_window: int = 5000,
        check_interval: int = 100,
    ):
        self.window_size = window_size
        self.reference_window = reference_window
        self.check_interval = check_interval

        # Data buffers
        self._reference_data: Deque[np.ndarray] = deque(maxlen=reference_window)
        self._current_data: Deque[np.ndarray] = deque(maxlen=window_size)
        self._predictions: Deque[str] = deque(maxlen=window_size)
        self._confidences: Deque[float] = deque(maxlen=window_size)

        # Detectors
        self._data_detector: Optional[DriftDetector] = None
        self._prediction_detector: Optional[DriftDetector] = None
        self._confidence_detector: Optional[DriftDetector] = None

        self._check_count = 0
        self._last_results: List[DriftResult] = []

    def set_reference(self, data: np.ndarray) -> None:
        """Set reference data distribution."""
        self._reference_data.clear()
        for sample in data:
            self._reference_data.append(np.array(sample))

        # Fit detectors
        ref_array = np.vstack(list(self._reference_data))

        self._data_detector = PSIDetector(drift_type=DriftType.DATA_DRIFT)
        self._data_detector.fit(ref_array.flatten())

        self._confidence_detector = PageHinkleyDetector(drift_type=DriftType.PREDICTION_DRIFT)

        logger.info(f"Drift monitor initialized with {len(self._reference_data)} reference samples")

    def add_sample(
        self,
        features: np.ndarray,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Optional[List[DriftResult]]:
        """
        Add a new sample and check for drift.

        Returns drift results if check is performed, None otherwise.
        """
        self._current_data.append(np.array(features))

        if prediction:
            self._predictions.append(prediction)

        if confidence is not None:
            self._confidences.append(confidence)

        self._check_count += 1

        # Periodic drift check
        if self._check_count >= self.check_interval:
            self._check_count = 0
            return self.check_drift()

        return None

    def check_drift(self) -> List[DriftResult]:
        """Perform drift detection."""
        results = []

        # Data drift
        if self._data_detector and len(self._current_data) >= self.window_size // 2:
            current_array = np.vstack(list(self._current_data))
            result = self._data_detector.detect(current_array.flatten())
            results.append(result)

        # Prediction drift (using confidence as proxy)
        if self._confidence_detector and self._confidences:
            conf_array = np.array(list(self._confidences))
            result = self._confidence_detector.detect(conf_array)
            results.append(result)

        self._last_results = results
        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "reference_size": len(self._reference_data),
            "current_size": len(self._current_data),
            "predictions_tracked": len(self._predictions),
            "last_results": [r.to_dict() for r in self._last_results],
            "drift_detected": any(r.detected for r in self._last_results),
        }

    def reset(self) -> None:
        """Reset monitor state."""
        self._current_data.clear()
        self._predictions.clear()
        self._confidences.clear()
        self._check_count = 0
        self._last_results = []

        if self._data_detector:
            self._data_detector.reset()
        if self._confidence_detector:
            self._confidence_detector.reset()
