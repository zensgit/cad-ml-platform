"""
Confidence calibration for HybridClassifier.

Provides calibration methods to ensure predicted confidence scores
reflect true probabilities.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CalibrationMethod(str, Enum):
    """Calibration method types."""
    PLATT_SCALING = "platt_scaling"
    ISOTONIC = "isotonic"
    TEMPERATURE_SCALING = "temperature_scaling"
    HISTOGRAM_BINNING = "histogram_binning"
    BETA_CALIBRATION = "beta_calibration"


@dataclass
class CalibrationMetrics:
    """Calibration evaluation metrics."""
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float  # MCE
    brier_score: float
    reliability_diagram: Dict[str, List[float]]
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ece": self.expected_calibration_error,
            "mce": self.maximum_calibration_error,
            "brier_score": self.brier_score,
            "reliability_diagram": self.reliability_diagram,
            "n_samples": self.n_samples,
        }


class Calibrator(ABC):
    """Abstract base class for confidence calibrators."""

    @abstractmethod
    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> None:
        """Fit calibrator on validation data."""
        pass

    @abstractmethod
    def calibrate(self, confidence: float) -> float:
        """Calibrate a single confidence score."""
        pass

    def calibrate_batch(self, confidences: np.ndarray) -> np.ndarray:
        """Calibrate batch of confidence scores."""
        return np.array([self.calibrate(c) for c in confidences])


class PlattScaling(Calibrator):
    """
    Platt scaling (sigmoid calibration).

    Fits a logistic regression model to map raw scores to probabilities.
    """

    def __init__(self):
        self._a: float = 0.0
        self._b: float = 0.0
        self._fitted: bool = False

    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> None:
        """Fit Platt scaling parameters using MLE."""
        # Transform labels to {0, 1}
        y = labels.astype(float)

        # Use scipy for optimization if available
        try:
            from scipy.optimize import minimize

            def neg_log_likelihood(params):
                a, b = params
                p = 1.0 / (1.0 + np.exp(-(a * confidences + b)))
                p = np.clip(p, 1e-10, 1 - 1e-10)
                return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

            result = minimize(neg_log_likelihood, [1.0, 0.0], method='BFGS')
            self._a, self._b = result.x
        except ImportError:
            # Fallback: simple gradient descent
            self._a, self._b = self._gradient_descent(confidences, y)

        self._fitted = True
        logger.info(f"Platt scaling fitted: a={self._a:.4f}, b={self._b:.4f}")

    def _gradient_descent(
        self,
        confidences: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        n_iters: int = 1000,
    ) -> Tuple[float, float]:
        """Simple gradient descent for Platt scaling."""
        a, b = 1.0, 0.0

        for _ in range(n_iters):
            p = 1.0 / (1.0 + np.exp(-(a * confidences + b)))
            p = np.clip(p, 1e-10, 1 - 1e-10)

            error = p - labels
            grad_a = np.mean(error * confidences)
            grad_b = np.mean(error)

            a -= lr * grad_a
            b -= lr * grad_b

        return a, b

    def calibrate(self, confidence: float) -> float:
        """Apply Platt scaling."""
        if not self._fitted:
            return confidence
        return float(1.0 / (1.0 + np.exp(-(self._a * confidence + self._b))))


class IsotonicCalibration(Calibrator):
    """
    Isotonic regression calibration.

    Non-parametric calibration using isotonic regression.
    """

    def __init__(self):
        self._x_cal: Optional[np.ndarray] = None
        self._y_cal: Optional[np.ndarray] = None
        self._fitted: bool = False

    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic regression."""
        try:
            from sklearn.isotonic import IsotonicRegression

            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(confidences, labels)
            self._ir = ir
            self._fitted = True
        except ImportError:
            # Fallback: simple binning
            self._fit_binning(confidences, labels)

        logger.info("Isotonic calibration fitted")

    def _fit_binning(self, confidences: np.ndarray, labels: np.ndarray) -> None:
        """Fallback binning-based calibration."""
        # Sort by confidence
        sorted_idx = np.argsort(confidences)
        self._x_cal = confidences[sorted_idx]
        self._y_cal = labels[sorted_idx].astype(float)

        # Apply PAVA (pool adjacent violators algorithm)
        self._y_cal = self._pava(self._y_cal)
        self._fitted = True

    def _pava(self, y: np.ndarray) -> np.ndarray:
        """Pool adjacent violators algorithm."""
        n = len(y)
        y = y.copy()

        while True:
            violation = False
            i = 0
            while i < n - 1:
                if y[i] > y[i + 1]:
                    # Pool
                    avg = (y[i] + y[i + 1]) / 2
                    y[i] = y[i + 1] = avg
                    violation = True
                i += 1
            if not violation:
                break

        return y

    def calibrate(self, confidence: float) -> float:
        """Apply isotonic calibration."""
        if not self._fitted:
            return confidence

        if hasattr(self, '_ir'):
            return float(self._ir.predict([[confidence]])[0])

        # Fallback interpolation
        if self._x_cal is None or self._y_cal is None:
            return confidence

        idx = np.searchsorted(self._x_cal, confidence)
        if idx == 0:
            return float(self._y_cal[0])
        if idx >= len(self._y_cal):
            return float(self._y_cal[-1])

        # Linear interpolation
        x0, x1 = self._x_cal[idx - 1], self._x_cal[idx]
        y0, y1 = self._y_cal[idx - 1], self._y_cal[idx]
        if x1 == x0:
            return float(y0)
        return float(y0 + (y1 - y0) * (confidence - x0) / (x1 - x0))


class TemperatureScaling(Calibrator):
    """
    Temperature scaling calibration.

    Simple but effective for neural network outputs.
    """

    def __init__(self, initial_temperature: float = 1.0):
        self._temperature: float = initial_temperature
        self._fitted: bool = False

    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> None:
        """Fit temperature parameter."""
        try:
            from scipy.optimize import minimize_scalar

            def nll(T):
                scaled = confidences ** (1 / T)
                # Clip for numerical stability
                scaled = np.clip(scaled, 1e-10, 1 - 1e-10)
                return -np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled))

            result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
            self._temperature = result.x
        except ImportError:
            # Grid search fallback
            best_t, best_nll = 1.0, float('inf')
            for t in np.linspace(0.1, 10.0, 100):
                scaled = confidences ** (1 / t)
                scaled = np.clip(scaled, 1e-10, 1 - 1e-10)
                nll = -np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled))
                if nll < best_nll:
                    best_t, best_nll = t, nll
            self._temperature = best_t

        self._fitted = True
        logger.info(f"Temperature scaling fitted: T={self._temperature:.4f}")

    def calibrate(self, confidence: float) -> float:
        """Apply temperature scaling."""
        if not self._fitted:
            return confidence
        return float(confidence ** (1 / self._temperature))

    @property
    def temperature(self) -> float:
        return self._temperature


class HistogramBinning(Calibrator):
    """
    Histogram binning calibration.

    Non-parametric calibration using equal-width bins.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._bin_boundaries: Optional[np.ndarray] = None
        self._bin_values: Optional[np.ndarray] = None
        self._fitted: bool = False

    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> None:
        """Fit histogram binning."""
        self._bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        self._bin_values = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            low, high = self._bin_boundaries[i], self._bin_boundaries[i + 1]
            mask = (confidences >= low) & (confidences < high)
            if mask.sum() > 0:
                self._bin_values[i] = labels[mask].mean()
            else:
                # Use bin center as default
                self._bin_values[i] = (low + high) / 2

        self._fitted = True
        logger.info(f"Histogram binning fitted with {self.n_bins} bins")

    def calibrate(self, confidence: float) -> float:
        """Apply histogram binning."""
        if not self._fitted or self._bin_boundaries is None:
            return confidence

        # Find bin
        bin_idx = np.searchsorted(self._bin_boundaries[1:-1], confidence)
        return float(self._bin_values[bin_idx])


class BetaCalibration(Calibrator):
    """
    Beta calibration.

    Uses beta distribution for calibration, handles scores near 0 and 1 better.
    """

    def __init__(self):
        self._a: float = 1.0
        self._b: float = 1.0
        self._c: float = 0.0
        self._fitted: bool = False

    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> None:
        """Fit beta calibration parameters."""
        # Clip to avoid log(0)
        conf = np.clip(confidences, 1e-10, 1 - 1e-10)

        try:
            from scipy.optimize import minimize

            def neg_log_likelihood(params):
                a, b, c = params
                # Beta calibration: c + (1-c) * sigmoid(a * log(conf/(1-conf)) + b)
                logit = np.log(conf / (1 - conf))
                p = c + (1 - c) / (1 + np.exp(-(a * logit + b)))
                p = np.clip(p, 1e-10, 1 - 1e-10)
                return -np.sum(labels * np.log(p) + (1 - labels) * np.log(1 - p))

            result = minimize(neg_log_likelihood, [1.0, 0.0, 0.0], method='BFGS')
            self._a, self._b, self._c = result.x
        except ImportError:
            # Fallback to Platt scaling
            platt = PlattScaling()
            platt.fit(confidences, labels)
            self._a, self._b = platt._a, platt._b
            self._c = 0.0

        self._fitted = True
        logger.info(f"Beta calibration fitted: a={self._a:.4f}, b={self._b:.4f}, c={self._c:.4f}")

    def calibrate(self, confidence: float) -> float:
        """Apply beta calibration."""
        if not self._fitted:
            return confidence

        conf = np.clip(confidence, 1e-10, 1 - 1e-10)
        logit = np.log(conf / (1 - conf))
        return float(self._c + (1 - self._c) / (1 + np.exp(-(self._a * logit + self._b))))


class ConfidenceCalibrator:
    """
    Main confidence calibration interface.

    Supports multiple calibration methods and per-source calibration.
    """

    METHODS = {
        CalibrationMethod.PLATT_SCALING: PlattScaling,
        CalibrationMethod.ISOTONIC: IsotonicCalibration,
        CalibrationMethod.TEMPERATURE_SCALING: TemperatureScaling,
        CalibrationMethod.HISTOGRAM_BINNING: HistogramBinning,
        CalibrationMethod.BETA_CALIBRATION: BetaCalibration,
    }

    def __init__(
        self,
        method: CalibrationMethod = CalibrationMethod.TEMPERATURE_SCALING,
        per_source: bool = True,
    ):
        self.method = method
        self.per_source = per_source
        self._global_calibrator: Optional[Calibrator] = None
        self._source_calibrators: Dict[str, Calibrator] = {}

    def fit(
        self,
        confidences: np.ndarray,
        labels: np.ndarray,
        sources: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit calibrator(s).

        Args:
            confidences: Raw confidence scores
            labels: True binary labels (1 if prediction was correct)
            sources: Optional source names for per-source calibration
        """
        calibrator_class = self.METHODS[self.method]

        if self.per_source and sources is not None:
            # Fit separate calibrator per source
            unique_sources = np.unique(sources)
            for source in unique_sources:
                mask = sources == source
                if mask.sum() > 10:  # Minimum samples
                    calibrator = calibrator_class()
                    calibrator.fit(confidences[mask], labels[mask])
                    self._source_calibrators[source] = calibrator

        # Also fit global calibrator
        self._global_calibrator = calibrator_class()
        self._global_calibrator.fit(confidences, labels)

        logger.info(f"Calibrator fitted: {len(self._source_calibrators)} source calibrators")

    def calibrate(
        self,
        confidence: float,
        source: Optional[str] = None,
    ) -> float:
        """
        Calibrate a confidence score.

        Args:
            confidence: Raw confidence score
            source: Optional source name

        Returns:
            Calibrated confidence
        """
        if self.per_source and source and source in self._source_calibrators:
            return self._source_calibrators[source].calibrate(confidence)

        if self._global_calibrator:
            return self._global_calibrator.calibrate(confidence)

        return confidence

    def evaluate(
        self,
        confidences: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """
        Evaluate calibration quality.

        Args:
            confidences: Confidence scores
            labels: True labels

        Returns:
            CalibrationMetrics
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences >= low) & (confidences < high)
            if mask.sum() > 0:
                bin_acc = labels[mask].mean()
                bin_conf = confidences[mask].mean()
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append((low + high) / 2)
                bin_counts.append(0)

        # ECE: Expected Calibration Error
        total = sum(bin_counts)
        ece = sum(
            count * abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
        ) / total if total > 0 else 0.0

        # MCE: Maximum Calibration Error
        mce = max(
            abs(acc - conf)
            for acc, conf in zip(bin_accuracies, bin_confidences)
        ) if bin_accuracies else 0.0

        # Brier Score
        brier = np.mean((confidences - labels) ** 2)

        return CalibrationMetrics(
            expected_calibration_error=float(ece),
            maximum_calibration_error=float(mce),
            brier_score=float(brier),
            reliability_diagram={
                "bin_accuracies": bin_accuracies,
                "bin_confidences": bin_confidences,
                "bin_counts": bin_counts,
            },
            n_samples=len(confidences),
        )
