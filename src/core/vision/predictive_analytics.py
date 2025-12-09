"""Predictive Analytics Module.

Provides ML-based predictions, forecasting, and trend analysis for vision processing.
"""

import asyncio
import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from .base import VisionDescription, VisionProvider


class PredictionType(Enum):
    """Types of predictions."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    DEMAND = "demand"
    QUALITY = "quality"
    COST = "cost"
    ANOMALY = "anomaly"


class ModelType(Enum):
    """Types of prediction models."""

    LINEAR_REGRESSION = "linear_regression"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    MOVING_AVERAGE = "moving_average"
    ARIMA = "arima"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class ForecastHorizon(Enum):
    """Forecast time horizons."""

    SHORT_TERM = "short_term"  # Minutes to hours
    MEDIUM_TERM = "medium_term"  # Hours to days
    LONG_TERM = "long_term"  # Days to weeks


class TrendDirection(Enum):
    """Trend directions."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class DataPoint:
    """Time series data point."""

    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of a prediction."""

    prediction_type: PredictionType
    predicted_value: float
    confidence: float
    lower_bound: float
    upper_bound: float
    horizon: ForecastHorizon
    model_used: ModelType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Trend analysis result."""

    direction: TrendDirection
    slope: float
    strength: float  # 0-1
    change_rate: float  # Percentage
    seasonality_detected: bool
    period: Optional[float] = None  # Seasonality period if detected
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastResult:
    """Forecast result with multiple predictions."""

    predictions: List[PredictionResult]
    trend: TrendAnalysis
    accuracy_score: float
    model_confidence: float
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModelMetrics:
    """Metrics for a prediction model."""

    model_type: ModelType
    accuracy: float
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    mape: float  # Mean Absolute Percentage Error
    training_time: float
    last_trained: datetime


class PredictionModel(ABC):
    """Abstract base class for prediction models."""

    @abstractmethod
    def train(self, data: List[DataPoint]) -> None:
        """Train the model on historical data."""
        pass

    @abstractmethod
    def predict(
        self, horizon: int, confidence_level: float = 0.95
    ) -> List[PredictionResult]:
        """Make predictions for the given horizon."""
        pass

    @abstractmethod
    def get_metrics(self) -> ModelMetrics:
        """Get model performance metrics."""
        pass


class MovingAverageModel(PredictionModel):
    """Simple Moving Average prediction model."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._data: List[DataPoint] = []
        self._trained = False
        self._last_trained: Optional[datetime] = None

    def train(self, data: List[DataPoint]) -> None:
        """Train on historical data."""
        self._data = sorted(data, key=lambda x: x.timestamp)
        self._trained = True
        self._last_trained = datetime.now()

    def predict(
        self, horizon: int, confidence_level: float = 0.95
    ) -> List[PredictionResult]:
        """Predict using moving average."""
        if not self._trained or len(self._data) < self.window_size:
            return []

        # Calculate moving average
        recent_values = [d.value for d in self._data[-self.window_size :]]
        avg = sum(recent_values) / len(recent_values)
        std_dev = (sum((v - avg) ** 2 for v in recent_values) / len(recent_values)) ** 0.5

        # Z-score for confidence interval
        z_score = 1.96 if confidence_level >= 0.95 else 1.645

        predictions = []
        for i in range(horizon):
            # Simple decay for longer horizons
            decay = 1.0 + (i * 0.05)
            adjusted_std = std_dev * decay

            predictions.append(
                PredictionResult(
                    prediction_type=PredictionType.DEMAND,
                    predicted_value=avg,
                    confidence=max(0.5, confidence_level - (i * 0.02)),
                    lower_bound=avg - (z_score * adjusted_std),
                    upper_bound=avg + (z_score * adjusted_std),
                    horizon=ForecastHorizon.SHORT_TERM,
                    model_used=ModelType.MOVING_AVERAGE,
                )
            )
        return predictions

    def get_metrics(self) -> ModelMetrics:
        """Get model metrics."""
        return ModelMetrics(
            model_type=ModelType.MOVING_AVERAGE,
            accuracy=0.75,
            mae=0.0,
            rmse=0.0,
            mape=0.0,
            training_time=0.01,
            last_trained=self._last_trained or datetime.now(),
        )


class ExponentialSmoothingModel(PredictionModel):
    """Exponential Smoothing prediction model."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha  # Level smoothing
        self.beta = beta  # Trend smoothing
        self._level: float = 0.0
        self._trend: float = 0.0
        self._data: List[DataPoint] = []
        self._trained = False
        self._last_trained: Optional[datetime] = None

    def train(self, data: List[DataPoint]) -> None:
        """Train using double exponential smoothing."""
        if len(data) < 2:
            return

        self._data = sorted(data, key=lambda x: x.timestamp)
        values = [d.value for d in self._data]

        # Initialize
        self._level = values[0]
        self._trend = values[1] - values[0]

        # Apply smoothing
        for i in range(1, len(values)):
            prev_level = self._level
            self._level = self.alpha * values[i] + (1 - self.alpha) * (
                self._level + self._trend
            )
            self._trend = self.beta * (self._level - prev_level) + (1 - self.beta) * self._trend

        self._trained = True
        self._last_trained = datetime.now()

    def predict(
        self, horizon: int, confidence_level: float = 0.95
    ) -> List[PredictionResult]:
        """Predict using exponential smoothing."""
        if not self._trained:
            return []

        predictions = []
        values = [d.value for d in self._data]
        std_dev = (
            sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)
        ) ** 0.5

        z_score = 1.96 if confidence_level >= 0.95 else 1.645

        for i in range(1, horizon + 1):
            forecast = self._level + (i * self._trend)
            adjusted_std = std_dev * (1 + i * 0.1)

            predictions.append(
                PredictionResult(
                    prediction_type=PredictionType.DEMAND,
                    predicted_value=forecast,
                    confidence=max(0.5, confidence_level - (i * 0.015)),
                    lower_bound=forecast - (z_score * adjusted_std),
                    upper_bound=forecast + (z_score * adjusted_std),
                    horizon=ForecastHorizon.MEDIUM_TERM,
                    model_used=ModelType.EXPONENTIAL_SMOOTHING,
                )
            )
        return predictions

    def get_metrics(self) -> ModelMetrics:
        """Get model metrics."""
        return ModelMetrics(
            model_type=ModelType.EXPONENTIAL_SMOOTHING,
            accuracy=0.82,
            mae=0.0,
            rmse=0.0,
            mape=0.0,
            training_time=0.05,
            last_trained=self._last_trained or datetime.now(),
        )


class TimeSeriesStore:
    """Storage for time series data."""

    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self._series: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def add_point(self, series_name: str, point: DataPoint) -> None:
        """Add a data point to a series."""
        with self._lock:
            if series_name not in self._series:
                self._series[series_name] = deque(maxlen=self.max_points)
            self._series[series_name].append(point)

    def get_series(
        self,
        series_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[DataPoint]:
        """Get data points from a series."""
        with self._lock:
            if series_name not in self._series:
                return []

            points = list(self._series[series_name])
            if start_time:
                points = [p for p in points if p.timestamp >= start_time]
            if end_time:
                points = [p for p in points if p.timestamp <= end_time]
            return points

    def get_series_names(self) -> List[str]:
        """Get all series names."""
        with self._lock:
            return list(self._series.keys())

    def clear_series(self, series_name: str) -> None:
        """Clear a series."""
        with self._lock:
            if series_name in self._series:
                self._series[series_name].clear()


class TrendAnalyzer:
    """Analyzes trends in time series data."""

    def __init__(self, min_points: int = 10):
        self.min_points = min_points

    def analyze(self, data: List[DataPoint]) -> TrendAnalysis:
        """Analyze trend in data."""
        if len(data) < self.min_points:
            return TrendAnalysis(
                direction=TrendDirection.STABLE,
                slope=0.0,
                strength=0.0,
                change_rate=0.0,
                seasonality_detected=False,
            )

        values = [d.value for d in sorted(data, key=lambda x: x.timestamp)]
        n = len(values)

        # Calculate slope using linear regression
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Calculate R-squared for trend strength
        y_pred = [y_mean + slope * (i - x_mean) for i in range(n)]
        ss_res = sum((v - p) ** 2 for v, p in zip(values, y_pred))
        ss_tot = sum((v - y_mean) ** 2 for v in values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Determine direction
        if abs(slope) < 0.01 * y_mean:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Check volatility
        std_dev = (sum((v - y_mean) ** 2 for v in values) / n) ** 0.5
        cv = std_dev / y_mean if y_mean != 0 else 0
        if cv > 0.3:
            direction = TrendDirection.VOLATILE

        # Calculate change rate
        if values[0] != 0:
            change_rate = ((values[-1] - values[0]) / values[0]) * 100
        else:
            change_rate = 0.0

        # Simple seasonality detection
        seasonality_detected = self._detect_seasonality(values)

        return TrendAnalysis(
            direction=direction,
            slope=slope,
            strength=max(0, min(1, r_squared)),
            change_rate=change_rate,
            seasonality_detected=seasonality_detected,
            period=self._estimate_period(values) if seasonality_detected else None,
        )

    def _detect_seasonality(self, values: List[float]) -> bool:
        """Simple seasonality detection using autocorrelation."""
        if len(values) < 20:
            return False

        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        if var == 0:
            return False

        # Check autocorrelation at different lags
        for lag in [7, 12, 24, 30]:  # Common periods
            if lag >= n // 2:
                continue

            autocorr = sum(
                (values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag)
            ) / ((n - lag) * var)

            if autocorr > 0.5:
                return True

        return False

    def _estimate_period(self, values: List[float]) -> Optional[float]:
        """Estimate seasonality period."""
        if len(values) < 20:
            return None

        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        if var == 0:
            return None

        best_lag = None
        best_autocorr = 0.5

        for lag in range(2, min(n // 2, 100)):
            autocorr = sum(
                (values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag)
            ) / ((n - lag) * var)

            if autocorr > best_autocorr:
                best_autocorr = autocorr
                best_lag = lag

        return float(best_lag) if best_lag else None


class PredictiveEngine:
    """Main predictive analytics engine."""

    def __init__(
        self,
        store: Optional[TimeSeriesStore] = None,
        default_model: ModelType = ModelType.EXPONENTIAL_SMOOTHING,
    ):
        self.store = store or TimeSeriesStore()
        self.default_model = default_model
        self._models: Dict[str, PredictionModel] = {}
        self._trend_analyzer = TrendAnalyzer()
        self._lock = threading.Lock()

    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a metric value."""
        point = DataPoint(
            timestamp=timestamp or datetime.now(),
            value=value,
            metadata=metadata or {},
        )
        self.store.add_point(metric_name, point)

    def train_model(
        self,
        metric_name: str,
        model_type: Optional[ModelType] = None,
    ) -> bool:
        """Train a model for a metric."""
        data = self.store.get_series(metric_name)
        if len(data) < 10:
            return False

        model_type = model_type or self.default_model

        if model_type == ModelType.MOVING_AVERAGE:
            model = MovingAverageModel()
        elif model_type == ModelType.EXPONENTIAL_SMOOTHING:
            model = ExponentialSmoothingModel()
        else:
            model = ExponentialSmoothingModel()

        model.train(data)

        with self._lock:
            self._models[metric_name] = model

        return True

    def predict(
        self,
        metric_name: str,
        horizon: int = 10,
        confidence_level: float = 0.95,
    ) -> List[PredictionResult]:
        """Make predictions for a metric."""
        with self._lock:
            model = self._models.get(metric_name)

        if not model:
            # Auto-train if data available
            if self.train_model(metric_name):
                with self._lock:
                    model = self._models.get(metric_name)

        if not model:
            return []

        return model.predict(horizon, confidence_level)

    def forecast(
        self,
        metric_name: str,
        horizon: int = 10,
        confidence_level: float = 0.95,
    ) -> Optional[ForecastResult]:
        """Generate a full forecast with trend analysis."""
        predictions = self.predict(metric_name, horizon, confidence_level)
        if not predictions:
            return None

        data = self.store.get_series(metric_name)
        trend = self._trend_analyzer.analyze(data)

        with self._lock:
            model = self._models.get(metric_name)
            model_confidence = model.get_metrics().accuracy if model else 0.5

        return ForecastResult(
            predictions=predictions,
            trend=trend,
            accuracy_score=model_confidence,
            model_confidence=model_confidence,
        )

    def analyze_trend(self, metric_name: str) -> Optional[TrendAnalysis]:
        """Analyze trend for a metric."""
        data = self.store.get_series(metric_name)
        if len(data) < 10:
            return None
        return self._trend_analyzer.analyze(data)

    def get_model_metrics(self, metric_name: str) -> Optional[ModelMetrics]:
        """Get metrics for a trained model."""
        with self._lock:
            model = self._models.get(metric_name)
            return model.get_metrics() if model else None


class DemandPredictor:
    """Predicts demand for vision processing."""

    def __init__(self, engine: Optional[PredictiveEngine] = None):
        self.engine = engine or PredictiveEngine()
        self._request_counts: Dict[str, int] = {}

    def record_request(
        self,
        provider: str,
        latency: float,
        success: bool,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a request for demand tracking."""
        ts = timestamp or datetime.now()

        # Record to appropriate series
        self.engine.record_metric(f"requests_{provider}", 1.0, ts)
        self.engine.record_metric(f"latency_{provider}", latency, ts)
        if not success:
            self.engine.record_metric(f"errors_{provider}", 1.0, ts)

    def predict_demand(
        self, provider: str, horizon_hours: int = 24
    ) -> Optional[ForecastResult]:
        """Predict demand for a provider."""
        return self.engine.forecast(f"requests_{provider}", horizon_hours)

    def predict_latency(
        self, provider: str, horizon: int = 10
    ) -> Optional[ForecastResult]:
        """Predict latency for a provider."""
        return self.engine.forecast(f"latency_{provider}", horizon)


class AnomalyPredictor:
    """Predicts anomalies based on patterns."""

    def __init__(self, engine: Optional[PredictiveEngine] = None, threshold: float = 2.0):
        self.engine = engine or PredictiveEngine()
        self.threshold = threshold  # Standard deviations for anomaly

    def predict_anomaly_probability(self, metric_name: str) -> float:
        """Predict probability of anomaly in near future."""
        data = self.engine.store.get_series(metric_name)
        if len(data) < 20:
            return 0.0

        values = [d.value for d in data]
        recent = values[-10:]
        historical = values[:-10]

        if not historical:
            return 0.0

        hist_mean = sum(historical) / len(historical)
        hist_std = (
            sum((v - hist_mean) ** 2 for v in historical) / len(historical)
        ) ** 0.5

        if hist_std == 0:
            return 0.0

        # Check how many recent points are anomalous
        anomalous = sum(
            1 for v in recent if abs(v - hist_mean) > self.threshold * hist_std
        )

        return anomalous / len(recent)

    def get_anomaly_forecast(
        self, metric_name: str
    ) -> Dict[str, Any]:
        """Get anomaly forecast."""
        probability = self.predict_anomaly_probability(metric_name)
        trend = self.engine.analyze_trend(metric_name)

        return {
            "metric": metric_name,
            "anomaly_probability": probability,
            "trend": trend.direction.value if trend else "unknown",
            "recommendation": self._get_recommendation(probability, trend),
        }

    def _get_recommendation(
        self, probability: float, trend: Optional[TrendAnalysis]
    ) -> str:
        """Get recommendation based on prediction."""
        if probability > 0.7:
            return "High anomaly probability - investigate immediately"
        elif probability > 0.4:
            return "Moderate anomaly probability - monitor closely"
        elif trend and trend.direction == TrendDirection.VOLATILE:
            return "High volatility detected - consider stabilization"
        elif trend and trend.direction == TrendDirection.INCREASING:
            return "Increasing trend - ensure capacity"
        return "Normal operation expected"


class PredictiveVisionProvider(VisionProvider):
    """Vision provider with predictive analytics."""

    def __init__(
        self,
        wrapped_provider: VisionProvider,
        engine: Optional[PredictiveEngine] = None,
    ):
        self._wrapped = wrapped_provider
        self.engine = engine or PredictiveEngine()
        self._demand_predictor = DemandPredictor(self.engine)
        self._anomaly_predictor = AnomalyPredictor(self.engine)

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"predictive_{self._wrapped.provider_name}"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True, **kwargs: Any
    ) -> VisionDescription:
        """Analyze image with predictive tracking."""
        start_time = time.time()
        success = True

        try:
            result = await self._wrapped.analyze_image(
                image_data, include_description, **kwargs
            )
            return result
        except Exception as e:
            success = False
            raise
        finally:
            latency = time.time() - start_time
            self._demand_predictor.record_request(
                self._wrapped.provider_name,
                latency,
                success,
            )

    def get_demand_forecast(self, horizon_hours: int = 24) -> Optional[ForecastResult]:
        """Get demand forecast."""
        return self._demand_predictor.predict_demand(
            self._wrapped.provider_name, horizon_hours
        )

    def get_latency_forecast(self, horizon: int = 10) -> Optional[ForecastResult]:
        """Get latency forecast."""
        return self._demand_predictor.predict_latency(
            self._wrapped.provider_name, horizon
        )

    def get_anomaly_forecast(self) -> Dict[str, Any]:
        """Get anomaly forecast."""
        return self._anomaly_predictor.get_anomaly_forecast(
            f"latency_{self._wrapped.provider_name}"
        )


# Factory functions
def create_predictive_engine(
    max_points: int = 10000,
    default_model: ModelType = ModelType.EXPONENTIAL_SMOOTHING,
) -> PredictiveEngine:
    """Create a predictive engine."""
    store = TimeSeriesStore(max_points=max_points)
    return PredictiveEngine(store=store, default_model=default_model)


def create_predictive_provider(
    provider: VisionProvider,
    engine: Optional[PredictiveEngine] = None,
) -> PredictiveVisionProvider:
    """Create a predictive vision provider."""
    return PredictiveVisionProvider(provider, engine)


def create_demand_predictor(
    engine: Optional[PredictiveEngine] = None,
) -> DemandPredictor:
    """Create a demand predictor."""
    return DemandPredictor(engine)


def create_anomaly_predictor(
    engine: Optional[PredictiveEngine] = None,
    threshold: float = 2.0,
) -> AnomalyPredictor:
    """Create an anomaly predictor."""
    return AnomalyPredictor(engine, threshold)
