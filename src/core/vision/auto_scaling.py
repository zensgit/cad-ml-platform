"""Auto Scaling Module.

Provides dynamic resource scaling based on demand prediction and metrics.
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base import VisionDescription, VisionProvider


class ScalingDirection(Enum):
    """Scaling directions."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"


class ScalingPolicy(Enum):
    """Scaling policies."""

    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    HYBRID = "hybrid"


class ResourceType(Enum):
    """Types of resources to scale."""

    WORKERS = "workers"
    MEMORY = "memory"
    CONNECTIONS = "connections"
    THROUGHPUT = "throughput"
    INSTANCES = "instances"


class ScalingState(Enum):
    """Scaling states."""

    STABLE = "stable"
    SCALING_UP = "scaling_up"
    SCALING_DOWN = "scaling_down"
    COOLDOWN = "cooldown"


@dataclass
class ScalingConfig:
    """Configuration for auto scaling."""

    min_capacity: int = 1
    max_capacity: int = 100
    desired_capacity: int = 1
    scale_up_threshold: float = 0.8  # Utilization threshold
    scale_down_threshold: float = 0.3
    scale_up_increment: int = 2
    scale_down_increment: int = 1
    cooldown_period_seconds: int = 300
    evaluation_periods: int = 3
    metric_name: str = "utilization"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""

    current_utilization: float = 0.0
    average_latency: float = 0.0
    request_rate: float = 0.0  # Requests per second
    error_rate: float = 0.0
    queue_depth: int = 0
    active_connections: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingDecision:
    """Result of a scaling decision."""

    direction: ScalingDirection
    current_capacity: int
    target_capacity: int
    reason: str
    confidence: float
    metrics: ScalingMetrics
    policy_used: ScalingPolicy
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingEvent:
    """Record of a scaling event."""

    event_id: str
    direction: ScalingDirection
    from_capacity: int
    to_capacity: int
    reason: str
    success: bool
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapacityPlan:
    """Planned capacity schedule."""

    plan_id: str
    scheduled_time: datetime
    target_capacity: int
    reason: str
    recurring: bool = False
    recurrence_pattern: Optional[str] = None  # e.g., "daily", "weekly"


class MetricsCollector:
    """Collects and aggregates metrics for scaling decisions."""

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._metrics: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record_metrics(self, metrics: ScalingMetrics) -> None:
        """Record metrics."""
        with self._lock:
            self._metrics.append(metrics)

    def get_average_metrics(self) -> ScalingMetrics:
        """Get averaged metrics over the window."""
        with self._lock:
            if not self._metrics:
                return ScalingMetrics()

            total = len(self._metrics)
            return ScalingMetrics(
                current_utilization=sum(m.current_utilization for m in self._metrics) / total,
                average_latency=sum(m.average_latency for m in self._metrics) / total,
                request_rate=sum(m.request_rate for m in self._metrics) / total,
                error_rate=sum(m.error_rate for m in self._metrics) / total,
                queue_depth=int(sum(m.queue_depth for m in self._metrics) / total),
                active_connections=int(sum(m.active_connections for m in self._metrics) / total),
            )

    def get_latest_metrics(self) -> Optional[ScalingMetrics]:
        """Get most recent metrics."""
        with self._lock:
            return self._metrics[-1] if self._metrics else None

    def get_trend(self) -> float:
        """Get utilization trend (-1 to 1)."""
        with self._lock:
            if len(self._metrics) < 2:
                return 0.0

            recent = list(self._metrics)
            half = len(recent) // 2
            old_avg = sum(m.current_utilization for m in recent[:half]) / half
            new_avg = sum(m.current_utilization for m in recent[half:]) / (len(recent) - half)

            if old_avg == 0:
                return 0.0
            return (new_avg - old_avg) / old_avg


class ScalingPredictor:
    """Predicts future capacity needs."""

    def __init__(self, history_size: int = 1000):
        self._history: deque = deque(maxlen=history_size)
        self._patterns: Dict[int, List[float]] = {}  # hour -> utilization pattern
        self._lock = threading.Lock()

    def record_observation(self, utilization: float, timestamp: datetime) -> None:
        """Record an observation for pattern learning."""
        with self._lock:
            self._history.append((utilization, timestamp))

            # Update hourly pattern
            hour = timestamp.hour
            if hour not in self._patterns:
                self._patterns[hour] = []
            self._patterns[hour].append(utilization)

            # Keep patterns bounded
            if len(self._patterns[hour]) > 100:
                self._patterns[hour] = self._patterns[hour][-100:]

    def predict_utilization(self, hours_ahead: int = 1) -> float:
        """Predict utilization for a future time."""
        with self._lock:
            target_hour = (datetime.now().hour + hours_ahead) % 24

            if target_hour in self._patterns and self._patterns[target_hour]:
                return sum(self._patterns[target_hour]) / len(self._patterns[target_hour])

            # Default to recent average
            if self._history:
                recent = [u for u, _ in list(self._history)[-100:]]
                return sum(recent) / len(recent)

            return 0.5

    def predict_capacity_needed(self, current_capacity: int, hours_ahead: int = 1) -> int:
        """Predict capacity needed in the future."""
        predicted_utilization = self.predict_utilization(hours_ahead)

        # If predicted utilization is high, need more capacity
        if predicted_utilization > 0.8:
            return int(current_capacity * 1.5)
        elif predicted_utilization > 0.6:
            return int(current_capacity * 1.2)
        elif predicted_utilization < 0.3:
            return max(1, int(current_capacity * 0.8))

        return current_capacity


class ReactiveScaler:
    """Reactive scaling based on current metrics."""

    def __init__(self, config: ScalingConfig):
        self.config = config

    def evaluate(self, metrics: ScalingMetrics, current_capacity: int) -> ScalingDecision:
        """Evaluate if scaling is needed."""
        if metrics.current_utilization >= self.config.scale_up_threshold:
            target = min(
                current_capacity + self.config.scale_up_increment,
                self.config.max_capacity,
            )
            return ScalingDecision(
                direction=ScalingDirection.SCALE_UP
                if target > current_capacity
                else ScalingDirection.NO_CHANGE,
                current_capacity=current_capacity,
                target_capacity=target,
                reason=f"Utilization {metrics.current_utilization:.2%} exceeds threshold {self.config.scale_up_threshold:.2%}",
                confidence=0.9,
                metrics=metrics,
                policy_used=ScalingPolicy.REACTIVE,
            )

        elif metrics.current_utilization <= self.config.scale_down_threshold:
            target = max(
                current_capacity - self.config.scale_down_increment,
                self.config.min_capacity,
            )
            return ScalingDecision(
                direction=ScalingDirection.SCALE_DOWN
                if target < current_capacity
                else ScalingDirection.NO_CHANGE,
                current_capacity=current_capacity,
                target_capacity=target,
                reason=f"Utilization {metrics.current_utilization:.2%} below threshold {self.config.scale_down_threshold:.2%}",
                confidence=0.85,
                metrics=metrics,
                policy_used=ScalingPolicy.REACTIVE,
            )

        return ScalingDecision(
            direction=ScalingDirection.NO_CHANGE,
            current_capacity=current_capacity,
            target_capacity=current_capacity,
            reason="Utilization within acceptable range",
            confidence=0.95,
            metrics=metrics,
            policy_used=ScalingPolicy.REACTIVE,
        )


class PredictiveScaler:
    """Predictive scaling based on forecasts."""

    def __init__(self, config: ScalingConfig, predictor: Optional[ScalingPredictor] = None):
        self.config = config
        self.predictor = predictor or ScalingPredictor()

    def evaluate(
        self, metrics: ScalingMetrics, current_capacity: int, hours_ahead: int = 1
    ) -> ScalingDecision:
        """Evaluate scaling based on predictions."""
        # Record current state
        self.predictor.record_observation(metrics.current_utilization, datetime.now())

        # Predict future need
        predicted_capacity = self.predictor.predict_capacity_needed(current_capacity, hours_ahead)
        predicted_utilization = self.predictor.predict_utilization(hours_ahead)

        if predicted_capacity > current_capacity:
            target = min(predicted_capacity, self.config.max_capacity)
            return ScalingDecision(
                direction=ScalingDirection.SCALE_UP
                if target > current_capacity
                else ScalingDirection.NO_CHANGE,
                current_capacity=current_capacity,
                target_capacity=target,
                reason=f"Predicted utilization {predicted_utilization:.2%} in {hours_ahead}h",
                confidence=0.75,
                metrics=metrics,
                policy_used=ScalingPolicy.PREDICTIVE,
            )

        elif predicted_capacity < current_capacity:
            target = max(predicted_capacity, self.config.min_capacity)
            return ScalingDecision(
                direction=ScalingDirection.SCALE_DOWN
                if target < current_capacity
                else ScalingDirection.NO_CHANGE,
                current_capacity=current_capacity,
                target_capacity=target,
                reason=f"Predicted lower demand in {hours_ahead}h",
                confidence=0.7,
                metrics=metrics,
                policy_used=ScalingPolicy.PREDICTIVE,
            )

        return ScalingDecision(
            direction=ScalingDirection.NO_CHANGE,
            current_capacity=current_capacity,
            target_capacity=current_capacity,
            reason="Predicted stable demand",
            confidence=0.8,
            metrics=metrics,
            policy_used=ScalingPolicy.PREDICTIVE,
        )


class ScheduledScaler:
    """Scheduled scaling based on time-based plans."""

    def __init__(self, config: ScalingConfig):
        self.config = config
        self._plans: List[CapacityPlan] = []
        self._lock = threading.Lock()

    def add_plan(self, plan: CapacityPlan) -> None:
        """Add a capacity plan."""
        with self._lock:
            self._plans.append(plan)
            self._plans.sort(key=lambda p: p.scheduled_time)

    def remove_plan(self, plan_id: str) -> bool:
        """Remove a capacity plan."""
        with self._lock:
            initial_count = len(self._plans)
            self._plans = [p for p in self._plans if p.plan_id != plan_id]
            return len(self._plans) < initial_count

    def get_active_plan(self) -> Optional[CapacityPlan]:
        """Get the currently active plan."""
        now = datetime.now()
        with self._lock:
            for plan in self._plans:
                if plan.scheduled_time <= now:
                    return plan
        return None

    def evaluate(self, current_capacity: int) -> ScalingDecision:
        """Evaluate if scheduled scaling should occur."""
        plan = self.get_active_plan()

        if plan and plan.target_capacity != current_capacity:
            target = max(
                self.config.min_capacity,
                min(plan.target_capacity, self.config.max_capacity),
            )
            direction = (
                ScalingDirection.SCALE_UP
                if target > current_capacity
                else ScalingDirection.SCALE_DOWN
                if target < current_capacity
                else ScalingDirection.NO_CHANGE
            )
            return ScalingDecision(
                direction=direction,
                current_capacity=current_capacity,
                target_capacity=target,
                reason=f"Scheduled: {plan.reason}",
                confidence=1.0,
                metrics=ScalingMetrics(),
                policy_used=ScalingPolicy.SCHEDULED,
            )

        return ScalingDecision(
            direction=ScalingDirection.NO_CHANGE,
            current_capacity=current_capacity,
            target_capacity=current_capacity,
            reason="No active schedule",
            confidence=1.0,
            metrics=ScalingMetrics(),
            policy_used=ScalingPolicy.SCHEDULED,
        )

    def get_plans(self) -> List[CapacityPlan]:
        """Get all plans."""
        with self._lock:
            return list(self._plans)


class AutoScaler:
    """Main auto scaling engine."""

    def __init__(
        self,
        config: Optional[ScalingConfig] = None,
        policy: ScalingPolicy = ScalingPolicy.HYBRID,
    ):
        self.config = config or ScalingConfig()
        self.policy = policy
        self._current_capacity = self.config.desired_capacity
        self._state = ScalingState.STABLE
        self._last_scaling_time: Optional[datetime] = None
        self._events: deque = deque(maxlen=1000)

        self._metrics_collector = MetricsCollector()
        self._reactive_scaler = ReactiveScaler(self.config)
        self._predictive_scaler = PredictiveScaler(self.config)
        self._scheduled_scaler = ScheduledScaler(self.config)

        self._scale_up_callback: Optional[Callable[[int, int], bool]] = None
        self._scale_down_callback: Optional[Callable[[int, int], bool]] = None

        self._lock = threading.Lock()

    def set_scale_callbacks(
        self,
        scale_up: Optional[Callable[[int, int], bool]] = None,
        scale_down: Optional[Callable[[int, int], bool]] = None,
    ) -> None:
        """Set callbacks for scaling operations."""
        self._scale_up_callback = scale_up
        self._scale_down_callback = scale_down

    def record_metrics(self, metrics: ScalingMetrics) -> None:
        """Record current metrics."""
        self._metrics_collector.record_metrics(metrics)

    def evaluate(self) -> ScalingDecision:
        """Evaluate if scaling is needed."""
        metrics = self._metrics_collector.get_average_metrics()

        # Check cooldown
        if self._is_in_cooldown():
            return ScalingDecision(
                direction=ScalingDirection.NO_CHANGE,
                current_capacity=self._current_capacity,
                target_capacity=self._current_capacity,
                reason="In cooldown period",
                confidence=1.0,
                metrics=metrics,
                policy_used=self.policy,
            )

        # Get decisions from different policies
        decisions: List[ScalingDecision] = []

        if self.policy in (ScalingPolicy.REACTIVE, ScalingPolicy.HYBRID):
            decisions.append(self._reactive_scaler.evaluate(metrics, self._current_capacity))

        if self.policy in (ScalingPolicy.PREDICTIVE, ScalingPolicy.HYBRID):
            decisions.append(self._predictive_scaler.evaluate(metrics, self._current_capacity))

        if self.policy in (ScalingPolicy.SCHEDULED, ScalingPolicy.HYBRID):
            decisions.append(self._scheduled_scaler.evaluate(self._current_capacity))

        # Select best decision
        return self._select_decision(decisions, metrics)

    def _select_decision(
        self, decisions: List[ScalingDecision], metrics: ScalingMetrics
    ) -> ScalingDecision:
        """Select the best decision from multiple policies."""
        if not decisions:
            return ScalingDecision(
                direction=ScalingDirection.NO_CHANGE,
                current_capacity=self._current_capacity,
                target_capacity=self._current_capacity,
                reason="No decisions available",
                confidence=0.5,
                metrics=metrics,
                policy_used=self.policy,
            )

        # Prioritize scale-up decisions for safety
        scale_ups = [d for d in decisions if d.direction == ScalingDirection.SCALE_UP]
        if scale_ups:
            # Return the most aggressive scale-up
            return max(scale_ups, key=lambda d: d.target_capacity)

        # For scale-down, be more conservative
        scale_downs = [d for d in decisions if d.direction == ScalingDirection.SCALE_DOWN]
        if scale_downs and all(d.direction == ScalingDirection.SCALE_DOWN for d in decisions):
            # Only scale down if all policies agree, use most conservative
            return max(scale_downs, key=lambda d: d.target_capacity)

        # Default to no change
        return ScalingDecision(
            direction=ScalingDirection.NO_CHANGE,
            current_capacity=self._current_capacity,
            target_capacity=self._current_capacity,
            reason="No consensus on scaling action",
            confidence=0.7,
            metrics=metrics,
            policy_used=self.policy,
        )

    def execute(self, decision: ScalingDecision) -> ScalingEvent:
        """Execute a scaling decision."""
        event_id = f"scale_{int(time.time() * 1000)}"
        start_time = time.time()
        success = True

        with self._lock:
            from_capacity = self._current_capacity

            if decision.direction == ScalingDirection.SCALE_UP:
                self._state = ScalingState.SCALING_UP
                if self._scale_up_callback:
                    success = self._scale_up_callback(from_capacity, decision.target_capacity)
                if success:
                    self._current_capacity = decision.target_capacity

            elif decision.direction == ScalingDirection.SCALE_DOWN:
                self._state = ScalingState.SCALING_DOWN
                if self._scale_down_callback:
                    success = self._scale_down_callback(from_capacity, decision.target_capacity)
                if success:
                    self._current_capacity = decision.target_capacity

            if decision.direction != ScalingDirection.NO_CHANGE:
                self._last_scaling_time = datetime.now()
                self._state = ScalingState.COOLDOWN

            duration = time.time() - start_time
            self._state = ScalingState.STABLE

        event = ScalingEvent(
            event_id=event_id,
            direction=decision.direction,
            from_capacity=from_capacity,
            to_capacity=self._current_capacity,
            reason=decision.reason,
            success=success,
            duration_seconds=duration,
        )
        self._events.append(event)

        return event

    def _is_in_cooldown(self) -> bool:
        """Check if in cooldown period."""
        if not self._last_scaling_time:
            return False
        elapsed = (datetime.now() - self._last_scaling_time).total_seconds()
        return elapsed < self.config.cooldown_period_seconds

    def get_current_capacity(self) -> int:
        """Get current capacity."""
        with self._lock:
            return self._current_capacity

    def get_state(self) -> ScalingState:
        """Get current scaling state."""
        return self._state

    def get_events(self, limit: int = 100) -> List[ScalingEvent]:
        """Get recent scaling events."""
        return list(self._events)[-limit:]

    def add_schedule(self, plan: CapacityPlan) -> None:
        """Add a scheduled capacity plan."""
        self._scheduled_scaler.add_plan(plan)

    def remove_schedule(self, plan_id: str) -> bool:
        """Remove a scheduled plan."""
        return self._scheduled_scaler.remove_plan(plan_id)


class AutoScaledVisionProvider(VisionProvider):
    """Vision provider with auto-scaling capabilities."""

    def __init__(
        self,
        provider: VisionProvider,
        auto_scaler: Optional[AutoScaler] = None,
    ):
        self._provider = provider
        self.auto_scaler = auto_scaler or AutoScaler()
        self._request_count = 0
        self._active_requests = 0
        self._total_latency = 0.0
        self._errors = 0
        self._lock = threading.Lock()

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"auto_scaled_{self._provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True, **kwargs: Any
    ) -> VisionDescription:
        """Analyze image with auto-scaling metrics collection."""
        with self._lock:
            self._request_count += 1
            self._active_requests += 1

        start_time = time.time()

        try:
            result = await self._provider.analyze_image(image_data, include_description, **kwargs)
            return result
        except Exception as e:
            with self._lock:
                self._errors += 1
            raise
        finally:
            latency = time.time() - start_time
            with self._lock:
                self._active_requests -= 1
                self._total_latency += latency

            # Record metrics
            self._record_metrics()

    def _record_metrics(self) -> None:
        """Record metrics for auto-scaling."""
        with self._lock:
            capacity = self.auto_scaler.get_current_capacity()
            utilization = self._active_requests / capacity if capacity > 0 else 0
            avg_latency = (
                self._total_latency / self._request_count if self._request_count > 0 else 0
            )
            error_rate = self._errors / self._request_count if self._request_count > 0 else 0

        metrics = ScalingMetrics(
            current_utilization=min(1.0, utilization),
            average_latency=avg_latency,
            request_rate=self._request_count / 60.0,  # Simplified
            error_rate=error_rate,
            active_connections=self._active_requests,
        )
        self.auto_scaler.record_metrics(metrics)

    def check_scaling(self) -> ScalingDecision:
        """Check if scaling is needed."""
        return self.auto_scaler.evaluate()

    def apply_scaling(self, decision: ScalingDecision) -> ScalingEvent:
        """Apply a scaling decision."""
        return self.auto_scaler.execute(decision)

    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        return {
            "current_capacity": self.auto_scaler.get_current_capacity(),
            "state": self.auto_scaler.get_state().value,
            "request_count": self._request_count,
            "active_requests": self._active_requests,
            "error_rate": self._errors / max(1, self._request_count),
            "recent_events": [
                {
                    "direction": e.direction.value,
                    "from": e.from_capacity,
                    "to": e.to_capacity,
                    "success": e.success,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in self.auto_scaler.get_events(10)
            ],
        }


# Factory functions
def create_scaling_config(
    min_capacity: int = 1,
    max_capacity: int = 100,
    scale_up_threshold: float = 0.8,
    scale_down_threshold: float = 0.3,
    cooldown_seconds: int = 300,
) -> ScalingConfig:
    """Create a scaling configuration."""
    return ScalingConfig(
        min_capacity=min_capacity,
        max_capacity=max_capacity,
        scale_up_threshold=scale_up_threshold,
        scale_down_threshold=scale_down_threshold,
        cooldown_period_seconds=cooldown_seconds,
    )


def create_auto_scaler(
    config: Optional[ScalingConfig] = None,
    policy: ScalingPolicy = ScalingPolicy.HYBRID,
) -> AutoScaler:
    """Create an auto scaler."""
    return AutoScaler(config=config, policy=policy)


def create_auto_scaled_provider(
    provider: VisionProvider,
    auto_scaler: Optional[AutoScaler] = None,
) -> AutoScaledVisionProvider:
    """Create an auto-scaled vision provider."""
    return AutoScaledVisionProvider(provider, auto_scaler)


def create_capacity_plan(
    plan_id: str,
    target_capacity: int,
    scheduled_time: datetime,
    reason: str,
    recurring: bool = False,
) -> CapacityPlan:
    """Create a capacity plan."""
    return CapacityPlan(
        plan_id=plan_id,
        scheduled_time=scheduled_time,
        target_capacity=target_capacity,
        reason=reason,
        recurring=recurring,
    )
