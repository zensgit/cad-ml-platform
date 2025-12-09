"""Chaos engineering for Vision Provider system.

This module provides chaos engineering features including:
- Fault injection
- Latency injection
- Error simulation
- Resource exhaustion simulation
- Chaos experiments
"""

import asyncio
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

from .base import VisionDescription, VisionProvider


class FaultType(Enum):
    """Fault type."""

    LATENCY = "latency"
    ERROR = "error"
    TIMEOUT = "timeout"
    PARTIAL_FAILURE = "partial_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CORRUPTION = "corruption"
    NETWORK_PARTITION = "network_partition"


class ExperimentStatus(Enum):
    """Experiment status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class TargetType(Enum):
    """Experiment target type."""

    PROVIDER = "provider"
    REQUEST = "request"
    RESPONSE = "response"
    DEPENDENCY = "dependency"


class InjectionStrategy(Enum):
    """Fault injection strategy."""

    RANDOM = "random"
    SEQUENTIAL = "sequential"
    BURST = "burst"
    GRADUAL = "gradual"


@dataclass
class FaultConfig:
    """Fault configuration."""

    fault_type: FaultType
    probability: float = 0.1  # 10% chance by default
    duration_ms: int = 0  # For latency faults
    error_message: str = "Injected fault"
    error_type: str = "RuntimeError"
    enabled: bool = True
    target: TargetType = TargetType.REQUEST
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_inject(self) -> bool:
        """Check if fault should be injected."""
        if not self.enabled:
            return False
        return random.random() < self.probability


@dataclass
class ExperimentConfig:
    """Chaos experiment configuration."""

    name: str
    description: str = ""
    faults: List[FaultConfig] = field(default_factory=list)
    duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    strategy: InjectionStrategy = InjectionStrategy.RANDOM
    max_impact_percentage: float = 50.0
    auto_abort_on_failure: bool = True
    steady_state_hypothesis: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentResult:
    """Chaos experiment result."""

    experiment_name: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_requests: int = 0
    affected_requests: int = 0
    injected_faults: Dict[str, int] = field(default_factory=dict)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    steady_state_validated: bool = True
    error: Optional[str] = None

    @property
    def impact_percentage(self) -> float:
        """Calculate impact percentage."""
        if self.total_requests == 0:
            return 0.0
        return self.affected_requests / self.total_requests * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_requests": self.total_requests,
            "affected_requests": self.affected_requests,
            "impact_percentage": self.impact_percentage,
            "injected_faults": dict(self.injected_faults),
            "observations": list(self.observations),
            "steady_state_validated": self.steady_state_validated,
            "error": self.error,
        }


class FaultInjector:
    """Injects faults into requests."""

    def __init__(self) -> None:
        """Initialize injector."""
        self._faults: List[FaultConfig] = []
        self._enabled = False
        self._lock = threading.Lock()
        self._injection_count: Dict[str, int] = {}

    def add_fault(self, fault: FaultConfig) -> None:
        """Add fault configuration.

        Args:
            fault: Fault configuration
        """
        with self._lock:
            self._faults.append(fault)

    def remove_fault(self, fault_type: FaultType) -> None:
        """Remove fault configuration.

        Args:
            fault_type: Fault type to remove
        """
        with self._lock:
            self._faults = [f for f in self._faults if f.fault_type != fault_type]

    def clear_faults(self) -> None:
        """Clear all faults."""
        with self._lock:
            self._faults.clear()

    def enable(self) -> None:
        """Enable fault injection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable fault injection."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if injection is enabled."""
        return self._enabled

    async def maybe_inject(self) -> Optional[FaultConfig]:
        """Maybe inject a fault.

        Returns:
            Injected fault config or None
        """
        if not self._enabled:
            return None

        with self._lock:
            for fault in self._faults:
                if fault.should_inject():
                    self._record_injection(fault.fault_type)
                    await self._apply_fault(fault)
                    return fault

        return None

    def _record_injection(self, fault_type: FaultType) -> None:
        """Record fault injection."""
        key = fault_type.value
        self._injection_count[key] = self._injection_count.get(key, 0) + 1

    async def _apply_fault(self, fault: FaultConfig) -> None:
        """Apply fault effect.

        Args:
            fault: Fault configuration
        """
        if fault.fault_type == FaultType.LATENCY:
            await asyncio.sleep(fault.duration_ms / 1000.0)

        elif fault.fault_type == FaultType.ERROR:
            error_class = {
                "RuntimeError": RuntimeError,
                "ValueError": ValueError,
                "TimeoutError": TimeoutError,
                "ConnectionError": ConnectionError,
            }.get(fault.error_type, RuntimeError)
            raise error_class(fault.error_message)

        elif fault.fault_type == FaultType.TIMEOUT:
            await asyncio.sleep(fault.duration_ms / 1000.0)
            raise TimeoutError(fault.error_message)

    def get_injection_stats(self) -> Dict[str, int]:
        """Get injection statistics."""
        return dict(self._injection_count)

    def reset_stats(self) -> None:
        """Reset injection statistics."""
        self._injection_count.clear()


class LatencyInjector:
    """Injects latency into requests."""

    def __init__(
        self,
        min_latency_ms: int = 100,
        max_latency_ms: int = 5000,
        probability: float = 0.1,
    ) -> None:
        """Initialize injector.

        Args:
            min_latency_ms: Minimum latency
            max_latency_ms: Maximum latency
            probability: Injection probability
        """
        self._min_latency = min_latency_ms
        self._max_latency = max_latency_ms
        self._probability = probability
        self._enabled = False
        self._total_injected = 0
        self._total_latency_ms = 0

    def enable(self) -> None:
        """Enable latency injection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable latency injection."""
        self._enabled = False

    async def maybe_inject(self) -> int:
        """Maybe inject latency.

        Returns:
            Injected latency in ms (0 if not injected)
        """
        if not self._enabled or random.random() >= self._probability:
            return 0

        latency = random.randint(self._min_latency, self._max_latency)
        await asyncio.sleep(latency / 1000.0)

        self._total_injected += 1
        self._total_latency_ms += latency

        return latency

    def get_stats(self) -> Dict[str, Any]:
        """Get injection statistics."""
        return {
            "total_injected": self._total_injected,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": (
                self._total_latency_ms / self._total_injected
                if self._total_injected > 0
                else 0
            ),
        }


class ErrorInjector:
    """Injects errors into requests."""

    def __init__(
        self,
        probability: float = 0.1,
        error_types: Optional[List[str]] = None,
    ) -> None:
        """Initialize injector.

        Args:
            probability: Error probability
            error_types: List of error types to inject
        """
        self._probability = probability
        self._error_types = error_types or ["RuntimeError"]
        self._enabled = False
        self._total_injected = 0

    def enable(self) -> None:
        """Enable error injection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable error injection."""
        self._enabled = False

    def maybe_inject(self) -> Optional[Exception]:
        """Maybe inject an error.

        Returns:
            Exception to raise or None
        """
        if not self._enabled or random.random() >= self._probability:
            return None

        self._total_injected += 1
        error_type = random.choice(self._error_types)

        error_class = {
            "RuntimeError": RuntimeError,
            "ValueError": ValueError,
            "TimeoutError": TimeoutError,
            "ConnectionError": ConnectionError,
            "IOError": IOError,
        }.get(error_type, RuntimeError)

        return error_class(f"Chaos injected {error_type}")

    def get_stats(self) -> Dict[str, int]:
        """Get injection statistics."""
        return {"total_injected": self._total_injected}


class ChaosExperiment:
    """A chaos experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
    ) -> None:
        """Initialize experiment.

        Args:
            config: Experiment configuration
        """
        self._config = config
        self._fault_injector = FaultInjector()
        self._result = ExperimentResult(experiment_name=config.name)
        self._started = False
        self._lock = threading.Lock()

        # Set up faults
        for fault in config.faults:
            self._fault_injector.add_fault(fault)

    def start(self) -> None:
        """Start experiment."""
        with self._lock:
            if self._started:
                return

            self._started = True
            self._result.status = ExperimentStatus.RUNNING
            self._result.started_at = datetime.now()
            self._fault_injector.enable()

    def stop(self) -> ExperimentResult:
        """Stop experiment.

        Returns:
            Experiment result
        """
        with self._lock:
            self._fault_injector.disable()
            self._result.completed_at = datetime.now()
            self._result.status = ExperimentStatus.COMPLETED
            self._result.injected_faults = self._fault_injector.get_injection_stats()
            return self._result

    def abort(self, reason: str = "") -> ExperimentResult:
        """Abort experiment.

        Args:
            reason: Abort reason

        Returns:
            Experiment result
        """
        with self._lock:
            self._fault_injector.disable()
            self._result.completed_at = datetime.now()
            self._result.status = ExperimentStatus.ABORTED
            self._result.error = reason
            self._result.injected_faults = self._fault_injector.get_injection_stats()
            return self._result

    async def process_request(self) -> Optional[FaultConfig]:
        """Process a request through experiment.

        Returns:
            Injected fault or None
        """
        with self._lock:
            self._result.total_requests += 1

        fault = await self._fault_injector.maybe_inject()

        if fault:
            with self._lock:
                self._result.affected_requests += 1

            # Check abort conditions
            if (
                self._config.auto_abort_on_failure
                and self._result.impact_percentage > self._config.max_impact_percentage
            ):
                self.abort(f"Impact exceeded {self._config.max_impact_percentage}%")

        return fault

    def add_observation(self, observation: Dict[str, Any]) -> None:
        """Add observation.

        Args:
            observation: Observation data
        """
        with self._lock:
            self._result.observations.append({
                "timestamp": datetime.now().isoformat(),
                **observation,
            })

    def get_result(self) -> ExperimentResult:
        """Get current result."""
        with self._lock:
            return ExperimentResult(
                experiment_name=self._result.experiment_name,
                status=self._result.status,
                started_at=self._result.started_at,
                completed_at=self._result.completed_at,
                total_requests=self._result.total_requests,
                affected_requests=self._result.affected_requests,
                injected_faults=dict(self._result.injected_faults),
                observations=list(self._result.observations),
                steady_state_validated=self._result.steady_state_validated,
                error=self._result.error,
            )

    def is_running(self) -> bool:
        """Check if experiment is running."""
        return self._result.status == ExperimentStatus.RUNNING


class ChaosManager:
    """Manages chaos experiments."""

    def __init__(self) -> None:
        """Initialize manager."""
        self._experiments: Dict[str, ChaosExperiment] = {}
        self._active_experiment: Optional[str] = None
        self._history: List[ExperimentResult] = []
        self._lock = threading.Lock()

    def create_experiment(self, config: ExperimentConfig) -> ChaosExperiment:
        """Create experiment.

        Args:
            config: Experiment configuration

        Returns:
            Created experiment
        """
        experiment = ChaosExperiment(config)
        with self._lock:
            self._experiments[config.name] = experiment
        return experiment

    def start_experiment(self, name: str) -> bool:
        """Start experiment.

        Args:
            name: Experiment name

        Returns:
            True if started
        """
        with self._lock:
            if self._active_experiment:
                return False

            experiment = self._experiments.get(name)
            if not experiment:
                return False

            experiment.start()
            self._active_experiment = name
            return True

    def stop_experiment(self, name: Optional[str] = None) -> Optional[ExperimentResult]:
        """Stop experiment.

        Args:
            name: Experiment name (uses active if not specified)

        Returns:
            Experiment result or None
        """
        with self._lock:
            exp_name = name or self._active_experiment
            if not exp_name:
                return None

            experiment = self._experiments.get(exp_name)
            if not experiment:
                return None

            result = experiment.stop()
            self._history.append(result)

            if exp_name == self._active_experiment:
                self._active_experiment = None

            return result

    def abort_experiment(
        self,
        name: Optional[str] = None,
        reason: str = "",
    ) -> Optional[ExperimentResult]:
        """Abort experiment.

        Args:
            name: Experiment name
            reason: Abort reason

        Returns:
            Experiment result or None
        """
        with self._lock:
            exp_name = name or self._active_experiment
            if not exp_name:
                return None

            experiment = self._experiments.get(exp_name)
            if not experiment:
                return None

            result = experiment.abort(reason)
            self._history.append(result)

            if exp_name == self._active_experiment:
                self._active_experiment = None

            return result

    def get_active_experiment(self) -> Optional[ChaosExperiment]:
        """Get active experiment."""
        with self._lock:
            if not self._active_experiment:
                return None
            return self._experiments.get(self._active_experiment)

    def get_experiment(self, name: str) -> Optional[ChaosExperiment]:
        """Get experiment by name."""
        return self._experiments.get(name)

    def list_experiments(self) -> List[str]:
        """List experiment names."""
        return list(self._experiments.keys())

    def get_history(self) -> List[ExperimentResult]:
        """Get experiment history."""
        return list(self._history)


class ChaosVisionProvider(VisionProvider):
    """Vision provider with chaos engineering support."""

    def __init__(
        self,
        provider: VisionProvider,
        chaos_manager: Optional[ChaosManager] = None,
    ) -> None:
        """Initialize provider.

        Args:
            provider: Underlying provider
            chaos_manager: Chaos manager
        """
        self._provider = provider
        self._chaos = chaos_manager or ChaosManager()
        self._latency_injector = LatencyInjector()
        self._error_injector = ErrorInjector()

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"chaos_{self._provider.provider_name}"

    def enable_latency_injection(
        self,
        min_ms: int = 100,
        max_ms: int = 5000,
        probability: float = 0.1,
    ) -> None:
        """Enable latency injection.

        Args:
            min_ms: Minimum latency
            max_ms: Maximum latency
            probability: Injection probability
        """
        self._latency_injector = LatencyInjector(min_ms, max_ms, probability)
        self._latency_injector.enable()

    def disable_latency_injection(self) -> None:
        """Disable latency injection."""
        self._latency_injector.disable()

    def enable_error_injection(
        self,
        probability: float = 0.1,
        error_types: Optional[List[str]] = None,
    ) -> None:
        """Enable error injection.

        Args:
            probability: Error probability
            error_types: Error types to inject
        """
        self._error_injector = ErrorInjector(probability, error_types)
        self._error_injector.enable()

    def disable_error_injection(self) -> None:
        """Disable error injection."""
        self._error_injector.disable()

    def get_chaos_manager(self) -> ChaosManager:
        """Get chaos manager."""
        return self._chaos

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with chaos injection.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        # Check active experiment
        experiment = self._chaos.get_active_experiment()
        if experiment:
            fault = await experiment.process_request()
            if fault and fault.fault_type == FaultType.ERROR:
                # Fault already raised in process_request
                pass

        # Inject latency
        await self._latency_injector.maybe_inject()

        # Inject error
        error = self._error_injector.maybe_inject()
        if error:
            raise error

        return await self._provider.analyze_image(image_data, include_description)


def create_chaos_provider(
    provider: VisionProvider,
    latency_probability: float = 0.0,
    error_probability: float = 0.0,
) -> ChaosVisionProvider:
    """Create chaos vision provider.

    Args:
        provider: Provider to wrap
        latency_probability: Latency injection probability
        error_probability: Error injection probability

    Returns:
        Chaos provider
    """
    chaos_provider = ChaosVisionProvider(provider)

    if latency_probability > 0:
        chaos_provider.enable_latency_injection(probability=latency_probability)

    if error_probability > 0:
        chaos_provider.enable_error_injection(probability=error_probability)

    return chaos_provider


def create_experiment(
    name: str,
    faults: Optional[List[FaultConfig]] = None,
    duration_minutes: int = 5,
    max_impact: float = 50.0,
) -> ExperimentConfig:
    """Create experiment configuration.

    Args:
        name: Experiment name
        faults: Fault configurations
        duration_minutes: Experiment duration
        max_impact: Maximum impact percentage

    Returns:
        Experiment configuration
    """
    return ExperimentConfig(
        name=name,
        faults=faults or [],
        duration=timedelta(minutes=duration_minutes),
        max_impact_percentage=max_impact,
    )
