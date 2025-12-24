"""Deployment strategies for Vision Provider system.

This module provides deployment patterns including:
- Blue-green deployments
- Canary releases
- Rolling updates
- A/B testing
- Feature toggles for deployment
"""

import asyncio
import hashlib
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

from .base import VisionDescription, VisionProvider


class DeploymentStrategy(Enum):
    """Deployment strategy type."""

    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    AB_TEST = "ab_test"
    SHADOW = "shadow"


class DeploymentPhase(Enum):
    """Deployment phase."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"


class EnvironmentType(Enum):
    """Environment type."""

    BLUE = "blue"
    GREEN = "green"
    CANARY = "canary"
    PRODUCTION = "production"
    STAGING = "staging"


class TrafficSplitMethod(Enum):
    """Traffic split method."""

    PERCENTAGE = "percentage"
    HEADER_BASED = "header_based"
    USER_ID_HASH = "user_id_hash"
    COOKIE_BASED = "cookie_based"
    RANDOM = "random"


@dataclass
class DeploymentVersion:
    """Deployment version information."""

    version: str
    provider: VisionProvider
    deployed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "provider": self.provider.provider_name,
            "deployed_at": self.deployed_at.isoformat(),
            "metadata": dict(self.metadata),
            "health_status": self.health_status,
        }


@dataclass
class DeploymentConfig:
    """Deployment configuration."""

    strategy: DeploymentStrategy
    canary_percentage: float = 10.0
    rollout_steps: List[float] = field(default_factory=lambda: [10, 25, 50, 100])
    validation_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    auto_rollback: bool = True
    health_check_interval: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    max_failure_rate: float = 0.1
    min_success_rate: float = 0.9


@dataclass
class DeploymentMetrics:
    """Deployment metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    version_requests: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    def record_request(
        self,
        version: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Record a request.

        Args:
            version: Version that handled request
            success: Whether request succeeded
            latency_ms: Request latency
        """
        self.total_requests += 1
        self.total_latency_ms += latency_ms

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.version_requests[version] = self.version_requests.get(version, 0) + 1


@dataclass
class DeploymentState:
    """Current deployment state."""

    phase: DeploymentPhase = DeploymentPhase.PENDING
    current_version: Optional[str] = None
    target_version: Optional[str] = None
    traffic_percentage: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "current_version": self.current_version,
            "target_version": self.target_version,
            "traffic_percentage": self.traffic_percentage,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "message": self.message,
        }


class TrafficRouter:
    """Routes traffic between versions."""

    def __init__(
        self,
        method: TrafficSplitMethod = TrafficSplitMethod.PERCENTAGE,
    ) -> None:
        """Initialize router.

        Args:
            method: Traffic split method
        """
        self._method = method
        self._versions: Dict[str, DeploymentVersion] = {}
        self._weights: Dict[str, float] = {}
        self._lock = threading.Lock()

    def add_version(
        self,
        version: DeploymentVersion,
        weight: float = 0.0,
    ) -> None:
        """Add version with weight.

        Args:
            version: Deployment version
            weight: Traffic weight (0-100)
        """
        with self._lock:
            self._versions[version.version] = version
            self._weights[version.version] = weight

    def remove_version(self, version: str) -> None:
        """Remove version.

        Args:
            version: Version string
        """
        with self._lock:
            self._versions.pop(version, None)
            self._weights.pop(version, None)

    def set_weight(self, version: str, weight: float) -> None:
        """Set version weight.

        Args:
            version: Version string
            weight: Traffic weight (0-100)
        """
        with self._lock:
            if version in self._versions:
                self._weights[version] = weight

    def route(
        self,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[DeploymentVersion]:
        """Route request to a version.

        Args:
            request_context: Optional request context

        Returns:
            Selected version or None
        """
        with self._lock:
            if not self._versions:
                return None

            if self._method == TrafficSplitMethod.PERCENTAGE:
                return self._route_by_percentage()
            elif self._method == TrafficSplitMethod.USER_ID_HASH:
                user_id = (request_context or {}).get("user_id", "")
                return self._route_by_hash(user_id)
            elif self._method == TrafficSplitMethod.RANDOM:
                return random.choice(list(self._versions.values()))
            else:
                return self._route_by_percentage()

    def _route_by_percentage(self) -> Optional[DeploymentVersion]:
        """Route by percentage weights."""
        total_weight = sum(self._weights.values())
        if total_weight == 0:
            return list(self._versions.values())[0] if self._versions else None

        rand = random.uniform(0, total_weight)
        cumulative = 0.0

        for version, weight in self._weights.items():
            cumulative += weight
            if rand <= cumulative:
                return self._versions[version]

        return list(self._versions.values())[-1]

    def _route_by_hash(self, key: str) -> Optional[DeploymentVersion]:
        """Route by hash of key."""
        if not self._versions:
            return None

        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        total_weight = sum(self._weights.values())

        if total_weight == 0:
            return list(self._versions.values())[0]

        target = hash_val % int(total_weight)
        cumulative = 0.0

        for version, weight in self._weights.items():
            cumulative += weight
            if target < cumulative:
                return self._versions[version]

        return list(self._versions.values())[-1]

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        with self._lock:
            return dict(self._weights)


class DeploymentManager:
    """Manages deployments."""

    def __init__(
        self,
        config: Optional[DeploymentConfig] = None,
    ) -> None:
        """Initialize manager.

        Args:
            config: Deployment configuration
        """
        self._config = config or DeploymentConfig(strategy=DeploymentStrategy.CANARY)
        self._versions: Dict[str, DeploymentVersion] = {}
        self._active_version: Optional[str] = None
        self._state = DeploymentState()
        self._metrics = DeploymentMetrics()
        self._router = TrafficRouter()
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[DeploymentState], None]] = []

    def register_version(
        self,
        version: str,
        provider: VisionProvider,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeploymentVersion:
        """Register a new version.

        Args:
            version: Version string
            provider: Vision provider
            metadata: Version metadata

        Returns:
            Deployment version
        """
        deployment = DeploymentVersion(
            version=version,
            provider=provider,
            metadata=metadata or {},
        )

        with self._lock:
            self._versions[version] = deployment

            if self._active_version is None:
                self._active_version = version
                self._router.add_version(deployment, 100.0)

        return deployment

    def start_deployment(
        self,
        target_version: str,
        strategy: Optional[DeploymentStrategy] = None,
    ) -> bool:
        """Start deployment to target version.

        Args:
            target_version: Target version
            strategy: Override strategy

        Returns:
            True if deployment started
        """
        with self._lock:
            if target_version not in self._versions:
                return False

            if self._state.phase == DeploymentPhase.IN_PROGRESS:
                return False

            self._state = DeploymentState(
                phase=DeploymentPhase.IN_PROGRESS,
                current_version=self._active_version,
                target_version=target_version,
                traffic_percentage=0.0,
                started_at=datetime.now(),
                message="Deployment started",
            )

            # Add target to router with initial weight
            target = self._versions[target_version]
            effective_strategy = strategy or self._config.strategy

            if effective_strategy == DeploymentStrategy.BLUE_GREEN:
                # Immediate switch
                self._router.add_version(target, 100.0)
                if self._active_version:
                    self._router.set_weight(self._active_version, 0.0)
                self._state.traffic_percentage = 100.0
            elif effective_strategy == DeploymentStrategy.CANARY:
                # Start with canary percentage
                self._router.add_version(target, self._config.canary_percentage)
                if self._active_version:
                    self._router.set_weight(
                        self._active_version,
                        100.0 - self._config.canary_percentage,
                    )
                self._state.traffic_percentage = self._config.canary_percentage
            else:
                # Rolling - start with first step
                initial = self._config.rollout_steps[0] if self._config.rollout_steps else 10.0
                self._router.add_version(target, initial)
                if self._active_version:
                    self._router.set_weight(self._active_version, 100.0 - initial)
                self._state.traffic_percentage = initial

        self._notify_callbacks()
        return True

    def advance_deployment(self) -> bool:
        """Advance deployment to next step.

        Returns:
            True if advanced
        """
        with self._lock:
            if self._state.phase != DeploymentPhase.IN_PROGRESS:
                return False

            target_version = self._state.target_version
            if not target_version:
                return False

            # Find next step
            current_pct = self._state.traffic_percentage
            next_pct = 100.0

            for step in self._config.rollout_steps:
                if step > current_pct:
                    next_pct = step
                    break

            # Update weights
            self._router.set_weight(target_version, next_pct)
            if self._active_version:
                self._router.set_weight(self._active_version, 100.0 - next_pct)

            self._state.traffic_percentage = next_pct

            if next_pct >= 100.0:
                self._state.phase = DeploymentPhase.VALIDATING
                self._state.message = "Validating deployment"

        self._notify_callbacks()
        return True

    def complete_deployment(self) -> bool:
        """Complete deployment.

        Returns:
            True if completed
        """
        with self._lock:
            if self._state.phase not in [
                DeploymentPhase.IN_PROGRESS,
                DeploymentPhase.VALIDATING,
            ]:
                return False

            target_version = self._state.target_version
            if not target_version:
                return False

            # Remove old version from routing
            if self._active_version and self._active_version != target_version:
                self._router.remove_version(self._active_version)

            # Set target as active
            self._active_version = target_version
            self._router.set_weight(target_version, 100.0)

            self._state.phase = DeploymentPhase.COMPLETED
            self._state.completed_at = datetime.now()
            self._state.traffic_percentage = 100.0
            self._state.message = "Deployment completed"

        self._notify_callbacks()
        return True

    def rollback(self) -> bool:
        """Rollback deployment.

        Returns:
            True if rolled back
        """
        with self._lock:
            if self._state.phase not in [
                DeploymentPhase.IN_PROGRESS,
                DeploymentPhase.VALIDATING,
            ]:
                return False

            previous_version = self._state.current_version
            target_version = self._state.target_version

            if not previous_version:
                return False

            # Restore traffic to previous version
            self._router.set_weight(previous_version, 100.0)
            if target_version:
                self._router.remove_version(target_version)

            self._state.phase = DeploymentPhase.ROLLING_BACK
            self._state.traffic_percentage = 0.0
            self._state.message = "Rolled back to previous version"

        self._notify_callbacks()
        return True

    def get_state(self) -> DeploymentState:
        """Get current deployment state."""
        with self._lock:
            return DeploymentState(
                phase=self._state.phase,
                current_version=self._state.current_version,
                target_version=self._state.target_version,
                traffic_percentage=self._state.traffic_percentage,
                started_at=self._state.started_at,
                completed_at=self._state.completed_at,
                message=self._state.message,
            )

    def get_metrics(self) -> DeploymentMetrics:
        """Get deployment metrics."""
        with self._lock:
            return DeploymentMetrics(
                total_requests=self._metrics.total_requests,
                successful_requests=self._metrics.successful_requests,
                failed_requests=self._metrics.failed_requests,
                total_latency_ms=self._metrics.total_latency_ms,
                version_requests=dict(self._metrics.version_requests),
            )

    def should_auto_rollback(self) -> bool:
        """Check if auto-rollback should trigger.

        Returns:
            True if should rollback
        """
        if not self._config.auto_rollback:
            return False

        metrics = self.get_metrics()
        return metrics.failure_rate > self._config.max_failure_rate

    def route_request(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[VisionProvider]:
        """Route request to appropriate provider.

        Args:
            context: Request context

        Returns:
            Provider to use or None
        """
        version = self._router.route(context)
        return version.provider if version else None

    def record_request(
        self,
        version: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Record request metrics.

        Args:
            version: Version that handled request
            success: Whether request succeeded
            latency_ms: Request latency
        """
        with self._lock:
            self._metrics.record_request(version, success, latency_ms)

    def add_callback(self, callback: Callable[[DeploymentState], None]) -> None:
        """Add state change callback.

        Args:
            callback: Callback function
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self) -> None:
        """Notify callbacks of state change."""
        state = self.get_state()
        for callback in self._callbacks:
            try:
                callback(state)
            except Exception:
                pass


class BlueGreenDeployment:
    """Blue-green deployment manager."""

    def __init__(self) -> None:
        """Initialize manager."""
        self._blue: Optional[DeploymentVersion] = None
        self._green: Optional[DeploymentVersion] = None
        self._active: EnvironmentType = EnvironmentType.BLUE
        self._lock = threading.Lock()

    def set_blue(self, version: DeploymentVersion) -> None:
        """Set blue environment.

        Args:
            version: Deployment version
        """
        with self._lock:
            self._blue = version

    def set_green(self, version: DeploymentVersion) -> None:
        """Set green environment.

        Args:
            version: Deployment version
        """
        with self._lock:
            self._green = version

    def switch(self) -> EnvironmentType:
        """Switch active environment.

        Returns:
            New active environment
        """
        with self._lock:
            if self._active == EnvironmentType.BLUE:
                self._active = EnvironmentType.GREEN
            else:
                self._active = EnvironmentType.BLUE
            return self._active

    def get_active(self) -> Optional[DeploymentVersion]:
        """Get active version.

        Returns:
            Active deployment version
        """
        with self._lock:
            if self._active == EnvironmentType.BLUE:
                return self._blue
            return self._green

    def get_inactive(self) -> Optional[DeploymentVersion]:
        """Get inactive version.

        Returns:
            Inactive deployment version
        """
        with self._lock:
            if self._active == EnvironmentType.BLUE:
                return self._green
            return self._blue

    def get_active_environment(self) -> EnvironmentType:
        """Get active environment type."""
        with self._lock:
            return self._active


class CanaryRelease:
    """Canary release manager."""

    def __init__(
        self,
        initial_percentage: float = 5.0,
        max_percentage: float = 100.0,
        step_percentage: float = 10.0,
    ) -> None:
        """Initialize manager.

        Args:
            initial_percentage: Initial canary percentage
            max_percentage: Maximum canary percentage
            step_percentage: Step increment
        """
        self._initial = initial_percentage
        self._max = max_percentage
        self._step = step_percentage
        self._current_percentage = 0.0
        self._stable: Optional[DeploymentVersion] = None
        self._canary: Optional[DeploymentVersion] = None
        self._lock = threading.Lock()

    def set_stable(self, version: DeploymentVersion) -> None:
        """Set stable version.

        Args:
            version: Stable version
        """
        with self._lock:
            self._stable = version

    def start_canary(self, version: DeploymentVersion) -> None:
        """Start canary release.

        Args:
            version: Canary version
        """
        with self._lock:
            self._canary = version
            self._current_percentage = self._initial

    def increase_traffic(self) -> float:
        """Increase canary traffic.

        Returns:
            New canary percentage
        """
        with self._lock:
            self._current_percentage = min(
                self._max,
                self._current_percentage + self._step,
            )
            return self._current_percentage

    def promote(self) -> None:
        """Promote canary to stable."""
        with self._lock:
            self._stable = self._canary
            self._canary = None
            self._current_percentage = 0.0

    def abort(self) -> None:
        """Abort canary release."""
        with self._lock:
            self._canary = None
            self._current_percentage = 0.0

    def get_canary_percentage(self) -> float:
        """Get current canary percentage."""
        with self._lock:
            return self._current_percentage

    def route(self) -> Optional[DeploymentVersion]:
        """Route request.

        Returns:
            Version to use
        """
        with self._lock:
            if self._canary and random.uniform(0, 100) < self._current_percentage:
                return self._canary
            return self._stable


class RollingUpdate:
    """Rolling update manager."""

    def __init__(
        self,
        batch_size: int = 1,
        max_unavailable: int = 1,
    ) -> None:
        """Initialize manager.

        Args:
            batch_size: Number of instances to update at once
            max_unavailable: Maximum unavailable instances
        """
        self._batch_size = batch_size
        self._max_unavailable = max_unavailable
        self._instances: List[DeploymentVersion] = []
        self._updated: Set[int] = set()
        self._lock = threading.Lock()

    def set_instances(self, instances: List[DeploymentVersion]) -> None:
        """Set instances to update.

        Args:
            instances: List of instances
        """
        with self._lock:
            self._instances = list(instances)
            self._updated.clear()

    def get_next_batch(self) -> List[int]:
        """Get next batch to update.

        Returns:
            List of instance indices
        """
        with self._lock:
            batch = []
            for i, _ in enumerate(self._instances):
                if i not in self._updated and len(batch) < self._batch_size:
                    batch.append(i)
            return batch

    def mark_updated(self, index: int) -> None:
        """Mark instance as updated.

        Args:
            index: Instance index
        """
        with self._lock:
            self._updated.add(index)

    def is_complete(self) -> bool:
        """Check if update is complete.

        Returns:
            True if all instances updated
        """
        with self._lock:
            return len(self._updated) == len(self._instances)

    def progress(self) -> float:
        """Get update progress.

        Returns:
            Progress percentage
        """
        with self._lock:
            if not self._instances:
                return 100.0
            return len(self._updated) / len(self._instances) * 100


class DeploymentVisionProvider(VisionProvider):
    """Vision provider with deployment management."""

    def __init__(
        self,
        manager: DeploymentManager,
    ) -> None:
        """Initialize provider.

        Args:
            manager: Deployment manager
        """
        self._manager = manager

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "deployment_managed"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image using deployment-managed provider.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        provider = self._manager.route_request()
        if not provider:
            raise RuntimeError("No provider available")

        state = self._manager.get_state()
        version = state.target_version or state.current_version or "unknown"

        start_time = time.time()

        try:
            result = await provider.analyze_image(image_data, include_description)
            latency_ms = (time.time() - start_time) * 1000
            self._manager.record_request(version, True, latency_ms)

            # Check for auto-rollback
            if self._manager.should_auto_rollback():
                self._manager.rollback()

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._manager.record_request(version, False, latency_ms)

            if self._manager.should_auto_rollback():
                self._manager.rollback()

            raise


def create_blue_green_provider(
    blue_provider: VisionProvider,
    green_provider: VisionProvider,
    active: EnvironmentType = EnvironmentType.BLUE,
) -> DeploymentVisionProvider:
    """Create blue-green deployment provider.

    Args:
        blue_provider: Blue environment provider
        green_provider: Green environment provider
        active: Initially active environment

    Returns:
        Deployment provider
    """
    config = DeploymentConfig(strategy=DeploymentStrategy.BLUE_GREEN)
    manager = DeploymentManager(config)

    manager.register_version("blue", blue_provider)
    manager.register_version("green", green_provider)

    if active == EnvironmentType.GREEN:
        manager.start_deployment("green")
        manager.complete_deployment()

    return DeploymentVisionProvider(manager)


def create_canary_provider(
    stable_provider: VisionProvider,
    canary_provider: VisionProvider,
    canary_percentage: float = 10.0,
) -> DeploymentVisionProvider:
    """Create canary deployment provider.

    Args:
        stable_provider: Stable provider
        canary_provider: Canary provider
        canary_percentage: Canary traffic percentage

    Returns:
        Deployment provider
    """
    config = DeploymentConfig(
        strategy=DeploymentStrategy.CANARY,
        canary_percentage=canary_percentage,
    )
    manager = DeploymentManager(config)

    manager.register_version("stable", stable_provider)
    manager.register_version("canary", canary_provider)
    manager.start_deployment("canary")

    return DeploymentVisionProvider(manager)
