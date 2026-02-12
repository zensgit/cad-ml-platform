"""Canary Release Manager.

Provides progressive rollout with automatic rollback based on
health metrics and error rates.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CanaryStatus(str, Enum):
    """Canary release status."""

    PENDING = "pending"  # Not yet started
    ROLLING_OUT = "rolling_out"  # Progressive rollout in progress
    PAUSED = "paused"  # Manually paused
    COMPLETED = "completed"  # Successfully rolled out to 100%
    ROLLED_BACK = "rolled_back"  # Automatically or manually rolled back
    FAILED = "failed"  # Rollout failed


class RolloutPhase(str, Enum):
    """Canary rollout phases."""

    CANARY = "canary"  # Initial canary (1-5%)
    EARLY_ADOPTER = "early_adopter"  # Early adopter (5-20%)
    PROGRESSIVE = "progressive"  # Progressive rollout (20-50%)
    MAJORITY = "majority"  # Majority rollout (50-80%)
    FULL = "full"  # Full rollout (100%)


@dataclass
class RolloutStep:
    """A step in the canary rollout."""

    percentage: int
    phase: RolloutPhase
    duration_minutes: int = 15  # How long to wait at this step
    min_requests: int = 100  # Minimum requests before proceeding
    max_error_rate: float = 0.01  # Maximum error rate threshold
    max_latency_p99_ms: int = 5000  # Maximum P99 latency threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "percentage": self.percentage,
            "phase": self.phase.value,
            "duration_minutes": self.duration_minutes,
            "min_requests": self.min_requests,
            "max_error_rate": self.max_error_rate,
            "max_latency_p99_ms": self.max_latency_p99_ms,
        }


@dataclass
class CanaryMetrics:
    """Metrics collected during canary rollout."""

    requests: int = 0
    errors: int = 0
    latency_sum_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    success_rate: float = 1.0
    collected_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.errors / self.requests if self.requests > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        return self.latency_sum_ms / self.requests if self.requests > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requests": self.requests,
            "errors": self.errors,
            "error_rate": self.error_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "success_rate": self.success_rate,
            "collected_at": self.collected_at.isoformat(),
        }


@dataclass
class CanaryConfig:
    """Configuration for a canary release."""

    version: str
    steps: List[RolloutStep] = field(default_factory=list)
    auto_rollback: bool = True
    auto_promote: bool = True
    bake_time_minutes: int = 30  # Final bake time before completion

    def __post_init__(self):
        """Initialize default steps if not provided."""
        if not self.steps:
            self.steps = [
                RolloutStep(1, RolloutPhase.CANARY, 10, 50, 0.02, 3000),
                RolloutStep(5, RolloutPhase.CANARY, 15, 100, 0.01, 3000),
                RolloutStep(10, RolloutPhase.EARLY_ADOPTER, 15, 200, 0.01, 3000),
                RolloutStep(25, RolloutPhase.PROGRESSIVE, 20, 500, 0.01, 4000),
                RolloutStep(50, RolloutPhase.MAJORITY, 30, 1000, 0.005, 4000),
                RolloutStep(75, RolloutPhase.MAJORITY, 30, 2000, 0.005, 5000),
                RolloutStep(100, RolloutPhase.FULL, 0, 0, 0.005, 5000),
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
            "auto_rollback": self.auto_rollback,
            "auto_promote": self.auto_promote,
            "bake_time_minutes": self.bake_time_minutes,
        }


@dataclass
class CanaryRelease:
    """A canary release in progress."""

    id: str
    config: CanaryConfig
    status: CanaryStatus = CanaryStatus.PENDING
    current_step_index: int = 0
    current_percentage: int = 0
    baseline_version: str = ""

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    step_started_at: Optional[datetime] = None

    # Metrics
    canary_metrics: CanaryMetrics = field(default_factory=CanaryMetrics)
    baseline_metrics: CanaryMetrics = field(default_factory=CanaryMetrics)
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)

    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def current_step(self) -> Optional[RolloutStep]:
        """Get current rollout step."""
        if 0 <= self.current_step_index < len(self.config.steps):
            return self.config.steps[self.current_step_index]
        return None

    @property
    def current_phase(self) -> Optional[RolloutPhase]:
        """Get current rollout phase."""
        step = self.current_step
        return step.phase if step else None

    @property
    def progress_percentage(self) -> float:
        """Get overall progress percentage."""
        if not self.config.steps:
            return 0.0
        return (self.current_step_index / len(self.config.steps)) * 100

    def add_event(self, event_type: str, message: str, **kwargs) -> None:
        """Add an event to the release history."""
        self.events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "message": message,
            **kwargs,
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "current_percentage": self.current_percentage,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "baseline_version": self.baseline_version,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress_percentage": self.progress_percentage,
            "canary_metrics": self.canary_metrics.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "events": self.events[-20:],  # Last 20 events
        }


class CanaryManager:
    """Manages canary releases.

    Features:
    - Progressive rollout with configurable steps
    - Automatic health monitoring
    - Auto-rollback on threshold breach
    - Comparison with baseline metrics
    """

    def __init__(
        self,
        get_metrics_fn: Optional[Callable[[str], CanaryMetrics]] = None,
        set_traffic_fn: Optional[Callable[[str, int], bool]] = None,
        rollback_fn: Optional[Callable[[str], bool]] = None,
    ):
        """Initialize Canary manager.

        Args:
            get_metrics_fn: Function to get metrics (version) -> CanaryMetrics
            set_traffic_fn: Function to set traffic split (version, percentage) -> success
            rollback_fn: Function to perform rollback (version) -> success
        """
        self._releases: Dict[str, CanaryRelease] = {}
        self._active_release: Optional[str] = None
        self._monitor_task: Optional[asyncio.Task] = None

        self._get_metrics_fn = get_metrics_fn or self._default_get_metrics
        self._set_traffic_fn = set_traffic_fn or self._default_set_traffic
        self._rollback_fn = rollback_fn or self._default_rollback

    def _default_get_metrics(self, version: str) -> CanaryMetrics:
        """Default metrics getter (returns empty metrics)."""
        return CanaryMetrics()

    def _default_set_traffic(self, version: str, percentage: int) -> bool:
        """Default traffic setter (logs only)."""
        logger.info(f"Setting traffic for {version} to {percentage}%")
        return True

    def _default_rollback(self, version: str) -> bool:
        """Default rollback (logs only)."""
        logger.info(f"Rolling back version {version}")
        return True

    async def create_release(
        self,
        release_id: str,
        version: str,
        baseline_version: str = "",
        config: Optional[CanaryConfig] = None,
    ) -> CanaryRelease:
        """Create a new canary release.

        Args:
            release_id: Unique release ID
            version: Version to roll out
            baseline_version: Current production version
            config: Rollout configuration

        Returns:
            Created canary release
        """
        if release_id in self._releases:
            raise ValueError(f"Release {release_id} already exists")

        release = CanaryRelease(
            id=release_id,
            config=config or CanaryConfig(version=version),
            baseline_version=baseline_version,
        )
        release.config.version = version
        self._releases[release_id] = release
        release.add_event("created", f"Canary release created for version {version}")
        logger.info(f"Created canary release: {release_id}")
        return release

    def get_release(self, release_id: str) -> Optional[CanaryRelease]:
        """Get release by ID."""
        return self._releases.get(release_id)

    def get_active_release(self) -> Optional[CanaryRelease]:
        """Get currently active release."""
        if self._active_release:
            return self._releases.get(self._active_release)
        return None

    async def start_release(self, release_id: str) -> bool:
        """Start a canary release.

        Args:
            release_id: Release ID to start

        Returns:
            True if started successfully
        """
        release = self._releases.get(release_id)
        if not release:
            return False

        if release.status != CanaryStatus.PENDING:
            logger.warning(f"Cannot start release {release_id} in status {release.status}")
            return False

        if self._active_release:
            logger.warning(f"Another release {self._active_release} is already active")
            return False

        release.status = CanaryStatus.ROLLING_OUT
        release.started_at = datetime.utcnow()
        release.step_started_at = datetime.utcnow()
        self._active_release = release_id

        # Start at first step
        first_step = release.config.steps[0] if release.config.steps else None
        if first_step:
            release.current_percentage = first_step.percentage
            self._set_traffic_fn(release.config.version, first_step.percentage)
            release.add_event(
                "started",
                f"Started rollout at {first_step.percentage}%",
                phase=first_step.phase.value,
            )

        # Start monitoring task
        self._start_monitoring()
        logger.info(f"Started canary release: {release_id}")
        return True

    async def pause_release(self, release_id: str) -> bool:
        """Pause a canary release."""
        release = self._releases.get(release_id)
        if not release or release.status != CanaryStatus.ROLLING_OUT:
            return False

        release.status = CanaryStatus.PAUSED
        release.add_event("paused", f"Rollout paused at {release.current_percentage}%")
        logger.info(f"Paused canary release: {release_id}")
        return True

    async def resume_release(self, release_id: str) -> bool:
        """Resume a paused canary release."""
        release = self._releases.get(release_id)
        if not release or release.status != CanaryStatus.PAUSED:
            return False

        release.status = CanaryStatus.ROLLING_OUT
        release.step_started_at = datetime.utcnow()
        release.add_event("resumed", f"Rollout resumed at {release.current_percentage}%")
        self._start_monitoring()
        logger.info(f"Resumed canary release: {release_id}")
        return True

    async def promote_release(self, release_id: str) -> bool:
        """Manually promote to next step."""
        release = self._releases.get(release_id)
        if not release or release.status not in (
            CanaryStatus.ROLLING_OUT,
            CanaryStatus.PAUSED,
        ):
            return False

        return await self._advance_step(release)

    async def rollback_release(self, release_id: str, reason: str = "") -> bool:
        """Rollback a canary release.

        Args:
            release_id: Release ID to rollback
            reason: Reason for rollback

        Returns:
            True if rollback successful
        """
        release = self._releases.get(release_id)
        if not release:
            return False

        # Set traffic back to 0 for canary version
        self._set_traffic_fn(release.config.version, 0)
        self._rollback_fn(release.config.version)

        release.status = CanaryStatus.ROLLED_BACK
        release.completed_at = datetime.utcnow()
        release.add_event("rolled_back", f"Rollback completed: {reason}")

        if self._active_release == release_id:
            self._active_release = None
            self._stop_monitoring()

        logger.info(f"Rolled back canary release: {release_id}, reason: {reason}")
        return True

    async def _advance_step(self, release: CanaryRelease) -> bool:
        """Advance to next rollout step."""
        next_index = release.current_step_index + 1

        if next_index >= len(release.config.steps):
            # Rollout complete
            release.status = CanaryStatus.COMPLETED
            release.completed_at = datetime.utcnow()
            release.add_event("completed", "Canary rollout completed successfully")

            if self._active_release == release.id:
                self._active_release = None
                self._stop_monitoring()

            logger.info(f"Completed canary release: {release.id}")
            return True

        # Move to next step
        release.current_step_index = next_index
        release.step_started_at = datetime.utcnow()
        next_step = release.config.steps[next_index]
        release.current_percentage = next_step.percentage

        self._set_traffic_fn(release.config.version, next_step.percentage)
        release.add_event(
            "promoted",
            f"Advanced to {next_step.percentage}%",
            phase=next_step.phase.value,
        )

        logger.info(
            f"Advanced release {release.id} to step {next_index} ({next_step.percentage}%)"
        )
        return True

    def _check_thresholds(self, release: CanaryRelease) -> tuple[bool, str]:
        """Check if current metrics are within thresholds.

        Returns:
            (passed, reason) tuple
        """
        step = release.current_step
        if not step:
            return True, ""

        metrics = release.canary_metrics

        # Check error rate
        if metrics.error_rate > step.max_error_rate:
            return False, f"Error rate {metrics.error_rate:.2%} exceeds threshold {step.max_error_rate:.2%}"

        # Check latency
        if metrics.latency_p99_ms > step.max_latency_p99_ms:
            return False, f"P99 latency {metrics.latency_p99_ms}ms exceeds threshold {step.max_latency_p99_ms}ms"

        # Compare with baseline if available
        baseline = release.baseline_metrics
        if baseline.requests > 0:
            # Error rate should not be significantly worse than baseline
            if metrics.error_rate > baseline.error_rate * 2:
                return False, f"Error rate {metrics.error_rate:.2%} is 2x worse than baseline {baseline.error_rate:.2%}"

        return True, ""

    def _should_promote(self, release: CanaryRelease) -> bool:
        """Check if release should be promoted to next step."""
        step = release.current_step
        if not step:
            return False

        # Check minimum requests
        if release.canary_metrics.requests < step.min_requests:
            return False

        # Check duration
        if release.step_started_at:
            elapsed_minutes = (datetime.utcnow() - release.step_started_at).total_seconds() / 60
            if elapsed_minutes < step.duration_minutes:
                return False

        return True

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._active_release:
            try:
                release = self._releases.get(self._active_release)
                if not release or release.status != CanaryStatus.ROLLING_OUT:
                    break

                # Collect metrics
                release.canary_metrics = self._get_metrics_fn(release.config.version)
                if release.baseline_version:
                    release.baseline_metrics = self._get_metrics_fn(release.baseline_version)

                # Store metrics history
                release.metrics_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "canary": release.canary_metrics.to_dict(),
                    "baseline": release.baseline_metrics.to_dict(),
                })
                # Keep last 100 entries
                release.metrics_history = release.metrics_history[-100:]

                # Check thresholds
                passed, reason = self._check_thresholds(release)
                if not passed and release.config.auto_rollback:
                    logger.warning(f"Threshold breach: {reason}")
                    await self.rollback_release(release.id, reason)
                    break

                # Check for auto-promotion
                if passed and release.config.auto_promote and self._should_promote(release):
                    await self._advance_step(release)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in canary monitor loop: {e}")
                await asyncio.sleep(30)

    def _start_monitoring(self) -> None:
        """Start the monitoring task."""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    def _stop_monitoring(self) -> None:
        """Stop the monitoring task."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

    def get_status(self) -> Dict[str, Any]:
        """Get overall canary status."""
        active = self.get_active_release()
        return {
            "active_release": active.to_dict() if active else None,
            "total_releases": len(self._releases),
            "releases": {
                rid: r.status.value for rid, r in self._releases.items()
            },
        }


# Global manager instance
_canary_manager: Optional[CanaryManager] = None


def get_canary_manager() -> CanaryManager:
    """Get global canary manager."""
    global _canary_manager
    if _canary_manager is None:
        _canary_manager = CanaryManager()
    return _canary_manager
