"""Self Healing Module.

Provides automated recovery, self-repair, and system resilience capabilities.
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


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class IssueType(Enum):
    """Types of issues."""

    LATENCY_HIGH = "latency_high"
    ERROR_RATE_HIGH = "error_rate_high"
    MEMORY_PRESSURE = "memory_pressure"
    CONNECTION_POOL_EXHAUSTED = "connection_pool_exhausted"
    CIRCUIT_OPEN = "circuit_open"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED = "rate_limited"
    QUEUE_OVERFLOW = "queue_overflow"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"


class RecoveryAction(Enum):
    """Recovery actions."""

    RESTART_COMPONENT = "restart_component"
    RESET_CIRCUIT_BREAKER = "reset_circuit_breaker"
    CLEAR_CACHE = "clear_cache"
    DRAIN_QUEUE = "drain_queue"
    SCALE_RESOURCES = "scale_resources"
    FAILOVER = "failover"
    ROLLBACK = "rollback"
    THROTTLE_REQUESTS = "throttle_requests"
    SHED_LOAD = "shed_load"
    RECONNECT = "reconnect"


class RecoveryStatus(Enum):
    """Recovery operation status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class HealthCheck:
    """Health check result."""

    component: str
    status: HealthStatus
    message: str
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Issue:
    """Detected issue."""

    issue_id: str
    issue_type: IssueType
    component: str
    severity: float  # 0-1, higher = more severe
    description: str
    detected_at: datetime = field(default_factory=datetime.now)
    auto_recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPlan:
    """Plan for recovering from an issue."""

    plan_id: str
    issue: Issue
    actions: List[RecoveryAction]
    estimated_recovery_time_seconds: float
    confidence: float  # 0-1, likelihood of success
    requires_manual_intervention: bool = False


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""

    plan_id: str
    status: RecoveryStatus
    actions_executed: List[RecoveryAction]
    actions_failed: List[RecoveryAction]
    recovery_time_seconds: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HealingEvent:
    """Record of a healing event."""

    event_id: str
    issue: Issue
    recovery_result: RecoveryResult
    started_at: datetime
    completed_at: datetime
    success: bool


class HealthMonitor:
    """Monitors component health."""

    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        unhealthy_threshold: int = 3,
    ):
        self.check_interval_seconds = check_interval_seconds
        self.unhealthy_threshold = unhealthy_threshold
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._history: Dict[str, deque] = {}
        self._status: Dict[str, HealthStatus] = {}
        self._lock = threading.Lock()

    def register_check(
        self, component: str, check_func: Callable[[], HealthCheck]
    ) -> None:
        """Register a health check."""
        with self._lock:
            self._checks[component] = check_func
            self._history[component] = deque(maxlen=100)
            self._status[component] = HealthStatus.UNKNOWN

    def unregister_check(self, component: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(component, None)
            self._history.pop(component, None)
            self._status.pop(component, None)

    def run_check(self, component: str) -> HealthCheck:
        """Run a single health check."""
        with self._lock:
            check_func = self._checks.get(component)

        if not check_func:
            return HealthCheck(
                component=component,
                status=HealthStatus.UNKNOWN,
                message="No check registered",
            )

        try:
            start = time.time()
            result = check_func()
            result.latency_ms = (time.time() - start) * 1000
        except Exception as e:
            result = HealthCheck(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
            )

        with self._lock:
            if component in self._history:
                self._history[component].append(result)
            self._update_status(component)

        return result

    def run_all_checks(self) -> List[HealthCheck]:
        """Run all health checks."""
        results = []
        with self._lock:
            components = list(self._checks.keys())

        for component in components:
            results.append(self.run_check(component))

        return results

    def _update_status(self, component: str) -> None:
        """Update component status based on history."""
        history = self._history.get(component, [])
        if not history:
            self._status[component] = HealthStatus.UNKNOWN
            return

        recent = list(history)[-self.unhealthy_threshold:]
        unhealthy_count = sum(
            1 for h in recent if h.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL)
        )

        if unhealthy_count >= self.unhealthy_threshold:
            self._status[component] = HealthStatus.UNHEALTHY
        elif unhealthy_count > 0:
            self._status[component] = HealthStatus.DEGRADED
        else:
            self._status[component] = HealthStatus.HEALTHY

    def get_status(self, component: str) -> HealthStatus:
        """Get component status."""
        with self._lock:
            return self._status.get(component, HealthStatus.UNKNOWN)

    def get_all_status(self) -> Dict[str, HealthStatus]:
        """Get status of all components."""
        with self._lock:
            return dict(self._status)

    def get_history(self, component: str, limit: int = 10) -> List[HealthCheck]:
        """Get check history for a component."""
        with self._lock:
            history = self._history.get(component, [])
            return list(history)[-limit:]


class IssueDetector:
    """Detects issues from metrics and health checks."""

    def __init__(self):
        self._thresholds: Dict[str, Dict[str, float]] = {
            "latency_ms": {"warning": 500, "critical": 2000},
            "error_rate": {"warning": 0.05, "critical": 0.2},
            "memory_percent": {"warning": 80, "critical": 95},
            "queue_depth": {"warning": 100, "critical": 500},
        }
        self._detected_issues: Dict[str, Issue] = {}
        self._issue_counter = 0
        self._lock = threading.Lock()

    def detect_issues(
        self,
        metrics: Dict[str, float],
        health_status: Dict[str, HealthStatus],
    ) -> List[Issue]:
        """Detect issues from metrics and health status."""
        issues = []

        # Check metrics
        latency = metrics.get("latency_ms", 0)
        if latency > self._thresholds["latency_ms"]["critical"]:
            issues.append(self._create_issue(
                IssueType.LATENCY_HIGH,
                "system",
                0.9,
                f"Critical latency: {latency:.0f}ms",
            ))
        elif latency > self._thresholds["latency_ms"]["warning"]:
            issues.append(self._create_issue(
                IssueType.LATENCY_HIGH,
                "system",
                0.5,
                f"High latency: {latency:.0f}ms",
            ))

        error_rate = metrics.get("error_rate", 0)
        if error_rate > self._thresholds["error_rate"]["critical"]:
            issues.append(self._create_issue(
                IssueType.ERROR_RATE_HIGH,
                "system",
                0.9,
                f"Critical error rate: {error_rate:.2%}",
            ))
        elif error_rate > self._thresholds["error_rate"]["warning"]:
            issues.append(self._create_issue(
                IssueType.ERROR_RATE_HIGH,
                "system",
                0.6,
                f"High error rate: {error_rate:.2%}",
            ))

        memory = metrics.get("memory_percent", 0)
        if memory > self._thresholds["memory_percent"]["critical"]:
            issues.append(self._create_issue(
                IssueType.MEMORY_PRESSURE,
                "system",
                0.95,
                f"Critical memory usage: {memory:.0f}%",
            ))
        elif memory > self._thresholds["memory_percent"]["warning"]:
            issues.append(self._create_issue(
                IssueType.MEMORY_PRESSURE,
                "system",
                0.6,
                f"High memory usage: {memory:.0f}%",
            ))

        # Check component health
        for component, status in health_status.items():
            if status == HealthStatus.UNHEALTHY:
                issues.append(self._create_issue(
                    IssueType.PROVIDER_UNAVAILABLE,
                    component,
                    0.8,
                    f"Component unhealthy: {component}",
                ))
            elif status == HealthStatus.CRITICAL:
                issues.append(self._create_issue(
                    IssueType.PROVIDER_UNAVAILABLE,
                    component,
                    1.0,
                    f"Component critical: {component}",
                ))

        # Track detected issues
        with self._lock:
            for issue in issues:
                self._detected_issues[issue.issue_id] = issue

        return issues

    def _create_issue(
        self,
        issue_type: IssueType,
        component: str,
        severity: float,
        description: str,
    ) -> Issue:
        """Create an issue."""
        with self._lock:
            self._issue_counter += 1
            issue_id = f"issue_{self._issue_counter}_{int(time.time())}"

        return Issue(
            issue_id=issue_id,
            issue_type=issue_type,
            component=component,
            severity=severity,
            description=description,
        )

    def get_active_issues(self) -> List[Issue]:
        """Get active issues."""
        with self._lock:
            return list(self._detected_issues.values())

    def resolve_issue(self, issue_id: str) -> bool:
        """Mark an issue as resolved."""
        with self._lock:
            return self._detected_issues.pop(issue_id, None) is not None


class RecoveryPlanner:
    """Plans recovery actions for issues."""

    def __init__(self):
        self._action_map: Dict[IssueType, List[RecoveryAction]] = {
            IssueType.LATENCY_HIGH: [
                RecoveryAction.THROTTLE_REQUESTS,
                RecoveryAction.SCALE_RESOURCES,
            ],
            IssueType.ERROR_RATE_HIGH: [
                RecoveryAction.RESET_CIRCUIT_BREAKER,
                RecoveryAction.FAILOVER,
            ],
            IssueType.MEMORY_PRESSURE: [
                RecoveryAction.CLEAR_CACHE,
                RecoveryAction.SHED_LOAD,
            ],
            IssueType.CONNECTION_POOL_EXHAUSTED: [
                RecoveryAction.RECONNECT,
                RecoveryAction.RESTART_COMPONENT,
            ],
            IssueType.CIRCUIT_OPEN: [
                RecoveryAction.RESET_CIRCUIT_BREAKER,
            ],
            IssueType.PROVIDER_UNAVAILABLE: [
                RecoveryAction.FAILOVER,
                RecoveryAction.RECONNECT,
            ],
            IssueType.RATE_LIMITED: [
                RecoveryAction.THROTTLE_REQUESTS,
            ],
            IssueType.QUEUE_OVERFLOW: [
                RecoveryAction.DRAIN_QUEUE,
                RecoveryAction.SHED_LOAD,
            ],
            IssueType.RESOURCE_EXHAUSTION: [
                RecoveryAction.SCALE_RESOURCES,
                RecoveryAction.RESTART_COMPONENT,
            ],
            IssueType.DEPENDENCY_FAILURE: [
                RecoveryAction.FAILOVER,
                RecoveryAction.ROLLBACK,
            ],
        }
        self._plan_counter = 0

    def create_plan(self, issue: Issue) -> RecoveryPlan:
        """Create a recovery plan for an issue."""
        self._plan_counter += 1
        plan_id = f"plan_{self._plan_counter}_{int(time.time())}"

        actions = self._action_map.get(issue.issue_type, [])
        if not actions:
            actions = [RecoveryAction.RESTART_COMPONENT]

        # Estimate recovery time based on actions
        time_estimates = {
            RecoveryAction.RESTART_COMPONENT: 30,
            RecoveryAction.RESET_CIRCUIT_BREAKER: 5,
            RecoveryAction.CLEAR_CACHE: 10,
            RecoveryAction.DRAIN_QUEUE: 20,
            RecoveryAction.SCALE_RESOURCES: 60,
            RecoveryAction.FAILOVER: 15,
            RecoveryAction.ROLLBACK: 45,
            RecoveryAction.THROTTLE_REQUESTS: 5,
            RecoveryAction.SHED_LOAD: 10,
            RecoveryAction.RECONNECT: 10,
        }

        estimated_time = sum(time_estimates.get(a, 30) for a in actions)

        # Confidence based on issue type and severity
        confidence = max(0.5, 1.0 - issue.severity * 0.3)

        return RecoveryPlan(
            plan_id=plan_id,
            issue=issue,
            actions=actions,
            estimated_recovery_time_seconds=estimated_time,
            confidence=confidence,
            requires_manual_intervention=issue.severity > 0.9,
        )


class RecoveryExecutor:
    """Executes recovery actions."""

    def __init__(self):
        self._action_handlers: Dict[RecoveryAction, Callable[[], bool]] = {}
        self._lock = threading.Lock()

    def register_handler(
        self, action: RecoveryAction, handler: Callable[[], bool]
    ) -> None:
        """Register a handler for an action."""
        with self._lock:
            self._action_handlers[action] = handler

    def execute_plan(self, plan: RecoveryPlan) -> RecoveryResult:
        """Execute a recovery plan."""
        start_time = time.time()
        executed = []
        failed = []

        for action in plan.actions:
            with self._lock:
                handler = self._action_handlers.get(action)

            if handler:
                try:
                    success = handler()
                    if success:
                        executed.append(action)
                    else:
                        failed.append(action)
                except Exception:
                    failed.append(action)
            else:
                # No handler, simulate success
                executed.append(action)

            # Small delay between actions
            time.sleep(0.1)

        recovery_time = time.time() - start_time
        overall_success = len(failed) == 0

        return RecoveryResult(
            plan_id=plan.plan_id,
            status=RecoveryStatus.SUCCESS if overall_success else RecoveryStatus.FAILED,
            actions_executed=executed,
            actions_failed=failed,
            recovery_time_seconds=recovery_time,
            message="Recovery completed" if overall_success else f"Failed actions: {failed}",
        )


class SelfHealingEngine:
    """Main self-healing engine."""

    def __init__(
        self,
        auto_heal: bool = True,
        max_recovery_attempts: int = 3,
    ):
        self.auto_heal = auto_heal
        self.max_recovery_attempts = max_recovery_attempts

        self._health_monitor = HealthMonitor()
        self._issue_detector = IssueDetector()
        self._recovery_planner = RecoveryPlanner()
        self._recovery_executor = RecoveryExecutor()

        self._healing_events: deque = deque(maxlen=1000)
        self._recovery_attempts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def register_health_check(
        self, component: str, check_func: Callable[[], HealthCheck]
    ) -> None:
        """Register a health check."""
        self._health_monitor.register_check(component, check_func)

    def register_recovery_handler(
        self, action: RecoveryAction, handler: Callable[[], bool]
    ) -> None:
        """Register a recovery handler."""
        self._recovery_executor.register_handler(action, handler)

    def check_health(self) -> Dict[str, HealthStatus]:
        """Run health checks."""
        self._health_monitor.run_all_checks()
        return self._health_monitor.get_all_status()

    def detect_issues(self, metrics: Dict[str, float]) -> List[Issue]:
        """Detect issues from metrics."""
        health_status = self._health_monitor.get_all_status()
        return self._issue_detector.detect_issues(metrics, health_status)

    def heal(self, issue: Issue) -> Optional[HealingEvent]:
        """Attempt to heal an issue."""
        # Check retry limit
        with self._lock:
            attempts = self._recovery_attempts.get(issue.issue_id, 0)
            if attempts >= self.max_recovery_attempts:
                return None
            self._recovery_attempts[issue.issue_id] = attempts + 1

        # Create and execute plan
        start_time = datetime.now()
        plan = self._recovery_planner.create_plan(issue)

        if plan.requires_manual_intervention and not issue.auto_recoverable:
            return None

        result = self._recovery_executor.execute_plan(plan)
        end_time = datetime.now()

        # Create healing event
        event = HealingEvent(
            event_id=f"heal_{int(time.time() * 1000)}",
            issue=issue,
            recovery_result=result,
            started_at=start_time,
            completed_at=end_time,
            success=result.status == RecoveryStatus.SUCCESS,
        )

        self._healing_events.append(event)

        # Resolve issue if successful
        if event.success:
            self._issue_detector.resolve_issue(issue.issue_id)
            with self._lock:
                self._recovery_attempts.pop(issue.issue_id, None)

        return event

    def auto_heal_check(self, metrics: Dict[str, float]) -> List[HealingEvent]:
        """Perform auto-healing check and heal if needed."""
        if not self.auto_heal:
            return []

        events = []
        issues = self.detect_issues(metrics)

        for issue in issues:
            if issue.auto_recoverable and issue.severity > 0.5:
                event = self.heal(issue)
                if event:
                    events.append(event)

        return events

    def get_healing_history(self, limit: int = 100) -> List[HealingEvent]:
        """Get healing event history."""
        return list(self._healing_events)[-limit:]

    def get_active_issues(self) -> List[Issue]:
        """Get active unresolved issues."""
        return self._issue_detector.get_active_issues()

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        status = self._health_monitor.get_all_status()
        issues = self._issue_detector.get_active_issues()

        healthy = sum(1 for s in status.values() if s == HealthStatus.HEALTHY)
        degraded = sum(1 for s in status.values() if s == HealthStatus.DEGRADED)
        unhealthy = sum(1 for s in status.values() if s in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL))

        return {
            "overall_status": (
                HealthStatus.HEALTHY.value if unhealthy == 0 and degraded == 0
                else HealthStatus.DEGRADED.value if unhealthy == 0
                else HealthStatus.UNHEALTHY.value
            ),
            "components": {
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
                "total": len(status),
            },
            "active_issues": len(issues),
            "recent_healing_events": len(list(self._healing_events)[-10:]),
        }


class SelfHealingVisionProvider(VisionProvider):
    """Vision provider with self-healing capabilities."""

    def __init__(
        self,
        provider: VisionProvider,
        healing_engine: Optional[SelfHealingEngine] = None,
    ):
        self._provider = provider
        self.healing_engine = healing_engine or SelfHealingEngine()

        # Metrics for issue detection
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0
        self._lock = threading.Lock()

        # Register default health check
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""

        def provider_check() -> HealthCheck:
            with self._lock:
                if self._request_count == 0:
                    return HealthCheck(
                        component=self._provider.provider_name,
                        status=HealthStatus.HEALTHY,
                        message="No requests yet",
                    )

                error_rate = self._error_count / self._request_count
                avg_latency = self._total_latency / self._request_count

            if error_rate > 0.2:
                status = HealthStatus.UNHEALTHY
            elif error_rate > 0.05:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            return HealthCheck(
                component=self._provider.provider_name,
                status=status,
                message=f"Error rate: {error_rate:.2%}, Avg latency: {avg_latency:.0f}ms",
            )

        self.healing_engine.register_health_check(
            self._provider.provider_name, provider_check
        )

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"self_healing_{self._provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True, **kwargs: Any
    ) -> VisionDescription:
        """Analyze image with self-healing."""
        start_time = time.time()

        with self._lock:
            self._request_count += 1

        try:
            result = await self._provider.analyze_image(
                image_data, include_description, **kwargs
            )
            return result
        except Exception as e:
            with self._lock:
                self._error_count += 1

            # Trigger healing check
            metrics = self._get_metrics()
            self.healing_engine.auto_heal_check(metrics)

            raise
        finally:
            latency = (time.time() - start_time) * 1000
            with self._lock:
                self._total_latency += latency

    def _get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        with self._lock:
            error_rate = self._error_count / max(1, self._request_count)
            avg_latency = self._total_latency / max(1, self._request_count)

        return {
            "error_rate": error_rate,
            "latency_ms": avg_latency,
            "request_count": self._request_count,
        }

    def check_health(self) -> Dict[str, HealthStatus]:
        """Check health of all components."""
        return self.healing_engine.check_health()

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        return self.healing_engine.get_health_summary()

    def get_active_issues(self) -> List[Issue]:
        """Get active issues."""
        return self.healing_engine.get_active_issues()


# Factory functions
def create_health_monitor(
    check_interval_seconds: float = 30.0,
    unhealthy_threshold: int = 3,
) -> HealthMonitor:
    """Create a health monitor."""
    return HealthMonitor(
        check_interval_seconds=check_interval_seconds,
        unhealthy_threshold=unhealthy_threshold,
    )


def create_self_healing_engine(
    auto_heal: bool = True,
    max_recovery_attempts: int = 3,
) -> SelfHealingEngine:
    """Create a self-healing engine."""
    return SelfHealingEngine(
        auto_heal=auto_heal,
        max_recovery_attempts=max_recovery_attempts,
    )


def create_self_healing_provider(
    provider: VisionProvider,
    healing_engine: Optional[SelfHealingEngine] = None,
) -> SelfHealingVisionProvider:
    """Create a self-healing vision provider."""
    return SelfHealingVisionProvider(provider, healing_engine)


def create_health_check(
    component: str,
    status: HealthStatus,
    message: str,
) -> HealthCheck:
    """Create a health check result."""
    return HealthCheck(
        component=component,
        status=status,
        message=message,
    )
