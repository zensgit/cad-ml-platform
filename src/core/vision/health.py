"""Provider health monitoring for vision analysis.

Provides:
- Continuous health monitoring
- Automated health checks
- Status dashboards
- Alerting integration
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status of a provider."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class AlertSeverity(Enum):
    """Severity level for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetrics:
    """Health metrics for a provider."""

    provider: str
    status: HealthStatus
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    uptime_percentage: float = 100.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    success_rate: float = 100.0
    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    consecutive_failures: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "uptime_percentage": self.uptime_percentage,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "success_rate": self.success_rate,
            "total_checks": self.total_checks,
            "successful_checks": self.successful_checks,
            "failed_checks": self.failed_checks,
            "consecutive_failures": self.consecutive_failures,
            "error_message": self.error_message,
        }


@dataclass
class HealthAlert:
    """A health alert notification."""

    alert_id: str
    provider: str
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Optional[HealthMetrics] = None
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "provider": self.provider,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""

    interval_seconds: float = 60.0  # How often to check
    timeout_seconds: float = 10.0  # Timeout for each check
    failure_threshold: int = 3  # Failures before unhealthy
    degraded_threshold: int = 1  # Failures before degraded
    recovery_threshold: int = 2  # Successes needed to recover
    latency_degraded_ms: float = 5000.0  # Latency threshold for degraded
    latency_unhealthy_ms: float = 10000.0  # Latency threshold for unhealthy
    success_rate_degraded: float = 0.95  # Success rate for degraded
    success_rate_unhealthy: float = 0.80  # Success rate for unhealthy
    enabled: bool = True


@dataclass
class HealthDashboard:
    """Dashboard view of all provider health."""

    generated_at: datetime
    providers: Dict[str, HealthMetrics]
    overall_status: HealthStatus
    active_alerts: List[HealthAlert]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "overall_status": self.overall_status.value,
            "providers": {name: metrics.to_dict() for name, metrics in self.providers.items()},
            "active_alerts": [alert.to_dict() for alert in self.active_alerts],
            "summary": self.summary,
        }


# Alert callback type
AlertCallback = Callable[[HealthAlert], None]


class HealthMonitor:
    """
    Monitors health of vision providers.

    Features:
    - Periodic health checks
    - Latency tracking with percentiles
    - Success rate monitoring
    - Automated alerting
    - Dashboard generation
    """

    def __init__(
        self,
        config: Optional[HealthCheckConfig] = None,
    ):
        """
        Initialize health monitor.

        Args:
            config: Health check configuration
        """
        self._config = config or HealthCheckConfig()
        self._providers: Dict[str, VisionProvider] = {}
        self._metrics: Dict[str, HealthMetrics] = {}
        self._latency_samples: Dict[str, List[float]] = {}
        self._alerts: List[HealthAlert] = []
        self._alert_callbacks: List[AlertCallback] = []
        self._running = False
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._alert_counter = 0

    def register_provider(
        self,
        provider: VisionProvider,
        custom_config: Optional[HealthCheckConfig] = None,
    ) -> None:
        """
        Register a provider for health monitoring.

        Args:
            provider: VisionProvider to monitor
            custom_config: Optional custom config for this provider
        """
        name = provider.provider_name
        self._providers[name] = provider
        self._metrics[name] = HealthMetrics(
            provider=name,
            status=HealthStatus.UNKNOWN,
        )
        self._latency_samples[name] = []
        logger.info(f"Registered provider for health monitoring: {name}")

    def unregister_provider(self, provider_name: str) -> bool:
        """
        Unregister a provider from health monitoring.

        Args:
            provider_name: Name of provider to remove

        Returns:
            True if provider was found and removed
        """
        if provider_name in self._providers:
            del self._providers[provider_name]
            del self._metrics[provider_name]
            del self._latency_samples[provider_name]
            logger.info(f"Unregistered provider from health monitoring: {provider_name}")
            return True
        return False

    def add_alert_callback(self, callback: AlertCallback) -> None:
        """Add a callback for alert notifications."""
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: AlertCallback) -> bool:
        """Remove an alert callback."""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)
            return True
        return False

    async def start(self) -> None:
        """Start the health monitoring loop."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")

    async def stop(self) -> None:
        """Stop the health monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Background loop for periodic health checks."""
        while self._running:
            try:
                await self.check_all_providers()
                await asyncio.sleep(self._config.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)  # Brief delay before retry

    async def check_provider(self, provider_name: str) -> HealthMetrics:
        """
        Perform health check on a specific provider.

        Args:
            provider_name: Name of provider to check

        Returns:
            Updated HealthMetrics
        """
        if provider_name not in self._providers:
            raise ValueError(f"Provider not registered: {provider_name}")

        provider = self._providers[provider_name]
        metrics = self._metrics[provider_name]

        # Create minimal test image (1x1 PNG)
        test_image = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        start_time = time.time()
        metrics.last_check = datetime.now()
        metrics.total_checks += 1

        try:
            await asyncio.wait_for(
                provider.analyze_image(test_image, include_description=False),
                timeout=self._config.timeout_seconds,
            )
            latency_ms = (time.time() - start_time) * 1000

            # Record success
            metrics.successful_checks += 1
            metrics.last_success = datetime.now()
            metrics.consecutive_failures = 0
            metrics.error_message = None

            # Update latency samples
            self._latency_samples[provider_name].append(latency_ms)
            if len(self._latency_samples[provider_name]) > 100:
                self._latency_samples[provider_name].pop(0)

            # Calculate latency stats
            samples = self._latency_samples[provider_name]
            metrics.avg_latency_ms = sum(samples) / len(samples)
            sorted_samples = sorted(samples)
            if len(sorted_samples) >= 20:
                p95_idx = int(len(sorted_samples) * 0.95)
                p99_idx = int(len(sorted_samples) * 0.99)
                metrics.p95_latency_ms = sorted_samples[p95_idx]
                metrics.p99_latency_ms = sorted_samples[min(p99_idx, len(sorted_samples) - 1)]

        except asyncio.TimeoutError:
            metrics.failed_checks += 1
            metrics.last_failure = datetime.now()
            metrics.consecutive_failures += 1
            metrics.error_message = "Health check timeout"

        except Exception as e:
            metrics.failed_checks += 1
            metrics.last_failure = datetime.now()
            metrics.consecutive_failures += 1
            metrics.error_message = str(e)

        # Calculate success rate
        if metrics.total_checks > 0:
            metrics.success_rate = metrics.successful_checks / metrics.total_checks

        # Calculate uptime
        if metrics.total_checks > 0:
            metrics.uptime_percentage = (metrics.successful_checks / metrics.total_checks) * 100

        # Determine status
        old_status = metrics.status
        metrics.status = self._determine_status(metrics)

        # Check for status changes and generate alerts
        if old_status != metrics.status:
            await self._handle_status_change(metrics, old_status)

        return metrics

    async def check_all_providers(self) -> Dict[str, HealthMetrics]:
        """
        Check health of all registered providers.

        Returns:
            Dictionary of provider name to HealthMetrics
        """
        tasks = [self.check_provider(name) for name in self._providers.keys()]
        await asyncio.gather(*tasks, return_exceptions=True)
        return dict(self._metrics)

    def _determine_status(self, metrics: HealthMetrics) -> HealthStatus:
        """Determine health status based on metrics."""
        # Check consecutive failures
        if metrics.consecutive_failures >= self._config.failure_threshold:
            return HealthStatus.UNHEALTHY

        if metrics.consecutive_failures >= self._config.degraded_threshold:
            return HealthStatus.DEGRADED

        # Check success rate
        if metrics.total_checks >= 10:
            if metrics.success_rate < self._config.success_rate_unhealthy:
                return HealthStatus.UNHEALTHY
            if metrics.success_rate < self._config.success_rate_degraded:
                return HealthStatus.DEGRADED

        # Check latency
        if metrics.avg_latency_ms >= self._config.latency_unhealthy_ms:
            return HealthStatus.UNHEALTHY
        if metrics.avg_latency_ms >= self._config.latency_degraded_ms:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    async def _handle_status_change(
        self,
        metrics: HealthMetrics,
        old_status: HealthStatus,
    ) -> None:
        """Handle status change and generate alerts."""
        # Determine severity
        if metrics.status == HealthStatus.UNHEALTHY:
            severity = AlertSeverity.CRITICAL
        elif metrics.status == HealthStatus.DEGRADED:
            severity = AlertSeverity.WARNING
        elif old_status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED):
            severity = AlertSeverity.INFO  # Recovery
        else:
            return  # No alert needed

        # Create alert
        self._alert_counter += 1
        alert = HealthAlert(
            alert_id=f"health-{self._alert_counter}",
            provider=metrics.provider,
            severity=severity,
            message=self._generate_alert_message(metrics, old_status),
            metrics=metrics,
        )

        self._alerts.append(alert)

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        logger.warning(f"Health alert: {alert.severity.value} - {alert.message}")

    def _generate_alert_message(
        self,
        metrics: HealthMetrics,
        old_status: HealthStatus,
    ) -> str:
        """Generate alert message for status change."""
        if metrics.status == HealthStatus.UNHEALTHY:
            if metrics.error_message:
                return f"Provider {metrics.provider} is UNHEALTHY: " f"{metrics.error_message}"
            return (
                f"Provider {metrics.provider} is UNHEALTHY: "
                f"{metrics.consecutive_failures} consecutive failures"
            )

        elif metrics.status == HealthStatus.DEGRADED:
            if metrics.avg_latency_ms >= self._config.latency_degraded_ms:
                return (
                    f"Provider {metrics.provider} is DEGRADED: "
                    f"high latency ({metrics.avg_latency_ms:.0f}ms)"
                )
            return (
                f"Provider {metrics.provider} is DEGRADED: "
                f"success rate {metrics.success_rate:.1%}"
            )

        elif metrics.status == HealthStatus.HEALTHY:
            return f"Provider {metrics.provider} has RECOVERED"

        return f"Provider {metrics.provider} status changed to {metrics.status.value}"

    def get_metrics(self, provider_name: str) -> Optional[HealthMetrics]:
        """Get current metrics for a provider."""
        return self._metrics.get(provider_name)

    def get_all_metrics(self) -> Dict[str, HealthMetrics]:
        """Get all provider metrics."""
        return dict(self._metrics)

    def get_alerts(
        self,
        provider: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[HealthAlert]:
        """
        Get alerts with optional filtering.

        Args:
            provider: Filter by provider name
            severity: Filter by severity
            active_only: Only unresolved alerts
            limit: Maximum alerts to return

        Returns:
            List of matching alerts
        """
        alerts = self._alerts

        if provider:
            alerts = [a for a in alerts if a.provider == provider]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if active_only:
            alerts = [a for a in alerts if not a.resolved]

        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                return True
        return False

    def get_dashboard(self) -> HealthDashboard:
        """
        Generate a health dashboard.

        Returns:
            HealthDashboard with current status
        """
        # Determine overall status
        statuses = [m.status for m in self._metrics.values()]
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.UNKNOWN

        # Get active alerts
        active_alerts = [a for a in self._alerts if not a.resolved]

        # Calculate summary
        total_providers = len(self._metrics)
        healthy_count = sum(1 for m in self._metrics.values() if m.status == HealthStatus.HEALTHY)
        avg_success_rate = (
            sum(m.success_rate for m in self._metrics.values()) / total_providers
            if total_providers > 0
            else 0
        )

        summary = {
            "total_providers": total_providers,
            "healthy_count": healthy_count,
            "degraded_count": sum(
                1 for m in self._metrics.values() if m.status == HealthStatus.DEGRADED
            ),
            "unhealthy_count": sum(
                1 for m in self._metrics.values() if m.status == HealthStatus.UNHEALTHY
            ),
            "avg_success_rate": avg_success_rate,
            "active_alerts": len(active_alerts),
        }

        return HealthDashboard(
            generated_at=datetime.now(),
            providers=dict(self._metrics),
            overall_status=overall,
            active_alerts=active_alerts,
            summary=summary,
        )


class HealthAwareVisionProvider:
    """
    Wrapper that adds health awareness to a VisionProvider.

    Automatically updates health metrics on each request.
    """

    def __init__(
        self,
        provider: VisionProvider,
        health_monitor: HealthMonitor,
    ):
        """
        Initialize health-aware provider.

        Args:
            provider: The underlying vision provider
            health_monitor: HealthMonitor instance
        """
        self._provider = provider
        self._monitor = health_monitor

        # Register with monitor
        if provider.provider_name not in self._monitor._metrics:
            self._monitor.register_provider(provider)

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """
        Analyze image and update health metrics.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description

        Returns:
            VisionDescription with analysis results
        """
        provider_name = self._provider.provider_name
        metrics = self._monitor._metrics[provider_name]

        start_time = time.time()
        metrics.total_checks += 1

        try:
            result = await self._provider.analyze_image(image_data, include_description)
            latency_ms = (time.time() - start_time) * 1000

            # Record success
            metrics.successful_checks += 1
            metrics.last_success = datetime.now()
            metrics.consecutive_failures = 0
            metrics.error_message = None

            # Update latency
            self._monitor._latency_samples[provider_name].append(latency_ms)
            if len(self._monitor._latency_samples[provider_name]) > 100:
                self._monitor._latency_samples[provider_name].pop(0)

            samples = self._monitor._latency_samples[provider_name]
            metrics.avg_latency_ms = sum(samples) / len(samples)

            # Update success rate
            metrics.success_rate = metrics.successful_checks / metrics.total_checks
            metrics.uptime_percentage = metrics.success_rate * 100

            # Update status
            metrics.status = self._monitor._determine_status(metrics)

            return result

        except Exception as e:
            metrics.failed_checks += 1
            metrics.last_failure = datetime.now()
            metrics.consecutive_failures += 1
            metrics.error_message = str(e)

            # Update metrics
            if metrics.total_checks > 0:
                metrics.success_rate = metrics.successful_checks / metrics.total_checks
                metrics.uptime_percentage = metrics.success_rate * 100

            # Update status
            old_status = metrics.status
            metrics.status = self._monitor._determine_status(metrics)

            # Generate alert if status changed
            if old_status != metrics.status:
                asyncio.create_task(self._monitor._handle_status_change(metrics, old_status))

            raise

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name

    @property
    def health_monitor(self) -> HealthMonitor:
        """Get the health monitor."""
        return self._monitor

    def get_health(self) -> HealthMetrics:
        """Get current health metrics."""
        return self._monitor._metrics[self._provider.provider_name]


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """
    Get the global health monitor instance.

    Returns:
        HealthMonitor singleton
    """
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor


def create_health_aware_provider(
    provider: VisionProvider,
    health_monitor: Optional[HealthMonitor] = None,
) -> HealthAwareVisionProvider:
    """
    Factory to create a health-aware provider wrapper.

    Args:
        provider: The underlying vision provider
        health_monitor: Optional health monitor

    Returns:
        HealthAwareVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> monitored = create_health_aware_provider(provider)
        >>> await monitored.health_monitor.start()
        >>> result = await monitored.analyze_image(image_bytes)
        >>> print(f"Health: {monitored.get_health().status.value}")
    """
    monitor = health_monitor or get_health_monitor()
    return HealthAwareVisionProvider(provider=provider, health_monitor=monitor)
