"""
Health checking for model serving.

Provides:
- Model health monitoring
- Worker health checks
- System resource monitoring
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


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ModelHealth:
    """Health information for a model."""
    model_name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "consecutive_failures": self.consecutive_failures,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "message": self.message,
        }


@dataclass
class SystemHealth:
    """System-level health information."""
    status: HealthStatus = HealthStatus.UNKNOWN
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_percent: Optional[float] = None
    models_loaded: int = 0
    active_requests: int = 0
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "status": self.status.value,
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "models_loaded": self.models_loaded,
            "active_requests": self.active_requests,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }
        if self.gpu_memory_percent is not None:
            result["gpu_memory_percent"] = round(self.gpu_memory_percent, 1)
        return result


class HealthChecker:
    """
    Health checker for model serving infrastructure.

    Monitors:
    - Model health (latency, errors)
    - Worker health
    - System resources
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        failure_threshold: int = 3,
        latency_threshold_ms: float = 1000.0,
        error_rate_threshold: float = 0.1,
    ):
        """
        Initialize health checker.

        Args:
            check_interval: Interval between health checks in seconds
            failure_threshold: Consecutive failures before unhealthy
            latency_threshold_ms: Latency threshold for degraded status
            error_rate_threshold: Error rate threshold for degraded status
        """
        self._check_interval = check_interval
        self._failure_threshold = failure_threshold
        self._latency_threshold = latency_threshold_ms
        self._error_rate_threshold = error_rate_threshold

        self._model_health: Dict[str, ModelHealth] = {}
        self._health_checks: Dict[str, Callable[[], bool]] = {}
        self._start_time = time.time()
        self._running = False
        self._check_task: Optional[asyncio.Task] = None

    def register_model(
        self,
        model_name: str,
        check_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Register a model for health monitoring.

        Args:
            model_name: Model name
            check_fn: Optional custom health check function
        """
        self._model_health[model_name] = ModelHealth(model_name=model_name)
        if check_fn:
            self._health_checks[model_name] = check_fn

    def unregister_model(self, model_name: str) -> None:
        """Unregister a model."""
        self._model_health.pop(model_name, None)
        self._health_checks.pop(model_name, None)

    def record_request(
        self,
        model_name: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """
        Record a request result for health tracking.

        Args:
            model_name: Model name
            success: Whether request succeeded
            latency_ms: Request latency
        """
        if model_name not in self._model_health:
            self._model_health[model_name] = ModelHealth(model_name=model_name)

        health = self._model_health[model_name]

        if success:
            health.consecutive_failures = 0
            health.last_success = datetime.utcnow()
        else:
            health.consecutive_failures += 1

        # Update rolling averages (exponential moving average)
        alpha = 0.1
        health.avg_latency_ms = alpha * latency_ms + (1 - alpha) * health.avg_latency_ms
        error_value = 0.0 if success else 1.0
        health.error_rate = alpha * error_value + (1 - alpha) * health.error_rate

        # Update status
        self._update_model_status(model_name)

    def _update_model_status(self, model_name: str) -> None:
        """Update model health status based on metrics."""
        health = self._model_health.get(model_name)
        if not health:
            return

        # Check consecutive failures
        if health.consecutive_failures >= self._failure_threshold:
            health.status = HealthStatus.UNHEALTHY
            health.message = f"Consecutive failures: {health.consecutive_failures}"
            return

        # Check error rate
        if health.error_rate >= self._error_rate_threshold:
            health.status = HealthStatus.DEGRADED
            health.message = f"High error rate: {health.error_rate:.2%}"
            return

        # Check latency
        if health.avg_latency_ms >= self._latency_threshold:
            health.status = HealthStatus.DEGRADED
            health.message = f"High latency: {health.avg_latency_ms:.0f}ms"
            return

        health.status = HealthStatus.HEALTHY
        health.message = ""

    async def check_model(self, model_name: str) -> ModelHealth:
        """
        Perform health check on a model.

        Args:
            model_name: Model name

        Returns:
            ModelHealth
        """
        if model_name not in self._model_health:
            self._model_health[model_name] = ModelHealth(model_name=model_name)

        health = self._model_health[model_name]
        health.last_check = datetime.utcnow()

        # Run custom health check if registered
        check_fn = self._health_checks.get(model_name)
        if check_fn:
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    result = await check_fn()
                else:
                    result = check_fn()

                if result:
                    health.status = HealthStatus.HEALTHY
                    health.consecutive_failures = 0
                else:
                    health.consecutive_failures += 1
                    self._update_model_status(model_name)

            except Exception as e:
                health.consecutive_failures += 1
                health.message = str(e)
                self._update_model_status(model_name)

        return health

    async def check_all(self) -> Dict[str, ModelHealth]:
        """
        Check health of all registered models.

        Returns:
            Dict of model name to ModelHealth
        """
        results = {}
        for model_name in list(self._model_health.keys()):
            results[model_name] = await self.check_model(model_name)
        return results

    def get_model_health(self, model_name: str) -> Optional[ModelHealth]:
        """Get health status for a model."""
        return self._model_health.get(model_name)

    def get_system_health(self) -> SystemHealth:
        """
        Get overall system health.

        Returns:
            SystemHealth
        """
        health = SystemHealth(
            uptime_seconds=time.time() - self._start_time,
            models_loaded=len(self._model_health),
        )

        # Try to get system metrics
        try:
            import psutil

            health.cpu_percent = psutil.cpu_percent()
            health.memory_percent = psutil.virtual_memory().percent

        except ImportError:
            pass

        # Try to get GPU metrics
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                health.gpu_memory_percent = (allocated / total) * 100

        except (ImportError, RuntimeError):
            pass

        # Determine overall status
        unhealthy_count = sum(
            1 for h in self._model_health.values()
            if h.status == HealthStatus.UNHEALTHY
        )
        degraded_count = sum(
            1 for h in self._model_health.values()
            if h.status == HealthStatus.DEGRADED
        )

        if unhealthy_count > 0:
            health.status = HealthStatus.UNHEALTHY
        elif degraded_count > 0 or health.memory_percent > 90 or health.cpu_percent > 90:
            health.status = HealthStatus.DEGRADED
        else:
            health.status = HealthStatus.HEALTHY

        return health

    def get_all_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health report.

        Returns:
            Health report dict
        """
        return {
            "system": self.get_system_health().to_dict(),
            "models": {
                name: health.to_dict()
                for name, health in self._model_health.items()
            },
            "summary": {
                "total_models": len(self._model_health),
                "healthy": sum(
                    1 for h in self._model_health.values()
                    if h.status == HealthStatus.HEALTHY
                ),
                "degraded": sum(
                    1 for h in self._model_health.values()
                    if h.status == HealthStatus.DEGRADED
                ),
                "unhealthy": sum(
                    1 for h in self._model_health.values()
                    if h.status == HealthStatus.UNHEALTHY
                ),
            },
        }

    async def start_background_checks(self) -> None:
        """Start background health check loop."""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("Started background health checks")

    async def stop_background_checks(self) -> None:
        """Stop background health check loop."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped background health checks")

    async def _check_loop(self) -> None:
        """Background check loop."""
        while self._running:
            try:
                await self.check_all()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            await asyncio.sleep(self._check_interval)
