"""Health Check System for CAD ML Platform.

Provides:
- Deep health checks for all dependencies
- Self-healing mechanisms
- Dependency status tracking
- Kubernetes probes integration
"""

from src.core.health.checker import (
    HealthChecker,
    HealthCheck,
    HealthStatus,
    DependencyHealth,
    HealthCheckResult,
    get_health_checker,
)
from src.core.health.probes import (
    LivenessProbe,
    ReadinessProbe,
    StartupProbe,
    ProbeResult,
)
from src.core.health.self_healing import (
    SelfHealer,
    HealingAction,
    HealingStrategy,
    CircuitBreakerHealer,
)

__all__ = [
    # Checker
    "HealthChecker",
    "HealthCheck",
    "HealthStatus",
    "DependencyHealth",
    "HealthCheckResult",
    "get_health_checker",
    # Probes
    "LivenessProbe",
    "ReadinessProbe",
    "StartupProbe",
    "ProbeResult",
    # Self-healing
    "SelfHealer",
    "HealingAction",
    "HealingStrategy",
    "CircuitBreakerHealer",
]
