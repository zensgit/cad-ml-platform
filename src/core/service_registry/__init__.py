"""Service Registry Module.

Provides service discovery:
- Service registration
- Health checking
- Instance management
"""

from src.core.service_registry.core import (
    ServiceStatus,
    HealthCheckType,
    HealthCheckConfig,
    HealthCheckResult,
    ServiceInstance,
    ServiceDefinition,
    Service,
    generate_instance_id,
    ServiceQuery,
    ServiceEvent,
)
from src.core.service_registry.registry import (
    ServiceRegistry,
    InMemoryServiceRegistry,
    HealthChecker,
    HeartbeatManager,
    ServiceClient,
)

__all__ = [
    # Core
    "ServiceStatus",
    "HealthCheckType",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ServiceInstance",
    "ServiceDefinition",
    "Service",
    "generate_instance_id",
    "ServiceQuery",
    "ServiceEvent",
    # Registry
    "ServiceRegistry",
    "InMemoryServiceRegistry",
    "HealthChecker",
    "HeartbeatManager",
    "ServiceClient",
]
