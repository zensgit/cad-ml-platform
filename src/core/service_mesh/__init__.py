"""Service Mesh Integration Module.

Provides service mesh capabilities:
- Service discovery
- Load balancing
- Health aggregation
"""

from src.core.service_mesh.discovery import (
    ServiceStatus,
    ServiceInstance,
    ServiceDefinition,
    ServiceRegistry,
    InMemoryServiceRegistry,
    ServiceDiscovery,
    ServiceRegistrar,
)
from src.core.service_mesh.load_balancer import (
    LoadBalancerStats,
    LoadBalancer,
    RoundRobinBalancer,
    WeightedRoundRobinBalancer,
    LeastConnectionsBalancer,
    RandomBalancer,
    ConsistentHashBalancer,
    AdaptiveBalancer,
    LoadBalancerFactory,
)
from src.core.service_mesh.health import (
    HealthStatus,
    HealthCheckResult,
    AggregatedHealth,
    HealthCheck,
    HTTPHealthCheck,
    TCPHealthCheck,
    FunctionHealthCheck,
    HealthAggregator,
    DependencyHealthTracker,
)

__all__ = [
    # Discovery
    "ServiceStatus",
    "ServiceInstance",
    "ServiceDefinition",
    "ServiceRegistry",
    "InMemoryServiceRegistry",
    "ServiceDiscovery",
    "ServiceRegistrar",
    # Load Balancer
    "LoadBalancerStats",
    "LoadBalancer",
    "RoundRobinBalancer",
    "WeightedRoundRobinBalancer",
    "LeastConnectionsBalancer",
    "RandomBalancer",
    "ConsistentHashBalancer",
    "AdaptiveBalancer",
    "LoadBalancerFactory",
    # Health
    "HealthStatus",
    "HealthCheckResult",
    "AggregatedHealth",
    "HealthCheck",
    "HTTPHealthCheck",
    "TCPHealthCheck",
    "FunctionHealthCheck",
    "HealthAggregator",
    "DependencyHealthTracker",
]
