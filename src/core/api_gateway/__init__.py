"""API Gateway Module.

Provides API gateway functionality:
- Request routing
- Load balancing
- Middleware support
- Error handling
"""

from src.core.api_gateway.routing import (
    MatchType,
    RouteMatch,
    Route,
    Request,
    RouteMatcher,
    PathMatcher,
    HeaderMatcher,
    QueryMatcher,
    CompositeRouteMatcher,
    Router,
    RouteGroup,
    create_router_from_config,
)
from src.core.api_gateway.loadbalancer import (
    BackendState,
    Backend,
    BackendPool,
    LoadBalancer,
    RoundRobinBalancer,
    WeightedRoundRobinBalancer,
    LeastConnectionsBalancer,
    WeightedLeastConnectionsBalancer,
    RandomBalancer,
    WeightedRandomBalancer,
    IPHashBalancer,
    AdaptiveBalancer,
    LoadBalancerConfig,
    create_load_balancer,
)
from src.core.api_gateway.gateway import (
    GatewayErrorCode,
    Response,
    GatewayContext,
    Middleware,
    LoggingMiddleware,
    RequestIdMiddleware,
    TimeoutMiddleware,
    RetryMiddleware,
    BackendCaller,
    MockBackendCaller,
    Gateway,
    GatewayStats,
)

__all__ = [
    # Routing
    "MatchType",
    "RouteMatch",
    "Route",
    "Request",
    "RouteMatcher",
    "PathMatcher",
    "HeaderMatcher",
    "QueryMatcher",
    "CompositeRouteMatcher",
    "Router",
    "RouteGroup",
    "create_router_from_config",
    # Load Balancer
    "BackendState",
    "Backend",
    "BackendPool",
    "LoadBalancer",
    "RoundRobinBalancer",
    "WeightedRoundRobinBalancer",
    "LeastConnectionsBalancer",
    "WeightedLeastConnectionsBalancer",
    "RandomBalancer",
    "WeightedRandomBalancer",
    "IPHashBalancer",
    "AdaptiveBalancer",
    "LoadBalancerConfig",
    "create_load_balancer",
    # Gateway
    "GatewayErrorCode",
    "Response",
    "GatewayContext",
    "Middleware",
    "LoggingMiddleware",
    "RequestIdMiddleware",
    "TimeoutMiddleware",
    "RetryMiddleware",
    "BackendCaller",
    "MockBackendCaller",
    "Gateway",
    "GatewayStats",
]
