"""
API Gateway for Vision Provider.

This module provides API gateway patterns including:
- Request routing and load balancing
- API versioning
- Request/response transformation
- API composition
- Protocol translation
- API aggregation

Phase 10 Feature.
"""

import asyncio
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from urllib.parse import parse_qs, urlparse

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


# ============================================================================
# Gateway Enums
# ============================================================================


class HttpMethod(Enum):
    """HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"
    LEAST_LATENCY = "least_latency"


class ProtocolType(Enum):
    """Protocol types."""

    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    GRAPHQL = "graphql"


class ResponseStatus(Enum):
    """Response status codes."""

    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    INTERNAL_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


# ============================================================================
# Gateway Data Classes
# ============================================================================


@dataclass
class GatewayRequest:
    """API Gateway request."""

    method: HttpMethod
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    client_ip: str = ""
    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate request ID if not provided."""
        if not self.request_id:
            self.request_id = hashlib.md5(
                f"{self.path}{self.timestamp.isoformat()}".encode()
            ).hexdigest()[:16]


@dataclass
class GatewayResponse:
    """API Gateway response."""

    status: ResponseStatus
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    request_id: str = ""
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if response is successful."""
        return 200 <= self.status.value < 300


@dataclass
class RouteConfig:
    """Route configuration."""

    path_pattern: str
    methods: List[HttpMethod]
    handler: str  # Handler name or service name
    version: str = "v1"
    timeout_ms: int = 30000
    rate_limit: Optional[int] = None
    auth_required: bool = False
    cache_ttl_seconds: int = 0
    transform_request: Optional[str] = None
    transform_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceEndpoint:
    """Backend service endpoint."""

    host: str
    port: int
    protocol: ProtocolType = ProtocolType.HTTP
    weight: int = 1
    healthy: bool = True
    active_connections: int = 0
    avg_latency_ms: float = 0.0
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        """Get full URL."""
        return f"{self.protocol.value}://{self.host}:{self.port}"


@dataclass
class ApiVersion:
    """API version configuration."""

    version: str
    deprecated: bool = False
    sunset_date: Optional[datetime] = None
    routes: List[RouteConfig] = field(default_factory=list)
    transformers: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# Request/Response Transformers
# ============================================================================


class RequestTransformer(ABC):
    """Abstract request transformer."""

    @abstractmethod
    def transform(self, request: GatewayRequest) -> GatewayRequest:
        """Transform the request."""
        pass


class ResponseTransformer(ABC):
    """Abstract response transformer."""

    @abstractmethod
    def transform(self, response: GatewayResponse, request: GatewayRequest) -> GatewayResponse:
        """Transform the response."""
        pass


class HeaderInjectionTransformer(RequestTransformer):
    """Inject headers into requests."""

    def __init__(self, headers: Dict[str, str]) -> None:
        """Initialize transformer."""
        self._headers = headers

    def transform(self, request: GatewayRequest) -> GatewayRequest:
        """Add headers to request."""
        request.headers.update(self._headers)
        return request


class PathRewriteTransformer(RequestTransformer):
    """Rewrite request paths."""

    def __init__(self, pattern: str, replacement: str) -> None:
        """Initialize transformer."""
        self._pattern = re.compile(pattern)
        self._replacement = replacement

    def transform(self, request: GatewayRequest) -> GatewayRequest:
        """Rewrite the path."""
        request.path = self._pattern.sub(self._replacement, request.path)
        return request


class JsonResponseTransformer(ResponseTransformer):
    """Transform response body to JSON format."""

    def __init__(self, envelope: bool = True) -> None:
        """Initialize transformer."""
        self._envelope = envelope

    def transform(self, response: GatewayResponse, request: GatewayRequest) -> GatewayResponse:
        """Wrap response in JSON envelope."""
        import json

        if response.body and self._envelope:
            try:
                data = json.loads(response.body.decode("utf-8"))
                wrapped = {
                    "status": response.status.value,
                    "data": data,
                    "request_id": response.request_id,
                    "latency_ms": response.latency_ms,
                }
                response.body = json.dumps(wrapped).encode("utf-8")
                response.headers["Content-Type"] = "application/json"
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        return response


# ============================================================================
# Load Balancer
# ============================================================================


class LoadBalancer:
    """Load balancer for distributing requests across endpoints."""

    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
    ) -> None:
        """Initialize load balancer."""
        self._strategy = strategy
        self._endpoints: List[ServiceEndpoint] = []
        self._current_index = 0
        self._lock = asyncio.Lock()

    def add_endpoint(self, endpoint: ServiceEndpoint) -> None:
        """Add an endpoint."""
        self._endpoints.append(endpoint)

    def remove_endpoint(self, host: str, port: int) -> bool:
        """Remove an endpoint."""
        for i, ep in enumerate(self._endpoints):
            if ep.host == host and ep.port == port:
                self._endpoints.pop(i)
                return True
        return False

    async def select_endpoint(
        self, request: Optional[GatewayRequest] = None
    ) -> Optional[ServiceEndpoint]:
        """Select an endpoint based on strategy."""
        healthy_endpoints = [ep for ep in self._endpoints if ep.healthy]

        if not healthy_endpoints:
            return None

        if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return await self._round_robin(healthy_endpoints)
        elif self._strategy == LoadBalanceStrategy.RANDOM:
            return self._random(healthy_endpoints)
        elif self._strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_endpoints)
        elif self._strategy == LoadBalanceStrategy.WEIGHTED:
            return self._weighted(healthy_endpoints)
        elif self._strategy == LoadBalanceStrategy.IP_HASH:
            return self._ip_hash(healthy_endpoints, request)
        elif self._strategy == LoadBalanceStrategy.LEAST_LATENCY:
            return self._least_latency(healthy_endpoints)

        return healthy_endpoints[0]

    async def _round_robin(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round robin selection."""
        async with self._lock:
            endpoint = endpoints[self._current_index % len(endpoints)]
            self._current_index += 1
            return endpoint

    def _random(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Random selection."""
        import random

        return random.choice(endpoints)

    def _least_connections(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select endpoint with least active connections."""
        return min(endpoints, key=lambda ep: ep.active_connections)

    def _weighted(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted random selection."""
        import random

        total_weight = sum(ep.weight for ep in endpoints)
        r = random.uniform(0, total_weight)
        cumulative = 0

        for ep in endpoints:
            cumulative += ep.weight
            if r <= cumulative:
                return ep

        return endpoints[-1]

    def _ip_hash(
        self,
        endpoints: List[ServiceEndpoint],
        request: Optional[GatewayRequest],
    ) -> ServiceEndpoint:
        """Hash-based selection using client IP."""
        if request and request.client_ip:
            hash_val = hash(request.client_ip)
            return endpoints[hash_val % len(endpoints)]
        return endpoints[0]

    def _least_latency(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select endpoint with lowest average latency."""
        return min(endpoints, key=lambda ep: ep.avg_latency_ms)

    def get_endpoints(self) -> List[ServiceEndpoint]:
        """Get all endpoints."""
        return list(self._endpoints)

    def get_healthy_count(self) -> int:
        """Get count of healthy endpoints."""
        return sum(1 for ep in self._endpoints if ep.healthy)


# ============================================================================
# Router
# ============================================================================


class Router:
    """Route requests to appropriate handlers."""

    def __init__(self) -> None:
        """Initialize router."""
        self._routes: Dict[str, List[RouteConfig]] = {}
        self._compiled_patterns: Dict[str, Pattern[str]] = {}

    def add_route(self, config: RouteConfig) -> None:
        """Add a route configuration."""
        # Convert path pattern to regex
        pattern = self._path_to_regex(config.path_pattern)
        self._compiled_patterns[config.path_pattern] = re.compile(pattern)

        version = config.version
        if version not in self._routes:
            self._routes[version] = []

        self._routes[version].append(config)

    def _path_to_regex(self, path: str) -> str:
        """Convert path pattern to regex."""
        # Replace {param} with named groups
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", path)
        # Replace * with wildcard
        pattern = pattern.replace("*", ".*")
        return f"^{pattern}$"

    def match_route(
        self,
        path: str,
        method: HttpMethod,
        version: str = "v1",
    ) -> Optional[Tuple[RouteConfig, Dict[str, str]]]:
        """Match a request to a route."""
        if version not in self._routes:
            return None

        for config in self._routes[version]:
            if method not in config.methods:
                continue

            pattern = self._compiled_patterns.get(config.path_pattern)
            if pattern:
                match = pattern.match(path)
                if match:
                    return config, match.groupdict()

        return None

    def get_routes(self, version: Optional[str] = None) -> List[RouteConfig]:
        """Get all routes, optionally filtered by version."""
        if version:
            return self._routes.get(version, [])

        all_routes: List[RouteConfig] = []
        for routes in self._routes.values():
            all_routes.extend(routes)
        return all_routes


# ============================================================================
# API Versioning
# ============================================================================


class ApiVersionManager:
    """Manage API versions."""

    def __init__(self, default_version: str = "v1") -> None:
        """Initialize version manager."""
        self._versions: Dict[str, ApiVersion] = {}
        self._default_version = default_version

    def register_version(self, version: ApiVersion) -> None:
        """Register an API version."""
        self._versions[version.version] = version
        logger.info(f"Registered API version: {version.version}")

    def deprecate_version(self, version: str, sunset_date: Optional[datetime] = None) -> None:
        """Mark a version as deprecated."""
        if version in self._versions:
            self._versions[version].deprecated = True
            self._versions[version].sunset_date = sunset_date
            logger.warning(f"API version {version} deprecated")

    def get_version(self, version: Optional[str] = None) -> Optional[ApiVersion]:
        """Get API version configuration."""
        target = version or self._default_version
        return self._versions.get(target)

    def extract_version(self, request: GatewayRequest) -> str:
        """Extract version from request."""
        # Check header
        if "X-API-Version" in request.headers:
            return request.headers["X-API-Version"]

        # Check Accept header
        accept = request.headers.get("Accept", "")
        match = re.search(r"version=(\w+)", accept)
        if match:
            return match.group(1)

        # Check path prefix
        path_match = re.match(r"^/(v\d+)/", request.path)
        if path_match:
            return path_match.group(1)

        # Check query param
        version_param = request.query_params.get("version")
        if version_param:
            return version_param

        return self._default_version

    def list_versions(self) -> List[str]:
        """List all registered versions."""
        return list(self._versions.keys())

    def get_active_versions(self) -> List[str]:
        """Get non-deprecated versions."""
        return [v for v, config in self._versions.items() if not config.deprecated]


# ============================================================================
# API Aggregator
# ============================================================================


class ApiAggregator:
    """Aggregate multiple API calls into single response."""

    def __init__(self) -> None:
        """Initialize aggregator."""
        self._aggregations: Dict[str, List[str]] = {}

    def register_aggregation(self, name: str, endpoints: List[str]) -> None:
        """Register an aggregation definition."""
        self._aggregations[name] = endpoints

    async def aggregate(
        self,
        name: str,
        request: GatewayRequest,
        handler: Callable[[GatewayRequest], Any],
    ) -> Dict[str, Any]:
        """Execute aggregated API calls."""
        if name not in self._aggregations:
            raise ValueError(f"Aggregation '{name}' not found")

        endpoints = self._aggregations[name]
        results: Dict[str, Any] = {}

        # Execute calls concurrently
        tasks = []
        for endpoint in endpoints:
            sub_request = GatewayRequest(
                method=request.method,
                path=endpoint,
                headers=request.headers.copy(),
                query_params=request.query_params.copy(),
                client_ip=request.client_ip,
            )
            tasks.append((endpoint, handler(sub_request)))

        # Gather results
        for endpoint, task in tasks:
            try:
                if asyncio.iscoroutine(task):
                    result = await task
                else:
                    result = task
                results[endpoint] = result
            except Exception as e:
                results[endpoint] = {"error": str(e)}

        return results


# ============================================================================
# Gateway Middleware
# ============================================================================


class GatewayMiddleware(ABC):
    """Abstract gateway middleware."""

    @abstractmethod
    async def process(
        self,
        request: GatewayRequest,
        next_handler: Callable[[GatewayRequest], Any],
    ) -> GatewayResponse:
        """Process request through middleware."""
        pass


class LoggingMiddleware(GatewayMiddleware):
    """Log all requests and responses."""

    async def process(
        self,
        request: GatewayRequest,
        next_handler: Callable[[GatewayRequest], Any],
    ) -> GatewayResponse:
        """Log request details."""
        start = time.time()
        logger.info(f"Request: {request.method.value} {request.path} " f"[{request.request_id}]")

        result = next_handler(request)
        if asyncio.iscoroutine(result):
            response = await result
        else:
            response = result

        elapsed = (time.time() - start) * 1000
        logger.info(f"Response: {response.status.value} [{request.request_id}] " f"{elapsed:.2f}ms")

        return response


class CorsMiddleware(GatewayMiddleware):
    """Handle CORS for cross-origin requests."""

    def __init__(
        self,
        allowed_origins: Optional[List[str]] = None,
        allowed_methods: Optional[List[HttpMethod]] = None,
        allowed_headers: Optional[List[str]] = None,
        max_age: int = 86400,
    ) -> None:
        """Initialize CORS middleware."""
        self._origins = allowed_origins or ["*"]
        self._methods = allowed_methods or list(HttpMethod)
        self._headers = allowed_headers or ["*"]
        self._max_age = max_age

    async def process(
        self,
        request: GatewayRequest,
        next_handler: Callable[[GatewayRequest], Any],
    ) -> GatewayResponse:
        """Add CORS headers."""
        # Handle preflight
        if request.method == HttpMethod.OPTIONS:
            return GatewayResponse(
                status=ResponseStatus.NO_CONTENT,
                headers=self._get_cors_headers(request),
                request_id=request.request_id,
            )

        result = next_handler(request)
        if asyncio.iscoroutine(result):
            response = await result
        else:
            response = result

        response.headers.update(self._get_cors_headers(request))
        return response

    def _get_cors_headers(self, request: GatewayRequest) -> Dict[str, str]:
        """Get CORS headers."""
        origin = request.headers.get("Origin", "*")
        if "*" not in self._origins and origin not in self._origins:
            origin = self._origins[0] if self._origins else ""

        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": ", ".join(m.value for m in self._methods),
            "Access-Control-Allow-Headers": ", ".join(self._headers),
            "Access-Control-Max-Age": str(self._max_age),
        }


class CompressionMiddleware(GatewayMiddleware):
    """Compress responses."""

    def __init__(self, min_size: int = 1024) -> None:
        """Initialize compression middleware."""
        self._min_size = min_size

    async def process(
        self,
        request: GatewayRequest,
        next_handler: Callable[[GatewayRequest], Any],
    ) -> GatewayResponse:
        """Compress response if appropriate."""
        result = next_handler(request)
        if asyncio.iscoroutine(result):
            response = await result
        else:
            response = result

        accept_encoding = request.headers.get("Accept-Encoding", "")

        if response.body and len(response.body) >= self._min_size and "gzip" in accept_encoding:
            import gzip

            response.body = gzip.compress(response.body)
            response.headers["Content-Encoding"] = "gzip"

        return response


# ============================================================================
# API Gateway
# ============================================================================


class ApiGateway:
    """
    Central API Gateway.

    Provides routing, load balancing, versioning, and middleware support.
    """

    def __init__(
        self,
        default_version: str = "v1",
    ) -> None:
        """Initialize API Gateway."""
        self._router = Router()
        self._version_manager = ApiVersionManager(default_version)
        self._load_balancers: Dict[str, LoadBalancer] = {}
        self._middleware: List[GatewayMiddleware] = []
        self._aggregator = ApiAggregator()
        self._request_transformers: Dict[str, RequestTransformer] = {}
        self._response_transformers: Dict[str, ResponseTransformer] = {}
        self._handlers: Dict[str, Callable[..., Any]] = {}
        self._metrics: Dict[str, Any] = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "latency_sum_ms": 0.0,
        }

    def add_route(self, config: RouteConfig) -> None:
        """Add a route."""
        self._router.add_route(config)

    def register_handler(self, name: str, handler: Callable[..., Any]) -> None:
        """Register a request handler."""
        self._handlers[name] = handler

    def add_middleware(self, middleware: GatewayMiddleware) -> None:
        """Add middleware to the chain."""
        self._middleware.append(middleware)

    def add_load_balancer(
        self,
        service: str,
        balancer: LoadBalancer,
    ) -> None:
        """Add a load balancer for a service."""
        self._load_balancers[service] = balancer

    def register_transformer(
        self,
        name: str,
        transformer: Union[RequestTransformer, ResponseTransformer],
    ) -> None:
        """Register a transformer."""
        if isinstance(transformer, RequestTransformer):
            self._request_transformers[name] = transformer
        else:
            self._response_transformers[name] = transformer

    def register_version(self, version: ApiVersion) -> None:
        """Register an API version."""
        self._version_manager.register_version(version)

    def register_aggregation(self, name: str, endpoints: List[str]) -> None:
        """Register an API aggregation."""
        self._aggregator.register_aggregation(name, endpoints)

    async def handle_request(self, request: GatewayRequest) -> GatewayResponse:
        """Handle an incoming request."""
        start = time.time()
        self._metrics["requests_total"] += 1

        try:
            # Extract version
            version = self._version_manager.extract_version(request)

            # Check version deprecation
            version_config = self._version_manager.get_version(version)
            if version_config and version_config.deprecated:
                logger.warning(f"Request using deprecated API version: {version}")

            # Match route
            match_result = self._router.match_route(request.path, request.method, version)

            if not match_result:
                return GatewayResponse(
                    status=ResponseStatus.NOT_FOUND,
                    request_id=request.request_id,
                    body=b'{"error": "Route not found"}',
                )

            route_config, path_params = match_result
            request.metadata["path_params"] = path_params

            # Apply request transformer
            if route_config.transform_request:
                transformer = self._request_transformers.get(route_config.transform_request)
                if transformer:
                    request = transformer.transform(request)

            # Execute through middleware chain
            response = await self._execute_with_middleware(request, route_config)

            # Apply response transformer
            if route_config.transform_response:
                transformer = self._response_transformers.get(route_config.transform_response)
                if transformer:
                    response = transformer.transform(response, request)

            elapsed = (time.time() - start) * 1000
            response.latency_ms = elapsed
            self._metrics["latency_sum_ms"] += elapsed

            if response.is_success:
                self._metrics["requests_success"] += 1
            else:
                self._metrics["requests_failed"] += 1

            return response

        except Exception as e:
            logger.error(f"Gateway error: {e}")
            self._metrics["requests_failed"] += 1
            return GatewayResponse(
                status=ResponseStatus.INTERNAL_ERROR,
                request_id=request.request_id,
                body=f'{{"error": "{str(e)}"}}'.encode(),
            )

    async def _execute_with_middleware(
        self,
        request: GatewayRequest,
        route_config: RouteConfig,
    ) -> GatewayResponse:
        """Execute request through middleware chain."""

        # Build middleware chain
        async def final_handler(req: GatewayRequest) -> GatewayResponse:
            return await self._route_to_handler(req, route_config)

        handler = final_handler

        # Wrap with middleware (reverse order)
        for mw in reversed(self._middleware):
            prev_handler = handler

            async def create_handler(
                middleware: GatewayMiddleware,
                next_h: Callable[[GatewayRequest], Any],
            ) -> Callable[[GatewayRequest], Any]:
                async def wrapped(req: GatewayRequest) -> GatewayResponse:
                    return await middleware.process(req, next_h)

                return wrapped

            handler = await create_handler(mw, prev_handler)

        return await handler(request)

    async def _route_to_handler(
        self,
        request: GatewayRequest,
        route_config: RouteConfig,
    ) -> GatewayResponse:
        """Route request to appropriate handler."""
        handler_name = route_config.handler

        # Check for load balancer
        if handler_name in self._load_balancers:
            balancer = self._load_balancers[handler_name]
            endpoint = await balancer.select_endpoint(request)

            if not endpoint:
                return GatewayResponse(
                    status=ResponseStatus.SERVICE_UNAVAILABLE,
                    request_id=request.request_id,
                    body=b'{"error": "No healthy endpoints"}',
                )

            # In real implementation, forward to endpoint
            request.metadata["endpoint"] = endpoint.url

        # Execute handler
        handler = self._handlers.get(handler_name)
        if not handler:
            return GatewayResponse(
                status=ResponseStatus.NOT_FOUND,
                request_id=request.request_id,
                body=b'{"error": "Handler not found"}',
            )

        try:
            result = handler(request)
            if asyncio.iscoroutine(result):
                result = await result

            if isinstance(result, GatewayResponse):
                return result

            # Convert result to response
            import json

            body = json.dumps(result).encode() if result else None
            return GatewayResponse(
                status=ResponseStatus.OK,
                request_id=request.request_id,
                body=body,
                headers={"Content-Type": "application/json"},
            )

        except Exception as e:
            logger.error(f"Handler error: {e}")
            return GatewayResponse(
                status=ResponseStatus.INTERNAL_ERROR,
                request_id=request.request_id,
                body=f'{{"error": "{str(e)}"}}'.encode(),
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get gateway metrics."""
        total = self._metrics["requests_total"]
        return {
            **self._metrics,
            "avg_latency_ms": (self._metrics["latency_sum_ms"] / total if total > 0 else 0),
            "success_rate": (self._metrics["requests_success"] / total if total > 0 else 0),
        }


# ============================================================================
# Vision Gateway
# ============================================================================


class VisionApiGateway(ApiGateway):
    """
    Specialized API Gateway for Vision services.

    Provides vision-specific routing and handling.
    """

    def __init__(
        self,
        providers: Optional[Dict[str, VisionProvider]] = None,
    ) -> None:
        """Initialize Vision API Gateway."""
        super().__init__()
        self._providers = providers or {}

        # Register default routes
        self._register_default_routes()

    def register_provider(self, name: str, provider: VisionProvider) -> None:
        """Register a vision provider."""
        self._providers[name] = provider

    def _register_default_routes(self) -> None:
        """Register default vision routes."""
        # Analyze endpoint
        self.add_route(
            RouteConfig(
                path_pattern="/vision/analyze",
                methods=[HttpMethod.POST],
                handler="analyze",
                version="v1",
            )
        )

        # Provider list endpoint
        self.add_route(
            RouteConfig(
                path_pattern="/vision/providers",
                methods=[HttpMethod.GET],
                handler="list_providers",
                version="v1",
            )
        )

        # Health check endpoint
        self.add_route(
            RouteConfig(
                path_pattern="/health",
                methods=[HttpMethod.GET],
                handler="health_check",
                version="v1",
            )
        )

        # Register handlers
        self.register_handler("analyze", self._handle_analyze)
        self.register_handler("list_providers", self._handle_list_providers)
        self.register_handler("health_check", self._handle_health_check)

    async def _handle_analyze(self, request: GatewayRequest) -> Dict[str, Any]:
        """Handle image analysis request."""
        import json

        if not request.body:
            return {"error": "No image data provided"}

        try:
            # Parse request body
            data = json.loads(request.body.decode("utf-8"))
            image_data = data.get("image_data", "")
            context = data.get("context")
            provider_name = data.get("provider", "default")

            # Get provider
            provider = self._providers.get(provider_name)
            if not provider:
                return {"error": f"Provider '{provider_name}' not found"}

            # Analyze image
            if isinstance(image_data, str):
                import base64

                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = bytes(image_data)

            description = await provider.analyze_image(image_bytes, context)

            return {
                "summary": description.summary,
                "details": description.details,
                "confidence": description.confidence,
                "provider": provider_name,
            }

        except Exception as e:
            return {"error": str(e)}

    def _handle_list_providers(self, request: GatewayRequest) -> Dict[str, Any]:
        """List available providers."""
        return {
            "providers": [
                {
                    "name": name,
                    "provider_name": provider.provider_name,
                }
                for name, provider in self._providers.items()
            ]
        }

    def _handle_health_check(self, request: GatewayRequest) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "providers_count": len(self._providers),
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# Factory Functions
# ============================================================================


def create_api_gateway(
    with_logging: bool = True,
    with_cors: bool = True,
    cors_origins: Optional[List[str]] = None,
) -> ApiGateway:
    """Create a configured API gateway."""
    gateway = ApiGateway()

    if with_logging:
        gateway.add_middleware(LoggingMiddleware())

    if with_cors:
        gateway.add_middleware(CorsMiddleware(allowed_origins=cors_origins))

    return gateway


def create_vision_gateway(
    providers: Optional[Dict[str, VisionProvider]] = None,
    with_logging: bool = True,
    with_cors: bool = True,
) -> VisionApiGateway:
    """Create a configured Vision API gateway."""
    gateway = VisionApiGateway(providers=providers)

    if with_logging:
        gateway.add_middleware(LoggingMiddleware())

    if with_cors:
        gateway.add_middleware(CorsMiddleware())

    return gateway


def create_load_balancer(
    endpoints: List[Tuple[str, int]],
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
) -> LoadBalancer:
    """Create a load balancer with endpoints."""
    balancer = LoadBalancer(strategy=strategy)

    for host, port in endpoints:
        balancer.add_endpoint(ServiceEndpoint(host=host, port=port))

    return balancer
