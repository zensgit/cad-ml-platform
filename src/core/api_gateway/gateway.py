"""API Gateway Core.

Provides API gateway functionality:
- Request handling
- Response transformation
- Error handling
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core.api_gateway.routing import Request, Route, RouteMatch, Router
from src.core.api_gateway.loadbalancer import (
    Backend,
    BackendPool,
    BackendState,
    LoadBalancer,
    RoundRobinBalancer,
)

logger = logging.getLogger(__name__)


class GatewayErrorCode(Enum):
    """Gateway error codes."""
    NO_ROUTE = "no_route"
    NO_BACKEND = "no_backend"
    BACKEND_ERROR = "backend_error"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    RATE_LIMITED = "rate_limited"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"


@dataclass
class Response:
    """Gateway response."""
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    backend: Optional[str] = None
    latency_ms: float = 0.0
    error_code: Optional[GatewayErrorCode] = None
    error_message: Optional[str] = None


@dataclass
class GatewayContext:
    """Context for request processing."""
    request: Request
    route: Optional[Route] = None
    backend: Optional[Backend] = None
    start_time: float = field(default_factory=time.time)
    attempt: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def request_id(self) -> str:
        return self.request.request_id or str(uuid.uuid4())

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


class Middleware:
    """Base middleware class."""

    async def before_route(self, ctx: GatewayContext) -> Optional[Response]:
        """Called before routing. Return Response to short-circuit."""
        return None

    async def after_route(self, ctx: GatewayContext) -> None:
        """Called after routing, before backend call."""
        pass

    async def before_backend(self, ctx: GatewayContext) -> Optional[Response]:
        """Called before backend call. Return Response to short-circuit."""
        return None

    async def after_backend(
        self,
        ctx: GatewayContext,
        response: Response,
    ) -> Response:
        """Called after backend response. Can modify response."""
        return response

    async def on_error(
        self,
        ctx: GatewayContext,
        error: Exception,
    ) -> Optional[Response]:
        """Called on error. Return Response to handle error."""
        return None


class LoggingMiddleware(Middleware):
    """Middleware for request logging."""

    async def before_route(self, ctx: GatewayContext) -> Optional[Response]:
        logger.info(
            f"[{ctx.request_id}] {ctx.request.method} {ctx.request.path}"
        )
        return None

    async def after_backend(
        self,
        ctx: GatewayContext,
        response: Response,
    ) -> Response:
        logger.info(
            f"[{ctx.request_id}] {response.status_code} "
            f"{ctx.elapsed_ms:.1f}ms backend={response.backend}"
        )
        return response


class RequestIdMiddleware(Middleware):
    """Middleware to ensure request ID."""

    async def before_route(self, ctx: GatewayContext) -> Optional[Response]:
        if not ctx.request.request_id:
            ctx.request.request_id = str(uuid.uuid4())
        return None

    async def after_backend(
        self,
        ctx: GatewayContext,
        response: Response,
    ) -> Response:
        response.headers["X-Request-ID"] = ctx.request_id
        return response


class TimeoutMiddleware(Middleware):
    """Middleware for timeout handling."""

    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout

    async def before_backend(self, ctx: GatewayContext) -> Optional[Response]:
        timeout = (
            ctx.route.timeout_seconds if ctx.route
            else self.default_timeout
        )
        ctx.metadata["timeout"] = timeout
        return None


class RetryMiddleware(Middleware):
    """Middleware for retry handling."""

    async def after_backend(
        self,
        ctx: GatewayContext,
        response: Response,
    ) -> Response:
        # Add retry info to headers
        if ctx.attempt > 0:
            response.headers["X-Retry-Count"] = str(ctx.attempt)
        return response


class BackendCaller:
    """Interface for calling backends."""

    async def call(
        self,
        backend: Backend,
        request: Request,
        timeout: float,
    ) -> Response:
        """Call backend and return response."""
        # This would make actual HTTP call in real implementation
        # Here we simulate for the interface

        start = time.time()

        # Simulate processing
        await asyncio.sleep(0.01)

        latency = (time.time() - start) * 1000

        return Response(
            status_code=200,
            headers={"X-Backend": backend.id},
            body=b'{"status": "ok"}',
            backend=backend.id,
            latency_ms=latency,
        )


class MockBackendCaller(BackendCaller):
    """Mock backend caller for testing."""

    def __init__(self, responses: Optional[Dict[str, Response]] = None):
        self._responses = responses or {}

    def set_response(self, backend_id: str, response: Response) -> None:
        self._responses[backend_id] = response

    async def call(
        self,
        backend: Backend,
        request: Request,
        timeout: float,
    ) -> Response:
        if backend.id in self._responses:
            return self._responses[backend.id]

        return Response(
            status_code=200,
            backend=backend.id,
            body=b'{"mock": true}',
        )


class Gateway:
    """API Gateway."""

    def __init__(
        self,
        router: Router,
        caller: Optional[BackendCaller] = None,
    ):
        self._router = router
        self._caller = caller or MockBackendCaller()
        self._backend_pools: Dict[str, BackendPool] = {}
        self._load_balancers: Dict[str, LoadBalancer] = {}
        self._middleware: List[Middleware] = []
        self._default_balancer = RoundRobinBalancer()

    def add_middleware(self, middleware: Middleware) -> "Gateway":
        """Add middleware."""
        self._middleware.append(middleware)
        return self

    def register_backend_pool(
        self,
        name: str,
        pool: BackendPool,
        load_balancer: Optional[LoadBalancer] = None,
    ) -> None:
        """Register a backend pool."""
        self._backend_pools[name] = pool
        if load_balancer:
            self._load_balancers[name] = load_balancer

    def get_load_balancer(self, pool_name: str) -> LoadBalancer:
        """Get load balancer for pool."""
        return self._load_balancers.get(pool_name, self._default_balancer)

    async def handle(self, request: Request) -> Response:
        """Handle incoming request."""
        ctx = GatewayContext(request=request)

        try:
            # Before route middleware
            for mw in self._middleware:
                response = await mw.before_route(ctx)
                if response:
                    return response

            # Route matching
            match = self._router.match(request)
            if not match.matched or not match.route:
                return Response(
                    status_code=404,
                    error_code=GatewayErrorCode.NO_ROUTE,
                    error_message=f"No route for {request.method} {request.path}",
                )

            ctx.route = match.route

            # After route middleware
            for mw in self._middleware:
                await mw.after_route(ctx)

            # Get backend
            pool = self._backend_pools.get(match.route.backend)
            if not pool:
                return Response(
                    status_code=503,
                    error_code=GatewayErrorCode.NO_BACKEND,
                    error_message=f"No backend pool: {match.route.backend}",
                )

            balancer = self.get_load_balancer(match.route.backend)

            # Retry loop
            max_attempts = match.route.retry_count + 1
            last_response: Optional[Response] = None

            while ctx.attempt < max_attempts:
                ctx.attempt += 1

                backend = balancer.select(pool)
                if not backend:
                    return Response(
                        status_code=503,
                        error_code=GatewayErrorCode.NO_BACKEND,
                        error_message="No healthy backends available",
                    )

                ctx.backend = backend

                # Before backend middleware
                for mw in self._middleware:
                    response = await mw.before_backend(ctx)
                    if response:
                        return response

                # Call backend
                balancer.on_request_start(backend)
                try:
                    timeout = ctx.metadata.get("timeout", 30.0)

                    response = await asyncio.wait_for(
                        self._caller.call(backend, request, timeout),
                        timeout=timeout,
                    )

                    success = response.status_code < 500
                    balancer.on_request_end(backend, success)

                    # After backend middleware
                    for mw in self._middleware:
                        response = await mw.after_backend(ctx, response)

                    # Check if should retry
                    if response.status_code >= 500 and ctx.attempt < max_attempts:
                        last_response = response
                        continue

                    return response

                except asyncio.TimeoutError:
                    balancer.on_request_end(backend, False)
                    last_response = Response(
                        status_code=504,
                        error_code=GatewayErrorCode.TIMEOUT,
                        error_message=f"Backend timeout after {timeout}s",
                        backend=backend.id,
                    )

                    if ctx.attempt < max_attempts:
                        continue

                except Exception as e:
                    balancer.on_request_end(backend, False)
                    logger.error(f"Backend error: {e}")

                    last_response = Response(
                        status_code=502,
                        error_code=GatewayErrorCode.BACKEND_ERROR,
                        error_message=str(e),
                        backend=backend.id,
                    )

                    if ctx.attempt < max_attempts:
                        continue

            return last_response or Response(
                status_code=503,
                error_code=GatewayErrorCode.BACKEND_ERROR,
                error_message="All retry attempts failed",
            )

        except Exception as e:
            # Error middleware
            for mw in self._middleware:
                response = await mw.on_error(ctx, e)
                if response:
                    return response

            logger.exception(f"Gateway error: {e}")
            return Response(
                status_code=500,
                error_code=GatewayErrorCode.BACKEND_ERROR,
                error_message=str(e),
            )


@dataclass
class GatewayStats:
    """Gateway statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    requests_by_route: Dict[str, int] = field(default_factory=dict)
    requests_by_backend: Dict[str, int] = field(default_factory=dict)
    errors_by_code: Dict[str, int] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.successful_requests / self.total_requests
