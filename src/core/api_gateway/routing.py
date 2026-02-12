"""API Gateway Routing.

Provides request routing capabilities:
- Path-based routing
- Header-based routing
- Query parameter routing
"""

from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

logger = logging.getLogger(__name__)


class MatchType(Enum):
    """Types of route matching."""
    EXACT = "exact"
    PREFIX = "prefix"
    REGEX = "regex"


@dataclass
class RouteMatch:
    """Result of route matching."""
    matched: bool
    route: Optional["Route"] = None
    path_params: Dict[str, str] = field(default_factory=dict)
    score: int = 0  # Higher score = better match


@dataclass
class Route:
    """A route definition."""
    name: str
    path: str
    backend: str  # Backend service identifier
    # Empty means "allow all methods" for backward compatibility.
    methods: List[str] = field(default_factory=list)
    match_type: MatchType = MatchType.PREFIX
    priority: int = 0
    headers: Dict[str, str] = field(default_factory=dict)  # Required headers
    query_params: Dict[str, str] = field(default_factory=dict)  # Required params
    rewrite_path: Optional[str] = None
    strip_prefix: bool = False
    timeout_seconds: float = 30.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    _path_pattern: Optional[Pattern] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.match_type == MatchType.REGEX:
            self._path_pattern = re.compile(self.path)
        elif "{" in self.path:
            # Convert path params to regex
            pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", self.path)
            self._path_pattern = re.compile(f"^{pattern}$")


@dataclass
class Request:
    """Incoming request representation."""
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    client_ip: Optional[str] = None
    request_id: Optional[str] = None


class RouteMatcher(ABC):
    """Abstract route matcher."""

    @abstractmethod
    def match(self, request: Request, route: Route) -> RouteMatch:
        """Match request against route."""
        pass


class PathMatcher(RouteMatcher):
    """Path-based route matching."""

    def match(self, request: Request, route: Route) -> RouteMatch:
        path = request.path

        # Method check
        if route.methods and request.method not in route.methods:
            return RouteMatch(matched=False)

        # Path matching
        if route.match_type == MatchType.EXACT:
            if path == route.path:
                return RouteMatch(
                    matched=True,
                    route=route,
                    score=100 + route.priority,
                )

        elif route.match_type == MatchType.PREFIX:
            if path.startswith(route.path):
                score = len(route.path) + route.priority
                return RouteMatch(
                    matched=True,
                    route=route,
                    score=score,
                )

        elif route.match_type == MatchType.REGEX and route._path_pattern:
            match = route._path_pattern.match(path)
            if match:
                return RouteMatch(
                    matched=True,
                    route=route,
                    path_params=match.groupdict(),
                    score=50 + route.priority,
                )

        return RouteMatch(matched=False)


class HeaderMatcher(RouteMatcher):
    """Header-based route matching."""

    def match(self, request: Request, route: Route) -> RouteMatch:
        if not route.headers:
            return RouteMatch(matched=True, route=route)

        for header_name, expected_value in route.headers.items():
            actual_value = request.headers.get(header_name, "")

            # Wildcard match
            if expected_value == "*":
                if not actual_value:
                    return RouteMatch(matched=False)
            # Exact match
            elif actual_value != expected_value:
                return RouteMatch(matched=False)

        return RouteMatch(matched=True, route=route, score=len(route.headers) * 10)


class QueryMatcher(RouteMatcher):
    """Query parameter based route matching."""

    def match(self, request: Request, route: Route) -> RouteMatch:
        if not route.query_params:
            return RouteMatch(matched=True, route=route)

        for param_name, expected_value in route.query_params.items():
            actual_value = request.query_params.get(param_name, "")

            if expected_value == "*":
                if not actual_value:
                    return RouteMatch(matched=False)
            elif actual_value != expected_value:
                return RouteMatch(matched=False)

        return RouteMatch(
            matched=True,
            route=route,
            score=len(route.query_params) * 5,
        )


class CompositeRouteMatcher(RouteMatcher):
    """Combines multiple matchers."""

    def __init__(self, matchers: List[RouteMatcher]):
        self.matchers = matchers

    def match(self, request: Request, route: Route) -> RouteMatch:
        total_score = 0
        path_params: Dict[str, str] = {}

        for matcher in self.matchers:
            result = matcher.match(request, route)
            if not result.matched:
                return RouteMatch(matched=False)

            total_score += result.score
            path_params.update(result.path_params)

        return RouteMatch(
            matched=True,
            route=route,
            path_params=path_params,
            score=total_score,
        )


class Router:
    """Request router."""

    def __init__(self, matcher: Optional[RouteMatcher] = None):
        self._routes: List[Route] = []
        self._matcher = matcher or CompositeRouteMatcher([
            PathMatcher(),
            HeaderMatcher(),
            QueryMatcher(),
        ])

    def add_route(self, route: Route) -> None:
        """Add a route."""
        self._routes.append(route)
        # Sort by priority (descending)
        self._routes.sort(key=lambda r: -r.priority)
        logger.debug(f"Added route: {route.name} -> {route.backend}")

    def remove_route(self, name: str) -> bool:
        """Remove a route by name."""
        initial_len = len(self._routes)
        self._routes = [r for r in self._routes if r.name != name]
        return len(self._routes) < initial_len

    def match(self, request: Request) -> RouteMatch:
        """Find best matching route."""
        best_match: Optional[RouteMatch] = None

        for route in self._routes:
            result = self._matcher.match(request, route)
            if result.matched:
                if best_match is None or result.score > best_match.score:
                    best_match = result

        return best_match or RouteMatch(matched=False)

    def get_routes(self) -> List[Route]:
        """Get all routes."""
        return self._routes.copy()

    def rewrite_path(self, request: Request, route: Route) -> str:
        """Apply path rewriting rules."""
        path = request.path

        if route.strip_prefix:
            if path.startswith(route.path):
                path = path[len(route.path):] or "/"

        if route.rewrite_path:
            # Support path param substitution
            rewrite = route.rewrite_path
            for param, value in self.match(request).path_params.items():
                rewrite = rewrite.replace(f"{{{param}}}", value)
            path = rewrite

        return path


@dataclass
class RouteGroup:
    """Group of routes with common configuration."""
    prefix: str
    routes: List[Route] = field(default_factory=list)
    middleware: List[str] = field(default_factory=list)
    default_timeout: float = 30.0
    default_retry_count: int = 0

    def add(
        self,
        name: str,
        path: str,
        backend: str,
        **kwargs,
    ) -> Route:
        """Add route to group."""
        full_path = f"{self.prefix}{path}"
        route = Route(
            name=name,
            path=full_path,
            backend=backend,
            timeout_seconds=kwargs.pop("timeout_seconds", self.default_timeout),
            retry_count=kwargs.pop("retry_count", self.default_retry_count),
            **kwargs,
        )
        self.routes.append(route)
        return route


def create_router_from_config(config: Dict[str, Any]) -> Router:
    """Create router from configuration dict."""
    router = Router()

    for route_config in config.get("routes", []):
        route = Route(
            name=route_config["name"],
            path=route_config["path"],
            backend=route_config["backend"],
            methods=route_config.get("methods", ["GET"]),
            match_type=MatchType(route_config.get("match_type", "prefix")),
            priority=route_config.get("priority", 0),
            headers=route_config.get("headers", {}),
            query_params=route_config.get("query_params", {}),
            rewrite_path=route_config.get("rewrite_path"),
            strip_prefix=route_config.get("strip_prefix", False),
            timeout_seconds=route_config.get("timeout_seconds", 30.0),
            retry_count=route_config.get("retry_count", 0),
            metadata=route_config.get("metadata", {}),
        )
        router.add_route(route)

    return router
