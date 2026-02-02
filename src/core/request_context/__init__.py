"""Request Context Module.

Provides context propagation:
- Request scoped context
- Correlation IDs
- Baggage propagation
"""

from __future__ import annotations

import contextvars
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

T = TypeVar("T")


# Context variable for the current request context
_current_context: contextvars.ContextVar["RequestContext"] = contextvars.ContextVar(
    "request_context"
)


@dataclass
class Baggage:
    """Baggage for propagating key-value pairs across service boundaries."""

    _items: Dict[str, str] = field(default_factory=dict)

    def set(self, key: str, value: str) -> None:
        """Set a baggage item."""
        self._items[key] = value

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a baggage item."""
        return self._items.get(key, default)

    def remove(self, key: str) -> None:
        """Remove a baggage item."""
        self._items.pop(key, None)

    def items(self) -> Iterator[tuple[str, str]]:
        """Iterate over baggage items."""
        return iter(self._items.items())

    def to_header(self) -> str:
        """Convert to HTTP header value (W3C baggage format)."""
        parts = [f"{k}={v}" for k, v in self._items.items()]
        return ",".join(parts)

    @classmethod
    def from_header(cls, header: str) -> "Baggage":
        """Parse from HTTP header value."""
        baggage = cls()
        if header:
            for part in header.split(","):
                part = part.strip()
                if "=" in part:
                    key, value = part.split("=", 1)
                    baggage.set(key.strip(), value.strip())
        return baggage

    def copy(self) -> "Baggage":
        """Create a copy of the baggage."""
        new_baggage = Baggage()
        new_baggage._items = self._items.copy()
        return new_baggage


@dataclass
class RequestContext:
    """Context for a request."""

    # Identifiers
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    # Request info
    service_name: Optional[str] = None
    operation_name: Optional[str] = None
    client_ip: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    start_timestamp: float = field(default_factory=time.time)

    # Baggage and attributes
    baggage: Baggage = field(default_factory=Baggage)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Deadline
    deadline: Optional[datetime] = None

    def __post_init__(self):
        if self.correlation_id is None:
            self.correlation_id = self.request_id

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_timestamp) * 1000

    @property
    def remaining_ms(self) -> Optional[float]:
        """Get remaining time until deadline in milliseconds."""
        if self.deadline is None:
            return None
        remaining = (self.deadline - datetime.utcnow()).total_seconds() * 1000
        return max(0, remaining)

    @property
    def is_expired(self) -> bool:
        """Check if deadline has passed."""
        if self.deadline is None:
            return False
        return datetime.utcnow() > self.deadline

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a context attribute."""
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get a context attribute."""
        return self.attributes.get(key, default)

    def child_context(
        self,
        operation_name: Optional[str] = None,
    ) -> "RequestContext":
        """Create a child context for a sub-operation."""
        return RequestContext(
            request_id=str(uuid.uuid4()),
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            span_id=generate_span_id(),
            parent_span_id=self.span_id,
            service_name=self.service_name,
            operation_name=operation_name,
            client_ip=self.client_ip,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            baggage=self.baggage.copy(),
            attributes=self.attributes.copy(),
            deadline=self.deadline,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "service_name": self.service_name,
            "operation_name": self.operation_name,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "elapsed_ms": self.elapsed_ms,
            "attributes": self.attributes,
        }

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        headers = {
            "X-Request-ID": self.request_id,
            "X-Correlation-ID": self.correlation_id or self.request_id,
        }

        if self.trace_id:
            headers["X-Trace-ID"] = self.trace_id
        if self.span_id:
            headers["X-Span-ID"] = self.span_id
        if self.user_id:
            headers["X-User-ID"] = self.user_id
        if self.tenant_id:
            headers["X-Tenant-ID"] = self.tenant_id

        baggage_header = self.baggage.to_header()
        if baggage_header:
            headers["Baggage"] = baggage_header

        return headers

    @classmethod
    def from_headers(
        cls,
        headers: Dict[str, str],
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
    ) -> "RequestContext":
        """Create context from HTTP headers."""
        # Normalize header names (case-insensitive)
        normalized = {k.lower(): v for k, v in headers.items()}

        ctx = cls(
            request_id=normalized.get("x-request-id", str(uuid.uuid4())),
            correlation_id=normalized.get("x-correlation-id"),
            trace_id=normalized.get("x-trace-id") or generate_trace_id(),
            span_id=generate_span_id(),
            parent_span_id=normalized.get("x-span-id"),
            service_name=service_name,
            operation_name=operation_name,
            user_id=normalized.get("x-user-id"),
            tenant_id=normalized.get("x-tenant-id"),
        )

        # Parse baggage
        if "baggage" in normalized:
            ctx.baggage = Baggage.from_header(normalized["baggage"])

        return ctx


def generate_trace_id() -> str:
    """Generate a trace ID."""
    return secrets.token_hex(16)


def generate_span_id() -> str:
    """Generate a span ID."""
    return secrets.token_hex(8)


def get_current_context() -> Optional[RequestContext]:
    """Get the current request context."""
    return _current_context.get(None)


def set_current_context(ctx: RequestContext) -> contextvars.Token:
    """Set the current request context."""
    return _current_context.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    """Reset context to previous value."""
    _current_context.reset(token)


class ContextManager:
    """Context manager for request context."""

    def __init__(self, ctx: RequestContext):
        self._ctx = ctx
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> RequestContext:
        self._token = set_current_context(self._ctx)
        return self._ctx

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token:
            reset_context(self._token)


def with_context(ctx: RequestContext) -> ContextManager:
    """Create context manager for request context."""
    return ContextManager(ctx)


def context_scope(
    service_name: Optional[str] = None,
    operation_name: Optional[str] = None,
    **kwargs,
) -> ContextManager:
    """Create a new context scope."""
    ctx = RequestContext(
        service_name=service_name,
        operation_name=operation_name,
        **kwargs,
    )
    return ContextManager(ctx)


def child_scope(
    operation_name: Optional[str] = None,
) -> ContextManager:
    """Create a child context scope."""
    current = get_current_context()
    if current:
        ctx = current.child_context(operation_name)
    else:
        ctx = RequestContext(operation_name=operation_name)
    return ContextManager(ctx)


# Convenience functions

def get_request_id() -> Optional[str]:
    """Get current request ID."""
    ctx = get_current_context()
    return ctx.request_id if ctx else None


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    ctx = get_current_context()
    return ctx.correlation_id if ctx else None


def get_trace_id() -> Optional[str]:
    """Get current trace ID."""
    ctx = get_current_context()
    return ctx.trace_id if ctx else None


def get_user_id() -> Optional[str]:
    """Get current user ID."""
    ctx = get_current_context()
    return ctx.user_id if ctx else None


def get_tenant_id() -> Optional[str]:
    """Get current tenant ID."""
    ctx = get_current_context()
    return ctx.tenant_id if ctx else None


def set_attribute(key: str, value: Any) -> None:
    """Set attribute on current context."""
    ctx = get_current_context()
    if ctx:
        ctx.set_attribute(key, value)


def get_attribute(key: str, default: Any = None) -> Any:
    """Get attribute from current context."""
    ctx = get_current_context()
    return ctx.get_attribute(key, default) if ctx else default


def set_baggage(key: str, value: str) -> None:
    """Set baggage on current context."""
    ctx = get_current_context()
    if ctx:
        ctx.baggage.set(key, value)


def get_baggage(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get baggage from current context."""
    ctx = get_current_context()
    return ctx.baggage.get(key, default) if ctx else default


# Decorators

def with_request_context(
    service_name: Optional[str] = None,
    operation_name: Optional[str] = None,
):
    """Decorator to run function with request context."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        import asyncio
        import functools

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            with context_scope(service_name, operation_name or func.__name__):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            with context_scope(service_name, operation_name or func.__name__):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def propagate_context(
    service_name: Optional[str] = None,
):
    """Decorator to propagate context to child operations."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        import asyncio
        import functools

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            with child_scope(func.__name__):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            with child_scope(func.__name__):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


__all__ = [
    "Baggage",
    "RequestContext",
    "generate_trace_id",
    "generate_span_id",
    "get_current_context",
    "set_current_context",
    "reset_context",
    "ContextManager",
    "with_context",
    "context_scope",
    "child_scope",
    "get_request_id",
    "get_correlation_id",
    "get_trace_id",
    "get_user_id",
    "get_tenant_id",
    "set_attribute",
    "get_attribute",
    "set_baggage",
    "get_baggage",
    "with_request_context",
    "propagate_context",
]
