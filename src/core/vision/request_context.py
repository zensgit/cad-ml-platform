"""Request context propagation for Vision Provider system.

This module provides context propagation including:
- Request context management
- Context propagation across async boundaries
- Correlation ID tracking
- Baggage items (key-value pairs)
- Context scoping and isolation
"""

import contextvars
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

from .base import VisionDescription, VisionProvider


# Context variable for request context
_request_context: contextvars.ContextVar["RequestContext"] = contextvars.ContextVar(
    "request_context"
)


class ContextScope(Enum):
    """Scope of context propagation."""

    REQUEST = "request"  # Single request
    SESSION = "session"  # User session
    GLOBAL = "global"  # Global context


@dataclass
class BaggageItem:
    """Baggage item in context."""

    key: str
    value: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    propagate: bool = True  # Whether to propagate to child contexts


@dataclass
class ContextSpan:
    """Span within context for tracking operations."""

    span_id: str
    operation_name: str
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.now()

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {},
        })

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000


@dataclass
class RequestContext:
    """Context for a request."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    scope: ContextScope = ContextScope.REQUEST
    baggage: Dict[str, BaggageItem] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    spans: List[ContextSpan] = field(default_factory=list)
    parent_context_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    _current_span: Optional[ContextSpan] = field(default=None, repr=False)

    def set_baggage(
        self,
        key: str,
        value: str,
        propagate: bool = True,
        **metadata: Any,
    ) -> None:
        """Set baggage item.

        Args:
            key: Baggage key
            value: Baggage value
            propagate: Whether to propagate to children
            **metadata: Additional metadata
        """
        self.baggage[key] = BaggageItem(
            key=key,
            value=value,
            metadata=metadata,
            propagate=propagate,
        )

    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage value.

        Args:
            key: Baggage key

        Returns:
            Baggage value or None
        """
        item = self.baggage.get(key)
        return item.value if item else None

    def remove_baggage(self, key: str) -> bool:
        """Remove baggage item.

        Args:
            key: Baggage key

        Returns:
            True if removed
        """
        if key in self.baggage:
            del self.baggage[key]
            return True
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        """Set context attribute.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get context attribute.

        Args:
            key: Attribute key
            default: Default value

        Returns:
            Attribute value or default
        """
        return self.attributes.get(key, default)

    def start_span(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> ContextSpan:
        """Start a new span.

        Args:
            operation_name: Name of operation
            attributes: Span attributes

        Returns:
            New span
        """
        parent_id = self._current_span.span_id if self._current_span else None

        span = ContextSpan(
            span_id=str(uuid.uuid4()),
            operation_name=operation_name,
            parent_span_id=parent_id,
            attributes=attributes or {},
        )

        self.spans.append(span)
        self._current_span = span
        return span

    def end_span(self) -> Optional[ContextSpan]:
        """End current span.

        Returns:
            Ended span or None
        """
        if self._current_span:
            self._current_span.end()
            ended = self._current_span

            # Find parent span
            if ended.parent_span_id:
                for span in reversed(self.spans):
                    if span.span_id == ended.parent_span_id:
                        self._current_span = span
                        break
                else:
                    self._current_span = None
            else:
                self._current_span = None

            return ended

        return None

    @property
    def current_span(self) -> Optional[ContextSpan]:
        """Get current span."""
        return self._current_span

    def create_child(self) -> "RequestContext":
        """Create child context.

        Returns:
            Child context
        """
        child = RequestContext(
            correlation_id=self.correlation_id or self.request_id,
            scope=self.scope,
            parent_context_id=self.request_id,
        )

        # Copy propagatable baggage
        for key, item in self.baggage.items():
            if item.propagate:
                child.baggage[key] = BaggageItem(
                    key=item.key,
                    value=item.value,
                    metadata=dict(item.metadata),
                    propagate=item.propagate,
                )

        return child

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "scope": self.scope.value,
            "baggage": {k: v.value for k, v in self.baggage.items()},
            "attributes": dict(self.attributes),
            "span_count": len(self.spans),
            "created_at": self.created_at.isoformat(),
        }


class ContextManager:
    """Manages request contexts."""

    def __init__(self) -> None:
        """Initialize context manager."""
        self._contexts: Dict[str, RequestContext] = {}
        self._lock = threading.Lock()

    def create_context(
        self,
        correlation_id: Optional[str] = None,
        scope: ContextScope = ContextScope.REQUEST,
        **attributes: Any,
    ) -> RequestContext:
        """Create a new context.

        Args:
            correlation_id: Optional correlation ID
            scope: Context scope
            **attributes: Initial attributes

        Returns:
            New context
        """
        context = RequestContext(
            correlation_id=correlation_id,
            scope=scope,
        )

        for key, value in attributes.items():
            context.set_attribute(key, value)

        with self._lock:
            self._contexts[context.request_id] = context

        return context

    def get_context(self, request_id: str) -> Optional[RequestContext]:
        """Get context by request ID.

        Args:
            request_id: Request ID

        Returns:
            Context or None
        """
        with self._lock:
            return self._contexts.get(request_id)

    def remove_context(self, request_id: str) -> bool:
        """Remove context.

        Args:
            request_id: Request ID

        Returns:
            True if removed
        """
        with self._lock:
            if request_id in self._contexts:
                del self._contexts[request_id]
                return True
            return False

    def get_by_correlation_id(
        self, correlation_id: str
    ) -> List[RequestContext]:
        """Get contexts by correlation ID.

        Args:
            correlation_id: Correlation ID

        Returns:
            List of contexts
        """
        with self._lock:
            return [
                ctx for ctx in self._contexts.values()
                if ctx.correlation_id == correlation_id
            ]

    def cleanup_expired(self, max_age_seconds: float = 3600) -> int:
        """Remove expired contexts.

        Args:
            max_age_seconds: Maximum context age

        Returns:
            Number of removed contexts
        """
        cutoff = datetime.now()
        removed = 0

        with self._lock:
            to_remove = []
            for request_id, context in self._contexts.items():
                age = (cutoff - context.created_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(request_id)

            for request_id in to_remove:
                del self._contexts[request_id]
                removed += 1

        return removed


def get_current_context() -> Optional[RequestContext]:
    """Get current request context.

    Returns:
        Current context or None
    """
    try:
        return _request_context.get()
    except LookupError:
        return None


def set_current_context(context: RequestContext) -> contextvars.Token:
    """Set current request context.

    Args:
        context: Context to set

    Returns:
        Token for resetting
    """
    return _request_context.set(context)


def reset_context(token: contextvars.Token) -> None:
    """Reset context to previous value.

    Args:
        token: Token from set_current_context
    """
    _request_context.reset(token)


class context_scope:
    """Context manager for request context scope."""

    def __init__(
        self,
        context: Optional[RequestContext] = None,
        **attributes: Any,
    ) -> None:
        """Initialize context scope.

        Args:
            context: Optional context to use
            **attributes: Attributes for new context
        """
        self._context = context
        self._attributes = attributes
        self._token: Optional[contextvars.Token] = None
        self._created = False

    def __enter__(self) -> RequestContext:
        """Enter context scope."""
        if self._context is None:
            self._context = RequestContext()
            for key, value in self._attributes.items():
                self._context.set_attribute(key, value)
            self._created = True

        self._token = set_current_context(self._context)
        return self._context

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context scope."""
        if self._token:
            reset_context(self._token)


class ContextAwareVisionProvider(VisionProvider):
    """Vision provider with context awareness."""

    def __init__(
        self,
        provider: VisionProvider,
        context_manager: Optional[ContextManager] = None,
        auto_create_context: bool = True,
    ) -> None:
        """Initialize context-aware provider.

        Args:
            provider: Underlying vision provider
            context_manager: Optional context manager
            auto_create_context: Auto-create context if none exists
        """
        self._provider = provider
        self._context_manager = context_manager or ContextManager()
        self._auto_create_context = auto_create_context

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"context_aware_{self._provider.provider_name}"

    @property
    def context_manager(self) -> ContextManager:
        """Return context manager."""
        return self._context_manager

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with context tracking.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        context = get_current_context()

        if context is None and self._auto_create_context:
            context = self._context_manager.create_context()
            token = set_current_context(context)
        else:
            token = None

        try:
            if context:
                span = context.start_span(
                    "analyze_image",
                    attributes={
                        "provider": self._provider.provider_name,
                        "image_size": len(image_data),
                        "include_description": include_description,
                    },
                )

            result = await self._provider.analyze_image(
                image_data, include_description
            )

            if context:
                span.add_event("analysis_complete", {
                    "confidence": result.confidence,
                })
                context.end_span()

            return result

        except Exception as e:
            if context and context.current_span:
                context.current_span.add_event("error", {"message": str(e)})
                context.end_span()
            raise

        finally:
            if token:
                reset_context(token)


# Global context manager
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get global context manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


def create_context_aware_provider(
    provider: VisionProvider,
    context_manager: Optional[ContextManager] = None,
    auto_create_context: bool = True,
) -> ContextAwareVisionProvider:
    """Create a context-aware vision provider.

    Args:
        provider: Underlying vision provider
        context_manager: Optional context manager
        auto_create_context: Auto-create context if none exists

    Returns:
        ContextAwareVisionProvider instance
    """
    if context_manager is None:
        context_manager = get_context_manager()

    return ContextAwareVisionProvider(
        provider=provider,
        context_manager=context_manager,
        auto_create_context=auto_create_context,
    )
