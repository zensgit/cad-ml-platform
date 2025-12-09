"""Middleware chain pattern for Vision Provider system.

This module provides middleware capabilities including:
- Chainable middleware components
- Pre/post processing hooks
- Error handling middleware
- Middleware ordering and priority
- Conditional middleware execution
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .base import VisionDescription, VisionProvider


class MiddlewarePhase(Enum):
    """Phase of middleware execution."""

    BEFORE = "before"  # Before provider call
    AFTER = "after"  # After provider call
    ERROR = "error"  # On error
    FINALLY = "finally"  # Always executed


class MiddlewarePriority(Enum):
    """Priority for middleware ordering."""

    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class MiddlewareContext:
    """Context passed through middleware chain."""

    request_id: str
    image_data: bytes
    include_description: bool = True
    provider_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    skip_remaining: bool = False
    error: Optional[Exception] = None
    result: Optional[VisionDescription] = None

    def set_result(self, result: VisionDescription) -> None:
        """Set the result."""
        self.result = result

    def set_error(self, error: Exception) -> None:
        """Set error."""
        self.error = error

    def skip(self) -> None:
        """Skip remaining middleware."""
        self.skip_remaining = True


@dataclass
class MiddlewareResult:
    """Result from middleware execution."""

    middleware_name: str
    phase: MiddlewarePhase
    duration_ms: float
    success: bool
    modified_context: bool = False
    error: Optional[str] = None


class Middleware(ABC):
    """Abstract base class for middleware."""

    def __init__(
        self,
        name: str,
        priority: MiddlewarePriority = MiddlewarePriority.NORMAL,
        enabled: bool = True,
    ) -> None:
        """Initialize middleware.

        Args:
            name: Middleware name
            priority: Execution priority
            enabled: Whether middleware is enabled
        """
        self._name = name
        self._priority = priority
        self._enabled = enabled

    @property
    def name(self) -> str:
        """Return middleware name."""
        return self._name

    @property
    def priority(self) -> MiddlewarePriority:
        """Return middleware priority."""
        return self._priority

    @property
    def enabled(self) -> bool:
        """Return whether middleware is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        self._enabled = value

    def before(self, context: MiddlewareContext) -> None:
        """Execute before provider call.

        Override to add pre-processing logic.
        """
        pass

    def after(self, context: MiddlewareContext) -> None:
        """Execute after provider call.

        Override to add post-processing logic.
        """
        pass

    def on_error(self, context: MiddlewareContext) -> None:
        """Execute on error.

        Override to add error handling logic.
        """
        pass

    def finally_execute(self, context: MiddlewareContext) -> None:
        """Execute always (like finally block).

        Override to add cleanup logic.
        """
        pass


class LambdaMiddleware(Middleware):
    """Middleware using lambda functions."""

    def __init__(
        self,
        name: str,
        before_fn: Optional[Callable[[MiddlewareContext], None]] = None,
        after_fn: Optional[Callable[[MiddlewareContext], None]] = None,
        error_fn: Optional[Callable[[MiddlewareContext], None]] = None,
        finally_fn: Optional[Callable[[MiddlewareContext], None]] = None,
        priority: MiddlewarePriority = MiddlewarePriority.NORMAL,
    ) -> None:
        """Initialize lambda middleware."""
        super().__init__(name, priority)
        self._before_fn = before_fn
        self._after_fn = after_fn
        self._error_fn = error_fn
        self._finally_fn = finally_fn

    def before(self, context: MiddlewareContext) -> None:
        """Execute before function."""
        if self._before_fn:
            self._before_fn(context)

    def after(self, context: MiddlewareContext) -> None:
        """Execute after function."""
        if self._after_fn:
            self._after_fn(context)

    def on_error(self, context: MiddlewareContext) -> None:
        """Execute error function."""
        if self._error_fn:
            self._error_fn(context)

    def finally_execute(self, context: MiddlewareContext) -> None:
        """Execute finally function."""
        if self._finally_fn:
            self._finally_fn(context)


class ConditionalMiddleware(Middleware):
    """Middleware that executes based on condition."""

    def __init__(
        self,
        name: str,
        middleware: Middleware,
        condition: Callable[[MiddlewareContext], bool],
        priority: MiddlewarePriority = MiddlewarePriority.NORMAL,
    ) -> None:
        """Initialize conditional middleware."""
        super().__init__(name, priority)
        self._middleware = middleware
        self._condition = condition

    def before(self, context: MiddlewareContext) -> None:
        """Execute before if condition is met."""
        if self._condition(context):
            self._middleware.before(context)

    def after(self, context: MiddlewareContext) -> None:
        """Execute after if condition is met."""
        if self._condition(context):
            self._middleware.after(context)

    def on_error(self, context: MiddlewareContext) -> None:
        """Execute on error if condition is met."""
        if self._condition(context):
            self._middleware.on_error(context)

    def finally_execute(self, context: MiddlewareContext) -> None:
        """Execute finally if condition is met."""
        if self._condition(context):
            self._middleware.finally_execute(context)


@dataclass
class MiddlewareChainStats:
    """Statistics for middleware chain execution."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration_ms: float = 0.0
    middleware_durations: Dict[str, float] = field(default_factory=dict)
    middleware_errors: Dict[str, int] = field(default_factory=dict)

    def record_execution(
        self,
        middleware_name: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record a middleware execution."""
        self.total_executions += 1
        self.total_duration_ms += duration_ms

        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
            self.middleware_errors[middleware_name] = (
                self.middleware_errors.get(middleware_name, 0) + 1
            )

        self.middleware_durations[middleware_name] = (
            self.middleware_durations.get(middleware_name, 0.0) + duration_ms
        )


class MiddlewareChain:
    """Chain of middleware components."""

    def __init__(self, name: str = "default") -> None:
        """Initialize middleware chain.

        Args:
            name: Chain name
        """
        self._name = name
        self._middlewares: List[Middleware] = []
        self._stats = MiddlewareChainStats()

    @property
    def name(self) -> str:
        """Return chain name."""
        return self._name

    @property
    def stats(self) -> MiddlewareChainStats:
        """Return chain stats."""
        return self._stats

    def add(self, middleware: Middleware) -> "MiddlewareChain":
        """Add middleware to chain.

        Args:
            middleware: Middleware to add

        Returns:
            Self for chaining
        """
        self._middlewares.append(middleware)
        self._middlewares.sort(key=lambda m: m.priority.value)
        return self

    def remove(self, name: str) -> bool:
        """Remove middleware by name.

        Args:
            name: Middleware name

        Returns:
            True if removed
        """
        for i, m in enumerate(self._middlewares):
            if m.name == name:
                self._middlewares.pop(i)
                return True
        return False

    def get(self, name: str) -> Optional[Middleware]:
        """Get middleware by name."""
        for m in self._middlewares:
            if m.name == name:
                return m
        return None

    def execute_before(self, context: MiddlewareContext) -> List[MiddlewareResult]:
        """Execute all before hooks.

        Args:
            context: Middleware context

        Returns:
            List of results
        """
        return self._execute_phase(context, MiddlewarePhase.BEFORE)

    def execute_after(self, context: MiddlewareContext) -> List[MiddlewareResult]:
        """Execute all after hooks.

        Args:
            context: Middleware context

        Returns:
            List of results
        """
        return self._execute_phase(context, MiddlewarePhase.AFTER)

    def execute_error(self, context: MiddlewareContext) -> List[MiddlewareResult]:
        """Execute all error hooks.

        Args:
            context: Middleware context

        Returns:
            List of results
        """
        return self._execute_phase(context, MiddlewarePhase.ERROR)

    def execute_finally(self, context: MiddlewareContext) -> List[MiddlewareResult]:
        """Execute all finally hooks.

        Args:
            context: Middleware context

        Returns:
            List of results
        """
        return self._execute_phase(context, MiddlewarePhase.FINALLY)

    def _execute_phase(
        self,
        context: MiddlewareContext,
        phase: MiddlewarePhase,
    ) -> List[MiddlewareResult]:
        """Execute a phase of middleware.

        Args:
            context: Middleware context
            phase: Phase to execute

        Returns:
            List of results
        """
        results: List[MiddlewareResult] = []

        for middleware in self._middlewares:
            if not middleware.enabled:
                continue

            if context.skip_remaining and phase != MiddlewarePhase.FINALLY:
                break

            start_time = time.time()
            success = True
            error_msg = None

            try:
                if phase == MiddlewarePhase.BEFORE:
                    middleware.before(context)
                elif phase == MiddlewarePhase.AFTER:
                    middleware.after(context)
                elif phase == MiddlewarePhase.ERROR:
                    middleware.on_error(context)
                elif phase == MiddlewarePhase.FINALLY:
                    middleware.finally_execute(context)

            except Exception as e:
                success = False
                error_msg = str(e)

            duration_ms = (time.time() - start_time) * 1000

            result = MiddlewareResult(
                middleware_name=middleware.name,
                phase=phase,
                duration_ms=duration_ms,
                success=success,
                error=error_msg,
            )
            results.append(result)

            self._stats.record_execution(middleware.name, success, duration_ms)

        return results


class MiddlewareVisionProvider(VisionProvider):
    """Vision provider with middleware chain support."""

    def __init__(
        self,
        provider: VisionProvider,
        chain: MiddlewareChain,
    ) -> None:
        """Initialize middleware provider.

        Args:
            provider: Underlying vision provider
            chain: Middleware chain
        """
        self._provider = provider
        self._chain = chain
        self._request_count = 0

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"middleware_{self._provider.provider_name}"

    @property
    def chain(self) -> MiddlewareChain:
        """Return middleware chain."""
        return self._chain

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with middleware chain.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        import uuid

        self._request_count += 1
        request_id = str(uuid.uuid4())

        context = MiddlewareContext(
            request_id=request_id,
            image_data=image_data,
            include_description=include_description,
            provider_name=self._provider.provider_name,
        )

        try:
            # Execute before hooks
            self._chain.execute_before(context)

            if context.skip_remaining and context.result:
                return context.result

            # Call provider
            result = await self._provider.analyze_image(
                context.image_data, context.include_description
            )
            context.set_result(result)

            # Execute after hooks
            self._chain.execute_after(context)

            return context.result or result

        except Exception as e:
            context.set_error(e)
            self._chain.execute_error(context)

            if context.result:
                return context.result

            raise

        finally:
            self._chain.execute_finally(context)


# Built-in middleware


class TimingMiddleware(Middleware):
    """Middleware that tracks timing."""

    def __init__(self) -> None:
        """Initialize timing middleware."""
        super().__init__("timing", MiddlewarePriority.HIGHEST)
        self._timings: Dict[str, float] = {}

    def before(self, context: MiddlewareContext) -> None:
        """Record start time."""
        context.metadata["_timing_start"] = time.time()

    def after(self, context: MiddlewareContext) -> None:
        """Record end time."""
        start = context.metadata.get("_timing_start")
        if start:
            duration = time.time() - start
            context.metadata["timing_ms"] = duration * 1000
            self._timings[context.request_id] = duration * 1000

    @property
    def timings(self) -> Dict[str, float]:
        """Return recorded timings."""
        return dict(self._timings)


class LoggingMiddleware(Middleware):
    """Middleware that logs requests."""

    def __init__(
        self,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize logging middleware."""
        super().__init__("logging", MiddlewarePriority.HIGH)
        self._log_fn = log_fn or print
        self._logs: List[str] = []

    def before(self, context: MiddlewareContext) -> None:
        """Log request start."""
        msg = f"[{context.request_id}] Starting analysis"
        self._logs.append(msg)
        self._log_fn(msg)

    def after(self, context: MiddlewareContext) -> None:
        """Log request complete."""
        msg = f"[{context.request_id}] Analysis complete"
        self._logs.append(msg)
        self._log_fn(msg)

    def on_error(self, context: MiddlewareContext) -> None:
        """Log error."""
        msg = f"[{context.request_id}] Error: {context.error}"
        self._logs.append(msg)
        self._log_fn(msg)

    @property
    def logs(self) -> List[str]:
        """Return logs."""
        return list(self._logs)


class MetadataMiddleware(Middleware):
    """Middleware that adds metadata."""

    def __init__(
        self,
        metadata: Dict[str, Any],
    ) -> None:
        """Initialize metadata middleware."""
        super().__init__("metadata", MiddlewarePriority.LOW)
        self._metadata = metadata

    def before(self, context: MiddlewareContext) -> None:
        """Add metadata to context."""
        context.metadata.update(self._metadata)


def create_middleware_provider(
    provider: VisionProvider,
    chain: Optional[MiddlewareChain] = None,
) -> MiddlewareVisionProvider:
    """Create a middleware vision provider.

    Args:
        provider: Underlying vision provider
        chain: Optional middleware chain

    Returns:
        MiddlewareVisionProvider instance
    """
    if chain is None:
        chain = MiddlewareChain()

    return MiddlewareVisionProvider(provider=provider, chain=chain)
