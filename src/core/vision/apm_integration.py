"""APM Integration Module for Vision System.

This module provides Application Performance Monitoring capabilities including:
- Transaction tracing and profiling
- Performance metrics collection
- Error tracking and diagnostics
- Dependency mapping and analysis
- Real-time performance insights
- Integration with popular APM tools

Phase 17: Advanced Observability & Monitoring
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import statistics
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Type, TypeVar, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class TransactionType(str, Enum):
    """Types of transactions."""

    REQUEST = "request"
    BACKGROUND = "background"
    SCHEDULED = "scheduled"
    MESSAGE = "message"
    CUSTOM = "custom"


class SpanKind(str, Enum):
    """Types of spans."""

    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span status codes."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class APMProvider(str, Enum):
    """Supported APM providers."""

    DATADOG = "datadog"
    NEW_RELIC = "newrelic"
    ELASTIC_APM = "elastic"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    OPENTELEMETRY = "opentelemetry"
    CUSTOM = "custom"


class ProfileType(str, Enum):
    """Types of profiling."""

    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    WALL_TIME = "wall_time"
    LOCK = "lock"


# ========================
# Data Classes
# ========================


@dataclass
class SpanContext:
    """Context for distributed tracing."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    sampled: bool = True


@dataclass
class Span:
    """A single span in a trace."""

    span_id: str
    trace_id: str
    name: str
    kind: SpanKind
    start_time: datetime
    end_time: Optional[datetime] = None
    parent_span_id: Optional[str] = None
    status: SpanStatus = SpanStatus.OK
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    stack_trace: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0


@dataclass
class Transaction:
    """A transaction representing a unit of work."""

    transaction_id: str
    name: str
    transaction_type: TransactionType
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.OK
    spans: List[Span] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Get transaction duration in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0


@dataclass
class ProfileSample:
    """A single profiling sample."""

    timestamp: datetime
    profile_type: ProfileType
    value: float
    unit: str
    call_stack: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics summary."""

    total_transactions: int = 0
    error_count: int = 0
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    throughput_rpm: float = 0.0  # Requests per minute
    error_rate: float = 0.0
    apdex_score: float = 0.0


@dataclass
class DependencyInfo:
    """Information about an external dependency."""

    name: str
    dependency_type: str  # database, http, cache, etc.
    target: str
    avg_duration_ms: float = 0.0
    call_count: int = 0
    error_count: int = 0
    last_called: Optional[datetime] = None


@dataclass
class APMConfig:
    """APM configuration."""

    provider: APMProvider = APMProvider.CUSTOM
    service_name: str = "vision-service"
    environment: str = "production"
    sample_rate: float = 1.0  # 1.0 = 100%
    apdex_threshold_ms: float = 500.0
    enable_profiling: bool = False
    enable_error_tracking: bool = True
    custom_tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ErrorInfo:
    """Information about an error."""

    error_id: str
    error_type: str
    message: str
    stack_trace: str
    timestamp: datetime
    transaction_id: Optional[str] = None
    span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    occurrences: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


# ========================
# Span Tracker
# ========================


class SpanTracker:
    """Tracks spans for distributed tracing."""

    def __init__(self):
        self._active_spans: Dict[str, Span] = {}
        self._completed_spans: deque = deque(maxlen=10000)
        self._lock = threading.Lock()

    def start_span(
        self,
        name: str,
        trace_id: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new span."""
        span = Span(
            span_id=f"span_{int(time.time() * 1000000)}",
            trace_id=trace_id,
            name=name,
            kind=kind,
            start_time=datetime.now(),
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )

        with self._lock:
            self._active_spans[span.span_id] = span

        return span

    def end_span(
        self, span: Span, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None
    ) -> None:
        """End a span."""
        span.end_time = datetime.now()
        span.status = status
        span.error = error

        if error:
            span.stack_trace = traceback.format_exc()

        with self._lock:
            self._active_spans.pop(span.span_id, None)
            self._completed_spans.append(span)

    def add_event(self, span: Span, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to a span."""
        span.events.append(
            {"name": name, "timestamp": datetime.now().isoformat(), "attributes": attributes or {}}
        )

    def get_active_spans(self, trace_id: Optional[str] = None) -> List[Span]:
        """Get active spans."""
        with self._lock:
            spans = list(self._active_spans.values())
        if trace_id:
            spans = [s for s in spans if s.trace_id == trace_id]
        return spans

    def get_completed_spans(self, trace_id: Optional[str] = None, limit: int = 100) -> List[Span]:
        """Get completed spans."""
        with self._lock:
            spans = list(self._completed_spans)
        if trace_id:
            spans = [s for s in spans if s.trace_id == trace_id]
        return spans[-limit:]


# ========================
# Transaction Tracker
# ========================


class TransactionTracker:
    """Tracks transactions."""

    def __init__(self, span_tracker: SpanTracker):
        self._span_tracker = span_tracker
        self._active_transactions: Dict[str, Transaction] = {}
        self._completed_transactions: deque = deque(maxlen=10000)
        self._lock = threading.Lock()

    def start_transaction(
        self,
        name: str,
        transaction_type: TransactionType = TransactionType.REQUEST,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Transaction:
        """Start a new transaction."""
        transaction = Transaction(
            transaction_id=f"txn_{int(time.time() * 1000000)}",
            name=name,
            transaction_type=transaction_type,
            start_time=datetime.now(),
            attributes=attributes or {},
        )

        with self._lock:
            self._active_transactions[transaction.transaction_id] = transaction

        return transaction

    def end_transaction(
        self,
        transaction: Transaction,
        status: SpanStatus = SpanStatus.OK,
        error: Optional[str] = None,
    ) -> None:
        """End a transaction."""
        transaction.end_time = datetime.now()
        transaction.status = status
        transaction.error = error

        # Collect spans for this transaction
        spans = self._span_tracker.get_completed_spans(trace_id=transaction.transaction_id)
        transaction.spans = spans

        with self._lock:
            self._active_transactions.pop(transaction.transaction_id, None)
            self._completed_transactions.append(transaction)

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get a transaction by ID."""
        with self._lock:
            if transaction_id in self._active_transactions:
                return self._active_transactions[transaction_id]
            for txn in self._completed_transactions:
                if txn.transaction_id == transaction_id:
                    return txn
            return None

    def get_recent_transactions(self, limit: int = 100) -> List[Transaction]:
        """Get recent transactions."""
        with self._lock:
            return list(self._completed_transactions)[-limit:]


# ========================
# Performance Analyzer
# ========================


class PerformanceAnalyzer:
    """Analyzes performance data."""

    def __init__(self, apdex_threshold_ms: float = 500.0):
        self._apdex_threshold = apdex_threshold_ms

    def analyze_transactions(self, transactions: List[Transaction]) -> PerformanceMetrics:
        """Analyze transaction performance."""
        if not transactions:
            return PerformanceMetrics()

        durations = [t.duration_ms for t in transactions if t.end_time]
        error_count = sum(1 for t in transactions if t.status == SpanStatus.ERROR)

        metrics = PerformanceMetrics(
            total_transactions=len(transactions),
            error_count=error_count,
            error_rate=error_count / len(transactions) if transactions else 0.0,
        )

        if durations:
            metrics.avg_duration_ms = statistics.mean(durations)
            sorted_durations = sorted(durations)
            metrics.p50_duration_ms = self._percentile(sorted_durations, 50)
            metrics.p95_duration_ms = self._percentile(sorted_durations, 95)
            metrics.p99_duration_ms = self._percentile(sorted_durations, 99)

        # Calculate throughput
        if len(transactions) >= 2:
            time_span = (
                transactions[-1].start_time - transactions[0].start_time
            ).total_seconds() / 60.0
            if time_span > 0:
                metrics.throughput_rpm = len(transactions) / time_span

        # Calculate Apdex score
        metrics.apdex_score = self._calculate_apdex(durations)

        return metrics

    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Calculate percentile."""
        if not sorted_values:
            return 0.0
        idx = (len(sorted_values) - 1) * p / 100
        lower = int(idx)
        upper = lower + 1
        if upper >= len(sorted_values):
            return sorted_values[-1]
        weight = idx - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    def _calculate_apdex(self, durations: List[float]) -> float:
        """Calculate Apdex score."""
        if not durations:
            return 1.0

        satisfied = sum(1 for d in durations if d <= self._apdex_threshold)
        tolerating = sum(
            1 for d in durations if self._apdex_threshold < d <= self._apdex_threshold * 4
        )

        return (satisfied + tolerating / 2) / len(durations)


# ========================
# Error Tracker
# ========================


class ErrorTracker:
    """Tracks and aggregates errors."""

    def __init__(self):
        self._errors: Dict[str, ErrorInfo] = {}
        self._error_history: deque = deque(maxlen=10000)
        self._lock = threading.Lock()

    def track_error(
        self,
        error: Exception,
        transaction_id: Optional[str] = None,
        span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> ErrorInfo:
        """Track an error."""
        error_type = type(error).__name__
        message = str(error)
        stack_trace = traceback.format_exc()

        # Create fingerprint for deduplication
        fingerprint = hashlib.sha256(f"{error_type}:{message}".encode()).hexdigest()[:12]

        now = datetime.now()

        with self._lock:
            if fingerprint in self._errors:
                existing = self._errors[fingerprint]
                existing.occurrences += 1
                existing.last_seen = now
                return existing

            error_info = ErrorInfo(
                error_id=f"err_{fingerprint}",
                error_type=error_type,
                message=message,
                stack_trace=stack_trace,
                timestamp=now,
                transaction_id=transaction_id,
                span_id=span_id,
                attributes=attributes or {},
                first_seen=now,
                last_seen=now,
            )

            self._errors[fingerprint] = error_info
            self._error_history.append(error_info)

            return error_info

    def get_errors(
        self,
        start_time: Optional[datetime] = None,
        error_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[ErrorInfo]:
        """Get tracked errors."""
        with self._lock:
            errors = list(self._errors.values())

        if start_time:
            errors = [e for e in errors if e.first_seen >= start_time]
        if error_type:
            errors = [e for e in errors if e.error_type == error_type]

        return sorted(errors, key=lambda x: x.occurrences, reverse=True)[:limit]

    def get_top_errors(self, limit: int = 10) -> List[ErrorInfo]:
        """Get most frequent errors."""
        with self._lock:
            errors = list(self._errors.values())
        return sorted(errors, key=lambda x: x.occurrences, reverse=True)[:limit]


# ========================
# Dependency Tracker
# ========================


class DependencyTracker:
    """Tracks external dependencies."""

    def __init__(self):
        self._dependencies: Dict[str, DependencyInfo] = {}
        self._call_durations: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_call(
        self, name: str, dependency_type: str, target: str, duration_ms: float, success: bool = True
    ) -> None:
        """Record a dependency call."""
        key = f"{name}:{target}"

        with self._lock:
            if key not in self._dependencies:
                self._dependencies[key] = DependencyInfo(
                    name=name, dependency_type=dependency_type, target=target
                )

            dep = self._dependencies[key]
            dep.call_count += 1
            if not success:
                dep.error_count += 1
            dep.last_called = datetime.now()

            self._call_durations[key].append(duration_ms)
            if len(self._call_durations[key]) > 1000:
                self._call_durations[key] = self._call_durations[key][-1000:]

            # Update average
            dep.avg_duration_ms = statistics.mean(self._call_durations[key])

    def get_dependencies(self) -> List[DependencyInfo]:
        """Get all tracked dependencies."""
        with self._lock:
            return list(self._dependencies.values())

    def get_dependency(self, name: str, target: str) -> Optional[DependencyInfo]:
        """Get a specific dependency."""
        key = f"{name}:{target}"
        with self._lock:
            return self._dependencies.get(key)


# ========================
# APM Manager
# ========================


class APMManager:
    """Main APM manager coordinating all APM operations."""

    def __init__(self, config: Optional[APMConfig] = None):
        self._config = config or APMConfig()
        self._span_tracker = SpanTracker()
        self._transaction_tracker = TransactionTracker(self._span_tracker)
        self._analyzer = PerformanceAnalyzer(self._config.apdex_threshold_ms)
        self._error_tracker = ErrorTracker()
        self._dependency_tracker = DependencyTracker()
        self._lock = threading.Lock()

    # Transaction API

    def start_transaction(
        self,
        name: str,
        transaction_type: TransactionType = TransactionType.REQUEST,
        **attributes: Any,
    ) -> Transaction:
        """Start a new transaction."""
        attrs = {**self._config.custom_tags, **attributes}
        return self._transaction_tracker.start_transaction(name, transaction_type, attrs)

    def end_transaction(
        self,
        transaction: Transaction,
        status: SpanStatus = SpanStatus.OK,
        error: Optional[str] = None,
    ) -> None:
        """End a transaction."""
        self._transaction_tracker.end_transaction(transaction, status, error)

    # Span API

    def start_span(
        self,
        name: str,
        transaction: Transaction,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_span: Optional[Span] = None,
        **attributes: Any,
    ) -> Span:
        """Start a new span."""
        return self._span_tracker.start_span(
            name=name,
            trace_id=transaction.transaction_id,
            kind=kind,
            parent_span_id=parent_span.span_id if parent_span else None,
            attributes=attributes,
        )

    def end_span(
        self, span: Span, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None
    ) -> None:
        """End a span."""
        self._span_tracker.end_span(span, status, error)

    # Error tracking

    def capture_error(
        self,
        error: Exception,
        transaction: Optional[Transaction] = None,
        span: Optional[Span] = None,
        **attributes: Any,
    ) -> ErrorInfo:
        """Capture an error."""
        return self._error_tracker.track_error(
            error,
            transaction_id=transaction.transaction_id if transaction else None,
            span_id=span.span_id if span else None,
            attributes=attributes,
        )

    # Dependency tracking

    def record_dependency(
        self, name: str, dependency_type: str, target: str, duration_ms: float, success: bool = True
    ) -> None:
        """Record a dependency call."""
        self._dependency_tracker.record_call(name, dependency_type, target, duration_ms, success)

    # Analytics

    def get_performance_metrics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """Get performance metrics."""
        transactions = self._transaction_tracker.get_recent_transactions(10000)

        if start_time:
            transactions = [t for t in transactions if t.start_time >= start_time]
        if end_time:
            transactions = [t for t in transactions if t.start_time <= end_time]

        return self._analyzer.analyze_transactions(transactions)

    def get_top_errors(self, limit: int = 10) -> List[ErrorInfo]:
        """Get most frequent errors."""
        return self._error_tracker.get_top_errors(limit)

    def get_dependencies(self) -> List[DependencyInfo]:
        """Get dependency information."""
        return self._dependency_tracker.get_dependencies()

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get a transaction by ID."""
        return self._transaction_tracker.get_transaction(transaction_id)

    def get_recent_transactions(self, limit: int = 100) -> List[Transaction]:
        """Get recent transactions."""
        return self._transaction_tracker.get_recent_transactions(limit)

    # Context managers

    @contextmanager
    def transaction_context(
        self,
        name: str,
        transaction_type: TransactionType = TransactionType.REQUEST,
        **attributes: Any,
    ) -> Generator[Transaction, None, None]:
        """Context manager for transactions."""
        transaction = self.start_transaction(name, transaction_type, **attributes)
        try:
            yield transaction
            self.end_transaction(transaction, SpanStatus.OK)
        except Exception as e:
            self.capture_error(e, transaction)
            self.end_transaction(transaction, SpanStatus.ERROR, str(e))
            raise

    @contextmanager
    def span_context(
        self,
        name: str,
        transaction: Transaction,
        kind: SpanKind = SpanKind.INTERNAL,
        **attributes: Any,
    ) -> Generator[Span, None, None]:
        """Context manager for spans."""
        span = self.start_span(name, transaction, kind, **attributes)
        try:
            yield span
            self.end_span(span, SpanStatus.OK)
        except Exception as e:
            self.end_span(span, SpanStatus.ERROR, str(e))
            raise


# ========================
# APM Vision Provider
# ========================


class APMVisionProvider(VisionProvider):
    """Vision provider with APM integration."""

    def __init__(self, base_provider: VisionProvider, apm_manager: Optional[APMManager] = None):
        self._base_provider = base_provider
        self._apm = apm_manager or APMManager()

    @property
    def provider_name(self) -> str:
        return f"apm_{self._base_provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, context: Optional[Dict[str, Any]] = None
    ) -> VisionDescription:
        """Analyze image with APM tracking."""
        with self._apm.transaction_context(
            f"vision.analyze.{self._base_provider.provider_name}",
            TransactionType.REQUEST,
            image_size=len(image_data),
        ) as transaction:
            with self._apm.span_context("analyze_image", transaction, SpanKind.INTERNAL) as span:
                start_time = time.time()

                try:
                    result = await self._base_provider.analyze_image(image_data, context)

                    duration_ms = (time.time() - start_time) * 1000

                    # Record as dependency
                    self._apm.record_dependency(
                        name=self._base_provider.provider_name,
                        dependency_type="vision_provider",
                        target="analyze_image",
                        duration_ms=duration_ms,
                        success=True,
                    )

                    return result

                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000

                    self._apm.record_dependency(
                        name=self._base_provider.provider_name,
                        dependency_type="vision_provider",
                        target="analyze_image",
                        duration_ms=duration_ms,
                        success=False,
                    )

                    raise

    def get_apm_manager(self) -> APMManager:
        """Get the APM manager."""
        return self._apm


# ========================
# Decorator
# ========================


T = TypeVar("T")


def apm_traced(
    apm_manager: APMManager,
    name: Optional[str] = None,
    transaction_type: TransactionType = TransactionType.REQUEST,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to trace functions with APM."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            with apm_manager.transaction_context(span_name, transaction_type) as transaction:
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            with apm_manager.transaction_context(span_name, transaction_type) as transaction:
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ========================
# Factory Functions
# ========================


def create_apm_manager(
    service_name: str = "vision-service",
    environment: str = "production",
    apdex_threshold_ms: float = 500.0,
) -> APMManager:
    """Create a new APM manager."""
    config = APMConfig(
        service_name=service_name, environment=environment, apdex_threshold_ms=apdex_threshold_ms
    )
    return APMManager(config)


def create_apm_config(
    provider: APMProvider = APMProvider.CUSTOM, service_name: str = "vision-service", **kwargs: Any
) -> APMConfig:
    """Create APM configuration."""
    return APMConfig(provider=provider, service_name=service_name, **kwargs)


def create_apm_provider(
    base_provider: VisionProvider, apm_manager: Optional[APMManager] = None
) -> APMVisionProvider:
    """Create an APM vision provider."""
    return APMVisionProvider(base_provider, apm_manager)
