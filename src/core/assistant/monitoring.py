"""
Logging and Monitoring Module for CAD Assistant.

Provides structured logging, metrics collection, and monitoring utilities.
"""

import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional


class LogLevel(Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry."""

    level: LogLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    logger_name: str = "cad_assistant"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "logger": self.logger_name,
            **self.extra,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class StructuredLogger:
    """
    Structured logging for the assistant.

    Provides consistent JSON logging with context support.

    Example:
        >>> logger = StructuredLogger("assistant.api")
        >>> logger.info("Request received", request_id="123", endpoint="/ask")
    """

    def __init__(
        self,
        name: str = "cad_assistant",
        level: LogLevel = LogLevel.INFO,
        handlers: Optional[List[logging.Handler]] = None,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            level: Minimum log level
            handlers: Custom handlers (uses default if None)
        """
        self.name = name
        self.level = level
        self._context: Dict[str, Any] = {}

        # Setup Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.value))

        if handlers:
            for handler in handlers:
                self._logger.addHandler(handler)
        elif not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def set_context(self, **kwargs) -> None:
        """Set persistent context for all log entries."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear persistent context."""
        self._context.clear()

    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary log context."""
        old_context = self._context.copy()
        self._context.update(kwargs)
        try:
            yield
        finally:
            self._context = old_context

    def _log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Internal logging method."""
        entry = LogEntry(
            level=level,
            message=message,
            logger_name=self.name,
            extra={**self._context, **kwargs},
        )

        log_func = getattr(self._logger, level.value.lower())
        log_func(entry.to_json())

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def exception(self, message: str, exc_info: Optional[Exception] = None, **kwargs) -> None:
        """Log exception with traceback."""
        import traceback

        if exc_info:
            kwargs["exception"] = str(exc_info)
            kwargs["traceback"] = traceback.format_exc()

        self._log(LogLevel.ERROR, message, **kwargs)


@dataclass
class MetricValue:
    """A metric value with timestamp."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and exposes metrics.

    Supports counters, gauges, and histograms.

    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.increment("api.requests", labels={"endpoint": "/ask"})
        >>> metrics.gauge("api.active_connections", 5)
        >>> with metrics.timer("api.request_duration"):
        ...     process_request()
    """

    def __init__(self, prefix: str = "cad_assistant"):
        """
        Initialize metrics collector.

        Args:
            prefix: Metric name prefix
        """
        self.prefix = prefix
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create metric key from name and labels."""
        key = f"{self.prefix}.{name}"
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            key = f"{key}{{{label_str}}}"
        return key

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value

    def decrement(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Decrement a counter."""
        self.increment(name, -value, labels)

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)
            # Keep last 1000 values
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]

    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.histogram(name, duration, labels)

    def timed(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator for timing functions."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.timer(name, labels):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            metrics = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }

            # Calculate histogram statistics
            for key, values in self._histograms.items():
                if values:
                    metrics["histograms"][key] = {
                        "count": len(values),
                        "sum": sum(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "p50": self._percentile(values, 50),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99),
                    }

            return metrics

    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        with self._lock:
            # Counters
            for key, value in self._counters.items():
                lines.append(f"{key}_total {value}")

            # Gauges
            for key, value in self._gauges.items():
                lines.append(f"{key} {value}")

            # Histograms (as summaries)
            for key, values in self._histograms.items():
                if values:
                    lines.append(f"{key}_count {len(values)}")
                    lines.append(f"{key}_sum {sum(values)}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        if f == c:
            return sorted_values[f]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


class HealthChecker:
    """
    Health check utility.

    Monitors component health and provides status.

    Example:
        >>> checker = HealthChecker()
        >>> checker.register("database", check_db_connection)
        >>> checker.register("cache", check_cache_connection)
        >>> status = checker.check_all()
    """

    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Callable[[], bool]] = {}
        self._status: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        check_fn: Callable[[], bool],
        critical: bool = True,
    ) -> None:
        """
        Register a health check.

        Args:
            name: Check name
            check_fn: Function that returns True if healthy
            critical: Whether failure is critical
        """
        self._checks[name] = {
            "fn": check_fn,
            "critical": critical,
        }

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)
        self._status.pop(name, None)

    def check(self, name: str) -> Dict[str, Any]:
        """Run a single health check."""
        if name not in self._checks:
            return {"status": "unknown", "error": "Check not found"}

        check = self._checks[name]
        try:
            start = time.time()
            healthy = check["fn"]()
            duration = time.time() - start

            self._status[name] = {
                "status": "healthy" if healthy else "unhealthy",
                "duration_ms": duration * 1000,
                "critical": check["critical"],
                "timestamp": time.time(),
            }
        except Exception as e:
            self._status[name] = {
                "status": "error",
                "error": str(e),
                "critical": check["critical"],
                "timestamp": time.time(),
            }

        return self._status[name]

    def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_healthy = True

        for name in self._checks:
            result = self.check(name)
            results[name] = result

            if result["status"] != "healthy" and self._checks[name]["critical"]:
                overall_healthy = False

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "checks": results,
            "timestamp": time.time(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get last known status without running checks."""
        overall_healthy = True

        for name, status in self._status.items():
            if status["status"] != "healthy" and self._checks.get(name, {}).get("critical", True):
                overall_healthy = False
                break

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "checks": self._status.copy(),
            "timestamp": time.time(),
        }


# Global instances
_logger: Optional[StructuredLogger] = None
_metrics: Optional[MetricsCollector] = None


def get_logger(name: str = "cad_assistant") -> StructuredLogger:
    """Get or create a structured logger."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger(name)
    return _logger


def get_metrics() -> MetricsCollector:
    """Get or create a metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
