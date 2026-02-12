"""Structured logging configuration.

Features:
- JSON formatted logs for aggregation
- Request correlation IDs
- Performance metrics
- Error tracking
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
tenant_id_var: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)

F = TypeVar("F", bound=Callable[..., Any])


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(
        self,
        service_name: str = "cad-ml-platform",
        environment: str = "production",
        include_stack_trace: bool = True,
    ):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.include_stack_trace = include_stack_trace

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
        }

        # Add location info
        log_entry["location"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add context variables
        if request_id := request_id_var.get():
            log_entry["request_id"] = request_id
        if user_id := user_id_var.get():
            log_entry["user_id"] = user_id
        if tenant_id := tenant_id_var.get():
            log_entry["tenant_id"] = tenant_id

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry["extra"] = record.extra_fields

        # Add exception info
        if record.exc_info and self.include_stack_trace:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, default=str)


class StructuredLogger:
    """Enhanced logger with structured output."""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._extra_fields: Dict[str, Any] = {}

    def _log(
        self, level: int, message: str, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(unknown)",
            0,
            message,
            (),
            None,
        )
        if extra:
            record.extra_fields = {**self._extra_fields, **extra}
        elif self._extra_fields:
            record.extra_fields = self._extra_fields
        self.logger.handle(record)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, kwargs if kwargs else None)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, message, kwargs if kwargs else None)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, kwargs if kwargs else None)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, kwargs if kwargs else None)

    def critical(self, message: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, message, kwargs if kwargs else None)

    def with_fields(self, **fields: Any) -> "StructuredLogger":
        """Return a new logger with additional context fields."""
        new_logger = StructuredLogger(self.logger.name, self.logger.level)
        new_logger._extra_fields = {**self._extra_fields, **fields}
        return new_logger


def setup_structured_logging(
    service_name: str = "cad-ml-platform",
    environment: str = "production",
    level: int = logging.INFO,
    json_output: bool = True,
) -> None:
    """Configure structured logging for the application."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if json_output:
        formatter = StructuredFormatter(
            service_name=service_name,
            environment=environment,
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)


def log_execution_time(logger: Optional[StructuredLogger] = None) -> Callable[[F], F]:
    """Decorator to log function execution time."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            log = logger or get_logger(func.__module__)
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                log.info(
                    f"{func.__name__} completed",
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    status="success",
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                log.error(
                    f"{func.__name__} failed",
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    status="error",
                    error=str(e),
                )
                raise

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            log = logger or get_logger(func.__module__)
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                log.info(
                    f"{func.__name__} completed",
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    status="success",
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                log.error(
                    f"{func.__name__} failed",
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    status="error",
                    error=str(e),
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> str:
    """Set request context for logging."""
    req_id = request_id or generate_request_id()
    request_id_var.set(req_id)
    if user_id:
        user_id_var.set(user_id)
    if tenant_id:
        tenant_id_var.set(tenant_id)
    return req_id


def clear_request_context() -> None:
    """Clear request context."""
    request_id_var.set(None)
    user_id_var.set(None)
    tenant_id_var.set(None)
