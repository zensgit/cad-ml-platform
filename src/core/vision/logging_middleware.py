"""Request/response logging middleware for vision providers.

Provides:
- Structured request/response logging
- Timing and performance metrics
- Error tracking and alerting
- Log aggregation support
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for vision operations."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogDestination(Enum):
    """Supported log destinations."""

    CONSOLE = "console"
    FILE = "file"
    JSON_FILE = "json_file"
    CALLBACK = "callback"
    STRUCTURED = "structured"


@dataclass
class RequestLog:
    """Log entry for a vision request."""

    request_id: str
    timestamp: datetime
    provider: str
    image_size_bytes: int
    include_description: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "image_size_bytes": self.image_size_bytes,
            "include_description": self.include_description,
            "metadata": self.metadata,
        }


@dataclass
class ResponseLog:
    """Log entry for a vision response."""

    request_id: str
    timestamp: datetime
    provider: str
    success: bool
    response_time_ms: float
    result_summary: Optional[str] = None
    confidence: Optional[float] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "success": self.success,
            "response_time_ms": self.response_time_ms,
            "result_summary": self.result_summary,
            "confidence": self.confidence,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "metadata": self.metadata,
        }


@dataclass
class LoggingConfig:
    """Configuration for logging middleware."""

    # Log levels
    request_log_level: LogLevel = LogLevel.INFO
    response_log_level: LogLevel = LogLevel.INFO
    error_log_level: LogLevel = LogLevel.ERROR

    # Content options
    log_image_hash: bool = True
    log_result_summary: bool = True
    max_summary_length: int = 200
    include_request_metadata: bool = True
    include_response_metadata: bool = True

    # Performance tracking
    slow_request_threshold_ms: float = 5000.0
    track_percentiles: bool = True

    # Destinations
    destinations: List[LogDestination] = field(
        default_factory=lambda: [LogDestination.CONSOLE]
    )

    # File logging
    log_file_path: Optional[str] = None
    json_log_file_path: Optional[str] = None
    max_log_file_size_mb: float = 100.0
    log_rotation_count: int = 5


class LogHandler(ABC):
    """Abstract base class for log handlers."""

    @abstractmethod
    def handle_request(self, log: RequestLog) -> None:
        """Handle a request log entry."""
        pass

    @abstractmethod
    def handle_response(self, log: ResponseLog) -> None:
        """Handle a response log entry."""
        pass


class ConsoleLogHandler(LogHandler):
    """Handler for console logging."""

    def __init__(self, config: LoggingConfig):
        self._config = config
        self._logger = logging.getLogger("vision.middleware")

    def _get_log_func(self, level: LogLevel) -> Callable:
        """Get the appropriate logging function."""
        level_map = {
            LogLevel.DEBUG: self._logger.debug,
            LogLevel.INFO: self._logger.info,
            LogLevel.WARNING: self._logger.warning,
            LogLevel.ERROR: self._logger.error,
            LogLevel.CRITICAL: self._logger.critical,
        }
        return level_map.get(level, self._logger.info)

    def handle_request(self, log: RequestLog) -> None:
        """Log request to console."""
        log_func = self._get_log_func(self._config.request_log_level)
        log_func(
            f"[{log.request_id}] Vision request to {log.provider} "
            f"(image: {log.image_size_bytes} bytes)"
        )

    def handle_response(self, log: ResponseLog) -> None:
        """Log response to console."""
        if log.success:
            log_func = self._get_log_func(self._config.response_log_level)
            status = "✓"
            details = f"confidence={log.confidence:.2f}" if log.confidence else ""
        else:
            log_func = self._get_log_func(self._config.error_log_level)
            status = "✗"
            details = f"error={log.error_type}: {log.error_message}"

        slow_marker = ""
        if log.response_time_ms > self._config.slow_request_threshold_ms:
            slow_marker = " [SLOW]"

        log_func(
            f"[{log.request_id}] Vision response from {log.provider} {status} "
            f"({log.response_time_ms:.0f}ms){slow_marker} {details}"
        )


class FileLogHandler(LogHandler):
    """Handler for file logging."""

    def __init__(self, config: LoggingConfig):
        self._config = config
        self._file_path = config.log_file_path

    def _write_log(self, message: str) -> None:
        """Write message to log file."""
        if not self._file_path:
            return
        try:
            with open(self._file_path, "a") as f:
                f.write(f"{message}\n")
        except IOError as e:
            logger.warning(f"Failed to write to log file: {e}")

    def handle_request(self, log: RequestLog) -> None:
        """Log request to file."""
        message = (
            f"{log.timestamp.isoformat()} REQUEST [{log.request_id}] "
            f"provider={log.provider} size={log.image_size_bytes}"
        )
        self._write_log(message)

    def handle_response(self, log: ResponseLog) -> None:
        """Log response to file."""
        status = "SUCCESS" if log.success else "ERROR"
        message = (
            f"{log.timestamp.isoformat()} RESPONSE [{log.request_id}] "
            f"provider={log.provider} status={status} "
            f"time_ms={log.response_time_ms:.2f}"
        )
        if log.error_message:
            message += f" error={log.error_message}"
        self._write_log(message)


class JSONLogHandler(LogHandler):
    """Handler for JSON file logging."""

    def __init__(self, config: LoggingConfig):
        self._config = config
        self._file_path = config.json_log_file_path

    def _write_json(self, data: Dict[str, Any]) -> None:
        """Write JSON to log file."""
        if not self._file_path:
            return
        try:
            with open(self._file_path, "a") as f:
                f.write(json.dumps(data) + "\n")
        except IOError as e:
            logger.warning(f"Failed to write to JSON log file: {e}")

    def handle_request(self, log: RequestLog) -> None:
        """Log request as JSON."""
        data = {"type": "request", **log.to_dict()}
        self._write_json(data)

    def handle_response(self, log: ResponseLog) -> None:
        """Log response as JSON."""
        data = {"type": "response", **log.to_dict()}
        self._write_json(data)


class CallbackLogHandler(LogHandler):
    """Handler that calls user-provided callbacks."""

    def __init__(
        self,
        request_callback: Optional[Callable[[RequestLog], None]] = None,
        response_callback: Optional[Callable[[ResponseLog], None]] = None,
    ):
        self._request_callback = request_callback
        self._response_callback = response_callback

    def handle_request(self, log: RequestLog) -> None:
        """Call request callback."""
        if self._request_callback:
            try:
                self._request_callback(log)
            except Exception as e:
                logger.warning(f"Request callback failed: {e}")

    def handle_response(self, log: ResponseLog) -> None:
        """Call response callback."""
        if self._response_callback:
            try:
                self._response_callback(log)
            except Exception as e:
                logger.warning(f"Response callback failed: {e}")


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time_ms: float = 0.0
    min_response_time_ms: float = float("inf")
    max_response_time_ms: float = 0.0
    slow_requests: int = 0

    # Percentile tracking
    _response_times: List[float] = field(default_factory=list)

    @property
    def avg_response_time_ms(self) -> float:
        """Calculate average response time."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time_ms / self.successful_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def record(
        self, success: bool, response_time_ms: float, is_slow: bool = False
    ) -> None:
        """Record a request result."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.total_response_time_ms += response_time_ms
            self.min_response_time_ms = min(
                self.min_response_time_ms, response_time_ms
            )
            self.max_response_time_ms = max(
                self.max_response_time_ms, response_time_ms
            )
            self._response_times.append(response_time_ms)
            # Keep only last 1000 samples for percentiles
            if len(self._response_times) > 1000:
                self._response_times.pop(0)
        else:
            self.failed_requests += 1

        if is_slow:
            self.slow_requests += 1

    def get_percentile(self, p: float) -> float:
        """Get response time percentile (0-100)."""
        if not self._response_times:
            return 0.0
        sorted_times = sorted(self._response_times)
        idx = int(len(sorted_times) * p / 100)
        idx = min(idx, len(sorted_times) - 1)
        return sorted_times[idx]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
            "min_response_time_ms": (
                self.min_response_time_ms
                if self.min_response_time_ms != float("inf")
                else 0.0
            ),
            "max_response_time_ms": self.max_response_time_ms,
            "slow_requests": self.slow_requests,
        }

        # Add percentiles if available
        if self._response_times:
            result["p50_ms"] = self.get_percentile(50)
            result["p90_ms"] = self.get_percentile(90)
            result["p95_ms"] = self.get_percentile(95)
            result["p99_ms"] = self.get_percentile(99)

        return result


class LoggingMiddleware:
    """
    Middleware for logging vision provider requests and responses.

    Features:
    - Multiple log destinations
    - Structured logging support
    - Performance metrics tracking
    - Slow request detection
    """

    def __init__(self, config: Optional[LoggingConfig] = None):
        """
        Initialize logging middleware.

        Args:
            config: Logging configuration
        """
        self._config = config or LoggingConfig()
        self._handlers: List[LogHandler] = []
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._global_metrics = PerformanceMetrics()

        # Initialize handlers based on config
        self._initialize_handlers()

    def _initialize_handlers(self) -> None:
        """Initialize log handlers based on configuration."""
        for dest in self._config.destinations:
            if dest == LogDestination.CONSOLE:
                self._handlers.append(ConsoleLogHandler(self._config))
            elif dest == LogDestination.FILE:
                if self._config.log_file_path:
                    self._handlers.append(FileLogHandler(self._config))
            elif dest == LogDestination.JSON_FILE:
                if self._config.json_log_file_path:
                    self._handlers.append(JSONLogHandler(self._config))

    def add_handler(self, handler: LogHandler) -> None:
        """Add a custom log handler."""
        self._handlers.append(handler)

    def add_callback_handler(
        self,
        request_callback: Optional[Callable[[RequestLog], None]] = None,
        response_callback: Optional[Callable[[ResponseLog], None]] = None,
    ) -> None:
        """Add callback-based handler."""
        handler = CallbackLogHandler(request_callback, response_callback)
        self._handlers.append(handler)

    def log_request(
        self,
        provider: str,
        image_size_bytes: int,
        include_description: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log a request and return request ID for correlation.

        Args:
            provider: Provider name
            image_size_bytes: Size of image data
            include_description: Whether description is requested
            metadata: Additional metadata

        Returns:
            Request ID for correlation with response
        """
        request_id = str(uuid.uuid4())[:8]
        log = RequestLog(
            request_id=request_id,
            timestamp=datetime.now(),
            provider=provider,
            image_size_bytes=image_size_bytes,
            include_description=include_description,
            metadata=metadata or {} if self._config.include_request_metadata else {},
        )

        for handler in self._handlers:
            try:
                handler.handle_request(log)
            except Exception as e:
                logger.warning(f"Log handler failed: {e}")

        return request_id

    def log_response(
        self,
        request_id: str,
        provider: str,
        success: bool,
        response_time_ms: float,
        result: Optional[VisionDescription] = None,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a response.

        Args:
            request_id: Correlation ID from log_request
            provider: Provider name
            success: Whether request succeeded
            response_time_ms: Response time in milliseconds
            result: Vision result if successful
            error: Exception if failed
            metadata: Additional metadata
        """
        result_summary = None
        confidence = None

        if result and self._config.log_result_summary:
            result_summary = result.summary[: self._config.max_summary_length]
            confidence = result.confidence

        error_message = None
        error_type = None
        if error:
            error_type = type(error).__name__
            error_message = str(error)

        is_slow = response_time_ms > self._config.slow_request_threshold_ms

        log = ResponseLog(
            request_id=request_id,
            timestamp=datetime.now(),
            provider=provider,
            success=success,
            response_time_ms=response_time_ms,
            result_summary=result_summary,
            confidence=confidence,
            error_message=error_message,
            error_type=error_type,
            metadata=metadata or {} if self._config.include_response_metadata else {},
        )

        for handler in self._handlers:
            try:
                handler.handle_response(log)
            except Exception as e:
                logger.warning(f"Log handler failed: {e}")

        # Update metrics
        self._global_metrics.record(success, response_time_ms, is_slow)

        if provider not in self._metrics:
            self._metrics[provider] = PerformanceMetrics()
        self._metrics[provider].record(success, response_time_ms, is_slow)

    def get_metrics(self, provider: Optional[str] = None) -> PerformanceMetrics:
        """
        Get performance metrics.

        Args:
            provider: Optional provider name for provider-specific metrics

        Returns:
            PerformanceMetrics instance
        """
        if provider:
            return self._metrics.get(provider, PerformanceMetrics())
        return self._global_metrics

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        return {
            "global": self._global_metrics.to_dict(),
            "by_provider": {
                name: metrics.to_dict() for name, metrics in self._metrics.items()
            },
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._global_metrics = PerformanceMetrics()
        self._metrics.clear()


class LoggingVisionProvider:
    """
    Wrapper that adds logging to any VisionProvider.

    Automatically logs requests and responses with timing.
    """

    def __init__(
        self,
        provider: VisionProvider,
        middleware: LoggingMiddleware,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize logging provider.

        Args:
            provider: The underlying vision provider
            middleware: LoggingMiddleware instance
            extra_metadata: Extra metadata to include in all logs
        """
        self._provider = provider
        self._middleware = middleware
        self._extra_metadata = extra_metadata or {}

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VisionDescription:
        """
        Analyze image with logging.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description
            metadata: Additional request metadata

        Returns:
            VisionDescription with analysis results

        Raises:
            VisionProviderError: If analysis fails
        """
        # Combine metadata
        combined_metadata = {**self._extra_metadata, **(metadata or {})}

        # Log request
        request_id = self._middleware.log_request(
            provider=self._provider.provider_name,
            image_size_bytes=len(image_data),
            include_description=include_description,
            metadata=combined_metadata,
        )

        start_time = time.time()
        try:
            result = await self._provider.analyze_image(
                image_data, include_description
            )
            response_time_ms = (time.time() - start_time) * 1000

            # Log successful response
            self._middleware.log_response(
                request_id=request_id,
                provider=self._provider.provider_name,
                success=True,
                response_time_ms=response_time_ms,
                result=result,
                metadata=combined_metadata,
            )

            return result

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            # Log failed response
            self._middleware.log_response(
                request_id=request_id,
                provider=self._provider.provider_name,
                success=False,
                response_time_ms=response_time_ms,
                error=e,
                metadata=combined_metadata,
            )

            raise

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name

    @property
    def middleware(self) -> LoggingMiddleware:
        """Get the logging middleware."""
        return self._middleware

    def get_metrics(self) -> PerformanceMetrics:
        """Get metrics for this provider."""
        return self._middleware.get_metrics(self._provider.provider_name)


# Global middleware instance
_global_middleware: Optional[LoggingMiddleware] = None


def get_logging_middleware() -> LoggingMiddleware:
    """
    Get the global logging middleware instance.

    Returns:
        LoggingMiddleware singleton
    """
    global _global_middleware
    if _global_middleware is None:
        _global_middleware = LoggingMiddleware()
    return _global_middleware


def create_logging_provider(
    provider: VisionProvider,
    config: Optional[LoggingConfig] = None,
    middleware: Optional[LoggingMiddleware] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> LoggingVisionProvider:
    """
    Factory to create a logging provider wrapper.

    Args:
        provider: The underlying vision provider
        config: Logging configuration (ignored if middleware provided)
        middleware: Optional existing middleware instance
        extra_metadata: Extra metadata for all logs

    Returns:
        LoggingVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> logged = create_logging_provider(
        ...     provider,
        ...     config=LoggingConfig(
        ...         destinations=[LogDestination.CONSOLE, LogDestination.JSON_FILE],
        ...         json_log_file_path="vision.log",
        ...         slow_request_threshold_ms=3000,
        ...     ),
        ... )
        >>> result = await logged.analyze_image(image_bytes)
        >>> print(logged.get_metrics().to_dict())
    """
    if middleware is None:
        middleware = LoggingMiddleware(config=config)

    return LoggingVisionProvider(
        provider=provider,
        middleware=middleware,
        extra_metadata=extra_metadata,
    )
