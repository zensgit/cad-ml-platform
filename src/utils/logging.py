"""Structured logging setup for CAD ML Platform.

Features:
- JSON formatted logs for aggregation
- Request correlation IDs
- Configurable log levels
- Environment-aware formatting
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

# Re-export from structured module for convenience
try:
    from src.core.logging.structured import (
        StructuredLogger,
        get_logger,
        set_request_context,
        clear_request_context,
        generate_request_id,
        log_execution_time,
        request_id_var,
        user_id_var,
        tenant_id_var,
    )
except ImportError:
    # Fallback if structured module not available
    StructuredLogger = None  # type: ignore
    get_logger = None  # type: ignore
    set_request_context = None  # type: ignore
    clear_request_context = None  # type: ignore
    generate_request_id = None  # type: ignore
    log_execution_time = None  # type: ignore
    request_id_var = None  # type: ignore
    user_id_var = None  # type: ignore
    tenant_id_var = None  # type: ignore


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(
        self,
        service_name: str = "cad-ml-platform",
        include_timestamp: bool = True,
        include_location: bool = False,
    ):
        super().__init__()
        self.service_name = service_name
        self.include_timestamp = include_timestamp
        self.include_location = include_location

    def format(self, record: logging.LogRecord) -> str:
        data: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "service": self.service_name,
        }

        if self.include_timestamp:
            data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if self.include_location:
            data["location"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add request context if available
        if request_id_var is not None:
            try:
                if request_id := request_id_var.get():
                    data["request_id"] = request_id
                if user_id_var and (user_id := user_id_var.get()):
                    data["user_id"] = user_id
                if tenant_id_var and (tenant_id := tenant_id_var.get()):
                    data["tenant_id"] = tenant_id
            except Exception:
                pass

        # Common structured fields emitted by subsystems
        structured_attrs = [
            # OCR subsystem
            "provider",
            "image_hash",
            "latency_ms",
            "fallback_level",
            "error_code",
            "error",
            "stage",
            "trace_id",
            "extraction_mode",
            "completeness",
            "calibrated_confidence",
            "dimensions_count",
            "symbols_count",
            "stages_latency_ms",
            # API subsystem
            "endpoint",
            "method",
            "status_code",
            "duration_ms",
            "client_ip",
            # ML subsystem
            "model",
            "model_version",
            "inference_time_ms",
            "batch_size",
            "confidence",
            # Batch processing
            "job_id",
            "job_status",
            "items_processed",
            "items_failed",
        ]

        for attr in structured_attrs:
            if hasattr(record, attr):
                data[attr] = getattr(record, attr)

        # Add extra fields if present
        if hasattr(record, "extra_fields") and record.extra_fields:
            data.update(record.extra_fields)

        # Add exception info
        if record.exc_info:
            import traceback

            data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(data, ensure_ascii=False, default=str)


class SimpleFormatter(logging.Formatter):
    """Simple text formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} - {record.name} - {record.levelname} - {record.getMessage()}"


def setup_logging(
    level: Optional[str] = None,
    json_output: Optional[bool] = None,
    service_name: str = "cad-ml-platform",
) -> None:
    """Setup logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to LOG_LEVEL env var or INFO.
        json_output: Use JSON formatting. Defaults to LOG_FORMAT env var or True in production.
        service_name: Service name for log entries.
    """
    # Determine log level
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Determine output format
    if json_output is None:
        log_format = os.getenv("LOG_FORMAT", "json").lower()
        json_output = log_format == "json"

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    if json_output:
        formatter = JsonFormatter(
            service_name=service_name,
            include_timestamp=True,
            include_location=os.getenv("LOG_INCLUDE_LOCATION", "false").lower() == "true",
        )
    else:
        formatter = SimpleFormatter()

    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Reduce noise from third-party libraries
    for lib in ["urllib3", "httpx", "httpcore", "asyncio"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


# Convenience exports
__all__ = [
    "setup_logging",
    "JsonFormatter",
    "SimpleFormatter",
    "StructuredLogger",
    "get_logger",
    "set_request_context",
    "clear_request_context",
    "generate_request_id",
    "log_execution_time",
]
