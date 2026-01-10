"""Centralized error mapping for OCR providers.

Maps exceptions to ErrorCode values for consistent metrics labeling.
"""

import asyncio
import logging
from typing import Optional

from src.core.errors import ErrorCode

logger = logging.getLogger(__name__)


def map_exception_to_error_code(exc: Exception) -> ErrorCode:
    """Map an exception to the appropriate ErrorCode.

    Args:
        exc: The exception to map

    Returns:
        ErrorCode enum value for metrics labeling

    Mapping:
        - MemoryError → RESOURCE_EXHAUSTED
        - TimeoutError/asyncio.TimeoutError → PROVIDER_TIMEOUT
        - ValueError (parse related) → PARSE_FAILED
        - IOError/OSError → IO_ERROR (if we add it)
        - ConnectionError → NETWORK_ERROR
        - AuthenticationError → AUTH_FAILED
        - Generic → INTERNAL_ERROR
    """
    # Memory exhaustion
    if isinstance(exc, MemoryError):
        return ErrorCode.RESOURCE_EXHAUSTED

    # Timeout errors
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return ErrorCode.PROVIDER_TIMEOUT

    # Parse errors (typically ValueError with parse-related message)
    if isinstance(exc, ValueError):
        error_msg = str(exc).lower()
        if any(term in error_msg for term in ["parse", "decode", "invalid format", "malformed"]):
            return ErrorCode.PARSE_FAILED
        # Other ValueErrors are input errors
        return ErrorCode.INPUT_ERROR

    # Network/connection errors
    if isinstance(exc, (ConnectionError, ConnectionRefusedError, ConnectionAbortedError)):
        return ErrorCode.NETWORK_ERROR

    # IO errors (file system, permissions)
    if isinstance(exc, (IOError, OSError)):
        error_msg = str(exc).lower()
        if "permission" in error_msg:
            return ErrorCode.AUTH_FAILED  # Permission denied treated as auth
        return ErrorCode.NETWORK_ERROR  # IO errors often network-related

    # Authentication/permission errors
    if type(exc).__name__ in ["AuthenticationError", "PermissionError"]:
        return ErrorCode.AUTH_FAILED

    # Quota/rate limit errors
    if type(exc).__name__ in ["QuotaExceededError", "RateLimitError"]:
        return ErrorCode.QUOTA_EXCEEDED

    # Model loading errors
    if type(exc).__name__ in ["ModelNotFoundError", "ModelLoadError"]:
        return ErrorCode.MODEL_LOAD_ERROR

    # Check exception message for additional patterns
    error_msg = str(exc).lower()

    # Network-related patterns
    if any(term in error_msg for term in ["network", "connection", "socket", "dns", "resolve"]):
        return ErrorCode.NETWORK_ERROR

    # Timeout patterns
    if any(term in error_msg for term in ["timeout", "timed out", "deadline"]):
        return ErrorCode.PROVIDER_TIMEOUT

    # Resource patterns
    if any(term in error_msg for term in ["memory", "resource", "exhausted", "oom"]):
        return ErrorCode.RESOURCE_EXHAUSTED

    # Auth patterns
    if any(term in error_msg for term in ["auth", "permission", "forbidden", "unauthorized"]):
        return ErrorCode.AUTH_FAILED

    # Parse patterns
    if any(term in error_msg for term in ["parse", "decode", "invalid", "malformed"]):
        return ErrorCode.PARSE_FAILED

    # Model patterns
    if any(term in error_msg for term in ["model", "load", "initialize"]):
        return ErrorCode.MODEL_LOAD_ERROR

    # Default to internal error
    return ErrorCode.INTERNAL_ERROR


def log_and_map_exception(
    exc: Exception, provider: str, stage: str, context: Optional[str] = None
) -> ErrorCode:
    """Log exception details and return mapped ErrorCode.

    Args:
        exc: The exception that occurred
        provider: Provider name for logging
        stage: Processing stage where error occurred
        context: Optional additional context

    Returns:
        ErrorCode for metrics labeling
    """
    error_code = map_exception_to_error_code(exc)

    # Log at appropriate level based on error type
    log_message = f"Provider {provider} error at {stage}: {type(exc).__name__}: {exc}"
    if context:
        log_message += f" | Context: {context}"

    # Log original exception details at debug level for troubleshooting
    logger.debug(f"Original exception details: {log_message}")

    # Log summary at appropriate level
    if error_code in [ErrorCode.RESOURCE_EXHAUSTED, ErrorCode.PROVIDER_TIMEOUT]:
        logger.error(log_message)
    elif error_code in [ErrorCode.NETWORK_ERROR, ErrorCode.AUTH_FAILED]:
        logger.warning(log_message)
    else:
        logger.info(log_message)

    return error_code


# Convenience functions for common error scenarios
def handle_inference_error(exc: Exception, provider: str) -> ErrorCode:
    """Handle errors during model inference."""
    return log_and_map_exception(exc, provider, "infer")


def handle_parse_error(exc: Exception, provider: str) -> ErrorCode:
    """Handle errors during result parsing."""
    # Parse errors are always PARSE_FAILED unless it's a different type
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
        return ErrorCode.PARSE_FAILED
    return log_and_map_exception(exc, provider, "parse")


def handle_init_error(exc: Exception, provider: str) -> ErrorCode:
    """Handle errors during provider initialization."""
    return log_and_map_exception(exc, provider, "init")


def handle_load_error(exc: Exception, provider: str) -> ErrorCode:
    """Handle errors during model loading."""
    # Loading errors default to MODEL_LOAD_ERROR unless memory-related
    if isinstance(exc, MemoryError):
        return ErrorCode.RESOURCE_EXHAUSTED
    return ErrorCode.MODEL_LOAD_ERROR
