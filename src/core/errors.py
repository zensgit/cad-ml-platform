"""Shared error codes for API responses.

Centralizes error code enumeration to ensure consistency across
Vision and OCR endpoints (replaces scattered Literal usages).
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    INPUT_ERROR = "INPUT_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    PROVIDER_DOWN = "PROVIDER_DOWN"
    RATE_LIMIT = "RATE_LIMIT"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    TIMEOUT = "TIMEOUT"
    # New detailed error codes for provider exceptions
    NETWORK_ERROR = "NETWORK_ERROR"  # Network connectivity issues
    PARSE_FAILED = "PARSE_FAILED"  # Response parsing failures
    AUTH_FAILED = "AUTH_FAILED"  # Authentication/API key issues
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"  # Service quota/limit exceeded
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"  # Unsupported file/image format
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"  # Provider-specific timeout
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"  # Model initialization failure
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"  # Memory/CPU resource limits
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"  # External service/provider errors


__all__ = ["ErrorCode"]
