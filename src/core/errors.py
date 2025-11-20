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


__all__ = ["ErrorCode"]
