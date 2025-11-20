"""Unified OCR exception types and codes (OCR_001-999 range)."""

from __future__ import annotations

from typing import Optional

from src.core.errors import ErrorCode


class OcrError(Exception):
    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        provider: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        super().__init__(message)
        self.code = code
        self.provider = provider
        self.stage = stage
        self.message = message

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "provider": self.provider,
            "stage": self.stage,
        }


# Legacy OCR_ERRORS dict - DEPRECATED
# Use ErrorCode enum directly from src.core.errors instead
# Kept temporarily for backward compatibility
OCR_ERRORS = {
    "PARSE_FAIL": ErrorCode.PARSE_FAILED.value,
    "TIMEOUT": ErrorCode.TIMEOUT.value,
    "PROVIDER_DOWN": ErrorCode.PROVIDER_DOWN.value,
    "INVALID_INPUT": ErrorCode.INPUT_ERROR.value,
}
