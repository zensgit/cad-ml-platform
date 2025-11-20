"""Unified OCR exception types and codes (OCR_001-999 range)."""

from __future__ import annotations

from typing import Optional
from src.core.errors import ErrorCode


class OcrError(Exception):
    def __init__(self, code: ErrorCode | str, message: str, provider: Optional[str] = None, stage: Optional[str] = None):
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


# Common codes reserved (extend as needed during Week1):
OCR_ERRORS = {
    "PARSE_FAIL": "OCR_001",
    "TIMEOUT": "OCR_002",
    "PROVIDER_DOWN": "OCR_003",
    "INVALID_INPUT": "OCR_004",
}
