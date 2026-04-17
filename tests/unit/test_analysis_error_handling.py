from __future__ import annotations

from fastapi import HTTPException
import pytest

from src.core.analysis_error_handling import (
    handle_analysis_http_exception,
    handle_analysis_options_json_error,
    handle_analysis_unexpected_exception,
)


class _Logger:
    def __init__(self) -> None:
        self.messages = []

    def error(self, message: str) -> None:
        self.messages.append(message)


def test_handle_analysis_options_json_error_raises_structured_http_exception():
    with pytest.raises(HTTPException) as exc_info:
        handle_analysis_options_json_error()

    exc = exc_info.value
    assert exc.status_code == 400
    assert exc.detail["code"] == "JSON_PARSE_ERROR"
    assert exc.detail["stage"] == "options"


def test_handle_analysis_http_exception_wraps_non_structured_detail():
    with pytest.raises(HTTPException) as exc_info:
        handle_analysis_http_exception(HTTPException(status_code=404, detail="missing"))

    exc = exc_info.value
    assert exc.status_code == 404
    assert exc.detail["code"] == "DATA_NOT_FOUND"
    assert exc.detail["stage"] == "analysis"


def test_handle_analysis_http_exception_preserves_structured_detail():
    original = HTTPException(status_code=422, detail={"code": "BUSINESS_RULE_VIOLATION"})

    with pytest.raises(HTTPException) as exc_info:
        handle_analysis_http_exception(original)

    assert exc_info.value is original


def test_handle_analysis_unexpected_exception_wraps_and_logs():
    logger = _Logger()

    with pytest.raises(HTTPException) as exc_info:
        handle_analysis_unexpected_exception(
            file_name="boom.dxf",
            exc=RuntimeError("boom"),
            logger_instance=logger,
        )

    exc = exc_info.value
    assert exc.status_code == 500
    assert exc.detail["code"] == "INTERNAL_ERROR"
    assert exc.detail["stage"] == "analysis"
    assert logger.messages == ["Analysis failed for boom.dxf: boom"]
