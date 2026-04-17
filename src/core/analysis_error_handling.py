"""Shared error handling helpers for analyze flows."""

from __future__ import annotations

from typing import Any, NoReturn

from fastapi import HTTPException

from src.core.errors_extended import ErrorCode, build_error
from src.utils.analysis_metrics import (
    analysis_error_code_total,
    analysis_errors_total,
    analysis_requests_total,
)


def handle_analysis_options_json_error() -> NoReturn:
    analysis_requests_total.labels(status="error").inc()
    analysis_errors_total.labels(stage="options", code="json_decode").inc()
    analysis_error_code_total.labels(code=ErrorCode.JSON_PARSE_ERROR.value).inc()
    err = build_error(
        ErrorCode.JSON_PARSE_ERROR,
        stage="options",
        message="Invalid options JSON format",
    )
    raise HTTPException(status_code=400, detail=err)


def handle_analysis_http_exception(exc: HTTPException) -> NoReturn:
    analysis_requests_total.labels(status="error").inc()

    code = ErrorCode.INTERNAL_ERROR
    if exc.status_code == 400:
        code = ErrorCode.INPUT_ERROR
    elif exc.status_code == 404:
        code = ErrorCode.DATA_NOT_FOUND
    elif exc.status_code == 413:
        code = ErrorCode.INPUT_SIZE_EXCEEDED
    elif exc.status_code == 422:
        code = ErrorCode.BUSINESS_RULE_VIOLATION

    analysis_errors_total.labels(stage="general", code=str(exc.status_code)).inc()
    if isinstance(exc.detail, dict):
        raise exc

    analysis_error_code_total.labels(code=code.value).inc()
    err = build_error(code, stage="analysis", message=str(exc.detail))
    raise HTTPException(status_code=exc.status_code, detail=err)


def handle_analysis_unexpected_exception(
    *,
    file_name: str,
    exc: Exception,
    logger_instance: Any,
) -> NoReturn:
    analysis_requests_total.labels(status="error").inc()
    analysis_errors_total.labels(
        stage="general",
        code=ErrorCode.INTERNAL_ERROR.value,
    ).inc()
    analysis_error_code_total.labels(code=ErrorCode.INTERNAL_ERROR.value).inc()
    logger_instance.error(f"Analysis failed for {file_name}: {str(exc)}")
    err = build_error(
        ErrorCode.INTERNAL_ERROR,
        stage="analysis",
        message=f"Analysis failed: {str(exc)}",
    )
    raise HTTPException(status_code=500, detail=err)


__all__ = [
    "handle_analysis_http_exception",
    "handle_analysis_options_json_error",
    "handle_analysis_unexpected_exception",
]
