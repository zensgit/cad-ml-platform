from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.core.legacy_redirect_pipeline import (
    build_legacy_redirect_exception,
    raise_legacy_redirect,
)


def test_build_legacy_redirect_exception_returns_structured_410():
    exc = build_legacy_redirect_exception(
        old_path="/api/v1/analyze/faiss/health",
        new_path="/api/v1/health/faiss",
        method="GET",
    )

    assert isinstance(exc, HTTPException)
    assert exc.status_code == 410
    assert exc.detail["code"] == "RESOURCE_GONE"
    assert exc.detail["context"]["deprecated_path"] == "/api/v1/analyze/faiss/health"
    assert exc.detail["context"]["new_path"] == "/api/v1/health/faiss"
    assert exc.detail["context"]["method"] == "GET"


def test_raise_legacy_redirect_raises_http_exception():
    with pytest.raises(HTTPException) as excinfo:
        raise_legacy_redirect(
            old_path="/api/v1/analyze/model/reload",
            new_path="/api/v1/model/reload",
            method="POST",
        )

    exc = excinfo.value
    assert exc.status_code == 410
    assert exc.detail["context"]["method"] == "POST"
