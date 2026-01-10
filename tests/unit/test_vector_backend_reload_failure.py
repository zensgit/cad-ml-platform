"""Tests: vector store backend reload failure returns structured error without altering backend."""

from __future__ import annotations

import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.core.errors_extended import ErrorCode
from src.main import app

client = TestClient(app)


def _read_metric(counter):
    try:
        return int(counter._value.get())  # type: ignore[attr-defined]
    except Exception:
        return None


def test_vector_backend_reload_failure_keeps_original_backend() -> None:
    original_backend = os.environ.get("VECTOR_STORE_BACKEND")
    os.environ["VECTOR_STORE_BACKEND"] = "memory"
    try:
        with patch("src.core.similarity.reload_vector_store_backend") as mock_reload:
            response = client.post(
                "/api/v1/vectors/backend/reload?backend=nonexistent",
                headers={"X-Admin-Token": "test"},
            )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        detail = data["detail"]
        assert detail["code"] == ErrorCode.INPUT_VALIDATION_FAILED.value
        assert detail["stage"] == "backend_reload"
        if "context" in detail:
            assert detail["context"].get("backend") == "nonexistent"
        mock_reload.assert_not_called()
        assert os.getenv("VECTOR_STORE_BACKEND") == "memory"
    finally:
        if original_backend is None:
            os.environ.pop("VECTOR_STORE_BACKEND", None)
        else:
            os.environ["VECTOR_STORE_BACKEND"] = original_backend


def test_vector_backend_reload_missing_admin_token() -> None:
    from src.utils.analysis_metrics import vector_store_reload_total

    counter = vector_store_reload_total.labels(status="error", reason="auth_failed")
    before = _read_metric(counter)

    response = client.post("/api/v1/vectors/backend/reload")
    assert response.status_code == 401
    data = response.json()
    assert "detail" in data
    detail = data["detail"]
    assert detail["code"] == ErrorCode.AUTHORIZATION_FAILED.value
    if before is not None:
        after = _read_metric(counter)
        assert after == before + 1
