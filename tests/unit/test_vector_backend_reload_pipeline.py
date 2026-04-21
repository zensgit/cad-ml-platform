from __future__ import annotations

import os

import pytest
from fastapi import HTTPException

from src.api.v1.vectors import VectorBackendReloadResponse
from src.core.vector_backend_reload_pipeline import run_vector_backend_reload_pipeline


class _Metric:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def labels(self, **kwargs):  # noqa: ANN003, ANN202
        self.calls.append(kwargs)
        return self

    def inc(self):  # noqa: ANN202
        return None


class _ErrorCode:
    class INPUT_VALIDATION_FAILED:
        value = "INPUT_VALIDATION_FAILED"

    class INTERNAL_ERROR:
        value = "INTERNAL_ERROR"


@pytest.mark.asyncio
async def test_vector_backend_reload_pipeline_rejects_invalid_backend():
    metric = _Metric()
    with pytest.raises(HTTPException) as excinfo:
        await run_vector_backend_reload_pipeline(
            backend="invalid",
            reload_backend_fn=lambda: True,
            reload_metric=metric,
            error_code_cls=_ErrorCode,
            build_error_fn=lambda *args, **kwargs: {"args": args, **kwargs},  # noqa: ARG005
            response_cls=VectorBackendReloadResponse,
        )

    assert excinfo.value.status_code == 400
    assert metric.calls[0]["reason"] == "invalid_backend"


@pytest.mark.asyncio
async def test_vector_backend_reload_pipeline_success_sets_backend_env(monkeypatch):
    metric = _Metric()
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "memory")

    result = await run_vector_backend_reload_pipeline(
        backend="faiss",
        reload_backend_fn=lambda: True,
        reload_metric=metric,
        error_code_cls=_ErrorCode,
        build_error_fn=lambda *args, **kwargs: {"args": args, **kwargs},  # noqa: ARG005
        response_cls=VectorBackendReloadResponse,
    )

    assert result.status == "ok"
    assert result.backend == "faiss"
    assert os.getenv("VECTOR_STORE_BACKEND") == "faiss"
    assert metric.calls[-1] == {"status": "success", "reason": "ok"}

