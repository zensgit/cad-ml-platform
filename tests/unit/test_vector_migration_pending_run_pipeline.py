from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.core.errors_extended import ErrorCode, build_error
from src.core.vector_migration_pending_run_pipeline import (
    run_vector_migration_pending_run_pipeline,
)


@pytest.mark.asyncio
async def test_pending_run_pipeline_raises_on_partial_qdrant_without_override():
    payload = SimpleNamespace(
        limit=10,
        dry_run=True,
        from_version_filter=None,
        allow_partial_scan=False,
    )

    async def _collect_pending_candidates(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["target_version"] == "v4"
        return {
            "backend": "qdrant",
            "distribution_complete": False,
            "scanned_vectors": 2,
            "scan_limit": 2,
            "pending_ids": ["vec2"],
        }

    with pytest.raises(HTTPException) as exc_info:
        await run_vector_migration_pending_run_pipeline(
            payload=payload,
            api_key="test",
            resolve_target_version_fn=lambda: "v4",
            collect_pending_candidates_fn=_collect_pending_candidates,
            migrate_vectors_fn=None,  # type: ignore[arg-type]
            request_cls=None,  # type: ignore[arg-type]
            error_code_cls=ErrorCode,
            build_error_fn=build_error,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == ErrorCode.CONSTRAINT_VIOLATION.value


@pytest.mark.asyncio
async def test_pending_run_pipeline_calls_migrate_vectors():
    payload = SimpleNamespace(
        limit=10,
        dry_run=True,
        from_version_filter="v2",
        allow_partial_scan=True,
    )
    captured = {}

    async def _collect_pending_candidates(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["target_version"] == "v4"
        return {
            "backend": "qdrant",
            "distribution_complete": False,
            "scanned_vectors": 2,
            "scan_limit": 2,
            "pending_ids": ["vec3"],
        }

    async def _migrate_vectors(request, *, api_key):  # noqa: ANN001, ANN202
        captured["request"] = request
        captured["api_key"] = api_key
        return {"ok": True}

    request_cls = lambda **kwargs: SimpleNamespace(**kwargs)  # noqa: E731

    result = await run_vector_migration_pending_run_pipeline(
        payload=payload,
        api_key="test",
        resolve_target_version_fn=lambda: "v4",
        collect_pending_candidates_fn=_collect_pending_candidates,
        migrate_vectors_fn=_migrate_vectors,
        request_cls=request_cls,
        error_code_cls=ErrorCode,
        build_error_fn=build_error,
    )

    assert result == {"ok": True}
    assert captured["request"].ids == ["vec3"]
    assert captured["request"].to_version == "v4"
    assert captured["request"].dry_run is True
    assert captured["api_key"] == "test"
