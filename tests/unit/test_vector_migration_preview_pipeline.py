from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.api.v1.vectors import VectorMigrationPreviewResponse
from src.core.errors_extended import ErrorCode, build_error
from src.core.vector_migration_preview_pipeline import run_vector_migration_preview_pipeline


@pytest.mark.asyncio
async def test_run_vector_migration_preview_pipeline_invalid_version():
    with pytest.raises(HTTPException) as exc_info:
        await run_vector_migration_preview_pipeline(
            to_version="v99",
            limit=10,
            response_cls=VectorMigrationPreviewResponse,
            error_code_cls=ErrorCode,
            build_error_fn=build_error,
            get_qdrant_store_fn=lambda: None,
            collect_qdrant_preview_samples_fn=None,  # type: ignore[arg-type]
            prepare_vector_for_upgrade_fn=None,  # type: ignore[arg-type]
            feature_extractor_cls=None,  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail["code"] == ErrorCode.INPUT_VALIDATION_FAILED.value


@pytest.mark.asyncio
async def test_run_vector_migration_preview_pipeline_qdrant_builds_distribution():
    class DummyExtractor:
        def __init__(self, *, feature_version):
            assert feature_version == "v4"

        def upgrade_vector(self, base_vector, *, current_version):
            assert current_version == "v3"
            return list(base_vector) + [9.0, 9.0]

    class DummyStore:
        pass

    async def _collect_qdrant_preview_samples(store, *, limit):  # noqa: ANN001, ANN202
        assert isinstance(store, DummyStore)
        assert limit == 2
        return [("vec1", [1.0] * 22, {"feature_version": "v3"})], 3, {"v3": 1, "v4": 2}

    def _prepare_vector_for_upgrade(extractor, vec, meta, from_version):  # noqa: ANN001, ANN202
        assert from_version == "v3"
        return vec, [], meta.get("vector_layout")

    response = await run_vector_migration_preview_pipeline(
        to_version="v4",
        limit=2,
        response_cls=VectorMigrationPreviewResponse,
        error_code_cls=ErrorCode,
        build_error_fn=build_error,
        get_qdrant_store_fn=lambda: DummyStore(),
        collect_qdrant_preview_samples_fn=_collect_qdrant_preview_samples,
        prepare_vector_for_upgrade_fn=_prepare_vector_for_upgrade,
        feature_extractor_cls=DummyExtractor,
    )

    assert response.total_vectors == 3
    assert response.by_version == {"v3": 1, "v4": 2}
    assert response.preview_items[0].status == "upgrade_preview"
    assert response.preview_items[0].dimension_before == 22
    assert response.preview_items[0].dimension_after == 24


@pytest.mark.asyncio
async def test_run_vector_migration_preview_pipeline_memory_limit_cap():
    class DummyExtractor:
        def __init__(self, *, feature_version):
            assert feature_version == "v3"

        def upgrade_vector(self, base_vector, *, current_version):
            return list(base_vector)

    vectors = {f"vec{i}": [0.1] * 7 for i in range(150)}
    meta = {f"vec{i}": {"feature_version": "v2"} for i in range(150)}
    with patch("src.core.similarity._VECTOR_STORE", vectors), patch("src.core.similarity._VECTOR_META", meta):
        response = await run_vector_migration_preview_pipeline(
            to_version="v3",
            limit=1000,
            response_cls=VectorMigrationPreviewResponse,
            error_code_cls=ErrorCode,
            build_error_fn=build_error,
            get_qdrant_store_fn=lambda: None,
            collect_qdrant_preview_samples_fn=None,  # type: ignore[arg-type]
            prepare_vector_for_upgrade_fn=lambda extractor, vec, meta, from_version: (vec, [], None),
            feature_extractor_cls=DummyExtractor,
        )

    assert response.total_vectors == 150
    assert len(response.preview_items) == 100
