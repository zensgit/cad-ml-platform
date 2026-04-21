from __future__ import annotations

from datetime import datetime

import pytest
from fastapi import HTTPException

from src.api.v1.vector_migration_models import VectorMigrateItem, VectorMigrateResponse
from src.core.vector_migrate_pipeline import run_vector_migrate_pipeline


class _Metric:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.observations: list[float] = []

    def labels(self, **kwargs):  # noqa: ANN003, ANN202
        self.calls.append(kwargs)
        return self

    def inc(self):  # noqa: ANN202
        return None

    def observe(self, value):  # noqa: ANN001, ANN202
        self.observations.append(float(value))


class _ErrorCode:
    class INPUT_VALIDATION_FAILED:
        value = "INPUT_VALIDATION_FAILED"


class _Extractor:
    def __init__(self, *, feature_version: str) -> None:
        self.feature_version = feature_version

    def upgrade_vector(self, existing, current_version=None):  # noqa: ANN001, ANN202
        return list(existing) + [9.0]

    def expected_dim(self, version: str) -> int:
        return {"v1": 7, "v2": 12, "v3": 24, "v4": 26}[version]


class _Payload:
    def __init__(self, ids: list[str], to_version: str, dry_run: bool) -> None:
        self.ids = ids
        self.to_version = to_version
        self.dry_run = dry_run


@pytest.mark.asyncio
async def test_run_vector_migrate_pipeline_rejects_invalid_target_version():
    with pytest.raises(HTTPException) as excinfo:
        await run_vector_migrate_pipeline(
            payload=_Payload(["vec1"], "v999", True),
            vector_store={},
            vector_meta={},
            qdrant_store=None,
            feature_extractor_cls=_Extractor,
            prepare_vector_for_upgrade_fn=lambda *args: ([1.0], [], "v1"),  # noqa: ARG005
            vector_layout_base="base",
            vector_layout_l3="l3",
            dimension_delta_metric=_Metric(),
            migrate_total_metric=_Metric(),
            analysis_error_code_total_metric=_Metric(),
            error_code_cls=_ErrorCode,
            build_error_fn=lambda *args, **kwargs: {"args": args, **kwargs},  # noqa: ARG005
            item_cls=VectorMigrateItem,
            response_cls=VectorMigrateResponse,
            history=[],
            uuid4_fn=lambda: "mig-id",
            utcnow_fn=lambda: datetime(2026, 4, 21, 10, 0, 0),
        )

    assert excinfo.value.status_code == 422


@pytest.mark.asyncio
async def test_run_vector_migrate_pipeline_memory_dry_run_records_history():
    history: list[dict[str, object]] = []
    response = await run_vector_migrate_pipeline(
        payload=_Payload(["vec1"], "v2", True),
        vector_store={"vec1": [1.0] * 7},
        vector_meta={"vec1": {"feature_version": "v1"}},
        qdrant_store=None,
        feature_extractor_cls=_Extractor,
        prepare_vector_for_upgrade_fn=lambda *args: ([1.0] * 7, [], "v1"),  # noqa: ARG005
        vector_layout_base="base",
        vector_layout_l3="l3",
        dimension_delta_metric=_Metric(),
        migrate_total_metric=_Metric(),
        analysis_error_code_total_metric=_Metric(),
        error_code_cls=_ErrorCode,
        build_error_fn=lambda *args, **kwargs: {"args": args, **kwargs},  # noqa: ARG005
        item_cls=VectorMigrateItem,
        response_cls=VectorMigrateResponse,
        history=history,
        uuid4_fn=lambda: "mig-id",
        utcnow_fn=lambda: datetime(2026, 4, 21, 10, 0, 0),
    )

    assert response.total == 1
    assert response.dry_run_total == 1
    assert response.items[0].status == "dry_run"
    assert history[-1]["migration_id"] == "mig-id"
    assert history[-1]["counts"]["dry_run"] == 1

