from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core import similarity as sim
from src.core.legacy_vector_migration_pipeline import (
    run_legacy_vector_migrate_pipeline,
    run_legacy_vector_migration_status_pipeline,
)


class _DummyExtractor:
    def __init__(self, feature_version: str):
        self.feature_version = feature_version

    async def extract(self, doc):  # noqa: ANN001
        return {"geometric": [1.0, 2.0], "semantic": [3.0]}

    def flatten(self, features):  # noqa: ANN001
        return [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_run_legacy_vector_migrate_pipeline_dry_run(monkeypatch):
    sim._VECTOR_STORE.clear()  # type: ignore[attr-defined]
    sim._VECTOR_META.clear()  # type: ignore[attr-defined]
    sim._VECTOR_STORE["legacy-mig"] = [0.1, 0.2, 0.3]  # type: ignore[attr-defined]
    sim._VECTOR_META["legacy-mig"] = {"feature_version": "v1"}  # type: ignore[attr-defined]
    async def _stub_get_cached_result(key):  # noqa: ANN001, ANN202
        return {
            "file_name": "legacy-mig.dxf",
            "file_format": "dxf",
            "statistics": {"bounding_box": {}},
        }

    monkeypatch.setattr(
        "src.core.legacy_vector_migration_pipeline.get_cached_result",
        _stub_get_cached_result,
    )
    monkeypatch.setattr(
        "src.core.legacy_vector_migration_pipeline.FeatureExtractor",
        _DummyExtractor,
    )

    result = await run_legacy_vector_migrate_pipeline(
        payload=SimpleNamespace(ids=["legacy-mig"], to_version="v2", dry_run=True)
    )

    assert result["total"] == 1
    assert result["migrated"] == 0
    assert result["skipped"] == 1
    assert result["dry_run_total"] == 1
    assert result["items"][0]["status"] == "dry_run"


def test_run_legacy_vector_migration_status_pipeline_reads_history(monkeypatch):
    sim._VECTOR_STORE.clear()  # type: ignore[attr-defined]
    sim._VECTOR_META.clear()  # type: ignore[attr-defined]
    sim._VECTOR_STORE["legacy-a"] = [0.1, 0.2]  # type: ignore[attr-defined]
    sim._VECTOR_META["legacy-a"] = {"feature_version": "v1"}  # type: ignore[attr-defined]
    sim._MIGRATION_STATUS = {  # type: ignore[attr-defined]
        "last_migration_id": "mig-1",
        "last_started_at": "2026-04-17T00:00:00+00:00",
        "last_finished_at": "2026-04-17T00:01:00+00:00",
        "last_total": 1,
        "last_migrated": 0,
        "last_skipped": 1,
        "history": [{"migration_id": "mig-1"}],
    }

    result = run_legacy_vector_migration_status_pipeline()

    assert result["last_migration_id"] == "mig-1"
    assert result["pending_vectors"] == 1
    assert result["feature_versions"] == {"v1": 1}
    assert result["history"] == [{"migration_id": "mig-1"}]
