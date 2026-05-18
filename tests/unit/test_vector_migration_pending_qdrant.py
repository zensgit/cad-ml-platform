from __future__ import annotations

from typing import Any

import pytest

from src.core.vector_migration_pending_qdrant import (
    collect_qdrant_migration_pending_candidates,
)


class _Item:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Point:
    def __init__(self, point_id: str, metadata: dict[str, Any] | None) -> None:
        self.id = point_id
        self.metadata = metadata


class _Store:
    def __init__(self, points: list[_Point], total: int | None = None) -> None:
        self.points = points
        self.total = len(points) if total is None else total
        self.calls: list[dict[str, Any]] = []

    async def count(self) -> int:
        return self.total

    async def list_vectors(
        self,
        offset: int,
        limit: int,
        with_vectors: bool,
    ) -> tuple[list[_Point], int]:
        self.calls.append(
            {"offset": offset, "limit": limit, "with_vectors": with_vectors}
        )
        return self.points[offset : offset + limit], self.total


@pytest.mark.asyncio
async def test_collect_qdrant_pending_candidates_exact_counts_and_page_limit() -> None:
    store = _Store(
        [
            _Point("vec1", {"feature_version": "v4"}),
            _Point("vec2", {"feature_version": "v3"}),
            _Point("vec3", {"feature_version": "v2"}),
        ]
    )

    result = await collect_qdrant_migration_pending_candidates(
        qdrant_store=store,
        limit=1,
        target_version="v4",
        normalized_filter=None,
        scan_limit=10,
        item_cls=_Item,
    )

    assert store.calls == [{"offset": 0, "limit": 3, "with_vectors": False}]
    assert result["backend"] == "qdrant"
    assert result["distribution_complete"] is True
    assert result["total_pending"] == 2
    assert result["observed_by_from_version"] == {"v3": 1, "v2": 1}
    assert result["pending_ids"] == ["vec2"]
    assert result["listed_count"] == 1
    assert result["items"][0].id == "vec2"


@pytest.mark.asyncio
async def test_collect_qdrant_pending_candidates_filter_and_unknown_version() -> None:
    store = _Store(
        [
            _Point("vec1", {"feature_version": "v3"}),
            _Point("vec2", {}),
            _Point("vec3", None),
        ]
    )

    result = await collect_qdrant_migration_pending_candidates(
        qdrant_store=store,
        limit=10,
        target_version="v4",
        normalized_filter="unknown",
        scan_limit=10,
        item_cls=_Item,
    )

    assert result["from_version_filter"] == "unknown"
    assert result["total_pending"] == 2
    assert result["observed_by_from_version"] == {"unknown": 2}
    assert [item.id for item in result["items"]] == ["vec2", "vec3"]


@pytest.mark.asyncio
async def test_collect_qdrant_pending_candidates_partial_scan_hides_total() -> None:
    store = _Store(
        [
            _Point("vec1", {"feature_version": "v4"}),
            _Point("vec2", {"feature_version": "v3"}),
            _Point("vec3", {"feature_version": "v2"}),
            _Point("vec4", {"feature_version": "v1"}),
        ],
        total=4,
    )

    result = await collect_qdrant_migration_pending_candidates(
        qdrant_store=store,
        limit=10,
        target_version="v4",
        normalized_filter=None,
        scan_limit=2,
        item_cls=_Item,
    )

    assert store.calls == [{"offset": 0, "limit": 2, "with_vectors": False}]
    assert result["scanned_vectors"] == 2
    assert result["distribution_complete"] is False
    assert result["total_pending"] is None
    assert [item.id for item in result["items"]] == ["vec2"]
