from __future__ import annotations

from typing import Any

from src.core.vector_migration_pending_memory import (
    collect_memory_migration_pending_candidates,
)


class _Item:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_collect_memory_pending_candidates_exact_counts_and_page_limit() -> None:
    result = collect_memory_migration_pending_candidates(
        limit=1,
        target_version="v4",
        normalized_filter=None,
        scan_limit=5000,
        item_cls=_Item,
        vector_meta={
            "vec1": {"feature_version": "v4"},
            "vec2": {"feature_version": "v3"},
            "vec3": {"feature_version": "v2"},
        },
        vector_store={
            "vec1": [1.0] * 24,
            "vec2": [2.0] * 22,
            "vec3": [3.0] * 12,
        },
    )

    assert result["backend"] == "memory"
    assert result["distribution_complete"] is True
    assert result["total_pending"] == 2
    assert result["scanned_vectors"] == 3
    assert result["observed_by_from_version"] == {"v3": 1, "v2": 1}
    assert result["pending_ids"] == ["vec2"]
    assert result["listed_count"] == 1
    assert result["items"][0].id == "vec2"


def test_collect_memory_pending_candidates_filter_and_unknown_version() -> None:
    result = collect_memory_migration_pending_candidates(
        limit=10,
        target_version="v4",
        normalized_filter="unknown",
        scan_limit=5000,
        item_cls=_Item,
        vector_meta={
            "vec1": {"feature_version": "v3"},
            "vec2": {},
            "vec3": {"feature_version": None},
            "vec4": {"feature_version": "v4"},
        },
        vector_store={
            "vec1": [1.0],
            "vec2": [2.0],
            "vec3": [3.0],
            "vec4": [4.0],
        },
    )

    assert result["from_version_filter"] == "unknown"
    assert result["total_pending"] == 2
    assert result["observed_by_from_version"] == {"unknown": 2}
    assert [item.id for item in result["items"]] == ["vec2", "vec3"]


def test_collect_memory_pending_candidates_skips_missing_vectors() -> None:
    result = collect_memory_migration_pending_candidates(
        limit=10,
        target_version="v4",
        normalized_filter=None,
        scan_limit=5000,
        item_cls=_Item,
        vector_meta={
            "vec1": {"feature_version": "v3"},
            "vec2": {"feature_version": "v2"},
        },
        vector_store={"vec1": [1.0]},
    )

    assert result["scanned_vectors"] == 1
    assert result["total_pending"] == 1
    assert result["pending_ids"] == ["vec1"]
