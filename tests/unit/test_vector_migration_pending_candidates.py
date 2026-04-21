from __future__ import annotations

from src.api.v1.vector_migration_models import VectorMigrationPendingItem
from src.core.vector_migration_pending_candidates import (
    collect_vector_migration_pending_candidates,
)


async def _collect_memory(**kwargs):  # noqa: ANN003, ANN202
    return await collect_vector_migration_pending_candidates(
        item_cls=VectorMigrationPendingItem,
        qdrant_store=None,
        scan_limit=5000,
        **kwargs,
    )


def test_collect_vector_migration_pending_candidates_memory_exact_counts():
    result = __import__("asyncio").run(
        _collect_memory(
            limit=10,
            target_version="v4",
            from_version_filter=None,
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
    )

    assert result["backend"] == "memory"
    assert result["total_pending"] == 2
    assert result["distribution_complete"] is True
    assert {item.id for item in result["items"]} == {"vec2", "vec3"}


def test_collect_vector_migration_pending_candidates_memory_applies_filter():
    result = __import__("asyncio").run(
        _collect_memory(
            limit=10,
            target_version="v4",
            from_version_filter="v2",
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
    )

    assert result["from_version_filter"] == "v2"
    assert result["total_pending"] == 1
    assert [item.id for item in result["items"]] == ["vec3"]


def test_collect_vector_migration_pending_candidates_qdrant_partial_hides_total():
    class DummyPoint:
        def __init__(self, point_id: str, metadata: dict[str, str]):
            self.id = point_id
            self.metadata = metadata

    class DummyQdrantStore:
        async def count(self):  # noqa: ANN202
            return 4

        async def list_vectors(self, offset=0, limit=50, with_vectors=False):  # noqa: ANN001, ANN202
            items = [
                DummyPoint("vec1", {"feature_version": "v4"}),
                DummyPoint("vec2", {"feature_version": "v3"}),
                DummyPoint("vec3", {"feature_version": "v2"}),
                DummyPoint("vec4", {"feature_version": "v1"}),
            ]
            return items[offset : offset + limit], 4

    result = __import__("asyncio").run(
        collect_vector_migration_pending_candidates(
            limit=10,
            target_version="v4",
            from_version_filter=None,
            qdrant_store=DummyQdrantStore(),
            scan_limit=2,
            item_cls=VectorMigrationPendingItem,
            vector_meta={},
            vector_store={},
        )
    )

    assert result["backend"] == "qdrant"
    assert result["scanned_vectors"] == 2
    assert result["distribution_complete"] is False
    assert result["total_pending"] is None
    assert [item.id for item in result["items"]] == ["vec2"]
