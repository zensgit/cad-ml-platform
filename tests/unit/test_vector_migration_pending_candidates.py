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


def test_collect_vector_migration_pending_candidates_delegates_qdrant_branch(monkeypatch):
    captured: dict[str, object] = {}

    async def _collect_qdrant(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {"backend": "qdrant"}

    monkeypatch.setattr(
        "src.core.vector_migration_pending_candidates."
        "collect_qdrant_migration_pending_candidates",
        _collect_qdrant,
    )

    result = __import__("asyncio").run(
        collect_vector_migration_pending_candidates(
            limit=5,
            target_version="v4",
            from_version_filter=" v3 ",
            qdrant_store="store",
            scan_limit=123,
            item_cls=VectorMigrationPendingItem,
            vector_meta={},
            vector_store={},
        )
    )

    assert result == {"backend": "qdrant"}
    assert captured["qdrant_store"] == "store"
    assert captured["limit"] == 5
    assert captured["target_version"] == "v4"
    assert captured["normalized_filter"] == "v3"
    assert captured["scan_limit"] == 123
    assert captured["item_cls"] is VectorMigrationPendingItem


def test_collect_vector_migration_pending_candidates_delegates_memory_branch(monkeypatch):
    captured: dict[str, object] = {}

    def _collect_memory(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {"backend": "memory"}

    monkeypatch.setattr(
        "src.core.vector_migration_pending_candidates."
        "collect_memory_migration_pending_candidates",
        _collect_memory,
    )

    result = __import__("asyncio").run(
        collect_vector_migration_pending_candidates(
            limit=7,
            target_version="v4",
            from_version_filter=" v2 ",
            qdrant_store=None,
            scan_limit=456,
            item_cls=VectorMigrationPendingItem,
            vector_meta={"vec2": {"feature_version": "v2"}},
            vector_store={"vec2": [2.0]},
        )
    )

    assert result == {"backend": "memory"}
    assert captured["limit"] == 7
    assert captured["target_version"] == "v4"
    assert captured["normalized_filter"] == "v2"
    assert captured["scan_limit"] == 456
    assert captured["item_cls"] is VectorMigrationPendingItem
    assert captured["vector_meta"] == {"vec2": {"feature_version": "v2"}}
    assert captured["vector_store"] == {"vec2": [2.0]}
