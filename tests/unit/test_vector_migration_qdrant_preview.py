from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.api.v1 import vectors as vectors_module
from src.core.vector_migration_qdrant_preview import collect_qdrant_preview_samples


class _QdrantStore:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.calls: list[dict[str, object]] = []

    async def count(self) -> int:
        return len(self._rows)

    async def list_vectors(
        self,
        *,
        offset: int,
        limit: int,
        with_vectors: bool,
    ) -> tuple[list[SimpleNamespace], int]:
        self.calls.append(
            {"offset": offset, "limit": limit, "with_vectors": with_vectors}
        )
        selected = self._rows[offset : offset + limit]
        return [
            SimpleNamespace(
                id=row["id"],
                metadata=row.get("metadata"),
                vector=row.get("vector"),
            )
            for row in selected
        ], len(self._rows)


@pytest.mark.asyncio
async def test_collect_qdrant_preview_samples_collects_samples_and_distribution() -> None:
    store = _QdrantStore(
        [
            {"id": 1, "metadata": {"feature_version": "v3"}, "vector": [1.0, 2.0]},
            {"id": "vec2", "metadata": {}, "vector": None},
            {"id": "vec3", "metadata": {"feature_version": "v4"}, "vector": [3.0]},
            {"id": "vec4", "metadata": None, "vector": [4.0]},
        ]
    )

    samples, total_available, by_version = await collect_qdrant_preview_samples(
        store,
        limit=2,
    )

    assert samples == [
        ("1", [1.0, 2.0], {"feature_version": "v3"}),
        ("vec2", [], {}),
    ]
    assert total_available == 4
    assert by_version == {"v3": 1, "v1": 2, "v4": 1}
    assert store.calls == [
        {"offset": 0, "limit": 2, "with_vectors": True},
        {"offset": 2, "limit": 2, "with_vectors": False},
    ]


@pytest.mark.asyncio
async def test_collect_qdrant_preview_samples_clamps_initial_limit() -> None:
    store = _QdrantStore(
        [{"id": "vec1", "metadata": {"feature_version": "v4"}, "vector": [1.0]}]
    )

    samples, total_available, by_version = await collect_qdrant_preview_samples(
        store,
        limit=0,
    )

    assert samples == [("vec1", [1.0], {"feature_version": "v4"})]
    assert total_available == 1
    assert by_version == {"v4": 1}
    assert store.calls == [{"offset": 0, "limit": 1, "with_vectors": True}]


@pytest.mark.asyncio
async def test_collect_qdrant_preview_samples_batches_distribution_scan() -> None:
    rows = [
        {"id": f"vec{i}", "metadata": {"feature_version": "v4"}, "vector": [float(i)]}
        for i in range(205)
    ]
    store = _QdrantStore(rows)

    samples, total_available, by_version = await collect_qdrant_preview_samples(
        store,
        limit=1,
    )

    assert samples == [("vec0", [0.0], {"feature_version": "v4"})]
    assert total_available == 205
    assert by_version == {"v4": 205}
    assert store.calls == [
        {"offset": 0, "limit": 1, "with_vectors": True},
        {"offset": 1, "limit": 200, "with_vectors": False},
        {"offset": 201, "limit": 4, "with_vectors": False},
    ]


def test_vectors_facade_preserves_qdrant_preview_samples_export() -> None:
    assert vectors_module._collect_qdrant_preview_samples is collect_qdrant_preview_samples
