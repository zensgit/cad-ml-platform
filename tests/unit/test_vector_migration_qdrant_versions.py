from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.v1 import vectors as vectors_module
from src.core.vector_migration_qdrant_versions import collect_qdrant_feature_versions


class _QdrantStore:
    def __init__(self, metadata_rows: list[dict[str, str] | None]) -> None:
        self._metadata_rows = metadata_rows
        self.calls: list[dict[str, object]] = []

    async def count(self) -> int:
        return len(self._metadata_rows)

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
        rows = self._metadata_rows[offset : offset + limit]
        return [SimpleNamespace(metadata=row) for row in rows], len(self._metadata_rows)


@pytest.mark.asyncio
async def test_collect_qdrant_feature_versions_counts_scanned_versions() -> None:
    store = _QdrantStore(
        [
            {"feature_version": "v4"},
            {"feature_version": "v3"},
            {"feature_version": "v4"},
            {},
        ]
    )

    versions, total_available, scanned = await collect_qdrant_feature_versions(
        store,
        scan_limit=4,
    )

    assert versions == {"v4": 2, "v3": 1, "unknown": 1}
    assert total_available == 4
    assert scanned == 4
    assert store.calls == [{"offset": 0, "limit": 4, "with_vectors": False}]


@pytest.mark.asyncio
async def test_collect_qdrant_feature_versions_respects_scan_limit_resolver() -> None:
    store = _QdrantStore(
        [
            {"feature_version": "v4"},
            {"feature_version": "v3"},
            {"feature_version": "v2"},
        ]
    )

    versions, total_available, scanned = await collect_qdrant_feature_versions(
        store,
        resolve_scan_limit_fn=lambda: 2,
    )

    assert versions == {"v4": 1, "v3": 1}
    assert total_available == 3
    assert scanned == 2
    assert store.calls == [{"offset": 0, "limit": 2, "with_vectors": False}]


@pytest.mark.asyncio
async def test_collect_qdrant_feature_versions_clamps_scan_limit() -> None:
    store = _QdrantStore([{"feature_version": "v4"}])

    versions, total_available, scanned = await collect_qdrant_feature_versions(
        store,
        scan_limit=0,
    )

    assert versions == {"v4": 1}
    assert total_available == 1
    assert scanned == 1
    assert store.calls == [{"offset": 0, "limit": 1, "with_vectors": False}]


@pytest.mark.asyncio
async def test_vectors_facade_collector_uses_facade_scan_limit_resolver(
    monkeypatch,
) -> None:
    store = _QdrantStore(
        [
            {"feature_version": "v4"},
            {"feature_version": "v3"},
            {"feature_version": "v2"},
        ]
    )
    monkeypatch.setattr(
        vectors_module,
        "_resolve_vector_migration_scan_limit",
        lambda: 2,
    )

    versions, total_available, scanned = await vectors_module._collect_qdrant_feature_versions(
        store,
    )

    assert versions == {"v4": 1, "v3": 1}
    assert total_available == 3
    assert scanned == 2
    assert store.calls == [{"offset": 0, "limit": 2, "with_vectors": False}]


@pytest.mark.asyncio
async def test_vectors_facade_collector_preserves_explicit_scan_limit(
    monkeypatch,
) -> None:
    store = _QdrantStore(
        [
            {"feature_version": "v4"},
            {"feature_version": "v3"},
        ]
    )

    def _fail_resolver() -> int:
        raise AssertionError("explicit scan_limit should not call resolver")

    monkeypatch.setattr(
        vectors_module,
        "_resolve_vector_migration_scan_limit",
        _fail_resolver,
    )

    versions, total_available, scanned = await vectors_module._collect_qdrant_feature_versions(
        store,
        scan_limit=1,
    )

    assert versions == {"v4": 1}
    assert total_available == 2
    assert scanned == 1
    assert store.calls == [{"offset": 0, "limit": 1, "with_vectors": False}]
