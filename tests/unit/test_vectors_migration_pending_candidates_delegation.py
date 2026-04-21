from __future__ import annotations

from unittest.mock import patch

from src.api.v1.vector_migration_models import VectorMigrationPendingItem
from src.api.v1.vectors import _collect_vector_migration_pending_candidates


def test_vectors_pending_candidates_wrapper_delegates_shared_helper():
    captured: dict[str, object] = {}

    async def _shared(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {"ok": True}

    with patch(
        "src.api.v1.vectors.collect_vector_migration_pending_candidates",
        _shared,
    ), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value="qdrant-store",
    ), patch(
        "src.api.v1.vectors._resolve_vector_migration_scan_limit",
        return_value=321,
    ), patch(
        "src.core.similarity._VECTOR_META",
        {"vec2": {"feature_version": "v3"}},
    ), patch(
        "src.core.similarity._VECTOR_STORE",
        {"vec2": [2.0] * 22},
    ):
        result = __import__("asyncio").run(
            _collect_vector_migration_pending_candidates(
                limit=5,
                target_version="v4",
                from_version_filter="v3",
            )
        )

    assert result == {"ok": True}
    assert captured["limit"] == 5
    assert captured["target_version"] == "v4"
    assert captured["from_version_filter"] == "v3"
    assert captured["qdrant_store"] == "qdrant-store"
    assert captured["scan_limit"] == 321
    assert captured["item_cls"] is VectorMigrationPendingItem
    assert captured["vector_meta"] == {"vec2": {"feature_version": "v3"}}
    assert captured["vector_store"] == {"vec2": [2.0] * 22}
