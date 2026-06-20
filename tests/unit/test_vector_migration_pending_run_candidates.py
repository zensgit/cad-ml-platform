from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.vector_migration_pending_run_candidates import (
    collect_pending_run_candidates,
)


@pytest.mark.asyncio
async def test_collect_pending_run_candidates_maps_payload_filters() -> None:
    captured: dict[str, object] = {}

    async def _collect_pending_candidates(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {"pending_ids": ["vec2"]}

    result = await collect_pending_run_candidates(
        payload=SimpleNamespace(limit=7, from_version_filter="v2"),
        target_version="v4",
        collect_pending_candidates_fn=_collect_pending_candidates,
    )

    assert result == {"pending_ids": ["vec2"]}
    assert captured == {
        "limit": 7,
        "target_version": "v4",
        "from_version_filter": "v2",
    }


@pytest.mark.asyncio
async def test_collect_pending_run_candidates_preserves_empty_filter() -> None:
    captured: dict[str, object] = {}

    async def _collect_pending_candidates(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {"pending_ids": []}

    result = await collect_pending_run_candidates(
        payload=SimpleNamespace(limit=3, from_version_filter=None),
        target_version="v4",
        collect_pending_candidates_fn=_collect_pending_candidates,
    )

    assert result == {"pending_ids": []}
    assert captured == {
        "limit": 3,
        "target_version": "v4",
        "from_version_filter": None,
    }
