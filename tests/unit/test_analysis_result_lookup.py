from __future__ import annotations

import pytest

from src.core.analysis_result_lookup import load_analysis_result_with_cache


@pytest.mark.asyncio
async def test_load_analysis_result_with_cache_returns_cached_result() -> None:
    cached_calls: list[str] = []
    load_calls: list[str] = []
    set_calls: list[tuple[str, dict[str, object], int]] = []

    async def _get_cached_result(key: str):  # noqa: ANN201
        cached_calls.append(key)
        return {"id": "cached"}

    async def _load_analysis_result(analysis_id: str):  # noqa: ANN201
        load_calls.append(analysis_id)
        return {"id": "loaded"}

    async def _set_cache(key: str, payload: dict[str, object], ttl_seconds: int):  # noqa: ANN201
        set_calls.append((key, payload, ttl_seconds))

    result = await load_analysis_result_with_cache(
        analysis_id="abc123",
        get_cached_result_fn=_get_cached_result,
        load_analysis_result_fn=_load_analysis_result,
        set_cache_fn=_set_cache,
    )

    assert result == {"id": "cached"}
    assert cached_calls == ["analysis_result:abc123"]
    assert load_calls == []
    assert set_calls == []


@pytest.mark.asyncio
async def test_load_analysis_result_with_cache_loads_and_sets_cache() -> None:
    set_calls: list[tuple[str, dict[str, object], int]] = []

    async def _get_cached_result(_key: str):  # noqa: ANN201
        return None

    async def _load_analysis_result(analysis_id: str):  # noqa: ANN201
        return {"id": analysis_id, "status": "ok"}

    async def _set_cache(key: str, payload: dict[str, object], ttl_seconds: int):  # noqa: ANN201
        set_calls.append((key, payload, ttl_seconds))

    result = await load_analysis_result_with_cache(
        analysis_id="xyz789",
        get_cached_result_fn=_get_cached_result,
        load_analysis_result_fn=_load_analysis_result,
        set_cache_fn=_set_cache,
        ttl_seconds=1800,
    )

    assert result == {"id": "xyz789", "status": "ok"}
    assert set_calls == [("analysis_result:xyz789", {"id": "xyz789", "status": "ok"}, 1800)]


@pytest.mark.asyncio
async def test_load_analysis_result_with_cache_returns_none_without_setting_cache() -> None:
    set_calls: list[tuple[str, dict[str, object], int]] = []

    async def _get_cached_result(_key: str):  # noqa: ANN201
        return None

    async def _load_analysis_result(_analysis_id: str):  # noqa: ANN201
        return None

    async def _set_cache(key: str, payload: dict[str, object], ttl_seconds: int):  # noqa: ANN201
        set_calls.append((key, payload, ttl_seconds))

    result = await load_analysis_result_with_cache(
        analysis_id="missing",
        get_cached_result_fn=_get_cached_result,
        load_analysis_result_fn=_load_analysis_result,
        set_cache_fn=_set_cache,
    )

    assert result is None
    assert set_calls == []
