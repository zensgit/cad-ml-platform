from __future__ import annotations

import json
from typing import Any

import pytest

from src.api.v1 import vectors as vectors_module
from src.core.vector_list_redis import list_vectors_redis


class _Item:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Response(dict):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class _Redis:
    def __init__(
        self,
        data: dict[Any, dict[Any, Any]],
        batches: list[tuple[int, list[Any]]],
    ) -> None:
        self._data = data
        self._batches = list(batches)
        self.scan_calls: list[dict[str, Any]] = []
        self.hgetall_keys: list[Any] = []

    async def scan(
        self,
        cursor: int = 0,
        match: str | None = None,
        count: int = 500,
    ) -> tuple[int, list[Any]]:
        self.scan_calls.append({"cursor": cursor, "match": match, "count": count})
        return self._batches.pop(0)

    async def hgetall(self, key: Any) -> dict[Any, Any]:
        self.hgetall_keys.append(key)
        return self._data[key]


def _label_contract(meta: dict[str, Any]) -> dict[str, Any]:
    return dict(meta.get("label_contract", {}))


@pytest.mark.asyncio
async def test_list_vectors_redis_paginates_and_builds_items() -> None:
    redis = _Redis(
        {
            b"vector:vec1": {
                b"v": "1,2",
                b"m": json.dumps(
                    {"material": "steel", "label_contract": {"part_type": "shaft"}}
                ),
            },
            b"vector:vec2": {
                b"v": "3,4,5",
                b"m": json.dumps(
                    {"material": "steel", "label_contract": {"part_type": "plate"}}
                ),
            },
            b"vector:vec3": {
                b"v": "6",
                b"m": json.dumps(
                    {"material": "steel", "label_contract": {"part_type": "bracket"}}
                ),
            },
        },
        [(0, [b"vector:vec1", b"vector:vec2", b"vector:vec3"])],
    )

    response = await list_vectors_redis(
        redis,
        offset=1,
        limit=1,
        scan_limit=0,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter=None,
        decision_source_filter=None,
        is_coarse_label_filter=None,
        item_cls=_Item,
        response_cls=_Response,
        matches_label_filters_fn=lambda **_kwargs: True,
        extract_label_contract_fn=_label_contract,
    )

    assert redis.scan_calls == [{"cursor": 0, "match": "vector:*", "count": 500}]
    assert response["total"] == 3
    assert len(response["vectors"]) == 1
    assert response["vectors"][0].id == "vec2"
    assert response["vectors"][0].dimension == 3
    assert response["vectors"][0].material == "steel"
    assert response["vectors"][0].part_type == "plate"


@pytest.mark.asyncio
async def test_list_vectors_redis_respects_scan_limit_and_filter_args() -> None:
    captured: list[dict[str, Any]] = []

    def _matches(**kwargs: Any) -> bool:
        captured.append(kwargs)
        return kwargs["meta"].get("material") == "steel"

    redis = _Redis(
        {
            "vector:vec1": {
                "v": "1,2",
                "m": json.dumps(
                    {"material": "steel", "label_contract": {"fine_part_type": "shaft"}}
                ),
            },
            "vector:missing-vector": {"m": "not-json"},
            "vector:not-scanned": {
                "v": "9",
                "m": json.dumps(
                    {"material": "steel", "label_contract": {"fine_part_type": "plate"}}
                ),
            },
        },
        [(0, ["vector:vec1", "vector:missing-vector", "vector:not-scanned"])],
    )

    response = await list_vectors_redis(
        redis,
        offset=0,
        limit=10,
        scan_limit=2,
        material_filter="steel",
        complexity_filter="high",
        fine_part_type_filter="shaft",
        coarse_part_type_filter="rotary",
        decision_source_filter="classifier",
        is_coarse_label_filter=False,
        item_cls=_Item,
        response_cls=_Response,
        matches_label_filters_fn=_matches,
        extract_label_contract_fn=_label_contract,
    )

    assert redis.hgetall_keys == ["vector:vec1", "vector:missing-vector"]
    assert response["total"] == 1
    assert response["vectors"][0].id == "vec1"
    assert len(captured) == 1
    assert captured[0]["material_filter"] == "steel"
    assert captured[0]["complexity_filter"] == "high"
    assert captured[0]["fine_part_type_filter"] == "shaft"
    assert captured[0]["coarse_part_type_filter"] == "rotary"
    assert captured[0]["decision_source_filter"] == "classifier"
    assert captured[0]["is_coarse_label_filter"] is False
    assert captured[0]["label_contract"] == {"fine_part_type": "shaft"}


@pytest.mark.asyncio
async def test_list_vectors_redis_handles_malformed_and_non_dict_meta() -> None:
    redis = _Redis(
        {
            "vector:bad-json": {"v": "1", "m": "not-json"},
            "vector:list-json": {"v": "2", "m": "[]"},
        },
        [(0, ["vector:bad-json", "vector:list-json"])],
    )

    response = await list_vectors_redis(
        redis,
        offset=0,
        limit=10,
        scan_limit=0,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter=None,
        decision_source_filter=None,
        is_coarse_label_filter=None,
        item_cls=_Item,
        response_cls=_Response,
        matches_label_filters_fn=lambda **_kwargs: True,
        extract_label_contract_fn=_label_contract,
    )

    assert response["total"] == 2
    assert [item.id for item in response["vectors"]] == ["bad-json", "list-json"]
    assert response["vectors"][0].material is None
    assert response["vectors"][1].part_type is None


@pytest.mark.asyncio
async def test_list_vectors_redis_scans_multiple_batches() -> None:
    redis = _Redis(
        {
            "vector:vec1": {"v": "1", "m": "{}"},
            "vector:vec2": {"v": "2,3", "m": "{}"},
        },
        [(42, ["vector:vec1"]), (0, ["vector:vec2"])],
    )

    response = await list_vectors_redis(
        redis,
        offset=0,
        limit=10,
        scan_limit=0,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter=None,
        decision_source_filter=None,
        is_coarse_label_filter=None,
        item_cls=_Item,
        response_cls=_Response,
        matches_label_filters_fn=lambda **_kwargs: True,
        extract_label_contract_fn=_label_contract,
    )

    assert [call["cursor"] for call in redis.scan_calls] == [0, 42]
    assert response["total"] == 2
    assert [item.id for item in response["vectors"]] == ["vec1", "vec2"]


@pytest.mark.asyncio
async def test_vectors_facade_redis_list_uses_facade_filter_helper(monkeypatch) -> None:
    monkeypatch.setattr(
        vectors_module,
        "_matches_vector_label_filters",
        lambda **kwargs: kwargs["meta"].get("material") == "steel",
    )
    redis = _Redis(
        {
            b"vector:vec1": {b"v": "1,2", b"m": json.dumps({"material": "steel"})},
            b"vector:vec2": {b"v": "3", b"m": json.dumps({"material": "aluminum"})},
        },
        [(0, [b"vector:vec1", b"vector:vec2"])],
    )

    response = await vectors_module._list_vectors_redis(
        redis,
        offset=0,
        limit=10,
        scan_limit=0,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter=None,
        decision_source_filter=None,
        is_coarse_label_filter=None,
    )

    assert response.total == 1
    assert response.vectors[0].id == "vec1"


@pytest.mark.asyncio
async def test_vectors_facade_redis_list_preserves_extractor_patch(monkeypatch) -> None:
    from src.core import similarity as similarity_module

    monkeypatch.setattr(
        similarity_module,
        "extract_vector_label_contract",
        lambda _meta: {"part_type": "patched"},
    )
    redis = _Redis(
        {"vector:vec1": {"v": "1", "m": "{}"}},
        [(0, ["vector:vec1"])],
    )

    response = await vectors_module._list_vectors_redis(
        redis,
        offset=0,
        limit=10,
        scan_limit=0,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter=None,
        decision_source_filter=None,
        is_coarse_label_filter=None,
    )

    assert response.total == 1
    assert response.vectors[0].part_type == "patched"


@pytest.mark.asyncio
async def test_vectors_facade_redis_list_preserves_json_loads_patch(monkeypatch) -> None:
    monkeypatch.setattr(
        vectors_module.json,
        "loads",
        lambda _raw: {"material": "patched-steel"},
    )
    redis = _Redis(
        {"vector:vec1": {"v": "1", "m": "opaque"}},
        [(0, ["vector:vec1"])],
    )

    response = await vectors_module._list_vectors_redis(
        redis,
        offset=0,
        limit=10,
        scan_limit=0,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter=None,
        decision_source_filter=None,
        is_coarse_label_filter=None,
    )

    assert response.total == 1
    assert response.vectors[0].material == "patched-steel"
