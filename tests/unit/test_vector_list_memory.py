from __future__ import annotations

from typing import Any

from src.api.v1 import vectors as vectors_module
from src.core.vector_list_memory import list_vectors_memory


class _Item:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Response(dict):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


def _label_contract(meta: dict[str, Any]) -> dict[str, Any]:
    return dict(meta.get("label_contract", {}))


def test_list_vectors_memory_paginates_and_builds_items() -> None:
    response = list_vectors_memory(
        {
            "vec1": [1.0, 2.0],
            "vec2": [3.0, 4.0, 5.0],
            "vec3": [6.0],
        },
        {
            "vec1": {"material": "steel", "label_contract": {"part_type": "shaft"}},
            "vec2": {"material": "steel", "label_contract": {"part_type": "plate"}},
            "vec3": {"material": "steel", "label_contract": {"part_type": "bracket"}},
        },
        offset=1,
        limit=1,
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

    assert response["total"] == 3
    assert len(response["vectors"]) == 1
    assert response["vectors"][0].id == "vec2"
    assert response["vectors"][0].dimension == 3
    assert response["vectors"][0].material == "steel"
    assert response["vectors"][0].part_type == "plate"


def test_list_vectors_memory_passes_filters_to_matcher() -> None:
    captured: list[dict[str, Any]] = []

    def _matches(**kwargs: Any) -> bool:
        captured.append(kwargs)
        return kwargs["meta"].get("material") == "steel"

    response = list_vectors_memory(
        {
            "vec1": [1.0],
            "vec2": [2.0],
        },
        {
            "vec1": {"material": "steel", "label_contract": {"fine_part_type": "shaft"}},
            "vec2": {"material": "aluminum", "label_contract": {"fine_part_type": "plate"}},
        },
        offset=0,
        limit=10,
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

    assert response["total"] == 1
    assert response["vectors"][0].id == "vec1"
    assert len(captured) == 2
    assert captured[0]["material_filter"] == "steel"
    assert captured[0]["complexity_filter"] == "high"
    assert captured[0]["fine_part_type_filter"] == "shaft"
    assert captured[0]["coarse_part_type_filter"] == "rotary"
    assert captured[0]["decision_source_filter"] == "classifier"
    assert captured[0]["is_coarse_label_filter"] is False
    assert captured[0]["label_contract"] == {"fine_part_type": "shaft"}


def test_vectors_facade_memory_list_uses_facade_filter_helper(monkeypatch) -> None:
    monkeypatch.setattr(
        vectors_module,
        "_matches_vector_label_filters",
        lambda **kwargs: kwargs["meta"].get("material") == "steel",
    )

    response = vectors_module._list_vectors_memory(
        {
            "vec1": [1.0],
            "vec2": [2.0, 3.0],
        },
        {
            "vec1": {"material": "steel", "part_type": "shaft"},
            "vec2": {"material": "aluminum", "part_type": "plate"},
        },
        offset=0,
        limit=10,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter=None,
        decision_source_filter=None,
        is_coarse_label_filter=None,
    )

    assert response.total == 1
    assert response.vectors[0].id == "vec1"
