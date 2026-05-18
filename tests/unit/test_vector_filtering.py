from __future__ import annotations

from types import SimpleNamespace

from src.api.v1 import vectors as vectors_module
from src.core.vector_filtering import (
    build_vector_filter_conditions,
    build_vector_search_filter_conditions,
    matches_vector_label_filters,
    matches_vector_search_filters,
    vector_item_payload,
)


def test_build_vector_filter_conditions_keeps_false_boolean_filter() -> None:
    assert build_vector_filter_conditions(
        material_filter="steel",
        complexity_filter="high",
        fine_part_type_filter="shaft",
        coarse_part_type_filter="rotary",
        decision_source_filter="classifier",
        is_coarse_label_filter=False,
    ) == {
        "material": "steel",
        "complexity": "high",
        "fine_part_type": "shaft",
        "coarse_part_type": "rotary",
        "decision_source": "classifier",
        "is_coarse_label": False,
    }


def test_build_vector_search_filter_conditions_reads_payload_fields() -> None:
    payload = SimpleNamespace(
        material_filter=None,
        complexity_filter="medium",
        fine_part_type_filter=None,
        coarse_part_type_filter="sheet",
        decision_source_filter=None,
        is_coarse_label_filter=True,
    )

    assert build_vector_search_filter_conditions(payload) == {
        "complexity": "medium",
        "coarse_part_type": "sheet",
        "is_coarse_label": True,
    }


def test_vector_item_payload_uses_metadata_and_label_contract() -> None:
    assert vector_item_payload(
        "vec-1",
        128,
        {"material": "steel", "complexity": "high", "format": "dxf"},
        {
            "part_type": "shaft",
            "fine_part_type": "stepped_shaft",
            "coarse_part_type": "rotary",
            "decision_source": "classifier",
            "is_coarse_label": False,
        },
    ) == {
        "id": "vec-1",
        "dimension": 128,
        "material": "steel",
        "complexity": "high",
        "format": "dxf",
        "part_type": "shaft",
        "fine_part_type": "stepped_shaft",
        "coarse_part_type": "rotary",
        "decision_source": "classifier",
        "is_coarse_label": False,
    }


def test_matches_vector_label_filters_uses_metadata_and_label_contract() -> None:
    meta = {"material": "steel", "complexity": "high"}
    label_contract = {
        "fine_part_type": "shaft",
        "coarse_part_type": "rotary",
        "decision_source": "classifier",
        "is_coarse_label": False,
    }

    assert matches_vector_label_filters(
        material_filter="steel",
        complexity_filter="high",
        fine_part_type_filter="shaft",
        coarse_part_type_filter="rotary",
        decision_source_filter="classifier",
        is_coarse_label_filter=False,
        meta=meta,
        label_contract=label_contract,
    )
    assert not matches_vector_label_filters(
        material_filter="aluminum",
        complexity_filter="high",
        fine_part_type_filter="shaft",
        coarse_part_type_filter="rotary",
        decision_source_filter="classifier",
        is_coarse_label_filter=False,
        meta=meta,
        label_contract=label_contract,
    )


def test_matches_vector_search_filters_reads_payload_fields() -> None:
    payload = SimpleNamespace(
        material_filter="steel",
        complexity_filter="high",
        fine_part_type_filter="shaft",
        coarse_part_type_filter="rotary",
        decision_source_filter="classifier",
        is_coarse_label_filter=False,
    )

    assert matches_vector_search_filters(
        payload,
        {"material": "steel", "complexity": "high"},
        {
            "fine_part_type": "shaft",
            "coarse_part_type": "rotary",
            "decision_source": "classifier",
            "is_coarse_label": False,
        },
    )


def test_vectors_facade_preserves_filtering_helper_exports() -> None:
    assert vectors_module._build_vector_filter_conditions is build_vector_filter_conditions
    assert (
        vectors_module._build_vector_search_filter_conditions
        is build_vector_search_filter_conditions
    )
    assert vectors_module._vector_item_payload is vector_item_payload
    assert vectors_module._matches_vector_label_filters is matches_vector_label_filters
    assert vectors_module._matches_vector_search_filters is matches_vector_search_filters
