from __future__ import annotations

from types import SimpleNamespace

from src.core.classification.vector_metadata import (
    build_vector_registration_metadata,
    extract_vector_label_contract,
)
from src.core.similarity import extract_vector_label_contract as extract_similarity_contract
from src.core.vector_layouts import VECTOR_LAYOUT_BASE, VECTOR_LAYOUT_L3


def test_build_vector_registration_metadata_preserves_classification_contract_fields():
    doc = SimpleNamespace(format="dxf", complexity_bucket=lambda: "medium")

    meta = build_vector_registration_metadata(
        material="steel",
        doc=doc,
        features={"geometric": [1.0, 2.0], "semantic": [3.0]},
        feature_vector=[1.0, 2.0, 3.0],
        feature_version="v7",
        vector_layout=VECTOR_LAYOUT_BASE,
        classification_meta={
            "part_type": "人孔",
            "fine_part_type": "人孔",
            "coarse_part_type": "开孔件",
            "final_decision_source": "hybrid",
            "is_coarse_label": False,
        },
    )

    assert meta == {
        "material": "steel",
        "complexity": "medium",
        "format": "dxf",
        "feature_version": "v7",
        "vector_layout": VECTOR_LAYOUT_BASE,
        "geometric_dim": "2",
        "semantic_dim": "1",
        "total_dim": "3",
        "part_type": "人孔",
        "fine_part_type": "人孔",
        "coarse_part_type": "开孔件",
        "is_coarse_label": "false",
        "final_decision_source": "hybrid",
    }


def test_build_vector_registration_metadata_supports_l3_without_classification_payload():
    doc = SimpleNamespace(format="step", complexity_bucket=lambda: "high")

    meta = build_vector_registration_metadata(
        material=None,
        doc=doc,
        features={"geometric": [1.0], "semantic": [2.0, 3.0]},
        feature_vector=[1.0, 2.0, 3.0, 4.0, 5.0],
        feature_version="v8",
        vector_layout=VECTOR_LAYOUT_L3,
        l3_dim=2,
    )

    assert meta == {
        "material": "unknown",
        "complexity": "high",
        "format": "step",
        "feature_version": "v8",
        "vector_layout": VECTOR_LAYOUT_L3,
        "geometric_dim": "1",
        "semantic_dim": "2",
        "total_dim": "5",
        "l3_3d_dim": "2",
    }


def test_similarity_contract_wrapper_matches_shared_vector_contract():
    payload = {
        "part_type": "法兰",
        "coarse_part_type": "法兰",
        "decision_source": "filename",
        "is_coarse_label": "true",
    }

    assert extract_vector_label_contract(payload) == extract_similarity_contract(payload)


def test_build_vector_registration_metadata_maps_decision_source_and_false_string():
    doc = SimpleNamespace(format="dxf", complexity_bucket=lambda: "low")

    meta = build_vector_registration_metadata(
        material="aluminum",
        doc=doc,
        features={"geometric": [1.0], "semantic": [2.0]},
        feature_vector=[1.0, 2.0],
        feature_version="v9",
        vector_layout=VECTOR_LAYOUT_BASE,
        classification_meta={
            "part_type": "法兰",
            "fine_part_type": "法兰",
            "coarse_part_type": "法兰",
            "decision_source": "filename",
            "is_coarse_label": "false",
        },
    )

    assert meta["final_decision_source"] == "filename"
    assert meta["is_coarse_label"] == "false"


def test_build_vector_registration_metadata_drops_blank_contract_fields():
    doc = SimpleNamespace(format="step", complexity_bucket=lambda: "simple")

    meta = build_vector_registration_metadata(
        material="steel",
        doc=doc,
        features={"geometric": [], "semantic": []},
        feature_vector=[],
        feature_version="v10",
        vector_layout=VECTOR_LAYOUT_BASE,
        classification_meta={
            "part_type": " ",
            "fine_part_type": "",
            "coarse_part_type": "  ",
            "final_decision_source": " ",
            "is_coarse_label": None,
        },
    )

    assert "part_type" not in meta
    assert "fine_part_type" not in meta
    assert "coarse_part_type" not in meta
    assert "final_decision_source" not in meta
    assert "is_coarse_label" not in meta
