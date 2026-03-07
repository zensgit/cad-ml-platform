from __future__ import annotations

from src.ml.vision_3d import prepare_brep_features_for_report


def test_prepare_brep_features_for_report_derives_surface_summary_and_hints() -> None:
    summary = prepare_brep_features_for_report(
        {
            "valid_3d": True,
            "faces": 20,
            "surface_types": {"cylinder": 15, "plane": 5},
            "embedding_vector": [0.1, 0.2, 0.3],
        }
    )

    assert summary["valid_3d"] is True
    assert summary["faces"] == 20
    assert summary["primary_surface_type"] == "cylinder"
    assert summary["primary_surface_ratio"] == 0.75
    assert summary["embedding_dim"] == 3
    assert summary["feature_hints"] == {"shaft": 0.6, "bolt": 0.4}
    assert summary["top_hint_label"] == "shaft"
    assert summary["top_hint_score"] == 0.6


def test_prepare_brep_features_for_report_prefers_existing_hint_payload() -> None:
    summary = prepare_brep_features_for_report(
        {
            "valid_3d": True,
            "faces": 6,
            "surface_types": {"plane": 6},
        },
        brep_feature_hints={"fixture": 0.91, "plate": 0.3},
    )

    assert summary["feature_hints"] == {"fixture": 0.91, "plate": 0.3}
    assert summary["top_hint_label"] == "fixture"
    assert summary["top_hint_score"] == 0.91
