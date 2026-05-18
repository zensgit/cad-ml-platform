from src.core.classification.decision_service import (
    DECISION_CONTRACT_VERSION,
    DecisionService,
)


def test_decision_service_builds_stable_contract_and_evidence():
    captured = {}

    def _finalize(payload, **kwargs):  # noqa: ANN001, ANN003, ANN201
        captured["kwargs"] = kwargs
        result = dict(payload or {})
        result.update(
            {
                "part_type": "人孔",
                "fine_part_type": "人孔",
                "coarse_part_type": "开孔件",
                "confidence": 0.93,
                "decision_source": "hybrid",
                "branch_conflicts": {"hybrid_vs_graph2d": True},
                "review_reasons": ["branch_conflict"],
                "knowledge_checks": [
                    {
                        "category": "material",
                        "rule_source": "materials_catalog",
                        "rule_version": "knowledge_grounding.v1",
                    }
                ],
                "violations": [],
                "standards_candidates": [
                    {
                        "type": "material",
                        "rule_source": "materials_catalog",
                        "rule_version": "knowledge_grounding.v1",
                    }
                ],
                "knowledge_hints": [
                    {
                        "label": "人孔",
                        "rule_source": "knowledge_manager",
                        "rule_version": "knowledge_grounding.v1",
                    }
                ],
                "needs_review": True,
            }
        )
        return result

    service = DecisionService(finalize_fn=_finalize)
    result = service.decide(
        {
            "part_type": "simple_plate",
            "confidence": 0.21,
            "confidence_source": "rules",
            "rule_version": "v1",
            "filename_prediction": {"label": "人孔", "confidence": 0.8},
            "titleblock_prediction": {"label": "人孔", "confidence": 0.7},
            "graph2d_prediction": {
                "label": "传动件",
                "confidence": 0.86,
                "status": "ok",
            },
            "part_classifier_prediction": {
                "label": "candidate_zero",
                "confidence": 0.0,
                "status": "ok",
            },
            "hybrid_decision": {
                "label": "人孔",
                "confidence": 0.93,
                "source": "hybrid",
            },
            "source_contributions": {"filename": 0.31, "graph2d": 0.22},
        },
        text_signals=[{"text": "人孔"}],
        text_items=["名称: 人孔"],
        geometric_features={"lines": 4},
        entity_counts={"LINE": 4},
        features_3d={
            "valid_3d": True,
            "faces": 8,
            "surface_types": {"plane": 6, "cylinder": 2},
            "feature_hints": {"opening": 0.72, "shaft": 0.2},
            "embedding_dim": 128,
        },
        vector_neighbors=[{"id": "neighbor-1", "score": 0.91}],
        active_learning_history={"prior_reviews": 2},
        low_confidence_threshold=0.55,
        high_confidence_threshold=0.9,
    )

    assert captured["kwargs"]["low_confidence_threshold"] == 0.55
    assert result["contract_version"] == DECISION_CONTRACT_VERSION
    assert result["decision_contract"]["contract_version"] == DECISION_CONTRACT_VERSION
    assert result["decision_contract"]["fine_part_type"] == "人孔"
    assert result["decision_contract"]["coarse_part_type"] == "开孔件"
    assert result["decision_contract"]["review_reasons"] == ["branch_conflict"]

    by_source = {row["source"]: row for row in result["evidence"]}
    assert by_source["baseline"]["kind"] == "decision"
    assert by_source["filename"]["contribution"] == 0.31
    assert by_source["graph2d"]["label"] == "传动件"
    assert by_source["part_classifier"]["confidence"] == 0.0
    assert by_source["hybrid"]["label"] == "人孔"
    assert by_source["brep"]["label"] == "opening"
    assert by_source["brep"]["confidence"] == 0.72
    assert by_source["knowledge"]["details"]["checks_count"] == 1
    assert by_source["knowledge"]["details"]["knowledge_hints_count"] == 1
    assert by_source["knowledge"]["details"]["rule_sources"] == [
        "materials_catalog",
        "knowledge_manager",
    ]
    assert by_source["knowledge"]["details"]["rule_versions"] == [
        "knowledge_grounding.v1"
    ]
    assert by_source["vector_neighbors"]["details"]["neighbor_count"] == 1
    assert by_source["active_learning_history"]["details"]["prior_reviews"] == 2


def test_decision_service_collects_fallback_flags():
    def _finalize(payload, **kwargs):  # noqa: ANN001, ANN003, ANN201
        result = dict(payload or {})
        result.update(
            {
                "fine_part_type": "unknown",
                "coarse_part_type": "unknown",
                "decision_source": "rules",
                "review_reasons": ["low_confidence"],
                "branch_conflicts": {},
            }
        )
        return result

    result = DecisionService(finalize_fn=_finalize).decide(
        {
            "part_type": "unknown",
            "confidence": 0.2,
            "confidence_source": "rules",
            "model_version": "ml_error",
            "hybrid_error": "provider unavailable",
            "graph2d_prediction": {"status": "model_unavailable"},
            "part_classifier_prediction": {"status": "timeout"},
        },
        features_3d={"valid_3d": False, "faces": 0},
    )

    assert result["fallback_flags"] == [
        "rules_baseline",
        "ml_unavailable",
        "hybrid_error",
        "graph2d_model_unavailable",
        "part_classifier_timeout",
        "brep_invalid",
    ]
    assert result["decision_contract"]["fallback_flags"] == result["fallback_flags"]
