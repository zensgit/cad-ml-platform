from src.core.classification.finalization import finalize_classification_payload


def test_finalize_classification_payload_builds_branch_conflicts_and_review(monkeypatch):
    def _stub_knowledge_summary(**kwargs):  # noqa: ANN003
        assert kwargs["fine_part_type"] == "人孔"
        assert kwargs["coarse_part_type"] == "开孔件"
        return {
            "knowledge_checks": [{"category": "material"}],
            "violations": [{"rule": "material_conflict"}],
            "standards_candidates": [{"designation": "GB/T 1804-M"}],
            "knowledge_hints": [{"label": "304"}],
        }

    monkeypatch.setattr(
        "src.core.classification.finalization.build_knowledge_summary",
        _stub_knowledge_summary,
    )

    payload = {
        "part_type": "人孔",
        "confidence": 0.91,
        "confidence_source": "hybrid",
        "rule_version": "HybridClassifier-v1",
        "fine_part_type": "人孔",
        "hybrid_decision": {"label": "人孔"},
        "graph2d_prediction": {"label": "传动件"},
        "filename_prediction": {"label": "人孔"},
        "titleblock_prediction": {"label": "人孔"},
        "history_prediction": {"label": "法兰"},
    }

    result = finalize_classification_payload(
        payload,
        text_signals=[{"text": "S30408"}],
        text_items=["名称:人孔", "材质:S30408"],
        geometric_features={"lines": 3},
        entity_counts={"LINE": 3},
        low_confidence_threshold=0.6,
        high_confidence_threshold=0.85,
    )

    assert result["decision_source"] == "hybrid"
    assert result["final_decision_source"] == "hybrid"
    assert result["coarse_part_type"] == "开孔件"
    assert result["coarse_fine_part_type"] == "开孔件"
    assert result["coarse_hybrid_label"] == "开孔件"
    assert result["coarse_graph2d_label"] == "传动件"
    assert result["coarse_history_label"] == "法兰"
    assert result["is_coarse_label"] is False
    assert result["has_branch_conflict"] is True
    assert result["branch_conflicts"] == {
        "hybrid_vs_graph2d": True,
        "filename_vs_graph2d": True,
        "titleblock_vs_graph2d": True,
        "history_vs_final": True,
    }
    assert result["knowledge_checks"] == [{"category": "material"}]
    assert result["violations"] == [{"rule": "material_conflict"}]
    assert result["standards_candidates"] == [{"designation": "GB/T 1804-M"}]
    assert result["knowledge_hints"] == [{"label": "304"}]
    assert result["needs_review"] is True
    assert result["review_priority"] == "critical"
    assert "knowledge_conflict" in result["review_reasons"]
    assert "branch_conflict" in result["review_reasons"]


def test_finalize_classification_payload_backfills_contract_without_text_items(monkeypatch):
    monkeypatch.setattr(
        "src.core.classification.finalization.build_knowledge_summary",
        lambda **_: {},
    )

    result = finalize_classification_payload(
        {
            "part_type": "bolt",
            "confidence": 0.72,
            "confidence_source": "fusion",
            "rule_version": "L2-Fusion-v1",
        },
        text_signals=None,
        text_items="ignored-non-list",
        geometric_features=None,
        entity_counts=None,
        low_confidence_threshold=0.6,
        high_confidence_threshold=0.85,
    )

    assert result["part_type"] == "bolt"
    assert result["fine_part_type"] == "bolt"
    assert result["coarse_part_type"] == "bolt"
    assert result["decision_source"] == "fusion"
    assert result["is_coarse_label"] is True
    assert result["knowledge_checks"] == []
    assert result["violations"] == []
    assert result["needs_review"] is False
    assert result["review_priority"] == "none"
