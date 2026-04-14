from src.core.classification import (
    build_classification_decision_contract,
    extract_label_decision_contract,
)


def test_build_classification_decision_contract_backfills_stable_fields():
    payload = {
        "part_type": "人孔",
        "confidence_source": "hybrid",
        "rule_version": "HybridClassifier-v1",
    }

    contract = build_classification_decision_contract(payload)

    assert contract["part_type"] == "人孔"
    assert contract["fine_part_type"] == "人孔"
    assert contract["coarse_part_type"] == "开孔件"
    assert contract["decision_source"] == "hybrid"
    assert contract["final_decision_source"] == "hybrid"
    assert contract["confidence_source"] == "hybrid"
    assert contract["is_coarse_label"] is False
    assert contract["rule_version"] == "HybridClassifier-v1"


def test_extract_label_decision_contract_preserves_explicit_fields():
    payload = {
        "part_type": "法兰",
        "fine_part_type": "法兰",
        "coarse_part_type": "法兰",
        "final_decision_source": "filename",
        "is_coarse_label": "true",
    }

    contract = extract_label_decision_contract(payload)

    assert contract == {
        "part_type": "法兰",
        "fine_part_type": "法兰",
        "coarse_part_type": "法兰",
        "decision_source": "filename",
        "final_decision_source": "filename",
        "is_coarse_label": True,
    }


def test_extract_label_decision_contract_uses_confidence_source_fallback():
    payload = {
        "part_type": "bolt",
        "confidence_source": "fusion",
    }

    contract = extract_label_decision_contract(payload)

    assert contract["part_type"] == "bolt"
    assert contract["fine_part_type"] == "bolt"
    assert contract["coarse_part_type"] == "bolt"
    assert contract["decision_source"] == "fusion"
    assert contract["final_decision_source"] == "fusion"
    assert contract["is_coarse_label"] is True
