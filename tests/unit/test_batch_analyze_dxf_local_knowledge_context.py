import json

from scripts.batch_analyze_dxf_local import _extract_knowledge_context


def test_extract_knowledge_context_serializes_categories_and_types() -> None:
    payload = _extract_knowledge_context(
        {
            "knowledge_checks": [
                {"category": "material", "item": "304"},
                {"category": "surface_finish", "item": "Ra3.2"},
            ],
            "violations": [
                {"category": "knowledge_conflict", "severity": "warn"},
            ],
            "standards_candidates": [
                {"type": "material", "designation": "304"},
                {"type": "surface_finish", "designation": "Ra 3.2"},
            ],
            "knowledge_hints": [
                {"label": "人孔", "score": 0.8},
            ],
        }
    )

    assert json.loads(payload["knowledge_checks"])[0]["category"] == "material"
    assert payload["knowledge_check_categories"] == "material;surface_finish"
    assert payload["knowledge_violation_categories"] == "knowledge_conflict"
    assert payload["knowledge_standard_types"] == "material;surface_finish"
    assert payload["knowledge_hint_labels"] == "人孔"
