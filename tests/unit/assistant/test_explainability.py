"""Tests for assistant explainability helpers."""

from src.core.assistant import (
    CADAssistant,
    RetrievalResult,
    RetrievalSource,
    build_assistant_evidence,
)


def test_build_assistant_evidence_sorts_and_extracts_key_facts():
    results = [
        RetrievalResult(
            source=RetrievalSource.THREADS,
            relevance=0.88,
            data={
                "designation": "M10",
                "nominal_diameter": 10.0,
                "pitch": 1.5,
                "tap_drill": 8.5,
            },
            summary="M10 thread specification",
            metadata={"match_type": "direct"},
        ),
        RetrievalResult(
            source=RetrievalSource.MATERIALS,
            relevance=0.95,
            data={
                "grade": "S30408",
                "properties": {
                    "name": "304不锈钢",
                    "tensile_strength": 520,
                    "yield_strength": 205,
                },
            },
            summary="304 stainless steel properties",
            metadata={"match_type": "keyword"},
        ),
    ]

    evidence = build_assistant_evidence(results)

    assert [item.reference_id for item in evidence] == ["E1", "E2"]
    assert evidence[0].source == "materials"
    assert evidence[0].relevance == 0.95
    assert "牌号: S30408" in evidence[0].key_facts
    assert "抗拉强度: 520 MPa" in evidence[0].key_facts
    assert evidence[1].source == "threads"
    assert "攻丝底孔: 8.5 mm" in evidence[1].key_facts


def test_assistant_ask_returns_structured_evidence():
    assistant = CADAssistant()

    response = assistant.ask("M10螺纹的底孔尺寸?")

    assert response.evidence
    assert response.evidence[0].reference_id == "E1"
    assert response.evidence[0].source == "threads"
    assert any("攻丝底孔" in fact for fact in response.evidence[0].key_facts)
    assert response.metadata["evidence_count"] == len(response.evidence)
