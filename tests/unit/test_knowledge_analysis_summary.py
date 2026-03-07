from src.core.knowledge.analysis_summary import build_knowledge_summary


def test_build_knowledge_summary_extracts_checks_and_conflicts() -> None:
    payload = build_knowledge_summary(
        text_signals="M10x1.25 ISO 2768-mK IT7 位置度 0.2 M A C B 人孔",
        geometric_features={"complexity_score": 12.0},
        entity_counts={"CIRCLE": 2, "LINE": 4},
        fine_part_type="人孔",
        coarse_part_type="开孔件",
    )

    categories = {item["category"] for item in payload["knowledge_checks"]}
    assert "thread_standard" in categories
    assert "general_tolerance" in categories
    assert "it_grade" in categories
    assert "gdt" in categories

    standards = payload["standards_candidates"]
    assert any(item.get("designation") == "M10x1.25" for item in standards)
    assert any(item.get("designation") == "ISO 2768-MK" for item in standards)

    violations = payload["violations"]
    assert violations
    assert violations[0]["category"] == "datum_sequence"
