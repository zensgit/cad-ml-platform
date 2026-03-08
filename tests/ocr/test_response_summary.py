from src.core.ocr.base import ProcessRequirements, TitleBlock
from src.core.ocr.response_summary import (
    build_engineering_signals,
    build_field_coverage,
    build_review_hints,
)


def test_build_review_hints_flags_missing_critical_fields():
    title_block = TitleBlock(drawing_number="DWG-001")
    field_coverage = build_field_coverage(
        title_block,
        ["drawing_number", "revision", "part_name", "material"],
    )
    engineering_signals = build_engineering_signals(
        title_block=title_block,
        dimensions=[],
        symbols=[],
        process_requirements=ProcessRequirements(),
    )

    review_hints = build_review_hints(
        title_block=title_block,
        identifiers=[],
        field_coverage=field_coverage,
        engineering_signals=engineering_signals,
    )

    assert review_hints["review_recommended"] is True
    assert review_hints["missing_critical_fields"] == ["part_name", "revision", "material"]
    assert "no_identifiers" in review_hints["review_reasons"]
    assert review_hints["primary_gap"] == "missing_critical_fields"
    assert review_hints["review_priority"] == "high"
    assert review_hints["automation_ready"] is False
    assert "fill_critical_title_block_fields" in review_hints["recommended_actions"]
    assert "manual_review_gate" in review_hints["recommended_actions"]
    assert review_hints["readiness_band"] == "low"


def test_build_review_hints_high_readiness_when_core_evidence_present():
    title_block = TitleBlock(
        drawing_number="DWG-001",
        revision="A",
        part_name="Bracket",
        material="Steel",
    )
    field_coverage = build_field_coverage(
        title_block,
        ["drawing_number", "revision", "part_name", "material"],
    )
    engineering_signals = build_engineering_signals(
        title_block=title_block,
        dimensions=[{"type": "diameter", "value": 10}],
        symbols=[],
        process_requirements=ProcessRequirements(general_notes=["按GB/T 1804执行"]),
    )

    review_hints = build_review_hints(
        title_block=title_block,
        identifiers=[{"identifier_type": "drawing_number", "value": "DWG-001"}],
        field_coverage=field_coverage,
        engineering_signals=engineering_signals,
    )

    assert review_hints["review_recommended"] is False
    assert review_hints["missing_critical_fields"] == []
    assert review_hints["primary_gap"] == "ready"
    assert review_hints["review_priority"] == "low"
    assert review_hints["automation_ready"] is True
    assert review_hints["recommended_actions"] == []
    assert review_hints["readiness_band"] == "high"
