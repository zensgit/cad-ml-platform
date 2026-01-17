"""Unit coverage for FusionAnalyzer MVP logic."""

from __future__ import annotations

from src.core.knowledge.fusion_analyzer import (
    FusionAnalyzer,
    build_doc_metadata,
    build_l2_features,
)
from src.core.knowledge.fusion_contracts import ConflictLevel, DecisionSource
from src.models.cad_document import BoundingBox, CadDocument


def test_invalid_format_fallback() -> None:
    analyzer = FusionAnalyzer(ai_confidence_threshold=0.7)
    decision = analyzer.analyze(
        doc_metadata={"valid_format": False},
        l2_features={},
        l3_features={},
        l4_prediction=None,
    )

    assert decision.primary_label == "Unknown"
    assert decision.confidence == 0.0
    assert decision.source == DecisionSource.RULE_BASED
    assert "Invalid file format" in " ".join(decision.reasons)


def test_ai_confident_no_conflict() -> None:
    analyzer = FusionAnalyzer(ai_confidence_threshold=0.7)
    decision = analyzer.analyze(
        doc_metadata={"valid_format": True},
        l2_features={"aspect_ratio": 2.0},
        l3_features={},
        l4_prediction={"label": "Slot", "confidence": 0.9},
    )

    assert decision.primary_label == "Slot"
    assert decision.source == DecisionSource.AI_MODEL
    assert decision.consistency_check == ConflictLevel.NONE
    assert decision.confidence == 0.9


def test_ai_conflict_fallback() -> None:
    analyzer = FusionAnalyzer(ai_confidence_threshold=0.7)
    decision = analyzer.analyze(
        doc_metadata={"valid_format": True},
        l2_features={"aspect_ratio": 1.0},
        l3_features={},
        l4_prediction={"label": "Slot", "confidence": 0.9},
    )

    assert decision.source == DecisionSource.RULE_BASED
    assert decision.consistency_check == ConflictLevel.HIGH
    assert decision.primary_label == "Standard_Part"


def test_rule_based_slot_fallback() -> None:
    analyzer = FusionAnalyzer(ai_confidence_threshold=0.9)
    decision = analyzer.analyze(
        doc_metadata={"valid_format": True},
        l2_features={"aspect_ratio": 2.2},
        l3_features={},
        l4_prediction={"label": "Slot", "confidence": 0.4},
    )

    assert decision.source == DecisionSource.RULE_BASED
    assert decision.primary_label == "Slot"


def test_build_doc_metadata_valid_format() -> None:
    doc = CadDocument(file_name="sample.step", format="step")
    meta = build_doc_metadata(doc)
    assert meta["valid_format"] is True


def test_build_doc_metadata_invalid_format() -> None:
    doc = CadDocument(file_name="sample.bin", format="bin")
    meta = build_doc_metadata(doc)
    assert meta["valid_format"] is False


def test_build_l2_features_aspect_ratio() -> None:
    doc = CadDocument(
        file_name="sample.step",
        format="step",
        bounding_box=BoundingBox(min_x=0, min_y=0, min_z=0, max_x=10, max_y=5, max_z=2),
    )
    l2 = build_l2_features(doc)
    assert l2["aspect_ratio"] == 2.0
    assert l2["bbox_width"] == 10.0
    assert l2["bbox_height"] == 5.0
