"""Tests for ProcessClassifier."""

import pytest

from src.core.ocr.base import (
    HeatTreatmentInfo,
    HeatTreatmentType,
    ProcessRequirements,
    SurfaceTreatmentInfo,
    SurfaceTreatmentType,
    WeldingInfo,
    WeldingType,
)
from src.ml.process_classifier import (
    ProcessClassifier,
    ProcessClassificationResult,
    get_process_classifier,
)


class TestProcessClassifier:
    """Test ProcessClassifier core functionality."""

    def test_empty_process_requirements(self):
        """Empty ProcessRequirements returns no suggestions."""
        classifier = ProcessClassifier()
        result = classifier.predict(None)

        assert result.suggested_labels == []
        assert result.confidence == 0.0
        assert result.features_found == {}

    def test_empty_process_requirements_object(self):
        """Empty ProcessRequirements object returns no suggestions."""
        classifier = ProcessClassifier()
        proc = ProcessRequirements()
        result = classifier.predict(proc)

        assert result.suggested_labels == []
        assert result.confidence == 0.0

    def test_heat_treatment_suggests_part_drawing(self):
        """Heat treatment suggests 零件图/机械制图."""
        classifier = ProcessClassifier()
        proc = ProcessRequirements(
            heat_treatments=[
                HeatTreatmentInfo(
                    type=HeatTreatmentType.quenching,
                    hardness="HRC58-62",
                )
            ]
        )
        result = classifier.predict(proc)

        assert "零件图" in result.suggested_labels
        assert "机械制图" in result.suggested_labels
        assert result.confidence > 0
        assert result.features_found["heat_treatment"] == 1

    def test_surface_treatment_suggests_part_drawing(self):
        """Surface treatment suggests 零件图/机械制图."""
        classifier = ProcessClassifier()
        proc = ProcessRequirements(
            surface_treatments=[
                SurfaceTreatmentInfo(
                    type=SurfaceTreatmentType.galvanizing,
                    thickness=10.0,
                )
            ]
        )
        result = classifier.predict(proc)

        assert "零件图" in result.suggested_labels
        assert result.confidence > 0
        assert result.features_found["surface_treatment"] == 1

    def test_welding_suggests_assembly_drawing(self):
        """Welding suggests 装配图/结构件/焊接件."""
        classifier = ProcessClassifier()
        proc = ProcessRequirements(
            welding=[
                WeldingInfo(
                    type=WeldingType.tig_welding,
                    filler_material="ER50-6",
                )
            ]
        )
        result = classifier.predict(proc)

        assert "装配图" in result.suggested_labels
        assert "焊接件" in result.suggested_labels
        assert result.confidence > 0
        assert result.features_found["welding"] == 1

    def test_general_notes_suggests_part_drawing(self):
        """General notes suggest 零件图/机械制图."""
        classifier = ProcessClassifier()
        proc = ProcessRequirements(
            general_notes=["未注公差按GB/T1804-m", "未注圆角R3"]
        )
        result = classifier.predict(proc)

        assert "零件图" in result.suggested_labels
        assert result.confidence > 0
        assert result.features_found["general_notes"] == 2

    def test_multiple_features_increase_confidence(self):
        """Multiple feature types increase confidence."""
        classifier = ProcessClassifier()

        # Single feature
        proc_single = ProcessRequirements(
            heat_treatments=[
                HeatTreatmentInfo(type=HeatTreatmentType.quenching)
            ]
        )
        result_single = classifier.predict(proc_single)

        # Multiple features
        proc_multi = ProcessRequirements(
            heat_treatments=[
                HeatTreatmentInfo(type=HeatTreatmentType.quenching)
            ],
            surface_treatments=[
                SurfaceTreatmentInfo(type=SurfaceTreatmentType.chromating)
            ],
            general_notes=["去毛刺"]
        )
        result_multi = classifier.predict(proc_multi)

        assert result_multi.confidence > result_single.confidence

    def test_confidence_bounded(self):
        """Confidence should be bounded by max_confidence."""
        classifier = ProcessClassifier(min_confidence=0.3, max_confidence=0.7)

        # Many features
        proc = ProcessRequirements(
            heat_treatments=[
                HeatTreatmentInfo(type=HeatTreatmentType.quenching),
                HeatTreatmentInfo(type=HeatTreatmentType.tempering),
            ],
            surface_treatments=[
                SurfaceTreatmentInfo(type=SurfaceTreatmentType.chromating),
                SurfaceTreatmentInfo(type=SurfaceTreatmentType.polishing),
            ],
            welding=[
                WeldingInfo(type=WeldingType.tig_welding),
            ],
            general_notes=["note1", "note2", "note3"]
        )
        result = classifier.predict(proc)

        assert result.confidence <= 0.7

    def test_welding_and_surface_mixed(self):
        """Mixed features result in combined suggestions."""
        classifier = ProcessClassifier()
        proc = ProcessRequirements(
            surface_treatments=[
                SurfaceTreatmentInfo(type=SurfaceTreatmentType.painting)
            ],
            welding=[
                WeldingInfo(type=WeldingType.spot_welding)
            ]
        )
        result = classifier.predict(proc)

        # Should have both part and assembly suggestions
        assert len(result.suggested_labels) > 2
        assert result.features_found["surface_treatment"] == 1
        assert result.features_found["welding"] == 1


class TestProcessClassifierFromText:
    """Test ProcessClassifier.predict_from_text method."""

    def test_predict_from_text_heat_treatment(self):
        """Predict from text with heat treatment."""
        classifier = ProcessClassifier()
        result = classifier.predict_from_text("整体淬火 HRC58-62")

        assert "零件图" in result.suggested_labels
        assert result.features_found.get("heat_treatment", 0) > 0

    def test_predict_from_text_welding(self):
        """Predict from text with welding."""
        classifier = ProcessClassifier()
        result = classifier.predict_from_text("氩弧焊 焊丝ER50-6 焊脚6mm")

        assert "装配图" in result.suggested_labels or "焊接件" in result.suggested_labels
        assert result.features_found.get("welding", 0) > 0

    def test_predict_from_text_empty(self):
        """Predict from empty text."""
        classifier = ProcessClassifier()
        result = classifier.predict_from_text("")

        assert result.suggested_labels == []
        assert result.confidence == 0.0

    def test_predict_from_text_no_process(self):
        """Predict from text without process info."""
        classifier = ProcessClassifier()
        result = classifier.predict_from_text("Φ20±0.02 R5 M10×1.5")

        assert result.suggested_labels == []
        assert result.confidence == 0.0


class TestProcessClassificationResult:
    """Test ProcessClassificationResult dataclass."""

    def test_to_dict(self):
        """to_dict returns correct structure."""
        result = ProcessClassificationResult(
            suggested_labels=["零件图", "机械制图"],
            confidence=0.5,
            features_found={"heat_treatment": 1, "welding": 0},
        )

        d = result.to_dict()
        assert d["suggested_labels"] == ["零件图", "机械制图"]
        assert d["confidence"] == 0.5
        assert d["features_found"]["heat_treatment"] == 1


class TestGetProcessClassifier:
    """Test singleton getter."""

    def test_singleton(self):
        """get_process_classifier returns singleton."""
        c1 = get_process_classifier()
        c2 = get_process_classifier()
        assert c1 is c2

    def test_returns_classifier(self):
        """get_process_classifier returns ProcessClassifier."""
        c = get_process_classifier()
        assert isinstance(c, ProcessClassifier)
