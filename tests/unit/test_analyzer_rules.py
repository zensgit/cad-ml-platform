import asyncio
from unittest.mock import MagicMock, patch

from src.core.analyzer import CADAnalyzer
from src.core.feature_extractor import FeatureExtractor
from src.models.cad_document import CadDocument, CadEntity


def test_analyzer_classification_simple():
    doc = CadDocument(file_name="a.stl", format="stl", entities=[CadEntity(kind="FACET")])
    extractor = FeatureExtractor()
    features = asyncio.run(extractor.extract(doc))
    analyzer = CADAnalyzer()
    result = asyncio.run(analyzer.classify_part(doc, features))
    assert result["type"] == "simple_plate"
    assert result["fine_type"] == "simple_plate"
    assert result["coarse_type"] == "simple_plate"
    assert result["is_coarse_type"] is True
    assert result["decision_source"] == "rule_based"
    assert result["review_reasons"] == []


def test_quality_empty():
    doc = CadDocument(file_name="empty.dxf", format="dxf")
    extractor = FeatureExtractor()
    features = asyncio.run(extractor.extract(doc))
    analyzer = CADAnalyzer()
    quality = asyncio.run(analyzer.check_quality(doc, features))
    assert "empty_document" in quality["issues"]


def test_analyzer_v16_classification_exposes_contract_fields():
    doc = CadDocument(
        file_name="a.dxf",
        format="dxf",
        entities=[CadEntity(kind="LINE")],
        metadata={"file_path": "/tmp/fake.dxf"},
    )
    object.__setattr__(doc, "file_path", "/tmp/fake.dxf")
    analyzer = CADAnalyzer()

    mock_result = MagicMock()
    mock_result.category = "人孔"
    mock_result.confidence = 0.96
    mock_result.probabilities = {"人孔": 0.96, "法兰": 0.04}
    mock_result.model_version = "v16"
    mock_result.needs_review = True
    mock_result.review_reason = "edge_case"
    mock_result.top2_category = "法兰"
    mock_result.top2_confidence = 0.04

    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = mock_result

    with patch("src.core.analyzer._get_v16_classifier", return_value=mock_classifier):
        result = asyncio.run(analyzer._classify_with_v16(doc))

    assert result is not None
    assert result["type"] == "人孔"
    assert result["fine_type"] == "人孔"
    assert result["coarse_type"] == "开孔件"
    assert result["is_coarse_type"] is False
    assert result["decision_source"] == "v16"
    assert result["needs_review"] is True
    assert result["review_reason"] == "edge_case"
    assert result["review_reasons"] == ["edge_case"]
    assert result["top2_category"] == "法兰"
    assert result["top2_confidence"] == 0.04
