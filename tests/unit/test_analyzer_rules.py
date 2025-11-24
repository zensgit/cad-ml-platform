from src.core.analyzer import CADAnalyzer
from src.core.feature_extractor import FeatureExtractor
from src.models.cad_document import CadDocument, CadEntity


import asyncio


def test_analyzer_classification_simple():
    doc = CadDocument(file_name="a.stl", format="stl", entities=[CadEntity(kind="FACET")])
    extractor = FeatureExtractor()
    features = asyncio.run(extractor.extract(doc))
    analyzer = CADAnalyzer()
    result = asyncio.run(analyzer.classify_part(doc, features))
    assert result["type"] == "simple_plate"


def test_quality_empty():
    doc = CadDocument(file_name="empty.dxf", format="dxf")
    extractor = FeatureExtractor()
    features = asyncio.run(extractor.extract(doc))
    analyzer = CADAnalyzer()
    quality = asyncio.run(analyzer.check_quality(doc, features))
    assert "empty_document" in quality["issues"]
