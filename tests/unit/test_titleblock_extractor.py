from __future__ import annotations

import pytest


def _build_doc():
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    msp = doc.modelspace()

    # establish bbox
    msp.add_line((0, 0), (100, 0))
    msp.add_line((0, 0), (0, 100))

    # title block texts in bottom-right quadrant
    msp.add_text("名称: 人孔", dxfattribs={"height": 2, "insert": (80, 10)})
    msp.add_text("材料: 304", dxfattribs={"height": 2, "insert": (80, 6)})

    return msp


def _build_doc_with_attribs():
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    msp = doc.modelspace()

    # establish bbox
    msp.add_line((0, 0), (100, 0))
    msp.add_line((0, 0), (0, 100))

    block = doc.blocks.new(name="TITLEBLOCK")
    block.add_attdef("图样名称", insert=(0, 0))
    block.add_attdef("图样代号", insert=(0, 0))
    block.add_attdef("材料", insert=(0, 0))

    insert = msp.add_blockref("TITLEBLOCK", (80, 10))
    insert.add_attrib("图样名称", "保护罩组件", insert=(80, 10))
    insert.add_attrib("图样代号", "BTJ02230301120-03", insert=(80, 8))
    insert.add_attrib("材料", "304", insert=(80, 6))

    return msp


def test_titleblock_extraction() -> None:
    from src.ml.titleblock_extractor import TitleBlockExtractor

    msp = _build_doc()
    extractor = TitleBlockExtractor(region_x_ratio=0.6, region_y_ratio=0.4)
    info = extractor.extract_from_msp(msp)

    assert info.part_name == "人孔"
    assert info.material == "304"
    assert info.raw_texts


def test_titleblock_classifier_matches() -> None:
    from src.ml.titleblock_extractor import TitleBlockClassifier

    msp = _build_doc()
    synonyms = {"人孔": ["人孔"]}
    classifier = TitleBlockClassifier(synonyms=synonyms)
    result = classifier.predict(list(msp))

    assert result["label"] == "人孔"
    assert result["confidence"] >= 0.6


def test_titleblock_extraction_from_attribs() -> None:
    from src.ml.titleblock_extractor import TitleBlockExtractor

    msp = _build_doc_with_attribs()
    extractor = TitleBlockExtractor(region_x_ratio=0.6, region_y_ratio=0.4)
    info = extractor.extract_from_msp(msp)

    assert info.part_name == "保护罩组件"
    assert info.drawing_number == "BTJ02230301120-03"
    assert info.material == "304"
