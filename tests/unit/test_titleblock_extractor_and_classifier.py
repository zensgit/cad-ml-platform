import ezdxf

from src.ml.titleblock_extractor import TitleBlockClassifier, TitleBlockExtractor
from src.utils.dxf_io import read_dxf_entities_from_bytes, write_dxf_document_to_bytes


def _make_doc_with_bbox() -> ezdxf.document.Drawing:
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    # Ensure bbox spans a reasonable range for titleblock region heuristics.
    msp.add_line((0, 0), (100, 100))
    return doc


def test_titleblock_classifier_normalizes_part_name_from_attrib_tag() -> None:
    doc = _make_doc_with_bbox()
    block = doc.blocks.new(name="TBLOCK")
    # ATTDEF required for attribs; tag is the key signal used by extractor.
    block.add_attdef(tag="名称", insert=(0, 0))

    msp = doc.modelspace()
    blockref = msp.add_blockref("TBLOCK", insert=(10, 10))
    blockref.add_attrib(tag="名称", text="拖车DN1500", insert=(10, 10))

    data = write_dxf_document_to_bytes(doc)
    entities = read_dxf_entities_from_bytes(data)

    clf = TitleBlockClassifier(synonyms={"拖车": ["拖车"]})
    pred = clf.predict(entities)

    assert pred["status"] == "matched"
    assert pred["label"] == "拖车"
    info = pred.get("title_block_info") or {}
    assert info.get("part_name") == "拖车DN1500"
    assert info.get("part_name_normalized") == "拖车"


def test_titleblock_extractor_reads_insert_virtual_entities() -> None:
    doc = _make_doc_with_bbox()
    block = doc.blocks.new(name="TBLOCK_TEXT")
    # Place a TEXT inside the block; titleblock extractor should inspect it via virtual_entities().
    block.add_text("名称:人孔", dxfattribs={"height": 2.5}).set_placement((0, 0))

    msp = doc.modelspace()
    # Insert into the titleblock region (x>=60, y<=40 for bbox 0..100).
    msp.add_blockref("TBLOCK_TEXT", insert=(80, 20))

    data = write_dxf_document_to_bytes(doc)
    entities = read_dxf_entities_from_bytes(data)

    extractor = TitleBlockExtractor()
    info = extractor.extract_from_entities(entities)

    assert info.part_name == "人孔"
