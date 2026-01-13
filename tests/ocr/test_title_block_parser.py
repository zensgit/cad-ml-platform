"""Tests for title block parsing helpers."""

from src.core.ocr.parsing.title_block_parser import (
    parse_title_block,
    parse_title_block_with_confidence,
)


def test_parse_title_block_english() -> None:
    text = (
        "Drawing No: DWG-2025-01 Rev A Part Name: Bracket "
        "Material: Aluminum Scale 1:2 Sheet 1 of 3 Date 2025-01-12 "
        "Weight 2.5kg Company ACME Projection third"
    )
    parsed = parse_title_block(text)
    assert parsed["drawing_number"] == "DWG-2025-01"
    assert parsed["revision"] == "A"
    assert parsed["part_name"] == "Bracket"
    assert parsed["material"] == "Aluminum"
    assert parsed["scale"] == "1:2"
    assert parsed["sheet"] == "1/3"
    assert parsed["date"] == "2025-01-12"
    assert parsed["weight"] == "2.5kg"
    assert parsed["company"] == "ACME"
    assert parsed["projection"] == "third"


def test_parse_title_block_chinese() -> None:
    text = (
        "图号: A-100 修订: B 名称: 支架 材料: 钢 比例 1:1 "
        "页 2/5 日期 2025/01/02 重量 3.2kg 公司: ACME 投影 第三角"
    )
    parsed = parse_title_block(text)
    assert parsed["drawing_number"] == "A-100"
    assert parsed["revision"] == "B"
    assert parsed["part_name"] == "支架"
    assert parsed["material"] == "钢"
    assert parsed["scale"] == "1:1"
    assert parsed["sheet"] == "2/5"
    assert parsed["date"] == "2025/01/02"
    assert parsed["weight"] == "3.2kg"
    assert parsed["company"] == "ACME"
    assert parsed["projection"] == "third"


def test_parse_title_block_with_confidence() -> None:
    lines = [
        {"text": "Drawing No: DWG-9", "score": 0.91},
        {"text": "Scale 1:2", "score": 0.8},
    ]
    values, confidences = parse_title_block_with_confidence(lines)
    assert values["drawing_number"] == "DWG-9"
    assert confidences["drawing_number"] == 0.91
    assert values["scale"] == "1:2"
    assert confidences["scale"] == 0.8


def test_parse_title_block_aliases() -> None:
    text = "DWG# X-42 REV. C DESC: Bracket MATL: Steel SHT 2 of 4 WT 1.2kg"
    parsed = parse_title_block(text)
    assert parsed["drawing_number"] == "X-42"
    assert parsed["revision"] == "C"
    assert parsed["part_name"] == "Bracket"
    assert parsed["material"] == "Steel"
    assert parsed["sheet"] == "2/4"
    assert parsed["weight"] == "1.2kg"


def test_parse_title_block_scale_nts() -> None:
    text = "Scale: N.T.S."
    parsed = parse_title_block(text)
    assert parsed["scale"] == "NTS"


def test_parse_title_block_scale_spacing_normalized() -> None:
    text = "Scale 1 : 2"
    parsed = parse_title_block(text)
    assert parsed["scale"] == "1:2"
