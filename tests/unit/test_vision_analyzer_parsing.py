from __future__ import annotations

import json

from src.core.vision_analyzer import VisionAnalyzer


def test_extracts_from_json_payload() -> None:
    analyzer = VisionAnalyzer()
    payload = {
        "objects": ["bolt", "nut"],
        "text": "OCR text",
        "dimensions": {"size": 10, "unit": "mm"},
    }
    text = json.dumps(payload)

    objects = analyzer._extract_objects(text)
    assert objects[0]["name"] == "bolt"
    assert analyzer._extract_text(text) == "OCR text"
    assert analyzer._extract_dimensions(text) == {"size": 10, "unit": "mm"}


def test_extracts_from_code_fence() -> None:
    analyzer = VisionAnalyzer()
    text = """```json
{"objects": [{"name": "gear"}], "text": "hi", "cad_elements": {"lines": 2}}
```"""

    objects = analyzer._extract_objects(text)
    assert objects == [{"name": "gear"}]
    assert analyzer._extract_text(text) == "hi"
    assert analyzer._extract_cad_elements(text) == {"lines": 2}


def test_extracts_from_plain_text() -> None:
    analyzer = VisionAnalyzer()
    text = "1. bracket\n2. spacer\nsize 10 mm +/- 0.1"

    objects = analyzer._extract_objects(text)
    assert objects[0]["name"] == "bracket"
    dimensions = analyzer._extract_dimensions(text)
    assert dimensions["values"][0]["value"] == 10.0
    assert dimensions["tolerances"][0]["value"] == 0.1


def test_extracts_symbol_tolerances_and_diameters() -> None:
    analyzer = VisionAnalyzer()
    text = "Ø10 ±0.05 and 20 mm +0.1/-0.05"

    dimensions = analyzer._extract_dimensions(text)
    values = dimensions["values"]
    assert any(item.get("type") == "diameter" for item in values)
    tolerances = dimensions["tolerances"]
    assert any(item.get("type") == "plus_minus" for item in tolerances)
    assert any(item.get("type") == "asymmetric" for item in tolerances)
