from src.core.ocr.parsing.identifier_parser import extract_identifiers


def test_extract_identifiers_from_same_line_with_bbox_and_confidence():
    identifiers = extract_identifiers(
        ocr_lines=[
            {"text": "图号: DWG-123", "bbox": [10, 10, 80, 12], "score": 0.95},
            {"text": "材料: Aluminum", "bbox": [10, 30, 90, 12], "score": 0.87},
        ]
    )

    drawing_number = next(item for item in identifiers if item.identifier_type == "drawing_number")
    material = next(item for item in identifiers if item.identifier_type == "material")

    assert drawing_number.value == "DWG-123"
    assert drawing_number.source == "ocr_line"
    assert drawing_number.bbox == [10, 10, 80, 12]
    assert drawing_number.confidence == 0.95

    assert material.value == "Aluminum"
    assert material.source_text == "材料: Aluminum"


def test_extract_identifiers_from_split_lines():
    identifiers = extract_identifiers(
        ocr_lines=[
            {"text": "图号", "bbox": [10, 10, 20, 12], "score": 0.91},
            {"text": "DWG-456", "bbox": [40, 10, 60, 12], "score": 0.93},
            {"text": "材料", "bbox": [10, 30, 20, 12], "score": 0.9},
            {"text": "45钢", "bbox": [40, 30, 30, 12], "score": 0.92},
        ]
    )

    drawing_number = next(item for item in identifiers if item.identifier_type == "drawing_number")
    material = next(item for item in identifiers if item.identifier_type == "material")

    assert drawing_number.value == "DWG-456"
    assert drawing_number.bbox == [40, 10, 60, 12]
    assert "图号" in drawing_number.source_text

    assert material.value == "45钢"
    assert material.source == "ocr_line"


def test_extract_identifiers_falls_back_to_text_and_deduplicates():
    identifiers = extract_identifiers(
        text="图号: DWG-789 材料: Steel",
        ocr_lines=[{"text": "图号: DWG-789", "bbox": [1, 1, 10, 10], "score": 0.9}],
    )

    drawing_numbers = [
        item for item in identifiers if item.identifier_type == "drawing_number"
    ]
    materials = [item for item in identifiers if item.identifier_type == "material"]

    assert len(drawing_numbers) == 1
    assert drawing_numbers[0].source == "ocr_line"
    assert len(materials) == 1
    assert materials[0].source == "regex_text"
