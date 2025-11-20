from src.core.ocr.parsing.bbox_mapper import assign_bboxes
from src.core.ocr.base import DimensionInfo, DimensionType, SymbolInfo, SymbolType


def test_assign_bboxes_matches_raw_and_value():
    dims = [
        DimensionInfo(type=DimensionType.diameter, value=20.0, raw="Φ20±0.02"),
        DimensionInfo(type=DimensionType.radius, value=5.0, raw="R5"),
    ]
    syms = [SymbolInfo(type=SymbolType.surface_roughness, value="3.2", normalized_form="surface_roughness",)]
    lines = [
        {"text": "Φ20±0.02", "bbox": [10, 10, 40, 10]},
        {"text": "R5", "bbox": [60, 10, 20, 10]},
        {"text": "Ra3.2", "bbox": [90, 10, 30, 10]},
    ]
    assign_bboxes(dims, syms, lines)
    assert dims[0].bbox == [10, 10, 40, 10]
    assert dims[1].bbox == [60, 10, 20, 10]
    assert syms[0].bbox == [90, 10, 30, 10]


def test_assign_bboxes_heuristic_path_when_no_exact_match():
    # Craft OCR lines that do not include exact raw or value substring for diameter 20.0
    dims = [
        DimensionInfo(type=DimensionType.diameter, value=20.0, raw="Φ20±0.02"),
    ]
    syms = []
    lines = [
        {"text": "phi 20.00 mm tol", "bbox": [5, 5, 50, 12]},
        {"text": "R 5.000", "bbox": [60, 5, 30, 12]},
    ]
    assign_bboxes(dims, syms, lines)
    # Heuristic should match first line due to similarity + numeric proximity + type hint
    assert dims[0].bbox == [5, 5, 50, 12]
