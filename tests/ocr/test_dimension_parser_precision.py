"""Precision tests for dual tolerance binding and thread pitch variants."""

from src.core.ocr.base import DimensionType
from src.core.ocr.parsing.dimension_parser import parse_dimensions_and_symbols


def test_dual_tolerance_binding_gap_limit():
    text = "Φ20 +0.02 -0.01"
    dims, _ = parse_dimensions_and_symbols(text)
    d = next((x for x in dims if x.type == DimensionType.diameter), None)
    assert d is not None
    assert d.tol_pos == 0.02
    assert d.tol_neg == 0.01
    assert d.tolerance == 0.02


def test_dual_tolerance_not_bound_if_far_gap():
    # Insert a long gap to exceed threshold and avoid binding
    text = "Φ20" + (" " * 50) + "+0.02 -0.01"
    dims, _ = parse_dimensions_and_symbols(text)
    d = next((x for x in dims if x.type == DimensionType.diameter), None)
    assert d is not None
    # Should not bind dual tolerance due to excessive gap
    assert d.tol_pos is None and d.tol_neg is None


def test_thread_pitch_variants():
    samples = [
        "M10×1.5",
        "M10x1.5",
        "M10X1.5",
        "M10*1.5",
    ]
    for s in samples:
        dims, _ = parse_dimensions_and_symbols(s)
        thr = next((x for x in dims if x.type == DimensionType.thread), None)
        assert thr is not None
        assert thr.value == 10.0
        assert abs(thr.pitch - 1.5) < 1e-6


def test_mixed_sequence_robustness():
    # Single-sided first, then dual tolerance nearby
    text = "R5 +0.01 some text M10x1.5 Φ20 +0.02 -0.01"
    dims, _ = parse_dimensions_and_symbols(text)
    dia = next((x for x in dims if x.type == DimensionType.diameter), None)
    assert dia is not None
    assert dia.tol_pos == 0.02 and dia.tol_neg == 0.01
    thr = next((x for x in dims if x.type == DimensionType.thread), None)
    assert thr is not None and abs(thr.pitch - 1.5) < 1e-6
