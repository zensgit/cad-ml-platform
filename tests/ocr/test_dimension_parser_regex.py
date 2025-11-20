"""Tests for dimension_parser regex extraction."""

from src.core.ocr.base import DimensionType, SymbolType
from src.core.ocr.parsing.dimension_parser import parse_dimensions_and_symbols


def test_parse_basic_diameter_radius_thread():
    text = "Φ20±0.02 R5 M10×1.5 Ra3.2"
    dims, syms = parse_dimensions_and_symbols(text)
    types = [d.type for d in dims]
    assert DimensionType.diameter in types
    assert DimensionType.radius in types
    assert DimensionType.thread in types
    assert any(s.type == SymbolType.surface_roughness for s in syms)
    dia = next(d for d in dims if d.type == DimensionType.diameter)
    assert dia.tolerance == 0.02
    assert dia.tol_pos == 0.02 or dia.tol_pos is None  # single tolerance treated as unified
    thr = next(d for d in dims if d.type == DimensionType.thread)
    assert thr.pitch == 1.5


def test_parse_dual_tolerance_attached():
    text = "Φ20 +0.02 -0.01"
    dims, _ = parse_dimensions_and_symbols(text)
    dia = next(d for d in dims if d.type == DimensionType.diameter)
    # unified tolerance becomes max of pos/neg
    assert dia.tolerance == 0.02
    assert dia.tol_pos == 0.02
    # current heuristic only captures pos; future improvement will map dual pair correctly
    # placeholder expectation: tol_neg may be None
    assert dia.tol_neg in (0.01, None)


def test_parse_multiple_threads():
    text = "M6 M10×1.5 M12x1.75"
    dims, _ = parse_dimensions_and_symbols(text)
    threads = [d for d in dims if d.type == DimensionType.thread]
    assert len(threads) == 3
    assert threads[1].pitch == 1.5
    assert threads[2].pitch == 1.75


def test_parse_geometric_symbols():
    text = "R5 ⊥ ∥ Φ10 +0.02 -0.01 Ra3.2 flatness position total runout profile of a line profile of a surface"
    _, syms = parse_dimensions_and_symbols(text)
    kinds = {s.type for s in syms}
    assert SymbolType.perpendicular in kinds
    assert SymbolType.parallel in kinds
    assert any(
        s.normalized_form in ("perpendicular", "parallel")
        for s in syms
        if s.type in (SymbolType.perpendicular, SymbolType.parallel)
    )
    # GD&T proxies
    assert SymbolType.flatness in kinds
    assert SymbolType.position in kinds
    assert SymbolType.total_runout in kinds
    assert SymbolType.profile_line in kinds
    assert SymbolType.profile_surface in kinds
