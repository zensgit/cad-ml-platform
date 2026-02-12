from pathlib import Path

import pytest

from src.core.knowledge.tolerance import get_limit_deviations

ISO286_PATH = Path("data/knowledge/iso286_deviations.json")


@pytest.mark.skipif(not ISO286_PATH.exists(), reason="iso286 deviations table not available")
def test_limit_deviations_hole_h7_positive_upper():
    lower, upper = get_limit_deviations("H", 7, 10.0)
    assert lower == 0.0
    assert upper > 0.0


@pytest.mark.skipif(not ISO286_PATH.exists(), reason="iso286 deviations table not available")
def test_limit_deviations_shaft_h6_upper_zero():
    lower, upper = get_limit_deviations("h", 6, 10.0)
    assert upper == 0.0
    assert lower < 0.0


@pytest.mark.skipif(not ISO286_PATH.exists(), reason="iso286 deviations table not available")
def test_limit_deviations_shaft_g6_negative_band():
    lower, upper = get_limit_deviations("g", 6, 10.0)
    assert upper < 0.0
    assert lower < upper


@pytest.mark.skipif(not ISO286_PATH.exists(), reason="iso286 deviations table not available")
def test_limit_deviations_hole_k7_crosses_zero():
    lower, upper = get_limit_deviations("K", 7, 6.0)
    assert lower < 0.0
    assert upper >= 0.0


@pytest.mark.skipif(not ISO286_PATH.exists(), reason="iso286 deviations table not available")
def test_limit_deviations_hole_p7_negative_band():
    lower, upper = get_limit_deviations("P", 7, 6.0)
    assert upper < 0.0
    assert lower < upper


@pytest.mark.skipif(not ISO286_PATH.exists(), reason="iso286 deviations table not available")
def test_limit_deviations_hole_js6_symmetric_table():
    lower, upper = get_limit_deviations("JS", 6, 6.0)
    assert lower == upper


@pytest.mark.skipif(not ISO286_PATH.exists(), reason="iso286 deviations table not available")
def test_limit_deviations_hole_cd6_normalizes_label():
    lower, upper = get_limit_deviations("CD", 6, 3.0)
    assert (lower, upper) == (34.0, 40.0)


@pytest.mark.skipif(not ISO286_PATH.exists(), reason="iso286 deviations table not available")
def test_limit_deviations_shaft_y6_normalizes_label():
    lower, upper = get_limit_deviations("y", 6, 24.0)
    assert (lower, upper) == (63.0, 76.0)
