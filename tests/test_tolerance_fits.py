import importlib
import json
from pathlib import Path

import pytest

import src.core.knowledge.tolerance.fits as fits
from src.core.knowledge.tolerance import get_tolerance_value


def test_get_fit_deviations_hole_basis_h7_g6():
    result = fits.get_fit_deviations("H7/g6", 25)
    assert result is not None
    assert result.hole_lower_deviation_um == 0
    assert result.hole_upper_deviation_um > 0
    assert result.max_clearance_um >= result.min_clearance_um


def test_get_fit_deviations_js_shaft_symmetry():
    result = fits.get_fit_deviations("H7/js6", 25)
    assert result is not None
    assert result.hole_lower_deviation_um == 0
    assert result.shaft_upper_deviation_um == -result.shaft_lower_deviation_um


def test_get_fit_deviations_non_h_hole():
    nominal = 25
    result = fits.get_fit_deviations("N9/h9", nominal)
    assert result is not None
    tolerance = get_tolerance_value(nominal, "IT9")
    assert tolerance is not None
    assert result.hole_upper_deviation_um - result.hole_lower_deviation_um == tolerance


def test_get_fit_deviations_non_h_hole_abc():
    nominal = 25
    tolerance = fits.get_tolerance_value(nominal, "IT9")
    for symbol in ("A", "B", "C"):
        result = fits.get_fit_deviations(f"{symbol}9/h9", nominal)
        assert result is not None
        expected = fits._get_hole_fundamental_deviation(symbol, nominal)
        assert result.hole_lower_deviation_um == expected
        assert result.hole_upper_deviation_um - result.hole_lower_deviation_um == tolerance


def test_hole_js_fundamental_deviation_symmetry():
    nominal = 25
    deviation = fits._get_hole_fundamental_deviation("JS", nominal)
    assert deviation is not None
    tolerance = fits.get_tolerance_value(nominal, "IT6")
    assert tolerance is not None
    assert deviation == pytest.approx(-tolerance / 2)


def test_hole_deviation_override(monkeypatch, tmp_path: Path):
    override_path = tmp_path / "iso286_hole_deviations.json"
    override_path.write_text(
        json.dumps(
            {
                "deviations": {
                    "D": [[30, -999.0]],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOLE_DEVIATIONS_PATH", str(override_path))
    importlib.reload(fits)
    nominal = 25
    result = fits.get_fit_deviations("D10/h9", nominal)
    assert result is not None
    tolerance = get_tolerance_value(nominal, "IT10")
    assert tolerance is not None
    assert result.hole_lower_deviation_um == -999.0
    assert result.hole_upper_deviation_um == -999.0 + tolerance
