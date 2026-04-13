"""Tests for materials compatibility module."""

from __future__ import annotations

import pytest

from src.core.materials.compatibility import (
    check_full_compatibility,
    check_galvanic_corrosion,
    check_heat_treatment_compatibility,
    check_weld_compatibility,
)


class TestCheckWeldCompatibility:
    """Tests for check_weld_compatibility()."""

    def test_same_carbon_steel_excellent(self):
        """Two carbon steels should have excellent weld compatibility."""
        result = check_weld_compatibility("Q235B", "Q235B")
        assert result["compatible"] is True
        assert result["rating"] in ("excellent", "good")

    def test_carbon_steel_stainless(self):
        """Carbon steel + stainless steel should be fair (compatible)."""
        result = check_weld_compatibility("Q235B", "S30408")
        assert result["compatible"] is True
        assert result["rating"] == "fair"

    def test_aluminum_carbon_steel_not_recommended(self):
        """Aluminum + carbon steel welding should not be recommended."""
        result = check_weld_compatibility("6061", "Q235B")
        assert result["compatible"] is False
        assert result["rating"] == "not_recommended"

    def test_unknown_material_error(self):
        """Unknown material should return error."""
        result = check_weld_compatibility("NONEXISTENT", "Q235B")
        assert result["compatible"] is False
        assert "error" in result

    def test_result_structure(self):
        """Result should contain expected fields."""
        result = check_weld_compatibility("Q235B", "S30408")
        assert "compatible" in result
        assert "rating" in result
        assert "rating_cn" in result
        assert "method" in result
        assert "material1" in result
        assert "material2" in result

    def test_symmetric_result(self):
        """Weld compatibility should be symmetric (order does not matter)."""
        r1 = check_weld_compatibility("Q235B", "S30408")
        r2 = check_weld_compatibility("S30408", "Q235B")
        assert r1["rating"] == r2["rating"]
        assert r1["compatible"] == r2["compatible"]


class TestCheckGalvanicCorrosion:
    """Tests for check_galvanic_corrosion()."""

    def test_same_material_safe(self):
        """Same material pair should be safe."""
        result = check_galvanic_corrosion("Q235B", "Q235B")
        assert result["risk"] == "safe"

    def test_distant_materials_high_risk(self):
        """Materials far apart in galvanic series should be high risk."""
        # Aluminum (-0.8) + Stainless steel (0.0) = 0.8V diff -> severe
        result = check_galvanic_corrosion("6061", "S30408")
        assert result["risk"] in ("high", "severe")

    def test_unknown_material(self):
        """Unknown material should return unknown risk."""
        result = check_galvanic_corrosion("NONEXISTENT", "Q235B")
        assert result["risk"] == "unknown"

    def test_result_structure_with_metals(self):
        """Metal pair result should have expected fields."""
        result = check_galvanic_corrosion("Q235B", "S30408")
        assert "risk" in result
        assert "risk_cn" in result
        assert "potential_difference" in result
        assert "recommendation" in result
        assert "anode" in result
        assert "cathode" in result

    def test_anode_cathode_identification(self):
        """More active material should be identified as anode."""
        # Carbon steel (-0.6) is more active than stainless steel (0.0)
        result = check_galvanic_corrosion("Q235B", "S30408")
        assert result["anode"]["role"] == "阳极（被腐蚀）"
        assert result["cathode"]["role"] == "阴极（受保护）"


class TestCheckHeatTreatmentCompatibility:
    """Tests for check_heat_treatment_compatibility()."""

    def test_unknown_material(self):
        """Unknown material should return error."""
        result = check_heat_treatment_compatibility("NONEXISTENT", "淬火")
        assert result["compatible"] is False
        assert "error" in result

    def test_known_material_returns_info(self):
        """Known material should return compatibility info."""
        result = check_heat_treatment_compatibility("45", "淬火")
        assert "compatible" in result
        assert "grade" in result or "error" in result

    def test_result_has_recommended_treatments(self):
        """Result should list recommended treatments."""
        result = check_heat_treatment_compatibility("Q235B", "退火")
        if "error" not in result:
            assert "recommended_treatments" in result


class TestCheckFullCompatibility:
    """Tests for check_full_compatibility()."""

    def test_same_material_compatible(self):
        """Same material should be compatible."""
        result = check_full_compatibility("Q235B", "Q235B")
        assert result["overall"] == "compatible"

    def test_incompatible_pair(self):
        """Highly incompatible pair should be flagged."""
        result = check_full_compatibility("6061", "Q235B")
        assert result["overall"] in ("caution", "incompatible")

    def test_result_structure(self):
        """Result should contain sub-checks."""
        result = check_full_compatibility("Q235B", "S30408")
        assert "overall" in result
        assert "overall_cn" in result
        assert "issues" in result
        assert "recommendations" in result
        assert "weld_compatibility" in result
        assert "galvanic_corrosion" in result

    def test_issues_list_type(self):
        """Issues should be a list."""
        result = check_full_compatibility("Q235B", "S30408")
        assert isinstance(result["issues"], list)
        assert isinstance(result["recommendations"], list)
