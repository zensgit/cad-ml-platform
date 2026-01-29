"""Tests for surface treatment knowledge module."""

import pytest


class TestElectroplating:
    """Tests for electroplating module."""

    def test_get_zinc_plating_parameters(self):
        """Test zinc plating parameter retrieval."""
        from src.core.knowledge.surface_treatment import get_plating_parameters, PlatingType

        params = get_plating_parameters(PlatingType.ZINC_YELLOW)

        assert params is not None
        assert params.thickness_typical > 0
        assert params.current_density_typical > 0

    def test_get_nickel_plating(self):
        """Test nickel plating parameters."""
        from src.core.knowledge.surface_treatment import get_plating_parameters

        params = get_plating_parameters("nickel_bright")

        assert params is not None
        assert params.hardness_hv is not None

    def test_get_hard_chrome(self):
        """Test hard chrome plating."""
        from src.core.knowledge.surface_treatment import get_plating_parameters

        params = get_plating_parameters("chrome_hard")

        assert params is not None
        assert params.hardness_hv[0] >= 800  # Hard chrome is very hard

    def test_plating_thickness_recommendation(self):
        """Test plating thickness recommendations."""
        from src.core.knowledge.surface_treatment import get_plating_thickness

        result = get_plating_thickness("fastener", "outdoor")

        assert "thickness_um" in result
        assert result["thickness_um"] >= 8

    def test_plating_recommendation(self):
        """Test plating type recommendation."""
        from src.core.knowledge.surface_treatment import recommend_plating_for_application

        recs = recommend_plating_for_application("bracket", ["corrosion", "耐磨"])

        assert len(recs) > 0
        assert all("plating" in r for r in recs)


class TestAnodizing:
    """Tests for anodizing module."""

    def test_get_type_ii_parameters(self):
        """Test Type II anodizing parameters."""
        from src.core.knowledge.surface_treatment import get_anodizing_parameters, AnodizingType

        params = get_anodizing_parameters(AnodizingType.TYPE_II)

        assert params is not None
        assert params.thickness_typical > 0
        assert "硫酸" in params.acid_type

    def test_get_hard_anodize(self):
        """Test hard anodize (Type III) parameters."""
        from src.core.knowledge.surface_treatment import get_anodizing_parameters

        params = get_anodizing_parameters("type_iii")

        assert params is not None
        assert params.thickness_typical >= 25  # Hard anodize is thick
        assert params.temperature_typical < 10  # Low temperature process

    def test_anodize_colors(self):
        """Test available anodize colors."""
        from src.core.knowledge.surface_treatment import get_anodizing_colors, AnodizingType

        colors = get_anodizing_colors(AnodizingType.TYPE_II)

        assert len(colors) > 0
        assert any(c["color_id"] == "black" for c in colors)

    def test_anodize_recommendation(self):
        """Test anodize type recommendation."""
        from src.core.knowledge.surface_treatment import recommend_anodizing_for_application

        recs = recommend_anodizing_for_application("bracket", ["wear"])

        assert len(recs) > 0
        # Should recommend Type III for wear
        from src.core.knowledge.surface_treatment import AnodizingType
        assert recs[0]["type"] == AnodizingType.TYPE_III

    def test_dimension_change(self):
        """Test anodize dimension change calculation."""
        from src.core.knowledge.surface_treatment.anodizing import calculate_dimension_change, AnodizingType

        result = calculate_dimension_change(25, AnodizingType.TYPE_II)

        assert "diameter_increase_um" in result
        assert result["diameter_increase_um"] > 0


class TestCoating:
    """Tests for coating module."""

    def test_get_powder_coating_parameters(self):
        """Test powder coating parameter retrieval."""
        from src.core.knowledge.surface_treatment import get_coating_parameters, CoatingType

        params = get_coating_parameters(CoatingType.POWDER_POLYESTER)

        assert params is not None
        assert params.cure_temperature is not None
        assert params.dft_recommended > 0

    def test_get_liquid_coating(self):
        """Test liquid coating parameters."""
        from src.core.knowledge.surface_treatment import get_coating_parameters

        params = get_coating_parameters("epoxy")

        assert params is not None
        assert params.air_dry_time is not None

    def test_coating_for_environment(self):
        """Test coating recommendation for environment."""
        from src.core.knowledge.surface_treatment import get_coating_for_environment

        result = get_coating_for_environment("C4")

        assert "min_dft_um" in result
        assert result["min_dft_um"] >= 200

    def test_coating_life_calculation(self):
        """Test coating life estimation."""
        from src.core.knowledge.surface_treatment import calculate_coating_life

        result = calculate_coating_life("polyurethane", "C3", 100)

        assert "estimated_life_years" in result
        assert result["estimated_life_years"] > 0

    def test_coating_system_recommendation(self):
        """Test complete coating system recommendation."""
        from src.core.knowledge.surface_treatment.coating import recommend_coating_system

        recs = recommend_coating_system("steel", "outdoor")

        assert len(recs) > 0
        assert "system" in recs[0]
        assert len(recs[0]["system"]) > 1  # Multi-layer system


class TestIntegration:
    """Integration tests for surface treatment module."""

    def test_module_imports(self):
        """Test all module imports work correctly."""
        from src.core.knowledge.surface_treatment import (
            PlatingType,
            PlatingParameters,
            get_plating_parameters,
            AnodizingType,
            AnodizingParameters,
            get_anodizing_parameters,
            CoatingType,
            CoatingParameters,
            get_coating_parameters,
        )

        # All imports should work
        assert PlatingType.ZINC_YELLOW is not None
        assert AnodizingType.TYPE_III is not None
        assert CoatingType.POWDER_POLYESTER is not None

    def test_cross_module_consistency(self):
        """Test consistency across modules."""
        from src.core.knowledge.surface_treatment import (
            get_plating_parameters,
            get_anodizing_parameters,
            get_coating_parameters,
        )

        # All should return proper parameter objects
        plating = get_plating_parameters("zinc_nickel")
        anodizing = get_anodizing_parameters("type_ii")
        coating = get_coating_parameters("powder_polyester")

        # All should have thickness info
        assert plating.thickness_typical > 0
        assert anodizing.thickness_typical > 0
        assert coating.dft_recommended > 0
