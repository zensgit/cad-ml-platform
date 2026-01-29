"""Tests for heat treatment knowledge module."""

import pytest


class TestHeatTreatmentParameters:
    """Tests for heat treatment parameters."""

    def test_get_quench_hardening_params(self):
        """Test quench hardening parameter retrieval."""
        from src.core.knowledge.heat_treatment import (
            get_heat_treatment_parameters,
            HeatTreatmentProcess,
        )

        params = get_heat_treatment_parameters("45", HeatTreatmentProcess.QUENCH_HARDENING)

        assert params is not None
        assert params.temperature_recommended > 800
        assert params.quench_media is not None

    def test_get_tempering_params(self):
        """Test tempering parameter retrieval."""
        from src.core.knowledge.heat_treatment import get_heat_treatment_parameters

        # Low temperature tempering
        params = get_heat_treatment_parameters("45", "tempering", variant="low")

        assert params is not None
        assert params.temperature_max < 300

    def test_get_normalizing_params(self):
        """Test normalizing parameter retrieval."""
        from src.core.knowledge.heat_treatment import get_heat_treatment_parameters

        params = get_heat_treatment_parameters("45", "normalizing")

        assert params is not None
        assert "正火" in params.notes_zh

    def test_aluminum_solution_treatment(self):
        """Test aluminum solution treatment."""
        from src.core.knowledge.heat_treatment import get_heat_treatment_parameters

        params = get_heat_treatment_parameters("6061", "solution_treatment")

        assert params is not None
        assert params.temperature_recommended < 600  # Aluminum temps are lower

    def test_invalid_material(self):
        """Test invalid material returns None."""
        from src.core.knowledge.heat_treatment import get_heat_treatment_parameters

        params = get_heat_treatment_parameters("INVALID", "quench_hardening")
        assert params is None


class TestQuenchMedia:
    """Tests for quench media data."""

    def test_get_water_quench(self):
        """Test water quench media data."""
        from src.core.knowledge.heat_treatment import get_quench_media

        media = get_quench_media("water")

        assert media is not None
        assert media["cooling_rate"][0] > 100  # Fast cooling

    def test_get_oil_quench(self):
        """Test oil quench media data."""
        from src.core.knowledge.heat_treatment import get_quench_media

        media = get_quench_media("oil")

        assert media is not None
        # Oil is slower than water
        water_media = get_quench_media("water")
        assert media["cooling_rate"][0] < water_media["cooling_rate"][0]


class TestHoldingTime:
    """Tests for holding time calculation."""

    def test_calculate_holding_time(self):
        """Test holding time calculation."""
        from src.core.knowledge.heat_treatment import calculate_holding_time

        time = calculate_holding_time(25, time_factor=1.5)

        assert time[0] > 0
        assert time[1] > time[0]

    def test_geometry_affects_time(self):
        """Test geometry factor affects holding time."""
        from src.core.knowledge.heat_treatment import calculate_holding_time

        solid_time = calculate_holding_time(25, geometry="solid")
        hollow_time = calculate_holding_time(25, geometry="hollow")

        # Hollow should need less time
        assert hollow_time[0] < solid_time[0]


class TestHardenability:
    """Tests for hardenability module."""

    def test_get_hardenability(self):
        """Test hardenability data lookup."""
        from src.core.knowledge.heat_treatment import get_hardenability, HardenabilityClass

        data = get_hardenability("45")

        assert data is not None
        assert data.hardenability_class == HardenabilityClass.LOW
        assert data.critical_diameter_water > 0

    def test_high_hardenability_steel(self):
        """Test high hardenability steel."""
        from src.core.knowledge.heat_treatment import get_hardenability, HardenabilityClass

        data = get_hardenability("Cr12MoV")

        assert data is not None
        assert data.hardenability_class == HardenabilityClass.VERY_HIGH
        assert data.critical_diameter_air is not None

    def test_tempering_temperature(self):
        """Test tempering temperature lookup."""
        from src.core.knowledge.heat_treatment import get_tempering_temperature

        temp = get_tempering_temperature("45", target_hardness=40)

        assert temp is not None
        assert temp[0] > 300  # Need medium temp for HRC 40

    def test_hardness_after_tempering(self):
        """Test hardness prediction after tempering."""
        from src.core.knowledge.heat_treatment import calculate_hardness_after_tempering

        hrc = calculate_hardness_after_tempering("45", 500)

        assert hrc is not None
        assert 30 < hrc < 40  # Typical range at 500°C


class TestAnnealing:
    """Tests for annealing module."""

    def test_get_full_annealing(self):
        """Test full annealing parameters."""
        from src.core.knowledge.heat_treatment import get_annealing_parameters, AnnealingType

        params = get_annealing_parameters("45", AnnealingType.FULL)

        assert params is not None
        assert params.cooling_method == "furnace"
        assert params.hardness_after is not None

    def test_get_spheroidizing(self):
        """Test spheroidizing annealing."""
        from src.core.knowledge.heat_treatment import get_annealing_parameters

        params = get_annealing_parameters("45", "spheroidizing")

        assert params is not None
        assert "球化" in params.notes_zh

    def test_stress_relief_parameters(self):
        """Test stress relief parameters."""
        from src.core.knowledge.heat_treatment import get_stress_relief_parameters

        params = get_stress_relief_parameters("carbon_steel", 50)

        assert "temperature_range" in params
        assert params["holding_time_minutes"] > 0

    def test_annealing_cycle_time(self):
        """Test annealing cycle time calculation."""
        from src.core.knowledge.heat_treatment import (
            calculate_annealing_cycle_time,
            AnnealingType,
        )

        result = calculate_annealing_cycle_time(
            temperature=830,
            section_thickness=50,
            annealing_type=AnnealingType.FULL,
        )

        assert result["total_time_hours"] > 0
        assert result["heating_time_hours"] > 0
        assert result["cooling_time_hours"] > 0

    def test_recommend_annealing(self):
        """Test annealing recommendation by purpose."""
        from src.core.knowledge.heat_treatment import recommend_annealing_for_purpose

        result = recommend_annealing_for_purpose("45", "machinability")

        assert result is not None
        assert result["recommended_type"] in ["spheroidizing", "full"]


class TestProcessRecommendation:
    """Tests for process recommendation."""

    def test_recommend_for_target_hardness(self):
        """Test process recommendation for target hardness."""
        from src.core.knowledge.heat_treatment import recommend_process_for_hardness

        result = recommend_process_for_hardness("45", target_hardness=35)

        assert result is not None
        assert len(result["recommendations"]) > 0

    def test_quench_recommendation(self):
        """Test quench media recommendation."""
        from src.core.knowledge.heat_treatment import recommend_quench_for_section

        result = recommend_quench_for_section("45", section_size=20)

        assert "recommendations" in result
        assert result["best_recommendation"] is not None
