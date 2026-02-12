"""Tests for welding knowledge module."""

import pytest


class TestWeldingParameters:
    """Tests for welding parameters."""

    def test_get_gmaw_parameters(self):
        """Test GMAW parameter retrieval."""
        from src.core.knowledge.welding import get_welding_parameters, WeldingProcess

        params = get_welding_parameters(WeldingProcess.GMAW, "carbon_steel", 6)

        assert params is not None
        assert params.process == WeldingProcess.GMAW
        assert params.current_recommended > 0
        assert params.voltage_recommended > 0

    def test_get_smaw_parameters(self):
        """Test SMAW parameter retrieval."""
        from src.core.knowledge.welding import get_welding_parameters

        params = get_welding_parameters("SMAW", "carbon_steel", 10)

        assert params is not None
        assert params.electrode_diameter > 0

    def test_get_gtaw_stainless(self):
        """Test GTAW parameters for stainless steel."""
        from src.core.knowledge.welding import get_welding_parameters

        params = get_welding_parameters("GTAW", "stainless_steel", 2)

        assert params is not None
        assert "DC" in params.current_type or "AC" in params.current_type

    def test_position_adjustment(self):
        """Test welding position affects parameters."""
        from src.core.knowledge.welding import (
            get_welding_parameters,
            WeldingPosition,
        )

        flat_params = get_welding_parameters(
            "GMAW", "carbon_steel", 6, WeldingPosition.FLAT
        )
        vertical_params = get_welding_parameters(
            "GMAW", "carbon_steel", 6, WeldingPosition.VERTICAL_UP
        )

        assert flat_params is not None
        assert vertical_params is not None
        # Vertical should have lower current
        assert vertical_params.current_recommended < flat_params.current_recommended

    def test_invalid_process(self):
        """Test invalid process returns None."""
        from src.core.knowledge.welding import get_welding_parameters

        params = get_welding_parameters("INVALID", "carbon_steel", 6)
        assert params is None


class TestFillerMaterial:
    """Tests for filler material recommendations."""

    def test_carbon_steel_filler(self):
        """Test carbon steel filler recommendations."""
        from src.core.knowledge.welding import get_filler_material

        fillers = get_filler_material("carbon_steel", "SMAW")

        assert len(fillers) > 0
        assert any("E70" in f for f in fillers)

    def test_stainless_filler(self):
        """Test stainless steel filler recommendations."""
        from src.core.knowledge.welding import get_filler_material

        fillers = get_filler_material("304_stainless", "GTAW")

        assert len(fillers) > 0
        assert any("308" in f for f in fillers)

    def test_aluminum_filler(self):
        """Test aluminum filler recommendations."""
        from src.core.knowledge.welding import get_filler_material

        fillers = get_filler_material("aluminum_6xxx", "GMAW")

        assert len(fillers) > 0


class TestHeatInput:
    """Tests for heat input calculation."""

    def test_calculate_heat_input(self):
        """Test heat input calculation."""
        from src.core.knowledge.welding import calculate_heat_input

        # V=28, I=230, S=350 mm/min
        heat_input = calculate_heat_input(28, 230, 350)

        assert heat_input > 0
        assert heat_input < 5  # Typical range

    def test_heat_input_zero_speed(self):
        """Test heat input with zero speed."""
        from src.core.knowledge.welding import calculate_heat_input

        heat_input = calculate_heat_input(28, 230, 0)
        assert heat_input == 0


class TestJointDesign:
    """Tests for joint design module."""

    def test_get_single_v_joint(self):
        """Test single V joint design."""
        from src.core.knowledge.welding import get_joint_design, GrooveType

        design = get_joint_design(GrooveType.SINGLE_V, 10)

        assert design is not None
        assert design.groove_angle == 60
        assert design.root_gap is not None

    def test_recommend_joint_thin_plate(self):
        """Test joint recommendation for thin plate."""
        from src.core.knowledge.welding import recommend_joint_for_thickness, GrooveType

        recommendations = recommend_joint_for_thickness(3)

        assert len(recommendations) > 0
        assert recommendations[0][0] == GrooveType.SQUARE

    def test_recommend_joint_thick_plate(self):
        """Test joint recommendation for thick plate."""
        from src.core.knowledge.welding import recommend_joint_for_thickness, GrooveType

        recommendations = recommend_joint_for_thickness(30, access_both_sides=True)

        assert len(recommendations) > 0
        # Should recommend double-V or U for thick plates
        groove_types = [r[0] for r in recommendations]
        assert GrooveType.DOUBLE_V in groove_types or GrooveType.DOUBLE_U in groove_types

    def test_minimum_fillet_size(self):
        """Test minimum fillet weld size."""
        from src.core.knowledge.welding import get_minimum_fillet_size

        size = get_minimum_fillet_size(10)
        assert size >= 3

        size_thick = get_minimum_fillet_size(25)
        assert size_thick > size


class TestWeldability:
    """Tests for weldability module."""

    def test_get_weldability(self):
        """Test weldability lookup."""
        from src.core.knowledge.welding import get_weldability, WeldabilityClass

        mat = get_weldability("Q235")

        assert mat is not None
        assert mat.weldability == WeldabilityClass.EXCELLENT

    def test_poor_weldability_material(self):
        """Test material with poor weldability."""
        from src.core.knowledge.welding import get_weldability, WeldabilityClass

        mat = get_weldability("40Cr")

        assert mat is not None
        assert mat.weldability == WeldabilityClass.POOR
        assert mat.preheat_required is True

    def test_preheat_temperature(self):
        """Test preheat temperature lookup."""
        from src.core.knowledge.welding import get_preheat_temperature

        temp = get_preheat_temperature("Q345", thickness=30)

        assert temp[0] >= 0
        assert temp[1] > temp[0]

    def test_preheat_by_carbon_equivalent(self):
        """Test preheat by CE value."""
        from src.core.knowledge.welding import get_preheat_temperature

        # High CE should require preheat
        temp = get_preheat_temperature(carbon_equivalent=0.55, thickness=40)

        assert temp[0] > 0

    def test_carbon_equivalent_calculation(self):
        """Test CE calculation."""
        from src.core.knowledge.welding import calculate_carbon_equivalent

        # Q235 typical composition
        ce = calculate_carbon_equivalent({
            "C": 0.18,
            "Mn": 0.5,
            "Si": 0.2,
        })

        assert 0.2 < ce < 0.4

    def test_material_compatibility(self):
        """Test material compatibility check."""
        from src.core.knowledge.welding import check_material_compatibility

        # Same material
        result = check_material_compatibility("304", "304")
        assert result["compatible"] is True

        # Dissimilar but compatible
        result = check_material_compatibility("304", "Q235")
        assert "notes" in result
