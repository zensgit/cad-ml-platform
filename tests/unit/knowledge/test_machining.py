"""Tests for machining knowledge module."""

import pytest
import math

from src.core.knowledge.machining import (
    # Cutting
    MachiningOperation,
    CuttingParameters,
    get_cutting_parameters,
    calculate_spindle_speed,
    calculate_feed_rate,
    calculate_metal_removal_rate,
    CUTTING_SPEED_TABLE,
    # Tooling
    ToolMaterial,
    ToolType,
    get_tool_recommendation,
    select_tool_for_material,
    TOOL_DATABASE,
    # Materials
    MachinabilityClass,
    get_machinability,
    get_material_cutting_data,
    MACHINABILITY_DATABASE,
)


class TestCuttingParameters:
    """Tests for cutting parameters functionality."""

    def test_get_cutting_parameters_basic(self):
        """Test basic parameter lookup."""
        params = get_cutting_parameters("turning_rough", "P", "coated_carbide")

        assert params is not None
        assert params.operation == MachiningOperation.TURNING_ROUGH
        assert params.cutting_speed_recommended > 0
        assert params.feed_recommended > 0
        assert params.depth_recommended > 0

    def test_get_cutting_parameters_different_materials(self):
        """Test parameters for different material groups."""
        # Steel should have higher speeds than superalloys
        steel_params = get_cutting_parameters("turning_rough", "P")
        super_params = get_cutting_parameters("turning_rough", "S")

        assert steel_params is not None
        assert super_params is not None
        assert steel_params.cutting_speed_recommended > super_params.cutting_speed_recommended

    def test_get_cutting_parameters_finishing_vs_roughing(self):
        """Test that finishing has different params than roughing."""
        rough = get_cutting_parameters("turning_rough", "P")
        finish = get_cutting_parameters("turning_finish", "P")

        assert rough is not None
        assert finish is not None
        # Finishing should have lower feed and depth
        assert finish.feed_recommended < rough.feed_recommended
        assert finish.depth_recommended < rough.depth_recommended

    def test_get_cutting_parameters_milling(self):
        """Test milling parameters include width of cut."""
        params = get_cutting_parameters("milling_face", "P")

        assert params is not None
        assert params.width_recommended is not None
        assert params.width_min is not None
        assert params.width_max is not None

    def test_get_cutting_parameters_invalid(self):
        """Test invalid inputs return None."""
        assert get_cutting_parameters("invalid_operation", "P") is None

    def test_calculate_spindle_speed(self):
        """Test spindle speed calculation."""
        # n = 1000 * Vc / (π * D)
        # For Vc=200 m/min, D=50mm: n = 1000*200/(π*50) ≈ 1273 rpm
        n = calculate_spindle_speed(200, 50)

        expected = round(1000 * 200 / (math.pi * 50))
        assert n == expected

    def test_calculate_spindle_speed_zero_diameter(self):
        """Test zero diameter returns 0."""
        assert calculate_spindle_speed(200, 0) == 0

    def test_calculate_feed_rate_turning(self):
        """Test feed rate calculation for turning."""
        # vf = n * f
        vf = calculate_feed_rate(1000, 0.2, 1)
        assert vf == 200.0

    def test_calculate_feed_rate_milling(self):
        """Test feed rate calculation for milling."""
        # vf = n * fz * z
        vf = calculate_feed_rate(1000, 0.1, 4)
        assert vf == 400.0

    def test_calculate_metal_removal_rate_turning(self):
        """Test MRR calculation for turning."""
        # Q = Vc * f * ap
        Q = calculate_metal_removal_rate(200, 0.3, 4)
        assert Q == 240.0

    def test_cutting_speed_table_completeness(self):
        """Test that speed table covers common combinations."""
        # Check all ISO groups have at least carbide entry
        for group in ["P", "M", "K", "N", "S", "H"]:
            assert (group, "carbide") in CUTTING_SPEED_TABLE or \
                   (group, "coated_carbide") in CUTTING_SPEED_TABLE


class TestTooling:
    """Tests for tooling functionality."""

    def test_get_tool_recommendation_basic(self):
        """Test basic tool recommendation."""
        rec = get_tool_recommendation("P", "roughing")

        assert rec is not None
        assert rec.tool_material is not None
        assert rec.tool_type is not None
        assert rec.geometry is not None
        assert rec.suitability > 0

    def test_get_tool_recommendation_different_operations(self):
        """Test recommendations for different operations."""
        rough = get_tool_recommendation("P", "roughing")
        finish = get_tool_recommendation("P", "finishing")

        assert rough is not None
        assert finish is not None
        # May have different tool materials
        assert rough.tool_material is not None
        assert finish.tool_material is not None

    def test_get_tool_recommendation_all_groups(self):
        """Test recommendations exist for all material groups."""
        for group in ["P", "M", "K", "N", "S", "H"]:
            rec = get_tool_recommendation(group, "roughing")
            assert rec is not None, f"No recommendation for {group}"

    def test_select_tool_for_material(self):
        """Test tool selection for a specific material."""
        tools = select_tool_for_material("low_carbon_steel")

        assert len(tools) > 0
        for tool in tools:
            assert "tool_id" in tool
            assert "suitability" in tool

    def test_select_tool_for_material_invalid(self):
        """Test invalid material returns empty list."""
        tools = select_tool_for_material("invalid_material")
        assert tools == []

    def test_tool_database_completeness(self):
        """Test tool database has required fields."""
        for tool_id, tool_data in TOOL_DATABASE.items():
            assert "type" in tool_data
            assert "suitable_materials" in tool_data
            assert "name_zh" in tool_data
            assert "name_en" in tool_data


class TestMaterials:
    """Tests for material machinability data."""

    def test_get_machinability_basic(self):
        """Test basic machinability lookup."""
        mat = get_machinability("low_carbon_steel")

        assert mat is not None
        assert mat.machinability_rating > 0
        assert mat.machinability_class is not None
        assert mat.cutting_speed_factor > 0

    def test_get_machinability_rating_order(self):
        """Test machinability ratings are logically ordered."""
        # Aluminum should be more machinable than titanium
        aluminum = get_machinability("aluminum_wrought")
        titanium = get_machinability("titanium_alloy")

        assert aluminum is not None
        assert titanium is not None
        assert aluminum.machinability_rating > titanium.machinability_rating

    def test_get_machinability_invalid(self):
        """Test invalid material returns None."""
        assert get_machinability("invalid_material") is None

    def test_get_material_cutting_data(self):
        """Test getting full cutting data."""
        data = get_material_cutting_data("austenitic_stainless")

        assert data is not None
        assert "machinability_rating" in data
        assert "cutting_speed_factor" in data
        assert "coolant" in data
        assert "characteristics" in data

    def test_machinability_classes_valid(self):
        """Test all materials have valid machinability classes."""
        for key, mat in MACHINABILITY_DATABASE.items():
            assert mat.machinability_class in MachinabilityClass
            # Check class matches rating
            rating = mat.machinability_rating
            if rating > 100:
                assert mat.machinability_class == MachinabilityClass.EXCELLENT
            elif rating >= 70:
                assert mat.machinability_class in [MachinabilityClass.EXCELLENT, MachinabilityClass.GOOD]

    def test_machinability_database_iso_groups(self):
        """Test materials are assigned to valid ISO groups."""
        valid_groups = {"P", "M", "K", "N", "S", "H"}
        for key, mat in MACHINABILITY_DATABASE.items():
            assert mat.material_group in valid_groups, f"{key} has invalid group {mat.material_group}"


class TestIntegration:
    """Integration tests across modules."""

    def test_material_to_cutting_parameters(self):
        """Test flow from material to cutting parameters."""
        mat = get_machinability("austenitic_stainless")
        assert mat is not None

        params = get_cutting_parameters("turning_rough", mat.material_group)
        assert params is not None

        # Adjust speed by material factor
        adjusted_speed = params.cutting_speed_recommended * mat.cutting_speed_factor
        assert adjusted_speed < params.cutting_speed_recommended  # Stainless needs lower speed

    def test_material_to_tool_selection(self):
        """Test flow from material to tool selection."""
        mat = get_machinability("titanium_alloy")
        assert mat is not None

        rec = get_tool_recommendation(mat.material_group, "roughing")
        assert rec is not None

    def test_full_parameter_calculation(self):
        """Test complete parameter calculation workflow."""
        # 1. Get material data
        mat = get_machinability("medium_carbon_steel")
        assert mat is not None

        # 2. Get base cutting parameters
        params = get_cutting_parameters("turning_rough", mat.material_group)
        assert params is not None

        # 3. Calculate spindle speed for D=40mm
        diameter = 40
        vc = params.cutting_speed_recommended * mat.cutting_speed_factor
        n = calculate_spindle_speed(vc, diameter)
        assert n > 0

        # 4. Calculate feed rate
        vf = calculate_feed_rate(n, params.feed_recommended)
        assert vf > 0

        # 5. Calculate MRR
        mrr = calculate_metal_removal_rate(vc, params.feed_recommended, params.depth_recommended)
        assert mrr > 0
