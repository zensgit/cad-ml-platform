"""Tests for GD&T (Geometric Dimensioning and Tolerancing) knowledge module."""

import pytest

from src.core.knowledge.gdt import (
    GDTCharacteristic,
    GDTCategory,
    ToleranceModifier,
    get_gdt_symbol,
    get_all_symbols,
    get_tolerance_zone,
    get_recommended_tolerance,
    calculate_bonus_tolerance,
    DatumFeatureType,
    create_datum_reference_frame,
    get_gdt_for_feature,
    get_inspection_method,
    interpret_feature_control_frame,
)
from src.core.knowledge.gdt.tolerances import ToleranceGrade, get_tolerance_relationship
from src.core.knowledge.gdt.application import FeatureType


class TestGDTSymbols:
    """Tests for GD&T symbols module."""

    def test_get_flatness_symbol(self):
        """Test getting flatness symbol info."""
        info = get_gdt_symbol(GDTCharacteristic.FLATNESS)

        assert info is not None
        assert info.name_zh == "平面度"
        assert info.category == GDTCategory.FORM
        assert info.requires_datum is False

    def test_get_position_symbol(self):
        """Test getting position symbol info."""
        info = get_gdt_symbol(GDTCharacteristic.POSITION)

        assert info is not None
        assert info.name_zh == "位置度"
        assert info.category == GDTCategory.LOCATION
        assert info.requires_datum is True

    def test_get_runout_symbol(self):
        """Test getting circular runout symbol info."""
        info = get_gdt_symbol(GDTCharacteristic.CIRCULAR_RUNOUT)

        assert info is not None
        assert info.name_zh == "圆跳动"
        assert info.category == GDTCategory.RUNOUT
        assert info.requires_datum is True

    def test_get_all_form_symbols(self):
        """Test getting all form tolerance symbols."""
        symbols = get_all_symbols(GDTCategory.FORM)

        # Form tolerances: straightness, flatness, circularity, cylindricity,
        # plus profile (line and surface) which can be form
        assert len(symbols) >= 4
        assert all(s.category == GDTCategory.FORM for s in symbols)

    def test_form_tolerances_no_datum(self):
        """Test that form tolerances don't require datum."""
        form_chars = [
            GDTCharacteristic.STRAIGHTNESS,
            GDTCharacteristic.FLATNESS,
            GDTCharacteristic.CIRCULARITY,
            GDTCharacteristic.CYLINDRICITY,
        ]

        for char in form_chars:
            info = get_gdt_symbol(char)
            assert info.requires_datum is False, f"{char} should not require datum"


class TestToleranceValues:
    """Tests for tolerance values and zones."""

    def test_get_tolerance_zone_flatness(self):
        """Test tolerance zone for flatness."""
        zone = get_tolerance_zone(GDTCharacteristic.FLATNESS, 0.05)

        assert zone is not None
        assert zone.value == 0.05
        assert "平面" in zone.description_zh

    def test_get_tolerance_zone_position(self):
        """Test tolerance zone for position (cylindrical)."""
        zone = get_tolerance_zone(GDTCharacteristic.POSITION, 0.2)

        assert zone is not None
        assert zone.value == 0.2
        assert zone.diameter_symbol is True
        assert "圆柱" in zone.description_zh

    def test_recommended_tolerance_flatness(self):
        """Test recommended flatness tolerance."""
        tol = get_recommended_tolerance(
            GDTCharacteristic.FLATNESS,
            50,  # 50mm nominal size
            ToleranceGrade.K,
        )

        assert tol is not None
        assert 0.1 <= tol <= 0.3  # Medium grade for 30-100mm

    def test_recommended_tolerance_circularity(self):
        """Test recommended circularity tolerance."""
        tol = get_recommended_tolerance(
            GDTCharacteristic.CIRCULARITY,
            25,
            ToleranceGrade.H,  # Fine grade
        )

        assert tol is not None
        assert tol < 0.05  # Fine grade should be tight

    def test_bonus_tolerance_mmc(self):
        """Test bonus tolerance calculation with MMC."""
        # Hole: Ø10 +0.1/0 with position Ø0.2(M)
        # Actual size: 10.08
        total = calculate_bonus_tolerance(
            geometric_tolerance=0.2,
            actual_size=10.08,
            mmc_size=10.0,
            modifier="M",
        )

        # Bonus = 10.08 - 10.0 = 0.08
        # Total = 0.2 + 0.08 = 0.28
        assert total == pytest.approx(0.28, rel=0.01)

    def test_tolerance_hierarchy_valid(self):
        """Test valid tolerance hierarchy."""
        result = get_tolerance_relationship(
            form_tolerance=0.02,
            orientation_tolerance=0.05,
            location_tolerance=0.1,
        )

        assert result["is_valid"] is True

    def test_tolerance_hierarchy_invalid(self):
        """Test invalid tolerance hierarchy."""
        result = get_tolerance_relationship(
            form_tolerance=0.1,  # Too large
            orientation_tolerance=0.05,
            location_tolerance=0.2,
        )

        assert result["is_valid"] is False
        assert len(result["issues"]) > 0


class TestDatums:
    """Tests for datum reference frames."""

    def test_create_simple_drf(self):
        """Test creating a simple datum reference frame."""
        drf = create_datum_reference_frame(
            primary=("A", DatumFeatureType.PLANE),
        )

        assert drf.primary is not None
        assert drf.primary.label == "A"
        assert drf.total_dof_constrained == 3  # Plane constrains 3 DOF

    def test_create_full_drf(self):
        """Test creating a fully constrained DRF."""
        drf = create_datum_reference_frame(
            primary=("A", DatumFeatureType.PLANE),
            secondary=("B", DatumFeatureType.AXIS),
            tertiary=("C", DatumFeatureType.PLANE),
        )

        assert drf.primary is not None
        assert drf.secondary is not None
        assert drf.tertiary is not None
        assert drf.total_dof_constrained >= 6
        assert drf.is_fully_constrained is True

    def test_axis_primary_datum(self):
        """Test axis as primary datum."""
        drf = create_datum_reference_frame(
            primary=("A", DatumFeatureType.AXIS),
        )

        assert drf.primary.feature_type == DatumFeatureType.AXIS
        assert drf.total_dof_constrained == 4  # Axis constrains 4 DOF


class TestApplications:
    """Tests for GD&T applications."""

    def test_gdt_for_hole(self):
        """Test GD&T recommendations for hole."""
        app = get_gdt_for_feature(FeatureType.HOLE)

        assert GDTCharacteristic.POSITION in app.recommended_characteristics
        assert len(app.inspection_methods) > 0

    def test_gdt_for_shaft(self):
        """Test GD&T recommendations for shaft."""
        app = get_gdt_for_feature(FeatureType.SHAFT)

        assert GDTCharacteristic.CYLINDRICITY in app.recommended_characteristics
        assert GDTCharacteristic.CIRCULAR_RUNOUT in app.recommended_characteristics

    def test_inspection_method_tight_tolerance(self):
        """Test inspection method for tight tolerance."""
        methods = get_inspection_method(
            GDTCharacteristic.CIRCULARITY,
            0.01,  # Tight tolerance
        )

        assert len(methods) > 0
        # Should recommend precise equipment
        cmm_method = next((m for m in methods if "三坐标" in m["method"]), None)
        if cmm_method:
            assert cmm_method["suitability"] == "推荐"

    def test_interpret_feature_control_frame(self):
        """Test parsing feature control frame."""
        fcf = interpret_feature_control_frame("位置度 0.2 M A B C")

        assert fcf is not None
        assert fcf.characteristic == GDTCharacteristic.POSITION
        assert fcf.tolerance_value == 0.2
        assert fcf.tolerance_modifier == ToleranceModifier.MMC
        assert fcf.primary_datum == "A"
        assert fcf.secondary_datum == "B"
        assert fcf.tertiary_datum == "C"

    def test_interpret_flatness_frame(self):
        """Test parsing flatness control frame (no datum)."""
        fcf = interpret_feature_control_frame("平面度 0.05")

        assert fcf is not None
        assert fcf.characteristic == GDTCharacteristic.FLATNESS
        assert fcf.tolerance_value == 0.05
        assert fcf.primary_datum is None


class TestIntegration:
    """Integration tests for GD&T module."""

    def test_module_imports(self):
        """Test all module imports work correctly."""
        from src.core.knowledge.gdt import (
            GDTCharacteristic,
            GDTCategory,
            ToleranceModifier,
            DatumModifier,
            get_gdt_symbol,
            get_all_symbols,
            GeometricTolerance,
            ToleranceZone,
            get_tolerance_zone,
            get_recommended_tolerance,
            DatumFeature,
            DatumReferenceFrame,
            create_datum_reference_frame,
            GDTApplication,
            get_gdt_for_feature,
            interpret_feature_control_frame,
        )

        # All imports should work
        assert GDTCharacteristic.FLATNESS is not None
        assert GDTCategory.FORM is not None
        assert ToleranceModifier.MMC is not None

    def test_end_to_end_hole_pattern(self):
        """Test end-to-end GD&T for hole pattern."""
        # Get recommendations
        app = get_gdt_for_feature(FeatureType.PATTERN)
        assert GDTCharacteristic.POSITION in app.recommended_characteristics

        # Get recommended tolerance
        tol = get_recommended_tolerance(
            GDTCharacteristic.PERPENDICULARITY,
            30,
            ToleranceGrade.K,
        )
        assert tol is not None

        # Create datum frame
        drf = create_datum_reference_frame(
            primary=("A", DatumFeatureType.PLANE),
            secondary=("B", DatumFeatureType.AXIS),
        )
        assert drf.total_dof_constrained >= 5

        # Parse control frame
        fcf = interpret_feature_control_frame("位置度 0.3 M A B")
        assert fcf.tolerance_value == 0.3

    def test_all_characteristics_have_zone(self):
        """Test all characteristics have tolerance zone defined."""
        for char in GDTCharacteristic:
            zone = get_tolerance_zone(char, 0.1)
            assert zone is not None, f"No zone for {char}"
            assert zone.value == 0.1
