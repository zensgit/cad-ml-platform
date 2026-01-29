"""Tests for design standards knowledge module."""

import pytest

from src.core.knowledge.design_standards import (
    SurfaceFinishGrade,
    get_ra_value,
    get_surface_finish_for_application,
    SURFACE_FINISH_TABLE,
    GeneralToleranceClass,
    get_linear_tolerance,
    get_angular_tolerance,
    get_general_tolerance_table,
    LINEAR_TOLERANCE_TABLE,
    get_preferred_diameter,
    get_standard_chamfer,
    get_standard_fillet,
    PREFERRED_DIAMETERS,
    STANDARD_CHAMFERS,
    STANDARD_FILLETS,
)


class TestSurfaceFinish:
    """Tests for surface finish standards."""

    def test_get_ra_value(self):
        """Test Ra value lookup."""
        assert get_ra_value(SurfaceFinishGrade.N7) == 1.6
        assert get_ra_value(SurfaceFinishGrade.N6) == 0.8
        assert get_ra_value(SurfaceFinishGrade.N8) == 3.2

    def test_get_ra_value_extreme_grades(self):
        """Test extreme grade values."""
        assert get_ra_value(SurfaceFinishGrade.N1) == 0.025
        assert get_ra_value(SurfaceFinishGrade.N12) == 50

    def test_get_surface_finish_for_application(self):
        """Test application-based surface finish lookup."""
        result = get_surface_finish_for_application("bearing_journal")

        assert result is not None
        assert "grade" in result
        assert "ra_range" in result
        assert result["ra_range"][0] < result["ra_range"][1]

    def test_get_surface_finish_for_invalid_application(self):
        """Test invalid application returns None."""
        assert get_surface_finish_for_application("invalid_app") is None

    def test_surface_finish_table_completeness(self):
        """Test all grades are in table."""
        for grade in SurfaceFinishGrade:
            assert grade in SURFACE_FINISH_TABLE

    def test_ra_values_decreasing(self):
        """Test Ra values decrease with grade number."""
        grades = list(SurfaceFinishGrade)
        for i in range(len(grades) - 1):
            ra_current = get_ra_value(grades[i])
            ra_next = get_ra_value(grades[i + 1])
            assert ra_current < ra_next


class TestGeneralTolerances:
    """Tests for general tolerance standards."""

    def test_get_linear_tolerance_basic(self):
        """Test basic linear tolerance lookup."""
        tol = get_linear_tolerance(50, GeneralToleranceClass.M)
        assert tol == 0.3

    def test_get_linear_tolerance_different_classes(self):
        """Test different tolerance classes."""
        tol_f = get_linear_tolerance(50, GeneralToleranceClass.F)
        tol_m = get_linear_tolerance(50, GeneralToleranceClass.M)
        tol_c = get_linear_tolerance(50, GeneralToleranceClass.C)

        # Fine < Medium < Coarse
        assert tol_f < tol_m < tol_c

    def test_get_linear_tolerance_size_ranges(self):
        """Test different size ranges."""
        # Larger sizes should have larger tolerances
        tol_small = get_linear_tolerance(10, GeneralToleranceClass.M)
        tol_large = get_linear_tolerance(200, GeneralToleranceClass.M)

        assert tol_small < tol_large

    def test_get_angular_tolerance(self):
        """Test angular tolerance lookup."""
        tol = get_angular_tolerance(80, GeneralToleranceClass.M)

        assert tol is not None
        assert "°" in tol

    def test_get_general_tolerance_table(self):
        """Test getting full tolerance table."""
        table = get_general_tolerance_table(GeneralToleranceClass.M)

        assert len(table) > 0
        for row in table:
            assert "range_min" in row
            assert "range_max" in row
            assert "tolerance_mm" in row


class TestDesignFeatures:
    """Tests for design features standards."""

    def test_get_preferred_diameter_nearest(self):
        """Test nearest preferred diameter."""
        result = get_preferred_diameter(23)
        # 22 or 24 should be the nearest
        assert result in [22, 24, 25]

    def test_get_preferred_diameter_up(self):
        """Test rounding up to preferred diameter."""
        result = get_preferred_diameter(23, "up")
        assert result >= 23
        assert result in PREFERRED_DIAMETERS

    def test_get_preferred_diameter_down(self):
        """Test rounding down to preferred diameter."""
        result = get_preferred_diameter(23, "down")
        assert result <= 23
        assert result in PREFERRED_DIAMETERS

    def test_get_standard_chamfer(self):
        """Test standard chamfer lookup."""
        result = get_standard_chamfer(1.8)

        assert result is not None
        assert "designation" in result
        assert "size" in result
        assert result["designation"] in STANDARD_CHAMFERS

    def test_get_standard_fillet(self):
        """Test standard fillet lookup."""
        result = get_standard_fillet(2.3)

        assert result is not None
        assert "designation" in result
        assert "size" in result
        assert result["designation"] in STANDARD_FILLETS

    def test_preferred_diameters_sorted(self):
        """Test preferred diameters are sorted."""
        for i in range(len(PREFERRED_DIAMETERS) - 1):
            assert PREFERRED_DIAMETERS[i] < PREFERRED_DIAMETERS[i + 1]

    def test_standard_chamfers_have_ranges(self):
        """Test all chamfers have valid ranges."""
        for name, data in STANDARD_CHAMFERS.items():
            assert "size" in data
            assert "range" in data
            assert data["range"][0] < data["size"] < data["range"][1]

    def test_standard_fillets_have_ranges(self):
        """Test all fillets have valid ranges."""
        for name, data in STANDARD_FILLETS.items():
            assert "size" in data
            assert "range" in data
            # Range should bracket the nominal size
            assert data["range"][0] <= data["size"] <= data["range"][1]


class TestIntegration:
    """Integration tests for design standards."""

    def test_surface_finish_process_consistency(self):
        """Test surface finish processes are consistent with Ra values."""
        # Finer finishes should require more advanced processes
        n6_data = SURFACE_FINISH_TABLE[SurfaceFinishGrade.N6]
        n10_data = SURFACE_FINISH_TABLE[SurfaceFinishGrade.N10]

        # N6 (fine grinding) Ra should be less than N10 (rough machining)
        assert n6_data[0] < n10_data[0]

    def test_tolerance_class_ordering(self):
        """Test tolerance classes are properly ordered."""
        for (min_s, max_s), tolerances in LINEAR_TOLERANCE_TABLE.items():
            f_tol = tolerances.get("f")
            m_tol = tolerances.get("m")
            c_tol = tolerances.get("c")
            v_tol = tolerances.get("v")

            # Where all exist, should be f < m < c < v
            if f_tol and m_tol:
                assert f_tol <= m_tol
            if m_tol and c_tol:
                assert m_tol <= c_tol
            if c_tol and v_tol:
                assert c_tol <= v_tol
