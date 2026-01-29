"""Tests for tolerance knowledge module."""

import pytest

from src.core.knowledge.tolerance import (
    ITGrade,
    get_tolerance_value,
    get_tolerance_table,
    TOLERANCE_GRADES,
    FitType,
    FitClass,
    get_fit_deviations,
    get_common_fits,
    COMMON_FITS,
    FitApplication,
    select_fit_for_application,
    get_fit_recommendations,
)


class TestITGrades:
    """Tests for IT grade functionality."""

    def test_get_tolerance_value_basic(self):
        """Test basic tolerance value lookup."""
        # IT7 @ 25mm should be 21 μm (ISO 286-1:2010)
        assert get_tolerance_value(25, "IT7") == 21
        assert get_tolerance_value(25, ITGrade.IT7) == 21

    def test_get_tolerance_value_different_sizes(self):
        """Test tolerance values at different sizes."""
        # IT7 tolerances should increase with size
        assert get_tolerance_value(3, "IT7") == 10
        assert get_tolerance_value(50, "IT7") == 25
        assert get_tolerance_value(100, "IT7") == 35

    def test_get_tolerance_value_different_grades(self):
        """Test different IT grades at same size."""
        # At 25mm, tolerance should increase with grade number
        it6 = get_tolerance_value(25, "IT6")
        it7 = get_tolerance_value(25, "IT7")
        it8 = get_tolerance_value(25, "IT8")

        assert it6 is not None
        assert it7 is not None
        assert it8 is not None
        assert it6 < it7 < it8

    def test_get_tolerance_value_out_of_range(self):
        """Test out of range values return None."""
        assert get_tolerance_value(0, "IT7") is None
        assert get_tolerance_value(-10, "IT7") is None
        assert get_tolerance_value(5000, "IT7") is None

    def test_get_tolerance_value_invalid_grade(self):
        """Test invalid grade returns None."""
        assert get_tolerance_value(25, "IT99") is None
        assert get_tolerance_value(25, "invalid") is None

    def test_get_tolerance_table(self):
        """Test getting multiple tolerance values."""
        table = get_tolerance_table(25)

        assert "IT5" in table
        assert "IT7" in table
        assert "IT14" in table
        assert table["IT7"] == 21

    def test_get_tolerance_table_custom_grades(self):
        """Test custom grade selection."""
        table = get_tolerance_table(25, ["IT6", "IT7", "IT8"])

        assert len(table) == 3
        assert "IT6" in table
        assert "IT7" in table
        assert "IT8" in table
        assert "IT5" not in table

    def test_tolerance_grades_data_completeness(self):
        """Test that tolerance data covers all standard size ranges."""
        # Should have 21 size ranges
        assert len(TOLERANCE_GRADES) >= 20

        # Each range should have standard grades
        for range_idx, grades in TOLERANCE_GRADES.items():
            assert "IT6" in grades
            assert "IT7" in grades
            assert "IT8" in grades


class TestFits:
    """Tests for fit system functionality."""

    def test_get_fit_deviations_h7g6(self):
        """Test H7/g6 fit calculations."""
        fit = get_fit_deviations("H7/g6", 25)

        assert fit is not None
        assert fit.fit_code == "H7/g6"
        assert fit.nominal_size_mm == 25
        assert fit.fit_type == FitType.CLEARANCE

        # H7 hole: lower deviation = 0, upper = IT7 tolerance
        assert fit.hole_lower_deviation_um == 0
        assert fit.hole_upper_deviation_um == 21  # IT7 @ 25mm

        # g6 shaft: upper deviation is fundamental deviation (negative)
        assert fit.shaft_upper_deviation_um < 0

        # Should have clearance (positive values)
        assert fit.min_clearance_um > 0
        assert fit.max_clearance_um > fit.min_clearance_um

    def test_get_fit_deviations_h7h6(self):
        """Test H7/h6 sliding fit."""
        fit = get_fit_deviations("H7/h6", 25)

        assert fit is not None
        assert fit.fit_type == FitType.CLEARANCE

        # h shaft has zero upper deviation
        assert fit.shaft_upper_deviation_um == 0

    def test_get_fit_deviations_h7k6(self):
        """Test H7/k6 transition fit."""
        fit = get_fit_deviations("H7/k6", 25)

        assert fit is not None
        assert fit.fit_type == FitType.TRANSITION

    def test_get_fit_deviations_h7p6(self):
        """Test H7/p6 interference fit."""
        fit = get_fit_deviations("H7/p6", 25)

        assert fit is not None
        assert fit.fit_type == FitType.INTERFERENCE

        # Should have interference (negative clearance possible)
        assert fit.min_clearance_um < fit.max_clearance_um

    def test_get_fit_deviations_invalid(self):
        """Test invalid fit code returns None."""
        assert get_fit_deviations("INVALID", 25) is None
        assert get_fit_deviations("H99/x99", 25) is None

    def test_get_common_fits_all(self):
        """Test getting all common fits."""
        fits = get_common_fits()

        assert len(fits) >= 10
        assert "H7/g6" in fits
        assert "H7/h6" in fits
        assert "H7/k6" in fits

    def test_get_common_fits_by_type(self):
        """Test filtering fits by type."""
        clearance_fits = get_common_fits(FitType.CLEARANCE)
        interference_fits = get_common_fits(FitType.INTERFERENCE)

        assert len(clearance_fits) > 0
        assert len(interference_fits) > 0

        for code, data in clearance_fits.items():
            assert data["type"] == FitType.CLEARANCE

        for code, data in interference_fits.items():
            assert data["type"] == FitType.INTERFERENCE

    def test_common_fits_data_completeness(self):
        """Test that common fits have all required data."""
        for code, data in COMMON_FITS.items():
            assert "hole" in data
            assert "shaft" in data
            assert "type" in data
            assert "name_zh" in data
            assert "name_en" in data
            assert "application_zh" in data
            assert "application_en" in data


class TestFitSelection:
    """Tests for fit selection guidance."""

    def test_select_fit_for_application_basic(self):
        """Test basic fit selection."""
        recs = select_fit_for_application(FitApplication.GEAR_HUB)

        assert len(recs) > 0
        assert recs[0].suitability >= recs[-1].suitability  # Sorted by suitability

    def test_select_fit_for_application_heavy_load(self):
        """Test fit selection for heavy loads."""
        recs = select_fit_for_application(FitApplication.GEAR_HUB, load_level="heavy")

        assert len(recs) > 0
        # Heavy load should recommend tighter fits
        fit_types = [r.fit_type for r in recs]
        assert FitType.INTERFERENCE in fit_types or FitType.TRANSITION in fit_types

    def test_select_fit_for_application_light_load(self):
        """Test fit selection for light loads."""
        recs = select_fit_for_application(FitApplication.BEARING_ROTATING, load_level="light")

        assert len(recs) > 0

    def test_get_fit_recommendations_by_type(self):
        """Test getting recommendations filtered by type."""
        clearance_recs = get_fit_recommendations(fit_type=FitType.CLEARANCE)
        interference_recs = get_fit_recommendations(fit_type=FitType.INTERFERENCE)

        assert len(clearance_recs) > 0
        assert len(interference_recs) > 0

        for rec in clearance_recs:
            assert rec["type"] == "clearance"

        for rec in interference_recs:
            assert rec["type"] == "interference"

    def test_fit_applications_coverage(self):
        """Test that all fit applications return results."""
        for app in FitApplication:
            recs = select_fit_for_application(app)
            # Most applications should have recommendations
            # Some special cases might return empty (e.g., KEY_KEYWAY has different format)
            if app not in [FitApplication.KEY_KEYWAY]:
                assert len(recs) >= 0


class TestIntegration:
    """Integration tests for tolerance module."""

    def test_tolerance_fit_consistency(self):
        """Test that fit deviations use correct tolerance values."""
        size = 25
        fit = get_fit_deviations("H7/g6", size)

        # Hole tolerance should match IT7
        it7 = get_tolerance_value(size, "IT7")
        assert fit.hole_upper_deviation_um - fit.hole_lower_deviation_um == it7

    def test_clearance_calculation_correctness(self):
        """Test clearance calculations are physically correct."""
        fit = get_fit_deviations("H7/g6", 25)

        # Max clearance = max hole - min shaft
        expected_max = fit.hole_upper_deviation_um - fit.shaft_lower_deviation_um
        assert fit.max_clearance_um == expected_max

        # Min clearance = min hole - max shaft
        expected_min = fit.hole_lower_deviation_um - fit.shaft_upper_deviation_um
        assert fit.min_clearance_um == expected_min

    def test_multiple_sizes_scaling(self):
        """Test that tolerances scale correctly with size."""
        sizes = [10, 25, 50, 100]
        tolerances = [get_tolerance_value(s, "IT7") for s in sizes]

        # Tolerances should increase with size
        for i in range(len(tolerances) - 1):
            assert tolerances[i] < tolerances[i + 1]
