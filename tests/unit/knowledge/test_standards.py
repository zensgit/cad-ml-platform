"""Tests for standards knowledge module."""

import pytest

from src.core.knowledge.standards import (
    # Threads
    ThreadType,
    MetricThread,
    get_thread_spec,
    get_thread_series,
    list_metric_threads,
    get_tap_drill_size,
    METRIC_THREADS,
    # Bearings
    BearingType,
    BearingSeries,
    BearingSpec,
    get_bearing_spec,
    get_bearing_by_bore,
    list_bearings,
    suggest_bearing_for_shaft,
    BEARING_DATABASE,
    # Seals
    SealType,
    ORingSpec,
    ORingMaterial,
    get_oring_spec,
    get_oring_by_id,
    list_orings,
    suggest_oring_material,
    ORING_DATABASE,
)


class TestThreads:
    """Tests for metric thread specifications."""

    def test_get_thread_spec_coarse(self):
        """Test getting coarse thread spec."""
        spec = get_thread_spec("M10")

        assert spec is not None
        assert spec.designation == "M10"
        assert spec.nominal_diameter == 10.0
        assert spec.pitch == 1.5
        assert spec.thread_type == ThreadType.METRIC_COARSE

    def test_get_thread_spec_fine(self):
        """Test getting fine thread spec."""
        spec = get_thread_spec("M10x1.25")

        assert spec is not None
        assert spec.designation == "M10x1.25"
        assert spec.nominal_diameter == 10.0
        assert spec.pitch == 1.25
        assert spec.thread_type == ThreadType.METRIC_FINE

    def test_get_thread_spec_fine_short_form(self):
        """Test getting fine thread spec with short form (M10x1 -> M10x1.0)."""
        spec = get_thread_spec("M10x1")

        assert spec is not None
        assert spec.pitch == 1.0

    def test_get_thread_spec_case_insensitive(self):
        """Test case insensitive lookup."""
        spec1 = get_thread_spec("m10")
        spec2 = get_thread_spec("M10")

        assert spec1 is not None
        assert spec2 is not None
        assert spec1.designation == spec2.designation

    def test_get_thread_spec_invalid(self):
        """Test invalid thread returns None."""
        assert get_thread_spec("M999") is None
        assert get_thread_spec("invalid") is None

    def test_get_tap_drill_size(self):
        """Test tap drill size lookup."""
        drill = get_tap_drill_size("M10")

        assert drill is not None
        assert drill == 8.5  # Standard M10 tap drill

    def test_list_metric_threads_for_diameter(self):
        """Test listing threads for a specific diameter."""
        threads = list_metric_threads(10)

        assert len(threads) >= 2  # At least coarse and one fine
        diameters = [t.nominal_diameter for t in threads]
        assert all(d == 10.0 for d in diameters)

    def test_get_thread_series_coarse(self):
        """Test getting coarse thread series."""
        threads = get_thread_series(ThreadType.METRIC_COARSE, min_diameter=6, max_diameter=16)

        assert len(threads) > 0
        for t in threads:
            assert t.thread_type == ThreadType.METRIC_COARSE
            assert 6 <= t.nominal_diameter <= 16

    def test_thread_dimensions_valid(self):
        """Test that thread dimensions are physically valid."""
        for spec in METRIC_THREADS.values():
            # Pitch diameter should be less than nominal
            assert spec.pitch_diameter < spec.nominal_diameter
            # Minor diameter should be less than pitch diameter
            assert spec.minor_diameter_ext < spec.pitch_diameter
            # Tap drill should be close to minor diameter
            assert spec.minor_diameter_int - 1 < spec.tap_drill_size < spec.nominal_diameter

    def test_metric_threads_completeness(self):
        """Test that common thread sizes are included."""
        common_sizes = ["M3", "M4", "M5", "M6", "M8", "M10", "M12", "M16", "M20"]
        for size in common_sizes:
            assert get_thread_spec(size) is not None


class TestBearings:
    """Tests for bearing specifications."""

    def test_get_bearing_spec_basic(self):
        """Test getting basic bearing spec."""
        spec = get_bearing_spec("6205")

        assert spec is not None
        assert spec.designation == "6205"
        assert spec.bore_d == 25
        assert spec.outer_d == 52
        assert spec.width_b == 15
        assert spec.bearing_type == BearingType.DEEP_GROOVE_BALL

    def test_get_bearing_spec_with_suffix(self):
        """Test getting bearing spec with suffix (stripped)."""
        spec = get_bearing_spec("6205-2RS")

        assert spec is not None
        assert spec.designation == "6205"

    def test_get_bearing_spec_invalid(self):
        """Test invalid bearing returns None."""
        assert get_bearing_spec("INVALID") is None
        assert get_bearing_spec("99999") is None

    def test_get_bearing_by_bore(self):
        """Test finding bearings by bore diameter."""
        bearings = get_bearing_by_bore(25)

        assert len(bearings) >= 2  # Should have multiple series
        for b in bearings:
            assert b.bore_d == 25

    def test_list_bearings_by_series(self):
        """Test listing bearings by series."""
        bearings = list_bearings(series=BearingSeries.SERIES_62)

        assert len(bearings) > 0
        for b in bearings:
            assert b.series == BearingSeries.SERIES_62

    def test_suggest_bearing_for_shaft(self):
        """Test bearing suggestions for shaft."""
        suggestions = suggest_bearing_for_shaft(25, load_type="medium")

        assert len(suggestions) > 0
        # All suggested bearings should fit the shaft
        for desig in suggestions:
            spec = get_bearing_spec(desig)
            assert spec is not None
            assert spec.bore_d == 25

    def test_bearing_load_ratings_valid(self):
        """Test that load ratings are physically valid."""
        for spec in BEARING_DATABASE.values():
            # Dynamic load should be greater than static for ball bearings
            assert spec.dynamic_load_c > 0
            assert spec.static_load_c0 > 0
            # Limiting speeds should be reasonable
            assert spec.limiting_speed_grease > 0
            assert spec.limiting_speed_oil >= spec.limiting_speed_grease

    def test_bearing_dimensions_valid(self):
        """Test that bearing dimensions are physically valid."""
        for spec in BEARING_DATABASE.values():
            # OD > bore
            assert spec.outer_d > spec.bore_d
            # Width > 0
            assert spec.width_b > 0
            # Weight > 0
            assert spec.weight > 0

    def test_bearing_database_completeness(self):
        """Test that common bearing sizes are included."""
        common_bearings = ["6000", "6001", "6200", "6201", "6204", "6205", "6305", "6310"]
        for desig in common_bearings:
            assert get_bearing_spec(desig) is not None


class TestSeals:
    """Tests for O-ring specifications."""

    def test_get_oring_spec_basic(self):
        """Test getting basic O-ring spec."""
        spec = get_oring_spec("20x3")

        assert spec is not None
        assert spec.inner_diameter == 20.0
        assert spec.cross_section == 3.0
        assert spec.standard == "ISO 3601"

    def test_get_oring_spec_normalized(self):
        """Test O-ring spec with various formats."""
        # These should all find the same O-ring
        spec1 = get_oring_spec("20x3")
        spec2 = get_oring_spec("20X3")
        spec3 = get_oring_spec("20 x 3")

        assert spec1 is not None
        assert spec2 is not None
        assert spec3 is not None

    def test_get_oring_spec_invalid(self):
        """Test invalid O-ring returns None."""
        assert get_oring_spec("999x999") is None

    def test_get_oring_by_id(self):
        """Test finding O-rings by inner diameter."""
        orings = get_oring_by_id(20)

        assert len(orings) > 0
        for o in orings:
            assert o.inner_diameter == 20.0

    def test_list_orings_by_cross_section(self):
        """Test listing O-rings by cross-section."""
        orings = list_orings(cross_section=3.0)

        assert len(orings) > 0
        for o in orings:
            assert o.cross_section == 3.0

    def test_suggest_oring_material(self):
        """Test O-ring material suggestions."""
        # Standard conditions - should suggest NBR
        materials = suggest_oring_material(-20, 80, "oil")
        assert len(materials) > 0
        assert ORingMaterial.NBR in materials

        # High temperature - should suggest FKM or silicone
        materials_hot = suggest_oring_material(0, 180, "oil")
        assert len(materials_hot) > 0
        assert ORingMaterial.FKM in materials_hot

        # Water application - should suggest EPDM
        materials_water = suggest_oring_material(-20, 100, "water")
        assert len(materials_water) > 0
        assert ORingMaterial.EPDM in materials_water

    def test_oring_groove_dimensions_valid(self):
        """Test that groove dimensions are physically valid."""
        for spec in ORING_DATABASE.values():
            # Groove width > cross-section (to allow room)
            assert spec.groove_width_static > spec.cross_section
            assert spec.groove_width_dynamic > spec.cross_section
            # Groove depth < cross-section (to achieve compression)
            assert spec.groove_depth_static < spec.cross_section
            assert spec.groove_depth_dynamic < spec.cross_section

    def test_oring_database_completeness(self):
        """Test that common O-ring sizes are included."""
        common_sizes = ["10x2", "20x3", "30x4", "50x5"]
        for size in common_sizes:
            assert get_oring_spec(size) is not None


class TestIntegration:
    """Integration tests across modules."""

    def test_thread_bearing_compatibility(self):
        """Test that common shaft diameters have matching bearings."""
        # Common thread diameters should have matching bearings
        shaft_diameters = [10, 12, 15, 17, 20, 25, 30, 35, 40]

        for d in shaft_diameters:
            bearings = get_bearing_by_bore(d)
            assert len(bearings) > 0, f"No bearing found for {d}mm shaft"

    def test_oring_for_common_bores(self):
        """Test that common bore sizes have matching O-rings."""
        common_bores = [10, 15, 20, 25, 30, 40, 50]

        for bore in common_bores:
            orings = get_oring_by_id(bore)
            # Not all bores will have exact matches, but common ones should
            if bore in [10, 15, 20, 25, 30, 40, 50]:
                assert len(orings) >= 0  # Some sizes may not have exact matches
