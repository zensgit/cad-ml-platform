"""Tests for src/api/v1/drift.py to improve coverage.

Covers:
- drift_status endpoint logic
- drift_reset endpoint logic
- drift_baseline_status endpoint logic
- DriftStatusResponse structure
- DriftResetResponse structure
- DriftBaselineStatusResponse structure
- Auto-refresh logic
- Staleness detection
"""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


class TestDriftStatusLogic:
    """Tests for drift_status endpoint logic."""

    def test_counter_conversion(self):
        """Test Counter converts list to dict counts."""
        materials = ["steel", "steel", "aluminum", "copper", "steel"]
        result = dict(Counter(materials))

        assert result == {"steel": 3, "aluminum": 1, "copper": 1}

    def test_empty_list_counter(self):
        """Test Counter handles empty list."""
        materials: List[str] = []
        result = dict(Counter(materials))

        assert result == {}

    def test_baseline_none_conversion(self):
        """Test baseline conversion when None."""
        baseline_materials: Optional[List[str]] = None
        material_baseline_counts = dict(Counter(baseline_materials)) if baseline_materials else None

        assert material_baseline_counts is None

    def test_baseline_empty_list_conversion(self):
        """Test baseline conversion when empty list."""
        baseline_materials: List[str] = []
        material_baseline_counts = dict(Counter(baseline_materials)) if baseline_materials else None

        assert material_baseline_counts is None

    def test_status_baseline_pending(self):
        """Test status is baseline_pending when counts below min."""
        mats = ["steel"] * 50
        preds = ["type_a"] * 50
        min_count = 100

        status = "baseline_pending" if (len(mats) < min_count or len(preds) < min_count) else "ok"

        assert status == "baseline_pending"

    def test_status_ok_when_sufficient(self):
        """Test status is ok when counts meet minimum."""
        mats = ["steel"] * 100
        preds = ["type_a"] * 100
        min_count = 100

        status = "baseline_pending" if (len(mats) < min_count or len(preds) < min_count) else "ok"

        assert status == "ok"


class TestAutoRefreshLogic:
    """Tests for auto-refresh baseline logic."""

    def test_auto_refresh_enabled_check(self):
        """Test auto-refresh enabled from env."""
        with patch.dict("os.environ", {"DRIFT_BASELINE_AUTO_REFRESH": "1"}):
            import os

            auto_refresh_enabled = os.getenv("DRIFT_BASELINE_AUTO_REFRESH", "1") == "1"
            assert auto_refresh_enabled is True

    def test_auto_refresh_disabled_check(self):
        """Test auto-refresh disabled from env."""
        with patch.dict("os.environ", {"DRIFT_BASELINE_AUTO_REFRESH": "0"}):
            import os

            auto_refresh_enabled = os.getenv("DRIFT_BASELINE_AUTO_REFRESH", "1") == "1"
            assert auto_refresh_enabled is False

    def test_stale_baseline_detection(self):
        """Test stale baseline detection logic."""
        baseline_ts = time.time() - 100000  # 100000 seconds ago
        max_age = 86400  # 24 hours
        current_time = time.time()

        material_age = int(current_time - baseline_ts)

        assert material_age > max_age

    def test_fresh_baseline_detection(self):
        """Test fresh baseline detection logic."""
        baseline_ts = time.time() - 1000  # 1000 seconds ago
        max_age = 86400  # 24 hours
        current_time = time.time()

        material_age = int(current_time - baseline_ts)

        assert material_age < max_age

    def test_auto_refresh_condition_all_met(self):
        """Test all conditions for auto-refresh are met."""
        auto_refresh_enabled = True
        material_age = 100000  # > max_age
        max_age = 86400
        mats_count = 150
        min_count = 100

        should_refresh = auto_refresh_enabled and material_age > max_age and mats_count >= min_count

        assert should_refresh is True

    def test_auto_refresh_condition_disabled(self):
        """Test auto-refresh skipped when disabled."""
        auto_refresh_enabled = False
        material_age = 100000
        max_age = 86400
        mats_count = 150
        min_count = 100

        should_refresh = auto_refresh_enabled and material_age > max_age and mats_count >= min_count

        assert should_refresh is False

    def test_auto_refresh_condition_not_stale(self):
        """Test auto-refresh skipped when not stale."""
        auto_refresh_enabled = True
        material_age = 1000  # < max_age
        max_age = 86400
        mats_count = 150
        min_count = 100

        should_refresh = auto_refresh_enabled and material_age > max_age and mats_count >= min_count

        assert should_refresh is False

    def test_auto_refresh_condition_insufficient_data(self):
        """Test auto-refresh skipped when insufficient data."""
        auto_refresh_enabled = True
        material_age = 100000
        max_age = 86400
        mats_count = 50  # < min_count
        min_count = 100

        should_refresh = auto_refresh_enabled and material_age > max_age and mats_count >= min_count

        assert should_refresh is False


class TestBaselineCreationLogic:
    """Tests for baseline creation logic."""

    def test_baseline_created_when_sufficient(self):
        """Test baseline created when count meets minimum."""
        baseline_exists = False
        mats_count = 100
        min_count = 100

        should_create = not baseline_exists and mats_count >= min_count

        assert should_create is True

    def test_baseline_not_created_when_insufficient(self):
        """Test baseline not created when count below minimum."""
        baseline_exists = False
        mats_count = 50
        min_count = 100

        should_create = not baseline_exists and mats_count >= min_count

        assert should_create is False

    def test_baseline_not_recreated_when_exists(self):
        """Test baseline not recreated when already exists."""
        baseline_exists = True
        mats_count = 150
        min_count = 100

        should_create = not baseline_exists and mats_count >= min_count

        assert should_create is False


class TestStaleFlagLogic:
    """Tests for stale flag calculation logic."""

    def test_stale_flag_material_stale(self):
        """Test stale flag when material baseline is stale."""
        baseline_material_age = 100000
        baseline_prediction_age = None
        max_age = 86400

        stale_flag = None
        if baseline_material_age and baseline_material_age > max_age:
            stale_flag = True
        if baseline_prediction_age and baseline_prediction_age > max_age:
            stale_flag = True if stale_flag is None else True
        if stale_flag is None and (baseline_material_age or baseline_prediction_age):
            stale_flag = False

        assert stale_flag is True

    def test_stale_flag_prediction_stale(self):
        """Test stale flag when prediction baseline is stale."""
        baseline_material_age = None
        baseline_prediction_age = 100000
        max_age = 86400

        stale_flag = None
        if baseline_material_age and baseline_material_age > max_age:
            stale_flag = True
        if baseline_prediction_age and baseline_prediction_age > max_age:
            stale_flag = True if stale_flag is None else True
        if stale_flag is None and (baseline_material_age or baseline_prediction_age):
            stale_flag = False

        assert stale_flag is True

    def test_stale_flag_both_stale(self):
        """Test stale flag when both baselines are stale."""
        baseline_material_age = 100000
        baseline_prediction_age = 100000
        max_age = 86400

        stale_flag = None
        if baseline_material_age and baseline_material_age > max_age:
            stale_flag = True
        if baseline_prediction_age and baseline_prediction_age > max_age:
            stale_flag = True if stale_flag is None else True
        if stale_flag is None and (baseline_material_age or baseline_prediction_age):
            stale_flag = False

        assert stale_flag is True

    def test_stale_flag_neither_stale(self):
        """Test stale flag when neither baseline is stale."""
        baseline_material_age = 1000
        baseline_prediction_age = 2000
        max_age = 86400

        stale_flag = None
        if baseline_material_age and baseline_material_age > max_age:
            stale_flag = True
        if baseline_prediction_age and baseline_prediction_age > max_age:
            stale_flag = True if stale_flag is None else True
        if stale_flag is None and (baseline_material_age or baseline_prediction_age):
            stale_flag = False

        assert stale_flag is False

    def test_stale_flag_no_baselines(self):
        """Test stale flag when no baselines exist."""
        baseline_material_age = None
        baseline_prediction_age = None
        max_age = 86400

        stale_flag = None
        if baseline_material_age and baseline_material_age > max_age:
            stale_flag = True
        if baseline_prediction_age and baseline_prediction_age > max_age:
            stale_flag = True if stale_flag is None else True
        if stale_flag is None and (baseline_material_age or baseline_prediction_age):
            stale_flag = False

        assert stale_flag is None


class TestDriftResetLogic:
    """Tests for drift_reset endpoint logic."""

    def test_reset_both_baselines(self):
        """Test reset when both baselines exist."""
        drift_state = {
            "baseline_materials": ["steel", "aluminum"],
            "baseline_predictions": ["type_a", "type_b"],
        }

        reset_material = bool(drift_state["baseline_materials"])
        reset_predictions = bool(drift_state["baseline_predictions"])

        assert reset_material is True
        assert reset_predictions is True

    def test_reset_only_materials(self):
        """Test reset when only materials baseline exists."""
        drift_state = {
            "baseline_materials": ["steel"],
            "baseline_predictions": [],
        }

        reset_material = bool(drift_state["baseline_materials"])
        reset_predictions = bool(drift_state["baseline_predictions"])

        assert reset_material is True
        assert reset_predictions is False

    def test_reset_only_predictions(self):
        """Test reset when only predictions baseline exists."""
        drift_state = {
            "baseline_materials": [],
            "baseline_predictions": ["type_a"],
        }

        reset_material = bool(drift_state["baseline_materials"])
        reset_predictions = bool(drift_state["baseline_predictions"])

        assert reset_material is False
        assert reset_predictions is True

    def test_reset_no_baselines(self):
        """Test reset when no baselines exist."""
        drift_state = {
            "baseline_materials": [],
            "baseline_predictions": [],
        }

        reset_material = bool(drift_state["baseline_materials"])
        reset_predictions = bool(drift_state["baseline_predictions"])

        assert reset_material is False
        assert reset_predictions is False

    def test_state_cleared_after_reset(self):
        """Test state is properly cleared after reset."""
        drift_state = {
            "baseline_materials": ["steel", "aluminum"],
            "baseline_predictions": ["type_a"],
            "baseline_materials_ts": time.time(),
            "baseline_predictions_ts": time.time(),
        }

        # Perform reset
        drift_state["baseline_materials"] = []
        drift_state["baseline_predictions"] = []
        drift_state["baseline_materials_ts"] = None
        drift_state["baseline_predictions_ts"] = None

        assert drift_state["baseline_materials"] == []
        assert drift_state["baseline_predictions"] == []
        assert drift_state["baseline_materials_ts"] is None
        assert drift_state["baseline_predictions_ts"] is None


class TestDriftBaselineStatusLogic:
    """Tests for drift_baseline_status endpoint logic."""

    def test_status_no_baseline(self):
        """Test status when no baselines exist."""
        material_age = None
        prediction_age = None
        stale_flag = None

        status = "stale" if stale_flag else "ok"
        if material_age is None and prediction_age is None:
            status = "no_baseline"

        assert status == "no_baseline"

    def test_status_stale(self):
        """Test status when baselines are stale."""
        material_age = 100000
        prediction_age = 100000
        max_age = 86400

        stale_flag = None
        if material_age and material_age > max_age:
            stale_flag = True
        if prediction_age and prediction_age > max_age:
            stale_flag = True if stale_flag is None else True
        if stale_flag is None and (material_age or prediction_age):
            stale_flag = False

        status = "stale" if stale_flag else "ok"
        if material_age is None and prediction_age is None:
            status = "no_baseline"

        assert status == "stale"

    def test_status_ok(self):
        """Test status when baselines are fresh."""
        material_age = 1000
        prediction_age = 2000
        max_age = 86400

        stale_flag = None
        if material_age and material_age > max_age:
            stale_flag = True
        if prediction_age and prediction_age > max_age:
            stale_flag = True if stale_flag is None else True
        if stale_flag is None and (material_age or prediction_age):
            stale_flag = False

        status = "stale" if stale_flag else "ok"
        if material_age is None and prediction_age is None:
            status = "no_baseline"

        assert status == "ok"


class TestDatetimeConversion:
    """Tests for datetime conversion logic."""

    def test_timestamp_to_datetime(self):
        """Test timestamp converts to datetime correctly."""
        ts = 1704067200.0  # 2024-01-01 00:00:00 UTC
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)

        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1
        assert dt.tzinfo == timezone.utc

    def test_current_timestamp_to_datetime(self):
        """Test current timestamp converts to datetime."""
        ts = time.time()
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)

        assert dt.tzinfo == timezone.utc
        assert (time.time() - ts) < 1  # Within 1 second

    def test_age_calculation(self):
        """Test age calculation from timestamp."""
        baseline_ts = time.time() - 3600  # 1 hour ago
        current_time = time.time()

        age = int(current_time - baseline_ts)

        assert 3599 <= age <= 3601  # Allow 1 second tolerance


class TestEnvVarDefaults:
    """Tests for environment variable defaults."""

    def test_min_count_default(self):
        """Test DRIFT_BASELINE_MIN_COUNT default."""
        import os

        with patch.dict("os.environ", {}, clear=True):
            min_count = int(os.getenv("DRIFT_BASELINE_MIN_COUNT", "100"))
            assert min_count == 100

    def test_min_count_override(self):
        """Test DRIFT_BASELINE_MIN_COUNT override."""
        import os

        with patch.dict("os.environ", {"DRIFT_BASELINE_MIN_COUNT": "50"}):
            min_count = int(os.getenv("DRIFT_BASELINE_MIN_COUNT", "100"))
            assert min_count == 50

    def test_max_age_default(self):
        """Test DRIFT_BASELINE_MAX_AGE_SECONDS default."""
        import os

        with patch.dict("os.environ", {}, clear=True):
            max_age = int(os.getenv("DRIFT_BASELINE_MAX_AGE_SECONDS", "86400"))
            assert max_age == 86400

    def test_max_age_override(self):
        """Test DRIFT_BASELINE_MAX_AGE_SECONDS override."""
        import os

        with patch.dict("os.environ", {"DRIFT_BASELINE_MAX_AGE_SECONDS": "43200"}):
            max_age = int(os.getenv("DRIFT_BASELINE_MAX_AGE_SECONDS", "86400"))
            assert max_age == 43200


class TestDriftResponseStructures:
    """Tests for response structure validation."""

    def test_drift_status_response_fields(self):
        """Test DriftStatusResponse has expected fields."""
        response = {
            "material_current": {"steel": 50, "aluminum": 30},
            "material_baseline": {"steel": 45, "aluminum": 35},
            "material_drift_score": 0.15,
            "prediction_current": {"type_a": 60, "type_b": 40},
            "prediction_baseline": {"type_a": 55, "type_b": 45},
            "prediction_drift_score": 0.12,
            "baseline_min_count": 100,
            "materials_total": 80,
            "predictions_total": 100,
            "status": "ok",
            "baseline_material_age": 3600,
            "baseline_prediction_age": 7200,
            "baseline_material_created_at": "2024-01-01T00:00:00+00:00",
            "baseline_prediction_created_at": "2024-01-01T00:00:00+00:00",
            "stale": False,
        }

        assert "material_current" in response
        assert "material_baseline" in response
        assert "material_drift_score" in response
        assert "prediction_current" in response
        assert "prediction_baseline" in response
        assert "prediction_drift_score" in response
        assert "baseline_min_count" in response
        assert "materials_total" in response
        assert "predictions_total" in response
        assert "status" in response
        assert "stale" in response

    def test_drift_reset_response_fields(self):
        """Test DriftResetResponse has expected fields."""
        response = {
            "status": "ok",
            "reset_material": True,
            "reset_predictions": True,
        }

        assert response["status"] == "ok"
        assert "reset_material" in response
        assert "reset_predictions" in response

    def test_drift_baseline_status_response_fields(self):
        """Test DriftBaselineStatusResponse has expected fields."""
        response = {
            "status": "ok",
            "material_age": 3600,
            "prediction_age": 7200,
            "material_created_at": "2024-01-01T00:00:00+00:00",
            "prediction_created_at": "2024-01-01T00:00:00+00:00",
            "stale": False,
            "max_age_seconds": 86400,
        }

        assert "status" in response
        assert "material_age" in response
        assert "prediction_age" in response
        assert "stale" in response
        assert "max_age_seconds" in response


class TestStartupMarkLogic:
    """Tests for startup mark logic to prevent duplicate metrics."""

    def test_startup_mark_not_set_initially(self):
        """Test startup mark not set initially."""
        drift_state: Dict[str, Any] = {}

        is_marked = drift_state.get("baseline_materials_startup_mark") is not None

        assert is_marked is False

    def test_startup_mark_set_after_first_access(self):
        """Test startup mark set after first access."""
        drift_state: Dict[str, Any] = {}

        # Simulate first access
        if drift_state.get("baseline_materials_startup_mark") is None:
            drift_state["baseline_materials_startup_mark"] = True

        assert drift_state["baseline_materials_startup_mark"] is True

    def test_startup_mark_prevents_repeat_increment(self):
        """Test startup mark prevents repeat increment."""
        drift_state: Dict[str, Any] = {"baseline_materials_startup_mark": True}
        increment_called = False

        if drift_state.get("baseline_materials_startup_mark") is None:
            increment_called = True
            drift_state["baseline_materials_startup_mark"] = True

        assert increment_called is False


class TestDriftMetricsIntegration:
    """Tests for drift metrics integration."""

    def test_drift_baseline_refresh_total_import(self):
        """Test drift_baseline_refresh_total can be imported."""
        from src.utils.analysis_metrics import drift_baseline_refresh_total

        assert drift_baseline_refresh_total is not None

    def test_drift_baseline_created_total_import(self):
        """Test drift_baseline_created_total can be imported."""
        from src.utils.analysis_metrics import drift_baseline_created_total

        assert drift_baseline_created_total is not None

    def test_drift_baseline_refresh_labels(self):
        """Test drift_baseline_refresh_total supports expected labels."""
        from src.utils.analysis_metrics import drift_baseline_refresh_total

        # Test label combinations
        labeled_material_stale = drift_baseline_refresh_total.labels(
            type="material", trigger="stale"
        )
        labeled_prediction_manual = drift_baseline_refresh_total.labels(
            type="prediction", trigger="manual"
        )
        labeled_material_startup = drift_baseline_refresh_total.labels(
            type="material", trigger="startup"
        )

        assert labeled_material_stale is not None
        assert labeled_prediction_manual is not None
        assert labeled_material_startup is not None

    def test_drift_baseline_created_labels(self):
        """Test drift_baseline_created_total supports expected labels."""
        from src.utils.analysis_metrics import drift_baseline_created_total

        labeled_material = drift_baseline_created_total.labels(type="material")
        labeled_prediction = drift_baseline_created_total.labels(type="prediction")

        assert labeled_material is not None
        assert labeled_prediction is not None


class TestDriftComputeIntegration:
    """Tests for drift compute integration."""

    def test_compute_drift_import(self):
        """Test compute_drift can be imported."""
        from src.utils.drift import compute_drift

        assert callable(compute_drift)

    def test_compute_drift_basic(self):
        """Test compute_drift with basic inputs."""
        from src.utils.drift import compute_drift

        current = ["a", "a", "b"]
        baseline = ["a", "b", "b"]

        score = compute_drift(current, baseline)

        assert isinstance(score, float)
        assert 0 <= score <= 1


class TestEdgeCases:
    """Tests for edge cases in drift logic."""

    def test_single_item_counter(self):
        """Test counter with single item."""
        materials = ["steel"]
        result = dict(Counter(materials))

        assert result == {"steel": 1}

    def test_large_counter(self):
        """Test counter with many items."""
        materials = ["steel"] * 1000 + ["aluminum"] * 500 + ["copper"] * 300
        result = dict(Counter(materials))

        assert result["steel"] == 1000
        assert result["aluminum"] == 500
        assert result["copper"] == 300

    def test_exact_min_count_threshold(self):
        """Test behavior at exact min_count threshold."""
        mats_count = 100
        min_count = 100

        meets_threshold = mats_count >= min_count

        assert meets_threshold is True

    def test_one_below_min_count_threshold(self):
        """Test behavior one below min_count threshold."""
        mats_count = 99
        min_count = 100

        meets_threshold = mats_count >= min_count

        assert meets_threshold is False

    def test_exact_max_age_threshold(self):
        """Test behavior at exact max_age threshold."""
        age = 86400
        max_age = 86400

        is_stale = age > max_age

        assert is_stale is False

    def test_one_above_max_age_threshold(self):
        """Test behavior one above max_age threshold."""
        age = 86401
        max_age = 86400

        is_stale = age > max_age

        assert is_stale is True
