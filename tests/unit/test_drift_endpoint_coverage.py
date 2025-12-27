"""Tests for src/api/v1/drift.py endpoint functions to improve coverage.

Directly calls the endpoint functions with mocked dependencies to achieve
high coverage of drift_status, drift_reset, and drift_baseline_status.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class TestDriftStatusEndpoint:
    """Tests for drift_status endpoint function."""

    @pytest.fixture(autouse=True)
    def reset_drift_state(self):
        """Reset drift state before each test."""
        from src.api.v1 import analyze as analyze_module

        # Save original state
        original_state = analyze_module._DRIFT_STATE.copy()

        # Reset to clean state
        analyze_module._DRIFT_STATE.update(
            {
                "materials": [],
                "predictions": [],
                "baseline_materials": [],
                "baseline_predictions": [],
                "baseline_materials_ts": None,
                "baseline_predictions_ts": None,
                "baseline_materials_startup_mark": None,
                "baseline_predictions_startup_mark": None,
            }
        )

        yield

        # Restore original state
        analyze_module._DRIFT_STATE.update(original_state)

    @pytest.mark.asyncio
    async def test_drift_status_empty_state_baseline_pending(self):
        """Test drift_status with empty state returns baseline_pending."""
        from src.api.v1.drift import drift_status

        result = await drift_status(api_key="test")

        assert result.status == "baseline_pending"
        assert result.material_current == {}
        assert result.prediction_current == {}
        assert result.material_baseline is None
        assert result.prediction_baseline is None
        assert result.material_drift_score is None
        assert result.prediction_drift_score is None
        assert result.materials_total == 0
        assert result.predictions_total == 0

    @pytest.mark.asyncio
    async def test_drift_status_creates_material_baseline(self):
        """Test drift_status creates material baseline when threshold reached."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_status

        # Set sufficient materials
        analyze_module._DRIFT_STATE["materials"] = ["steel"] * 60 + ["aluminum"] * 40
        analyze_module._DRIFT_STATE["predictions"] = ["simple"] * 100

        with patch.dict("os.environ", {"DRIFT_BASELINE_MIN_COUNT": "100"}):
            result = await drift_status(api_key="test")

        assert result.status == "ok"
        assert result.materials_total == 100
        assert result.predictions_total == 100
        # Baseline should have been created
        assert analyze_module._DRIFT_STATE["baseline_materials"] is not None
        assert len(analyze_module._DRIFT_STATE["baseline_materials"]) == 100

    @pytest.mark.asyncio
    async def test_drift_status_computes_drift_score(self):
        """Test drift_status computes drift score with existing baseline."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_status

        # Set different current vs baseline distributions
        analyze_module._DRIFT_STATE["materials"] = ["steel"] * 80 + ["aluminum"] * 20
        analyze_module._DRIFT_STATE["predictions"] = ["simple"] * 70 + ["complex"] * 30
        analyze_module._DRIFT_STATE["baseline_materials"] = ["steel"] * 50 + ["aluminum"] * 50
        analyze_module._DRIFT_STATE["baseline_predictions"] = ["simple"] * 50 + ["complex"] * 50
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = time.time()
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = time.time()

        result = await drift_status(api_key="test")

        assert result.material_drift_score is not None
        assert result.material_drift_score > 0
        assert result.prediction_drift_score is not None
        assert result.prediction_drift_score > 0
        assert result.material_baseline is not None
        assert result.prediction_baseline is not None

    @pytest.mark.asyncio
    async def test_drift_status_auto_refresh_stale_material_baseline(self):
        """Test drift_status auto-refreshes stale material baseline."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_status

        # Set stale baseline
        old_time = time.time() - 100000  # Very old
        analyze_module._DRIFT_STATE["materials"] = ["steel"] * 100
        analyze_module._DRIFT_STATE["predictions"] = ["simple"] * 100
        analyze_module._DRIFT_STATE["baseline_materials"] = ["aluminum"] * 100
        analyze_module._DRIFT_STATE["baseline_predictions"] = ["complex"] * 100
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = old_time
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = old_time

        with patch.dict(
            "os.environ",
            {
                "DRIFT_BASELINE_AUTO_REFRESH": "1",
                "DRIFT_BASELINE_MAX_AGE_SECONDS": "86400",
                "DRIFT_BASELINE_MIN_COUNT": "100",
            },
        ):
            result = await drift_status(api_key="test")

        # Should have been refreshed with current distribution
        assert analyze_module._DRIFT_STATE["baseline_materials_ts"] > old_time
        assert analyze_module._DRIFT_STATE["baseline_predictions_ts"] > old_time

    @pytest.mark.asyncio
    async def test_drift_status_no_auto_refresh_when_disabled(self):
        """Test drift_status does not auto-refresh when disabled."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_status

        old_time = time.time() - 100000
        analyze_module._DRIFT_STATE["materials"] = ["steel"] * 100
        analyze_module._DRIFT_STATE["predictions"] = ["simple"] * 100
        analyze_module._DRIFT_STATE["baseline_materials"] = ["aluminum"] * 100
        analyze_module._DRIFT_STATE["baseline_predictions"] = ["complex"] * 100
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = old_time
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = old_time

        with patch.dict(
            "os.environ",
            {
                "DRIFT_BASELINE_AUTO_REFRESH": "0",
                "DRIFT_BASELINE_MAX_AGE_SECONDS": "86400",
            },
        ):
            result = await drift_status(api_key="test")

        # Should NOT have been refreshed
        assert analyze_module._DRIFT_STATE["baseline_materials_ts"] == old_time
        assert analyze_module._DRIFT_STATE["baseline_predictions_ts"] == old_time
        # But stale flag should be True
        assert result.stale is True

    @pytest.mark.asyncio
    async def test_drift_status_stale_flag_true_for_old_baselines(self):
        """Test drift_status sets stale flag when baselines are old."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_status

        old_time = time.time() - 100000
        analyze_module._DRIFT_STATE["materials"] = ["steel"] * 50
        analyze_module._DRIFT_STATE["predictions"] = ["simple"] * 50
        analyze_module._DRIFT_STATE["baseline_materials"] = ["steel"] * 50
        analyze_module._DRIFT_STATE["baseline_predictions"] = ["simple"] * 50
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = old_time
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = old_time
        analyze_module._DRIFT_STATE["baseline_materials_startup_mark"] = True
        analyze_module._DRIFT_STATE["baseline_predictions_startup_mark"] = True

        with patch.dict(
            "os.environ",
            {
                "DRIFT_BASELINE_AUTO_REFRESH": "0",
                "DRIFT_BASELINE_MAX_AGE_SECONDS": "86400",
            },
        ):
            result = await drift_status(api_key="test")

        assert result.stale is True
        assert result.baseline_material_age is not None
        assert result.baseline_material_age > 86400

    @pytest.mark.asyncio
    async def test_drift_status_stale_flag_false_for_fresh_baselines(self):
        """Test drift_status sets stale flag False when baselines are fresh."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_status

        recent_time = time.time() - 1000  # 1000 seconds ago
        analyze_module._DRIFT_STATE["materials"] = ["steel"] * 100
        analyze_module._DRIFT_STATE["predictions"] = ["simple"] * 100
        analyze_module._DRIFT_STATE["baseline_materials"] = ["steel"] * 100
        analyze_module._DRIFT_STATE["baseline_predictions"] = ["simple"] * 100
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = recent_time
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = recent_time
        analyze_module._DRIFT_STATE["baseline_materials_startup_mark"] = True
        analyze_module._DRIFT_STATE["baseline_predictions_startup_mark"] = True

        result = await drift_status(api_key="test")

        assert result.stale is False
        assert result.baseline_material_age is not None
        assert result.baseline_material_age < 86400

    @pytest.mark.asyncio
    async def test_drift_status_startup_mark_set(self):
        """Test drift_status sets startup mark on first access."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_status

        analyze_module._DRIFT_STATE["materials"] = ["steel"] * 100
        analyze_module._DRIFT_STATE["predictions"] = ["simple"] * 100
        analyze_module._DRIFT_STATE["baseline_materials"] = ["steel"] * 100
        analyze_module._DRIFT_STATE["baseline_predictions"] = ["simple"] * 100
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = time.time()
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = time.time()
        analyze_module._DRIFT_STATE["baseline_materials_startup_mark"] = None
        analyze_module._DRIFT_STATE["baseline_predictions_startup_mark"] = None

        await drift_status(api_key="test")

        assert analyze_module._DRIFT_STATE["baseline_materials_startup_mark"] is True
        assert analyze_module._DRIFT_STATE["baseline_predictions_startup_mark"] is True

    @pytest.mark.asyncio
    async def test_drift_status_datetime_fields(self):
        """Test drift_status populates datetime fields correctly."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_status

        ts = time.time()
        analyze_module._DRIFT_STATE["materials"] = ["steel"] * 100
        analyze_module._DRIFT_STATE["predictions"] = ["simple"] * 100
        analyze_module._DRIFT_STATE["baseline_materials"] = ["steel"] * 100
        analyze_module._DRIFT_STATE["baseline_predictions"] = ["simple"] * 100
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = ts
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = ts
        analyze_module._DRIFT_STATE["baseline_materials_startup_mark"] = True
        analyze_module._DRIFT_STATE["baseline_predictions_startup_mark"] = True

        result = await drift_status(api_key="test")

        assert result.baseline_material_created_at is not None
        assert result.baseline_prediction_created_at is not None
        assert isinstance(result.baseline_material_created_at, datetime)
        assert result.baseline_material_created_at.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_drift_status_material_only_stale(self):
        """Test drift_status when only material baseline is stale."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_status

        old_time = time.time() - 100000
        recent_time = time.time() - 1000
        analyze_module._DRIFT_STATE["materials"] = ["steel"] * 50
        analyze_module._DRIFT_STATE["predictions"] = ["simple"] * 50
        analyze_module._DRIFT_STATE["baseline_materials"] = ["steel"] * 50
        analyze_module._DRIFT_STATE["baseline_predictions"] = ["simple"] * 50
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = old_time
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = recent_time
        analyze_module._DRIFT_STATE["baseline_materials_startup_mark"] = True
        analyze_module._DRIFT_STATE["baseline_predictions_startup_mark"] = True

        with patch.dict("os.environ", {"DRIFT_BASELINE_AUTO_REFRESH": "0"}):
            result = await drift_status(api_key="test")

        assert result.stale is True

    @pytest.mark.asyncio
    async def test_drift_status_prediction_only_stale(self):
        """Test drift_status when only prediction baseline is stale."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_status

        old_time = time.time() - 100000
        recent_time = time.time() - 1000
        analyze_module._DRIFT_STATE["materials"] = ["steel"] * 50
        analyze_module._DRIFT_STATE["predictions"] = ["simple"] * 50
        analyze_module._DRIFT_STATE["baseline_materials"] = ["steel"] * 50
        analyze_module._DRIFT_STATE["baseline_predictions"] = ["simple"] * 50
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = recent_time
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = old_time
        analyze_module._DRIFT_STATE["baseline_materials_startup_mark"] = True
        analyze_module._DRIFT_STATE["baseline_predictions_startup_mark"] = True

        with patch.dict("os.environ", {"DRIFT_BASELINE_AUTO_REFRESH": "0"}):
            result = await drift_status(api_key="test")

        assert result.stale is True


class TestDriftResetEndpoint:
    """Tests for drift_reset endpoint function."""

    @pytest.fixture(autouse=True)
    def reset_drift_state(self):
        """Reset drift state before each test."""
        from src.api.v1 import analyze as analyze_module

        original_state = analyze_module._DRIFT_STATE.copy()

        analyze_module._DRIFT_STATE.update(
            {
                "materials": [],
                "predictions": [],
                "baseline_materials": [],
                "baseline_predictions": [],
                "baseline_materials_ts": None,
                "baseline_predictions_ts": None,
            }
        )

        yield

        analyze_module._DRIFT_STATE.update(original_state)

    @pytest.mark.asyncio
    async def test_drift_reset_with_both_baselines(self):
        """Test drift_reset when both baselines exist."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_reset

        analyze_module._DRIFT_STATE["baseline_materials"] = ["steel"] * 50
        analyze_module._DRIFT_STATE["baseline_predictions"] = ["simple"] * 50
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = time.time()
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = time.time()

        result = await drift_reset(api_key="test")

        assert result.status == "ok"
        assert result.reset_material is True
        assert result.reset_predictions is True
        assert analyze_module._DRIFT_STATE["baseline_materials"] == []
        assert analyze_module._DRIFT_STATE["baseline_predictions"] == []
        assert analyze_module._DRIFT_STATE["baseline_materials_ts"] is None
        assert analyze_module._DRIFT_STATE["baseline_predictions_ts"] is None

    @pytest.mark.asyncio
    async def test_drift_reset_with_only_material_baseline(self):
        """Test drift_reset when only material baseline exists."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_reset

        analyze_module._DRIFT_STATE["baseline_materials"] = ["steel"] * 50
        analyze_module._DRIFT_STATE["baseline_predictions"] = []
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = time.time()

        result = await drift_reset(api_key="test")

        assert result.status == "ok"
        assert result.reset_material is True
        assert result.reset_predictions is False

    @pytest.mark.asyncio
    async def test_drift_reset_with_only_prediction_baseline(self):
        """Test drift_reset when only prediction baseline exists."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_reset

        analyze_module._DRIFT_STATE["baseline_materials"] = []
        analyze_module._DRIFT_STATE["baseline_predictions"] = ["simple"] * 50
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = time.time()

        result = await drift_reset(api_key="test")

        assert result.status == "ok"
        assert result.reset_material is False
        assert result.reset_predictions is True

    @pytest.mark.asyncio
    async def test_drift_reset_with_no_baselines(self):
        """Test drift_reset when no baselines exist."""
        from src.api.v1.drift import drift_reset

        result = await drift_reset(api_key="test")

        assert result.status == "ok"
        assert result.reset_material is False
        assert result.reset_predictions is False


class TestDriftBaselineStatusEndpoint:
    """Tests for drift_baseline_status endpoint function."""

    @pytest.fixture(autouse=True)
    def reset_drift_state(self):
        """Reset drift state before each test."""
        from src.api.v1 import analyze as analyze_module

        original_state = analyze_module._DRIFT_STATE.copy()

        analyze_module._DRIFT_STATE.update(
            {
                "materials": [],
                "predictions": [],
                "baseline_materials": [],
                "baseline_predictions": [],
                "baseline_materials_ts": None,
                "baseline_predictions_ts": None,
            }
        )

        yield

        analyze_module._DRIFT_STATE.update(original_state)

    @pytest.mark.asyncio
    async def test_drift_baseline_status_no_baseline(self):
        """Test drift_baseline_status with no baselines."""
        from src.api.v1.drift import drift_baseline_status

        result = await drift_baseline_status(api_key="test")

        assert result.status == "no_baseline"
        assert result.material_age is None
        assert result.prediction_age is None
        assert result.material_created_at is None
        assert result.prediction_created_at is None
        assert result.stale is None

    @pytest.mark.asyncio
    async def test_drift_baseline_status_fresh_baselines(self):
        """Test drift_baseline_status with fresh baselines."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_baseline_status

        recent_time = time.time() - 1000
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = recent_time
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = recent_time

        result = await drift_baseline_status(api_key="test")

        assert result.status == "ok"
        assert result.stale is False
        assert result.material_age is not None
        assert result.prediction_age is not None
        assert result.material_age < 86400
        assert result.prediction_age < 86400
        assert result.material_created_at is not None
        assert result.prediction_created_at is not None

    @pytest.mark.asyncio
    async def test_drift_baseline_status_stale_baselines(self):
        """Test drift_baseline_status with stale baselines."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_baseline_status

        old_time = time.time() - 100000
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = old_time
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = old_time

        result = await drift_baseline_status(api_key="test")

        assert result.status == "stale"
        assert result.stale is True
        assert result.material_age is not None
        assert result.prediction_age is not None
        assert result.material_age > 86400

    @pytest.mark.asyncio
    async def test_drift_baseline_status_only_material_exists(self):
        """Test drift_baseline_status when only material baseline exists."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_baseline_status

        recent_time = time.time() - 1000
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = recent_time

        result = await drift_baseline_status(api_key="test")

        assert result.status == "ok"
        assert result.material_age is not None
        assert result.prediction_age is None
        assert result.stale is False

    @pytest.mark.asyncio
    async def test_drift_baseline_status_only_prediction_exists(self):
        """Test drift_baseline_status when only prediction baseline exists."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_baseline_status

        recent_time = time.time() - 1000
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = recent_time

        result = await drift_baseline_status(api_key="test")

        assert result.status == "ok"
        assert result.material_age is None
        assert result.prediction_age is not None
        assert result.stale is False

    @pytest.mark.asyncio
    async def test_drift_baseline_status_material_stale_prediction_fresh(self):
        """Test drift_baseline_status when material is stale but prediction is fresh."""
        from src.api.v1 import analyze as analyze_module
        from src.api.v1.drift import drift_baseline_status

        old_time = time.time() - 100000
        recent_time = time.time() - 1000
        analyze_module._DRIFT_STATE["baseline_materials_ts"] = old_time
        analyze_module._DRIFT_STATE["baseline_predictions_ts"] = recent_time

        result = await drift_baseline_status(api_key="test")

        assert result.status == "stale"
        assert result.stale is True

    @pytest.mark.asyncio
    async def test_drift_baseline_status_max_age_from_env(self):
        """Test drift_baseline_status uses max_age from env."""
        from src.api.v1.drift import drift_baseline_status

        with patch.dict("os.environ", {"DRIFT_BASELINE_MAX_AGE_SECONDS": "43200"}):
            result = await drift_baseline_status(api_key="test")

        assert result.max_age_seconds == 43200


class TestDriftResponseModels:
    """Tests for drift response model creation."""

    def test_drift_status_response_model_creation(self):
        """Test DriftStatusResponse model can be created."""
        from src.api.v1.drift import DriftStatusResponse

        response = DriftStatusResponse(
            material_current={"steel": 50, "aluminum": 30},
            material_baseline={"steel": 45, "aluminum": 35},
            material_drift_score=0.15,
            prediction_current={"simple": 60},
            prediction_baseline={"simple": 55},
            prediction_drift_score=0.1,
            baseline_min_count=100,
            materials_total=80,
            predictions_total=60,
            status="ok",
            baseline_material_age=3600,
            baseline_prediction_age=3600,
            baseline_material_created_at=datetime.now(timezone.utc),
            baseline_prediction_created_at=datetime.now(timezone.utc),
            stale=False,
        )

        assert response.status == "ok"
        assert response.material_drift_score == 0.15

    def test_drift_status_response_with_none_values(self):
        """Test DriftStatusResponse with None optional values."""
        from src.api.v1.drift import DriftStatusResponse

        response = DriftStatusResponse(
            material_current={},
            prediction_current={},
            baseline_min_count=100,
            materials_total=0,
            predictions_total=0,
            status="baseline_pending",
        )

        assert response.material_baseline is None
        assert response.prediction_baseline is None
        assert response.material_drift_score is None
        assert response.prediction_drift_score is None
        assert response.stale is None

    def test_drift_reset_response_model_creation(self):
        """Test DriftResetResponse model can be created."""
        from src.api.v1.drift import DriftResetResponse

        response = DriftResetResponse(
            status="ok",
            reset_material=True,
            reset_predictions=False,
        )

        assert response.status == "ok"
        assert response.reset_material is True
        assert response.reset_predictions is False

    def test_drift_baseline_status_response_model_creation(self):
        """Test DriftBaselineStatusResponse model can be created."""
        from src.api.v1.drift import DriftBaselineStatusResponse

        response = DriftBaselineStatusResponse(
            status="ok",
            material_age=3600,
            prediction_age=7200,
            material_created_at=datetime.now(timezone.utc),
            prediction_created_at=datetime.now(timezone.utc),
            stale=False,
            max_age_seconds=86400,
        )

        assert response.status == "ok"
        assert response.max_age_seconds == 86400

    def test_drift_baseline_status_response_with_none_values(self):
        """Test DriftBaselineStatusResponse with None optional values."""
        from src.api.v1.drift import DriftBaselineStatusResponse

        response = DriftBaselineStatusResponse(
            status="no_baseline",
            max_age_seconds=86400,
        )

        assert response.material_age is None
        assert response.prediction_age is None
        assert response.material_created_at is None
        assert response.prediction_created_at is None
        assert response.stale is None


class TestDriftRouterExport:
    """Tests for router export."""

    def test_router_exported(self):
        """Test router is exported from module."""
        from src.api.v1.drift import router

        assert router is not None

    def test_all_contains_router(self):
        """Test __all__ contains router."""
        from src.api.v1 import drift

        assert "router" in drift.__all__
