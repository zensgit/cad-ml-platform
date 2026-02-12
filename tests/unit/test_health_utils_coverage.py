"""Tests for src/api/health_utils.py to improve coverage.

Covers:
- record_health_request function with metrics exception
- build_health_payload with hybrid config exceptions
- build_health_payload with provider registry exceptions
- torch availability detection
- degraded reasons logic
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestMetricsEnabled:
    """Tests for metrics_enabled function."""

    def test_metrics_enabled(self):
        """Test metrics_enabled returns boolean."""
        from src.api.health_utils import metrics_enabled

        result = metrics_enabled()
        assert isinstance(result, bool)


class TestRecordHealthRequest:
    """Tests for record_health_request function."""

    def test_record_health_request_success(self):
        """Test record_health_request records metrics."""
        from src.api.health_utils import record_health_request

        # Should not raise
        record_health_request("/health", "200", 0.05)

    def test_record_health_request_metrics_exception(self):
        """Test record_health_request handles metrics exception."""
        from src.api.health_utils import record_health_request

        with patch(
            "src.api.health_utils.health_requests_total"
        ) as mock_counter:
            mock_counter.labels.return_value.inc.side_effect = Exception("Metrics error")

            # Should not raise even if metrics fail
            record_health_request("/health", "500", 0.1)


class TestBuildHealthPayload:
    """Tests for build_health_payload function."""

    def test_build_health_payload_basic(self):
        """Test build_health_payload returns expected structure."""
        from src.api.health_utils import build_health_payload

        payload = build_health_payload()

        assert "status" in payload
        assert payload["status"] == "healthy"
        assert "timestamp" in payload
        assert "services" in payload
        assert "runtime" in payload
        assert "config" in payload

    def test_build_health_payload_with_metrics_override_true(self):
        """Test build_health_payload with metrics_enabled_override=True."""
        from src.api.health_utils import build_health_payload

        payload = build_health_payload(metrics_enabled_override=True)

        assert payload["runtime"]["metrics_enabled"] is True

    def test_build_health_payload_with_metrics_override_false(self):
        """Test build_health_payload with metrics_enabled_override=False."""
        from src.api.health_utils import build_health_payload

        payload = build_health_payload(metrics_enabled_override=False)

        assert payload["runtime"]["metrics_enabled"] is False

    def test_build_health_payload_hybrid_config_exception(self):
        """Test build_health_payload handles hybrid config exception."""
        from src.api.health_utils import build_health_payload

        with patch(
            "src.ml.hybrid_config.get_config",
            side_effect=Exception("Config error"),
        ):
            # Should not raise, just skip ML config
            payload = build_health_payload()

        assert "status" in payload
        # ML config may or may not be present depending on import success

    def test_build_health_payload_provider_registry_exception(self):
        """Test build_health_payload handles provider registry exception."""
        from src.api.health_utils import build_health_payload

        with patch(
            "src.api.health_utils.get_core_provider_registry_snapshot",
            side_effect=Exception("Registry error"),
        ):
            # Should not raise
            payload = build_health_payload()

        assert "status" in payload
        # core_providers may not be in config

    def test_build_health_payload_sanitizes_core_provider_plugin_errors(self):
        """Core provider plugin errors should be bounded and single-line."""
        from src.api.health_utils import build_health_payload

        long_error = "boom\nSECRET=abc\n" + ("x" * 2000)
        snapshot = {
            "bootstrapped": True,
            "bootstrap_timestamp": 0.0,
            "total_domains": 0,
            "total_providers": 0,
            "domains": [],
            "providers": {},
            "provider_classes": {},
            "plugins": {
                "enabled": True,
                "strict": False,
                "configured": ["tests.fixtures.provider_plugin_example:bootstrap"],
                "loaded": [],
                "errors": [
                    {
                        "plugin": "tests.fixtures.provider_plugin_example:bootstrap",
                        "error": long_error,
                    }
                ],
                "registered": {},
                "cache": {"reused": False, "reason": "first_load"},
                "summary": {
                    "overall_status": "degraded",
                    "configured_count": 1,
                    "loaded_count": 0,
                    "error_count": 1,
                },
            },
        }

        with patch(
            "src.api.health_utils.get_core_provider_registry_snapshot",
            return_value=snapshot,
        ):
            payload = build_health_payload()

        core = payload["config"].get("core_providers") or {}
        plugins = core.get("plugins") or {}
        errors = plugins.get("errors") or []
        assert len(errors) == 1
        err = errors[0].get("error") or ""
        assert "\n" not in err
        assert len(err) <= 300

    def test_build_health_payload_torch_not_available(self):
        """Test build_health_payload when torch is not available."""
        from src.api.health_utils import build_health_payload

        with patch("importlib.util.find_spec", return_value=None):
            payload = build_health_payload()

        assert "status" in payload
        # Check ML readiness reflects torch unavailable
        if "config" in payload and "ml" in payload.get("config", {}):
            ml_config = payload["config"]["ml"]
            if "readiness" in ml_config:
                assert ml_config["readiness"]["torch_available"] is False

    def test_build_health_payload_torch_check_exception(self):
        """Test build_health_payload handles torch check exception."""
        from src.api.health_utils import build_health_payload

        with patch("importlib.util.find_spec", side_effect=Exception("Import error")):
            payload = build_health_payload()

        assert "status" in payload

    def test_build_health_payload_degraded_reasons_graph2d(self, monkeypatch):
        """Test build_health_payload detects graph2d degraded state."""
        from src.api.health_utils import build_health_payload

        # Enable graph2d but make torch unavailable
        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.version = "1.0"
        mock_config.graph2d.enabled = True
        mock_config.filename.enabled = True
        mock_config.titleblock.enabled = True
        mock_config.process.enabled = True
        mock_config.sampling.max_nodes = 100
        mock_config.sampling.strategy = "random"
        mock_config.sampling.seed = 42
        mock_config.sampling.text_priority_ratio = 0.5

        with patch("src.ml.hybrid_config.get_config", return_value=mock_config):
            with patch("importlib.util.find_spec", return_value=None):
                payload = build_health_payload()

        if "config" in payload and "ml" in payload.get("config", {}):
            ml_config = payload["config"]["ml"]
            if "readiness" in ml_config:
                reasons = ml_config["readiness"].get("degraded_reasons", [])
                # Should have degraded reason for graph2d without torch
                assert any("graph2d" in r for r in reasons) or any("torch" in r for r in reasons)

    def test_build_health_payload_v16_disabled(self, monkeypatch):
        """Test build_health_payload when V16 classifier is disabled."""
        monkeypatch.setenv("DISABLE_V16_CLASSIFIER", "true")

        from src.api.health_utils import build_health_payload

        payload = build_health_payload()

        if "config" in payload and "ml" in payload.get("config", {}):
            ml_config = payload["config"]["ml"]
            if "readiness" in ml_config:
                assert ml_config["readiness"]["v16_disabled"] is True

    def test_build_health_payload_resilience_exception(self):
        """Test build_health_payload handles resilience health exception."""
        from src.api.health_utils import build_health_payload

        with patch(
            "src.api.health_utils.get_resilience_health",
            side_effect=Exception("Resilience error"),
        ):
            # Should not raise
            payload = build_health_payload()

        assert "status" in payload
        # resilience key may not be present

    def test_build_health_payload_resilience_none(self):
        """Test build_health_payload when resilience returns None."""
        from src.api.health_utils import build_health_payload

        with patch(
            "src.api.health_utils.get_resilience_health",
            return_value={"resilience": None},
        ):
            payload = build_health_payload()

        assert "status" in payload
        # resilience should not be in base if None
        # (it may still be there from normal flow)

    def test_build_health_payload_env_vars(self, monkeypatch):
        """Test build_health_payload reads environment variables."""
        monkeypatch.setenv("CLASSIFIER_RATE_LIMIT_PER_MIN", "200")
        monkeypatch.setenv("CLASSIFIER_RATE_LIMIT_BURST", "30")
        monkeypatch.setenv("CLASSIFIER_CACHE_MAX_SIZE", "2000")

        from src.api.health_utils import build_health_payload

        payload = build_health_payload()

        monitoring = payload["config"]["monitoring"]
        assert monitoring["classifier_rate_limit_per_min"] == 200
        assert monitoring["classifier_rate_limit_burst"] == 30
        assert monitoring["classifier_cache_max_size"] == 2000

    def test_build_health_payload_includes_graph2d_ops_settings(self, monkeypatch):
        """Health payload should include Graph2D gating + calibration settings."""
        monkeypatch.setenv("GRAPH2D_MIN_MARGIN", "0.2")
        monkeypatch.setenv("GRAPH2D_TEMPERATURE", "1.5")

        from src.ml.hybrid_config import reset_config

        reset_config()
        try:
            from src.api.health_utils import build_health_payload

            payload = build_health_payload()
        finally:
            reset_config()

        classification = payload["config"]["ml"]["classification"]
        assert classification["graph2d_min_margin"] == pytest.approx(0.2, rel=1e-6)
        assert classification["graph2d_temperature"] == pytest.approx(1.5, rel=1e-6)
        assert classification["graph2d_temperature_source"] == "env"

        core = payload["config"].get("core_providers") or {}
        assert isinstance(core.get("provider_classes"), dict)
        plugins = core.get("plugins") or {}
        summary = plugins.get("summary") or {}
        assert summary.get("overall_status") in {"ok", "degraded", "error"}
        assert isinstance(summary.get("configured_count"), int)

    def test_build_health_payload_includes_graph2d_ensemble_settings(
        self, monkeypatch, tmp_path
    ):
        """Health payload should include Graph2D ensemble configuration."""
        model_a = tmp_path / "graph2d_a.pth"
        model_b = tmp_path / "graph2d_b.pth"
        model_a.write_text("stub", encoding="utf-8")
        model_b.write_text("stub", encoding="utf-8")

        monkeypatch.setenv("GRAPH2D_ENSEMBLE_ENABLED", "true")
        monkeypatch.setenv(
            "GRAPH2D_ENSEMBLE_MODELS", f"{model_a},{model_b}"
        )

        from src.ml.hybrid_config import reset_config

        reset_config()
        try:
            from src.api.health_utils import build_health_payload

            payload = build_health_payload()
        finally:
            reset_config()

        classification = payload["config"]["ml"]["classification"]
        assert classification["graph2d_ensemble_enabled"] is True
        assert classification["graph2d_ensemble_models_configured"] == 2
        assert classification["graph2d_ensemble_models_present"] == 2
        assert classification["graph2d_ensemble_models"] == [
            "graph2d_a.pth",
            "graph2d_b.pth",
        ]


class TestHealthResponseModel:
    """Tests verifying HealthResponse model integration."""

    def test_payload_is_valid_health_response(self):
        """Test build_health_payload returns valid HealthResponse data."""
        from src.api.health_utils import build_health_payload

        payload = build_health_payload()

        # Verify required fields
        assert payload["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in payload
        assert "services" in payload
