"""Tests for src/core/resilience/resilience_manager.py to improve coverage.

Covers:
- ResilienceConfig dataclass
- ResilienceHealth dataclass
- ResilienceManager class
- protect method
- get_health method
- reset methods
- auto_scale method
- export/import config
- with_resilience decorator
"""

from __future__ import annotations

import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


class TestResilienceConfigDataclass:
    """Tests for ResilienceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.core.resilience.resilience_manager import ResilienceConfig

        config = ResilienceConfig()

        assert config.circuit_breaker_enabled is True
        assert config.circuit_failure_threshold == 5
        assert config.circuit_recovery_timeout == 60
        assert config.circuit_half_open_max_calls == 3

    def test_rate_limiter_defaults(self):
        """Test rate limiter default config."""
        from src.core.resilience.resilience_manager import ResilienceConfig

        config = ResilienceConfig()

        assert config.rate_limiter_enabled is True
        assert config.rate_limit == 100.0
        assert config.rate_burst == 150
        assert config.rate_algorithm == "token_bucket"

    def test_retry_defaults(self):
        """Test retry policy default config."""
        from src.core.resilience.resilience_manager import ResilienceConfig

        config = ResilienceConfig()

        assert config.retry_enabled is True
        assert config.retry_max_attempts == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 30.0
        assert config.retry_exponential_base == 2.0

    def test_bulkhead_defaults(self):
        """Test bulkhead default config."""
        from src.core.resilience.resilience_manager import ResilienceConfig

        config = ResilienceConfig()

        assert config.bulkhead_enabled is True
        assert config.bulkhead_max_concurrent == 10
        assert config.bulkhead_max_wait == 0.0
        assert config.bulkhead_type == "threadpool"

    def test_global_defaults(self):
        """Test global config defaults."""
        from src.core.resilience.resilience_manager import ResilienceConfig

        config = ResilienceConfig()

        assert config.metrics_enabled is True
        assert config.auto_scaling_enabled is False
        assert config.health_check_interval == 60

    def test_custom_config(self):
        """Test custom configuration values."""
        from src.core.resilience.resilience_manager import ResilienceConfig

        config = ResilienceConfig(
            circuit_breaker_enabled=False, rate_limit=50.0, retry_max_attempts=5
        )

        assert config.circuit_breaker_enabled is False
        assert config.rate_limit == 50.0
        assert config.retry_max_attempts == 5


class TestResilienceHealthDataclass:
    """Tests for ResilienceHealth dataclass."""

    def test_default_health(self):
        """Test default health values."""
        from src.core.resilience.resilience_manager import ResilienceHealth

        health = ResilienceHealth()

        assert health.healthy is True
        assert health.circuit_breakers == {}
        assert health.rate_limiters == {}
        assert health.retry_policies == {}
        assert health.bulkheads == {}
        assert health.overall_status == "healthy"
        assert health.issues == []
        assert health.timestamp == ""

    def test_health_with_issues(self):
        """Test health with issues list."""
        from src.core.resilience.resilience_manager import ResilienceHealth

        health = ResilienceHealth(
            healthy=False,
            overall_status="unhealthy",
            issues=["Circuit breaker open", "High rejection rate"],
        )

        assert health.healthy is False
        assert len(health.issues) == 2


class TestResilienceManagerSingleton:
    """Tests for ResilienceManager singleton pattern."""

    def test_singleton_pattern(self):
        """Test manager is singleton."""
        from src.core.resilience.resilience_manager import ResilienceManager

        # Reset singleton for test
        ResilienceManager._instance = None

        manager1 = ResilienceManager()
        manager2 = ResilienceManager()

        assert manager1 is manager2

    def test_initialized_once(self):
        """Test manager initializes only once."""
        from src.core.resilience.resilience_manager import ResilienceManager

        # Reset singleton
        ResilienceManager._instance = None

        manager = ResilienceManager()
        assert hasattr(manager, "initialized")
        assert manager.initialized is True


class TestResilienceManagerConfigure:
    """Tests for configure method."""

    def test_configure_updates_config(self):
        """Test configure updates manager config."""
        from src.core.resilience.resilience_manager import ResilienceConfig, ResilienceManager

        # Reset singleton
        ResilienceManager._instance = None
        manager = ResilienceManager()

        new_config = ResilienceConfig(rate_limit=50.0)
        manager.configure(new_config)

        assert manager.config.rate_limit == 50.0


class TestGetCircuitBreaker:
    """Tests for get_circuit_breaker method."""

    def test_creates_new_circuit_breaker(self):
        """Test creates new circuit breaker if not exists."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.circuit_breakers.clear()

        cb = manager.get_circuit_breaker("test_cb")

        assert cb is not None
        assert "test_cb" in manager.circuit_breakers

    def test_returns_existing_circuit_breaker(self):
        """Test returns existing circuit breaker."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.circuit_breakers.clear()

        cb1 = manager.get_circuit_breaker("test_cb")
        cb2 = manager.get_circuit_breaker("test_cb")

        assert cb1 is cb2

    def test_custom_threshold(self):
        """Test circuit breaker with custom threshold."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.circuit_breakers.clear()

        cb = manager.get_circuit_breaker("test_cb", failure_threshold=10)

        assert cb.failure_threshold == 10


class TestGetRateLimiter:
    """Tests for get_rate_limiter method."""

    def test_creates_new_rate_limiter(self):
        """Test creates new rate limiter if not exists."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.rate_limiters.clear()

        rl = manager.get_rate_limiter("test_rl")

        assert rl is not None
        assert "test_rl" in manager.rate_limiters

    def test_custom_rate(self):
        """Test rate limiter with custom rate."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.rate_limiters.clear()

        rl = manager.get_rate_limiter("test_rl", rate=50.0)

        assert rl.rate == 50.0


class TestGetRetryPolicy:
    """Tests for get_retry_policy method."""

    def test_creates_new_retry_policy(self):
        """Test creates new retry policy if not exists."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.retry_policies.clear()

        rp = manager.get_retry_policy("test_rp")

        assert rp is not None
        assert "test_rp" in manager.retry_policies

    def test_custom_max_attempts(self):
        """Test retry policy with custom max attempts."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.retry_policies.clear()

        rp = manager.get_retry_policy("test_rp", max_attempts=5)

        assert rp.max_attempts == 5


class TestGetBulkhead:
    """Tests for get_bulkhead method."""

    def test_creates_new_bulkhead(self):
        """Test creates new bulkhead if not exists."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.bulkheads.clear()

        bh = manager.get_bulkhead("test_bh")

        assert bh is not None
        assert "test_bh" in manager.bulkheads

    def test_custom_max_concurrent(self):
        """Test bulkhead with custom max concurrent."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.bulkheads.clear()

        bh = manager.get_bulkhead("test_bh", max_concurrent_calls=20)

        assert bh.max_concurrent_calls == 20


class TestProtectMethod:
    """Tests for protect method."""

    def test_protect_with_rate_limiter_rejection(self):
        """Test protect raises on rate limit exceeded."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.rate_limiters.clear()

        # Get rate limiter and drain it
        rl = manager.get_rate_limiter("test", rate=0.001, burst=1)

        # First call should succeed, drain tokens
        def func():
            return "success"

        # Drain the bucket
        for _ in range(5):
            try:
                rl.allow_request("test")
            except Exception:
                pass

        with pytest.raises(Exception, match="Rate limit exceeded"):
            manager.protect("test", func, use_bulkhead=False, use_retry=False)

    def test_protect_disabled_components(self):
        """Test protect with all components disabled."""
        from src.core.resilience.resilience_manager import ResilienceConfig, ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()

        # Disable all components
        manager.config = ResilienceConfig(
            circuit_breaker_enabled=False,
            rate_limiter_enabled=False,
            retry_enabled=False,
            bulkhead_enabled=False,
        )

        def func():
            return "success"

        result = manager.protect(
            "test",
            func,
            use_circuit_breaker=False,
            use_rate_limiter=False,
            use_retry=False,
            use_bulkhead=False,
        )

        assert result == "success"


class TestGetHealth:
    """Tests for get_health method."""

    def test_healthy_state(self):
        """Test healthy state with no issues."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.circuit_breakers.clear()
        manager.rate_limiters.clear()
        manager.retry_policies.clear()
        manager.bulkheads.clear()

        health = manager.get_health()

        assert health.healthy is True
        assert health.overall_status == "healthy"
        assert len(health.issues) == 0

    def test_health_timestamp(self):
        """Test health includes timestamp."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()

        health = manager.get_health()

        assert health.timestamp != ""


class TestResetComponent:
    """Tests for reset_component method."""

    def test_reset_circuit_breaker(self):
        """Test reset circuit breaker component."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.circuit_breakers.clear()

        cb = manager.get_circuit_breaker("test_cb")
        manager.reset_component("circuit_breaker", "test_cb")

        # Should not raise
        assert True

    def test_reset_rate_limiter(self):
        """Test reset rate limiter component."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.rate_limiters.clear()

        rl = manager.get_rate_limiter("test_rl")
        manager.reset_component("rate_limiter", "test_rl")

        assert True

    def test_reset_retry_policy(self):
        """Test reset retry policy component."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.retry_policies.clear()

        rp = manager.get_retry_policy("test_rp")
        manager.reset_component("retry_policy", "test_rp")

        assert True

    def test_reset_bulkhead(self):
        """Test reset bulkhead component."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.bulkheads.clear()

        bh = manager.get_bulkhead("test_bh")
        manager.reset_component("bulkhead", "test_bh")

        assert True

    def test_reset_nonexistent_component(self):
        """Test reset nonexistent component does nothing."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()

        # Should not raise
        manager.reset_component("circuit_breaker", "nonexistent")


class TestResetAll:
    """Tests for reset_all method."""

    def test_reset_all_components(self):
        """Test reset all components."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()

        # Create some components
        manager.get_circuit_breaker("cb1")
        manager.get_rate_limiter("rl1")
        manager.get_retry_policy("rp1")
        manager.get_bulkhead("bh1")

        # Should not raise
        manager.reset_all()


class TestAutoScale:
    """Tests for auto_scale method."""

    def test_auto_scale_disabled(self):
        """Test auto_scale does nothing when disabled."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.config.auto_scaling_enabled = False

        # Should return early
        manager.auto_scale()


class TestExportConfig:
    """Tests for export_config method."""

    def test_export_config_structure(self):
        """Test export_config returns correct structure."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.circuit_breakers.clear()
        manager.rate_limiters.clear()
        manager.retry_policies.clear()
        manager.bulkheads.clear()

        config = manager.export_config()

        assert "global_config" in config
        assert "circuit_breakers" in config
        assert "rate_limiters" in config
        assert "retry_policies" in config
        assert "bulkheads" in config

    def test_export_with_components(self):
        """Test export with created components."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.circuit_breakers.clear()
        manager.rate_limiters.clear()

        manager.get_circuit_breaker("test_cb")
        manager.get_rate_limiter("test_rl")

        config = manager.export_config()

        assert "test_cb" in config["circuit_breakers"]
        assert "test_rl" in config["rate_limiters"]


class TestImportConfig:
    """Tests for import_config method."""

    def test_import_global_config(self):
        """Test import global config."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()

        config_dict = {"global_config": {"rate_limit": 50.0, "retry_max_attempts": 5}}

        manager.import_config(config_dict)

        assert manager.config.rate_limit == 50.0
        assert manager.config.retry_max_attempts == 5


class TestGetMetricsSummary:
    """Tests for get_metrics_summary method."""

    def test_get_metrics_summary(self):
        """Test get_metrics_summary returns dict."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()

        summary = manager.get_metrics_summary()

        assert isinstance(summary, dict)


class TestWithResilienceDecorator:
    """Tests for with_resilience decorator."""

    def test_decorator_wraps_function(self):
        """Test decorator wraps function."""
        from src.core.resilience.resilience_manager import ResilienceManager, with_resilience

        ResilienceManager._instance = None

        @with_resilience(name="test_func")
        def my_function():
            return "result"

        assert hasattr(my_function, "resilience_name")
        assert my_function.resilience_name == "test_func"

    def test_decorator_auto_name(self):
        """Test decorator generates name from function."""
        from src.core.resilience.resilience_manager import ResilienceManager, with_resilience

        ResilienceManager._instance = None

        @with_resilience()
        def another_function():
            return "result"

        assert hasattr(another_function, "resilience_name")
        assert "another_function" in another_function.resilience_name


class TestGlobalInstance:
    """Tests for global resilience_manager instance."""

    def test_global_instance_exists(self):
        """Test global resilience_manager instance exists."""
        from src.core.resilience.resilience_manager import resilience_manager

        assert resilience_manager is not None

    def test_global_instance_is_manager(self):
        """Test global instance is ResilienceManager."""
        from src.core.resilience.resilience_manager import ResilienceManager, resilience_manager

        assert isinstance(resilience_manager, ResilienceManager)


class TestAutoScaleEnabled:
    """Tests for auto_scale method when enabled."""

    def test_auto_scale_enabled_adjusts_circuit_breaker(self):
        """Test auto_scale adjusts circuit breaker threshold on low failure rate."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.config.auto_scaling_enabled = True
        manager.circuit_breakers.clear()

        # Create a circuit breaker with mocked health
        cb = manager.get_circuit_breaker("test_cb", failure_threshold=5)
        initial_threshold = cb.failure_threshold

        # Mock very low failure rate
        with patch.object(
            cb,
            "get_health",
            return_value={"state": "closed", "failure_rate": 0.005, "failure_count": 0},  # < 0.01
        ):
            manager.auto_scale()

        # Threshold should increase
        assert cb.failure_threshold > initial_threshold

    def test_auto_scale_enabled_adjusts_rate_limiter_high_rejection(self):
        """Test auto_scale decreases rate on high rejection rate."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.config.auto_scaling_enabled = True
        manager.rate_limiters.clear()

        # Create a rate limiter
        rl = manager.get_rate_limiter("test_rl", rate=100.0)
        initial_rate = rl.rate

        # Mock high rejection rate
        with patch.object(
            rl, "get_health", return_value={"rejection_rate": 0.25, "current_rate": 100.0}  # > 0.2
        ):
            with patch.object(rl, "update_rate") as mock_update:
                manager.auto_scale()
                # Rate should be reduced by 10%
                mock_update.assert_called_once()
                new_rate = mock_update.call_args[0][0]
                assert new_rate < initial_rate

    def test_auto_scale_enabled_adjusts_rate_limiter_low_rejection(self):
        """Test auto_scale increases rate on low rejection rate."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.config.auto_scaling_enabled = True
        manager.rate_limiters.clear()

        # Create a rate limiter
        rl = manager.get_rate_limiter("test_rl", rate=100.0)
        initial_rate = rl.rate

        # Mock low rejection rate
        with patch.object(
            rl,
            "get_health",
            return_value={"rejection_rate": 0.005, "current_rate": 100.0},  # < 0.01
        ):
            with patch.object(rl, "update_rate") as mock_update:
                manager.auto_scale()
                # Rate should be increased by 10%
                mock_update.assert_called_once()
                new_rate = mock_update.call_args[0][0]
                assert new_rate > initial_rate

    def test_auto_scale_enabled_adjusts_bulkhead_high_utilization(self):
        """Test auto_scale increases bulkhead capacity on high utilization."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.config.auto_scaling_enabled = True
        manager.bulkheads.clear()

        # Create a bulkhead
        bh = manager.get_bulkhead("test_bh", max_concurrent_calls=10)

        # Mock high utilization
        with patch.object(
            bh, "get_health", return_value={"utilization": 0.95, "active_calls": 9}  # > 0.9
        ):
            with patch.object(bh, "resize") as mock_resize:
                manager.auto_scale()
                mock_resize.assert_called_once()
                new_capacity = mock_resize.call_args[0][0]
                assert new_capacity == 12  # 10 + 2

    def test_auto_scale_enabled_adjusts_bulkhead_low_utilization(self):
        """Test auto_scale decreases bulkhead capacity on low utilization."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.config.auto_scaling_enabled = True
        manager.bulkheads.clear()

        # Create a bulkhead
        bh = manager.get_bulkhead("test_bh", max_concurrent_calls=10)

        # Mock low utilization
        with patch.object(
            bh, "get_health", return_value={"utilization": 0.25, "active_calls": 2}  # < 0.3
        ):
            with patch.object(bh, "resize") as mock_resize:
                manager.auto_scale()
                mock_resize.assert_called_once()
                new_capacity = mock_resize.call_args[0][0]
                assert new_capacity == 9  # 10 - 1


class TestImportConfigComponents:
    """Tests for import_config with component configurations."""

    def test_import_config_with_circuit_breaker_config(self):
        """Test import_config imports circuit breaker config."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()

        config_dict = {
            "global_config": {"rate_limit": 50.0},
            "circuit_breakers": {"api_cb": {"failure_threshold": 10, "recovery_timeout": 120}},
        }

        with patch("src.core.resilience.resilience_manager.logger") as mock_logger:
            manager.import_config(config_dict)
            # Should log for each imported component
            assert mock_logger.info.called

    def test_import_config_with_all_components(self):
        """Test import_config imports all component types."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()

        config_dict = {
            "circuit_breakers": {"cb1": {"failure_threshold": 5}},
            "rate_limiters": {"rl1": {"rate": 50.0}},
            "retry_policies": {"rp1": {"max_attempts": 5}},
            "bulkheads": {"bh1": {"max_concurrent_calls": 20}},
        }

        with patch("src.core.resilience.resilience_manager.logger") as mock_logger:
            manager.import_config(config_dict)
            # Should log for each component type
            assert mock_logger.info.call_count == 4


class TestProtectPathsCoverage:
    """Tests for protect method different execution paths."""

    def test_protect_with_bulkhead_enabled(self):
        """Test protect uses bulkhead when enabled."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.bulkheads.clear()
        manager.rate_limiters.clear()

        def func():
            return "bulkhead_result"

        # Protect with bulkhead - need to mock rate limiter allow
        rl = manager.get_rate_limiter("test")
        with patch.object(rl, "allow_request", return_value=True):
            result = manager.protect(
                "test",
                func,
                use_circuit_breaker=False,
                use_rate_limiter=True,
                use_retry=False,
                use_bulkhead=True,
            )

        assert result == "bulkhead_result"

    def test_protect_with_circuit_breaker_only(self):
        """Test protect with only circuit breaker enabled."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.circuit_breakers.clear()
        manager.rate_limiters.clear()

        def func():
            return "cb_result"

        # Disable rate limiter to avoid rejection
        manager.config.rate_limiter_enabled = False

        result = manager.protect(
            "test",
            func,
            use_circuit_breaker=True,
            use_rate_limiter=False,
            use_retry=False,
            use_bulkhead=False,
        )

        assert result == "cb_result"

    def test_protect_with_retry_and_circuit_breaker(self):
        """Test protect with both retry and circuit breaker."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.circuit_breakers.clear()
        manager.retry_policies.clear()
        manager.rate_limiters.clear()

        call_count = 0

        def func():
            nonlocal call_count
            call_count += 1
            return "retry_cb_result"

        # Disable rate limiter
        manager.config.rate_limiter_enabled = False

        result = manager.protect(
            "test",
            func,
            use_circuit_breaker=True,
            use_rate_limiter=False,
            use_retry=True,
            use_bulkhead=False,
        )

        assert result == "retry_cb_result"

    def test_protect_with_retry_only(self):
        """Test protect with only retry enabled."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.retry_policies.clear()
        manager.rate_limiters.clear()

        def func():
            return "retry_result"

        # Disable rate limiter
        manager.config.rate_limiter_enabled = False
        manager.config.circuit_breaker_enabled = False

        result = manager.protect(
            "test",
            func,
            use_circuit_breaker=False,
            use_rate_limiter=False,
            use_retry=True,
            use_bulkhead=False,
        )

        assert result == "retry_result"


class TestGetHealthWithIssues:
    """Tests for get_health detecting issues."""

    def test_health_detects_open_circuit_breaker(self):
        """Test get_health detects open circuit breaker."""
        from src.core.resilience.circuit_breaker import CircuitState
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.circuit_breakers.clear()

        cb = manager.get_circuit_breaker("test_cb")

        # Mock open state
        with patch.object(
            cb,
            "get_health",
            return_value={
                "state": CircuitState.OPEN.value,
                "failure_rate": 0.5,
                "failure_count": 5,
            },
        ):
            health = manager.get_health()

        assert len(health.issues) > 0
        assert any("OPEN" in issue for issue in health.issues)

    def test_health_detects_high_rejection_rate(self):
        """Test get_health detects high rejection rate."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.rate_limiters.clear()

        rl = manager.get_rate_limiter("test_rl")

        # Mock high rejection rate
        with patch.object(
            rl, "get_health", return_value={"rejection_rate": 0.15, "current_rate": 100.0}  # > 0.1
        ):
            health = manager.get_health()

        assert len(health.issues) > 0
        assert any("rejection rate" in issue for issue in health.issues)

    def test_health_detects_low_retry_success_rate(self):
        """Test get_health detects low retry success rate."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.retry_policies.clear()

        rp = manager.get_retry_policy("test_rp")

        # Mock low success rate
        with patch.object(
            rp, "get_health", return_value={"success_rate": 0.4, "total_attempts": 100}  # < 0.5
        ):
            health = manager.get_health()

        assert len(health.issues) > 0
        assert any("success rate" in issue for issue in health.issues)

    def test_health_detects_high_bulkhead_utilization(self):
        """Test get_health detects high bulkhead utilization."""
        from src.core.resilience.resilience_manager import ResilienceManager

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.bulkheads.clear()

        bh = manager.get_bulkhead("test_bh")

        # Mock high utilization
        with patch.object(
            bh, "get_health", return_value={"utilization": 0.95, "active_calls": 9}  # > 0.9
        ):
            health = manager.get_health()

        assert len(health.issues) > 0
        assert any("utilization" in issue for issue in health.issues)


class TestWithResilienceDecoratorExecution:
    """Tests for with_resilience decorator execution."""

    def test_decorator_executes_through_manager(self):
        """Test decorator executes function through resilience manager."""
        from src.core.resilience.resilience_manager import ResilienceManager, with_resilience

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.config.rate_limiter_enabled = False
        manager.config.circuit_breaker_enabled = False
        manager.config.retry_enabled = False
        manager.config.bulkhead_enabled = False

        @with_resilience(name="test_decorated")
        def decorated_function(x, y):
            return x + y

        result = decorated_function(3, 5)

        assert result == 8

    def test_decorator_auto_name_execution(self):
        """Test decorator with auto-generated name executes correctly."""
        from src.core.resilience.resilience_manager import ResilienceManager, with_resilience

        ResilienceManager._instance = None
        manager = ResilienceManager()
        manager.config.rate_limiter_enabled = False
        manager.config.circuit_breaker_enabled = False
        manager.config.retry_enabled = False
        manager.config.bulkhead_enabled = False

        @with_resilience()
        def auto_named_function():
            return "auto_named_result"

        result = auto_named_function()

        assert result == "auto_named_result"
