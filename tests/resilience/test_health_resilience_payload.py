"""
Test Health Check Resilience Payload
测试健康检查中的韧性层状态信息
"""

import json
from datetime import datetime, timedelta

import pytest

from src.api.health_resilience import (
    CircuitBreakerStatus,
    CircuitState,
    RateLimiterStatus,
    ResilienceHealthCollector,
    get_resilience_health,
)


class TestResilienceHealth:
    """测试韧性层健康状态"""

    def test_empty_health_status(self):
        """测试空状态"""
        collector = ResilienceHealthCollector()
        status = collector.get_health_status()

        assert "resilience" in status
        assert status["resilience"]["status"] == "healthy"
        assert status["resilience"]["circuit_breakers"] == {}
        assert status["resilience"]["rate_limiters"] == {}

    def test_circuit_breaker_registration(self):
        """测试熔断器注册"""
        collector = ResilienceHealthCollector()

        # 模拟熔断器对象
        class MockCircuitBreaker:
            state = CircuitState.CLOSED
            failure_count = 2
            success_count = 10
            failure_threshold = 5
            last_failure_time = datetime.now()
            recovery_timeout = 60

        breaker = MockCircuitBreaker()
        collector.register_circuit_breaker("test_breaker", breaker)

        status = collector.get_health_status()
        cb_status = status["resilience"]["circuit_breakers"]["test_breaker"]

        assert cb_status["state"] == "closed"
        assert cb_status["failure_count"] == 2
        assert cb_status["success_count"] == 10
        assert cb_status["threshold"] == 5

    def test_open_circuit_degrades_status(self):
        """测试开路熔断器导致降级状态"""
        collector = ResilienceHealthCollector()

        # 添加一个开路的熔断器
        cb = CircuitBreakerStatus(
            name="failing_service",
            state=CircuitState.OPEN,
            failure_count=6,
            success_count=0,
            failure_threshold=5,
        )
        collector.circuit_breakers["failing_service"] = cb

        status = collector.get_health_status()
        assert status["resilience"]["status"] == "degraded"

    def test_rate_limiter_registration(self):
        """测试限流器注册"""
        collector = ResilienceHealthCollector()

        # 模拟限流器
        class MockRateLimiter:
            tokens = 45.5
            capacity = 100
            rate = 10
            algorithm = "token_bucket"

        limiter = MockRateLimiter()
        collector.register_rate_limiter("api_limiter", limiter)

        status = collector.get_health_status()
        rl_status = status["resilience"]["rate_limiters"]["api_limiter"]

        assert rl_status["current_tokens"] == 45.5
        assert rl_status["max_tokens"] == 100
        assert rl_status["utilization"] == 0.54  # 1 - (45.5/100)

    def test_stressed_status_with_exhausted_limiters(self):
        """测试限流器耗尽导致压力状态"""
        collector = ResilienceHealthCollector()

        # 添加一个几乎耗尽的限流器
        rl = RateLimiterStatus(
            name="exhausted_limiter",
            current_tokens=5,  # <10% of max
            max_tokens=100,
            refill_rate=10,
        )
        collector.rate_limiters["exhausted_limiter"] = rl

        status = collector.get_health_status()
        assert status["resilience"]["status"] == "stressed"

    def test_adaptive_status_update(self):
        """测试自适应状态更新"""
        collector = ResilienceHealthCollector()

        collector.update_adaptive_status(enabled=True, rate_multiplier=0.75, error_rate=0.025)

        status = collector.get_health_status()
        adaptive = status["resilience"]["adaptive"]

        assert adaptive["enabled"] is True
        assert adaptive["rate_multiplier"] == 0.75
        assert adaptive["actual_error_rate"] == 0.025
        assert adaptive["adjustments_made"] == 1

    def test_metrics_calculation(self):
        """测试指标计算"""
        collector = ResilienceHealthCollector()

        # 添加多个组件
        collector.circuit_breakers = {
            "cb1": CircuitBreakerStatus("cb1", CircuitState.CLOSED, 0, 10, 5),
            "cb2": CircuitBreakerStatus("cb2", CircuitState.OPEN, 5, 0, 5),
        }

        collector.rate_limiters = {
            "rl1": RateLimiterStatus("rl1", 80, 100, 10),
            "rl2": RateLimiterStatus("rl2", 40, 100, 10),
        }

        status = collector.get_health_status()
        metrics = status["resilience"]["metrics"]

        assert metrics["circuit_breaker_open_ratio"] == 0.5  # 1/2 open
        assert metrics["rate_limiter_avg_utilization"] == 0.4  # (0.2+0.6)/2

    def test_health_payload_json_serializable(self):
        """测试健康状态可JSON序列化"""
        collector = ResilienceHealthCollector()

        # 添加各种组件
        collector.register_circuit_breaker(
            "test_cb",
            type(
                "",
                (),
                {
                    "state": CircuitState.HALF_OPEN,
                    "failure_count": 3,
                    "success_count": 7,
                    "failure_threshold": 5,
                    "last_failure_time": datetime.now(),
                    "recovery_timeout": 30,
                },
            ),
        )

        collector.register_rate_limiter(
            "test_rl",
            type(
                "", (), {"tokens": 75, "capacity": 100, "rate": 10, "algorithm": "sliding_window"}
            ),
        )

        collector.update_adaptive_status(True, 1.2, 0.008)

        status = collector.get_health_status()

        # 确保可以序列化为JSON
        json_str = json.dumps(status, default=str)
        assert json_str

        # 确保可以反序列化
        parsed = json.loads(json_str)
        assert parsed["resilience"]["status"] in ["healthy", "degraded", "stressed"]

    def test_complex_scenario(self):
        """测试复杂场景"""
        collector = ResilienceHealthCollector()

        # 添加多个不同状态的组件
        collector.circuit_breakers = {
            "service_a": CircuitBreakerStatus("service_a", CircuitState.CLOSED, 1, 100, 5),
            "service_b": CircuitBreakerStatus(
                "service_b", CircuitState.OPEN, 10, 0, 5, last_failure_time=datetime.now()
            ),
            "service_c": CircuitBreakerStatus("service_c", CircuitState.HALF_OPEN, 4, 2, 5),
        }

        collector.rate_limiters = {
            "api_v1": RateLimiterStatus(
                "api_v1", 90, 100, 10, requests_accepted=1000, requests_rejected=5
            ),
            "api_v2": RateLimiterStatus(
                "api_v2", 3, 50, 5, requests_accepted=500, requests_rejected=50
            ),
        }

        collector.update_adaptive_status(True, 0.8, 0.015)

        status = collector.get_health_status()

        # 验证整体状态（有开路熔断器，应该是degraded）
        assert status["resilience"]["status"] == "degraded"

        # 验证熔断器细节
        assert len(status["resilience"]["circuit_breakers"]) == 3
        assert status["resilience"]["circuit_breakers"]["service_b"]["state"] == "open"

        # 验证限流器细节
        assert len(status["resilience"]["rate_limiters"]) == 2
        assert status["resilience"]["rate_limiters"]["api_v2"]["utilization"] == 0.94

        # 验证自适应状态
        assert status["resilience"]["adaptive"]["enabled"] is True
        assert status["resilience"]["adaptive"]["rate_multiplier"] == 0.8

        # 验证指标
        metrics = status["resilience"]["metrics"]
        assert metrics["circuit_breaker_open_ratio"] == pytest.approx(0.33, 0.01)


class TestAdditionalCoverage:
    """Additional tests to improve coverage."""

    def test_register_adaptive_rate_limiter(self):
        """Test register_adaptive_rate_limiter method."""
        collector = ResilienceHealthCollector()

        class MockAdaptiveLimiter:
            def get_status(self):
                return {
                    "phase": "normal",
                    "base_rate": 100,
                    "current_rate": 80,
                    "error_ema": 0.01,
                    "tokens_available": 50,
                    "consecutive_failures": 2,
                    "in_cooldown": False,
                    "recent_adjustments": ["adj1", "adj2"],
                }

        limiter = MockAdaptiveLimiter()
        collector.register_adaptive_rate_limiter("adaptive_test", limiter)

        assert "adaptive_test" in collector.adaptive_rate_limiters
        assert collector.adaptive_rate_limiters["adaptive_test"] is limiter

    def test_format_adaptive_rate_limiters_with_get_status(self):
        """Test _format_adaptive_rate_limiters when limiter has get_status."""
        collector = ResilienceHealthCollector()

        class MockAdaptiveLimiter:
            def get_status(self):
                return {
                    "phase": "recovery",
                    "base_rate": 200,
                    "current_rate": 150,
                    "error_ema": 0.02,
                    "tokens_available": 75,
                    "consecutive_failures": 0,
                    "in_cooldown": True,
                    "recent_adjustments": [],
                }

        collector.register_adaptive_rate_limiter("limiter1", MockAdaptiveLimiter())

        status = collector.get_health_status()
        adaptive_rl = status["resilience"]["adaptive_rate_limit"]

        assert "limiter1" in adaptive_rl
        assert adaptive_rl["limiter1"]["phase"] == "recovery"
        assert adaptive_rl["limiter1"]["base_rate"] == 200
        assert adaptive_rl["limiter1"]["current_rate"] == 150
        assert adaptive_rl["limiter1"]["in_cooldown"] is True

    def test_format_adaptive_rate_limiters_without_get_status(self):
        """Test _format_adaptive_rate_limiters when limiter has no get_status."""
        collector = ResilienceHealthCollector()

        class MockLimiterNoStatus:
            pass

        collector.register_adaptive_rate_limiter("no_status", MockLimiterNoStatus())

        status = collector.get_health_status()
        adaptive_rl = status["resilience"]["adaptive_rate_limit"]

        # Should not include limiter without get_status
        assert "no_status" not in adaptive_rl

    def test_format_retry_policies_with_data(self):
        """Test _format_retry_policies with actual retry policy data."""
        from src.api.health_resilience import RetryPolicyStatus

        collector = ResilienceHealthCollector()

        # Add retry policies
        collector.retry_policies = {
            "api_retry": RetryPolicyStatus(
                name="api_retry",
                max_attempts=3,
                current_attempt=1,
                strategy="exponential",
                base_delay=1.0,
                total_retries=100,
                successful_retries=85,
            ),
            "db_retry": RetryPolicyStatus(
                name="db_retry",
                max_attempts=5,
                current_attempt=0,
                strategy="linear",
                base_delay=0.5,
                total_retries=0,  # No retries yet
                successful_retries=0,
            ),
        }

        status = collector.get_health_status()
        rp = status["resilience"]["retry_policies"]

        assert "api_retry" in rp
        assert rp["api_retry"]["max_attempts"] == 3
        assert rp["api_retry"]["strategy"] == "exponential"
        assert rp["api_retry"]["total_retries"] == 100
        assert rp["api_retry"]["success_rate"] == 0.85

        assert "db_retry" in rp
        assert rp["db_retry"]["success_rate"] == 0  # No retries, so 0

    def test_format_bulkheads_with_data(self):
        """Test _format_bulkheads with actual bulkhead data."""
        from src.api.health_resilience import BulkheadStatus

        collector = ResilienceHealthCollector()

        collector.bulkheads = {
            "api_bulkhead": BulkheadStatus(
                name="api_bulkhead",
                max_concurrent=10,
                current_concurrent=7,
                queued_calls=3,
                rejected_calls=5,
                pool_type="thread",
            ),
            "db_bulkhead": BulkheadStatus(
                name="db_bulkhead",
                max_concurrent=5,
                current_concurrent=5,
                queued_calls=10,
                rejected_calls=20,
                pool_type="process",
            ),
        }

        status = collector.get_health_status()
        bh = status["resilience"]["bulkheads"]

        assert "api_bulkhead" in bh
        assert bh["api_bulkhead"]["max_concurrent"] == 10
        assert bh["api_bulkhead"]["current_concurrent"] == 7
        assert bh["api_bulkhead"]["utilization"] == 0.7
        assert bh["api_bulkhead"]["queued"] == 3
        assert bh["api_bulkhead"]["rejected"] == 5

        assert "db_bulkhead" in bh
        assert bh["db_bulkhead"]["utilization"] == 1.0

    def test_collect_metrics_with_adaptive_limiters(self):
        """Test _collect_metrics includes adaptive limiter phases."""
        collector = ResilienceHealthCollector()

        class MockAdaptiveLimiter:
            def __init__(self, phase):
                self._phase = phase

            def get_status(self):
                return {"phase": self._phase}

        collector.register_adaptive_rate_limiter("limiter1", MockAdaptiveLimiter("normal"))
        collector.register_adaptive_rate_limiter("limiter2", MockAdaptiveLimiter("normal"))
        collector.register_adaptive_rate_limiter("limiter3", MockAdaptiveLimiter("recovery"))

        status = collector.get_health_status()
        metrics = status["resilience"]["metrics"]

        assert "adaptive_limiter_phases" in metrics
        assert metrics["adaptive_limiter_phases"]["normal"] == 2
        assert metrics["adaptive_limiter_phases"]["recovery"] == 1

    def test_collect_metrics_bulkhead_rejections(self):
        """Test _collect_metrics sums bulkhead rejections."""
        from src.api.health_resilience import BulkheadStatus

        collector = ResilienceHealthCollector()

        collector.bulkheads = {
            "bh1": BulkheadStatus("bh1", 10, 5, 0, 10),
            "bh2": BulkheadStatus("bh2", 10, 5, 0, 15),
        }

        status = collector.get_health_status()
        metrics = status["resilience"]["metrics"]

        assert metrics["bulkhead_rejections"] == 25


class TestUpdateResilienceMetrics:
    """Tests for update_resilience_metrics function."""

    def test_update_resilience_metrics_circuit_breaker(self):
        """Test update_resilience_metrics with circuit_breaker component."""
        from src.api.health_resilience import update_resilience_metrics

        # Should not raise
        update_resilience_metrics("circuit_breaker", name="test", state="open")

    def test_update_resilience_metrics_rate_limiter(self):
        """Test update_resilience_metrics with rate_limiter component."""
        from src.api.health_resilience import update_resilience_metrics

        # Should not raise
        update_resilience_metrics("rate_limiter", name="test", tokens=50)

    def test_update_resilience_metrics_adaptive(self):
        """Test update_resilience_metrics with adaptive component."""
        from src.api.health_resilience import resilience_collector, update_resilience_metrics

        # Reset state
        resilience_collector.adaptive_status.adjustment_count = 0

        update_resilience_metrics(
            "adaptive", enabled=True, rate_multiplier=0.9, error_rate=0.02
        )

        assert resilience_collector.adaptive_status.enabled is True
        assert resilience_collector.adaptive_status.current_rate_multiplier == 0.9
        assert resilience_collector.adaptive_status.actual_error_rate == 0.02
        assert resilience_collector.adaptive_status.adjustment_count == 1

    def test_update_resilience_metrics_unknown_component(self):
        """Test update_resilience_metrics with unknown component."""
        from src.api.health_resilience import update_resilience_metrics

        # Should not raise for unknown component
        update_resilience_metrics("unknown_component", foo="bar")


class TestIntegration:
    """集成测试"""

    def test_global_health_function(self):
        """测试全局健康函数"""
        from src.api.health_resilience import resilience_collector

        # 清空状态
        resilience_collector.circuit_breakers.clear()
        resilience_collector.rate_limiters.clear()

        # 添加一些状态
        resilience_collector.circuit_breakers["test"] = CircuitBreakerStatus(
            "test", CircuitState.CLOSED, 0, 10, 5
        )

        # 获取健康状态
        health = get_resilience_health()

        assert health["resilience"]["status"] == "healthy"
        assert "test" in health["resilience"]["circuit_breakers"]

    @pytest.mark.parametrize(
        "state,expected_status",
        [
            (CircuitState.CLOSED, "healthy"),
            (CircuitState.OPEN, "degraded"),
            (CircuitState.HALF_OPEN, "healthy"),
        ],
    )
    def test_status_mapping(self, state, expected_status):
        """测试状态映射"""
        collector = ResilienceHealthCollector()
        collector.circuit_breakers["test"] = CircuitBreakerStatus("test", state, 0, 0, 5)

        status = collector.get_health_status()
        assert status["resilience"]["status"] == expected_status
