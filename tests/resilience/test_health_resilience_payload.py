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
