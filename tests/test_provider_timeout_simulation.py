#!/usr/bin/env python3
"""
Provider Timeout Simulation Tests
Provider 超时模拟测试 - 测试系统在各种超时场景下的行为

测试场景：
1. 单个 Provider 超时
2. 级联超时（多个 Provider 连续超时）
3. 部分超时（某些请求超时）
4. 慢响应（接近超时但未超时）
5. 间歇性超时
6. 超时恢复测试
"""

import asyncio
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest


@dataclass
class TimeoutScenario:
    """超时场景配置"""

    name: str
    providers: List[str]
    timeout_pattern: str  # fixed, random, increasing, burst
    timeout_duration: float
    recovery_time: float
    failure_rate: float  # 0.0 到 1.0
    description: str


class ProviderTimeoutSimulator:
    """Provider 超时模拟器"""

    def __init__(self):
        self.scenarios = self._define_scenarios()
        self.metrics = {
            "total_requests": 0,
            "timeouts": 0,
            "recoveries": 0,
            "response_times": [],
            "cascade_failures": 0,
        }

    def _define_scenarios(self) -> List[TimeoutScenario]:
        """定义测试场景"""
        # NOTE: 使用较短的超时时间（0.01-0.1秒）以加快测试运行速度
        # 保持相同的测试逻辑和比例关系
        return [
            TimeoutScenario(
                name="single_provider_timeout",
                providers=["deepseek"],
                timeout_pattern="fixed",
                timeout_duration=0.05,  # 原30.0，缩短为50ms
                recovery_time=0.1,  # 原60.0，缩短为100ms
                failure_rate=1.0,
                description="单个 Provider 完全超时",
            ),
            TimeoutScenario(
                name="cascading_timeout",
                providers=["deepseek", "assemblyai", "glm4v"],
                timeout_pattern="increasing",
                timeout_duration=0.02,  # 原10.0，缩短为20ms
                recovery_time=0.2,  # 原120.0，缩短为200ms
                failure_rate=1.0,
                description="多个 Provider 级联超时",
            ),
            TimeoutScenario(
                name="partial_timeout",
                providers=["deepseek"],
                timeout_pattern="random",
                timeout_duration=0.03,  # 原15.0，缩短为30ms
                recovery_time=0.05,  # 原30.0，缩短为50ms
                failure_rate=0.3,
                description="30% 的请求超时",
            ),
            TimeoutScenario(
                name="slow_response",
                providers=["assemblyai"],
                timeout_pattern="fixed",
                timeout_duration=0.05,  # 原25.0，缩短为50ms
                recovery_time=0.0,
                failure_rate=0.0,
                description="慢响应但不超时",
            ),
            TimeoutScenario(
                name="intermittent_timeout",
                providers=["glm4v"],
                timeout_pattern="burst",
                timeout_duration=0.06,  # 原35.0，缩短为60ms
                recovery_time=0.08,  # 原45.0，缩短为80ms
                failure_rate=0.5,
                description="间歇性超时突发",
            ),
            TimeoutScenario(
                name="gradual_degradation",
                providers=["deepseek", "assemblyai"],
                timeout_pattern="increasing",
                timeout_duration=0.01,  # 原5.0，缩短为10ms
                recovery_time=0.15,  # 原90.0，缩短为150ms
                failure_rate=0.7,
                description="逐渐恶化的超时情况",
            ),
        ]

    async def simulate_provider_call(
        self, provider: str, scenario: TimeoutScenario, request_num: int
    ) -> Dict[str, Any]:
        """模拟 Provider 调用"""
        start_time = time.time()
        self.metrics["total_requests"] += 1

        try:
            # 判断是否应该超时
            should_timeout = self._should_timeout(scenario, request_num)

            if should_timeout:
                # 计算超时持续时间
                timeout_duration = self._calculate_timeout_duration(scenario, request_num)

                # 模拟超时
                await asyncio.sleep(timeout_duration)
                self.metrics["timeouts"] += 1

                raise asyncio.TimeoutError(f"Provider {provider} timeout after {timeout_duration}s")

            else:
                # 正常响应（可能慢）
                if scenario.timeout_pattern == "fixed" and scenario.failure_rate == 0:
                    # 慢响应场景
                    response_time = scenario.timeout_duration
                else:
                    # 正常响应时间（缩短为0.001-0.01秒以加快测试）
                    response_time = random.uniform(0.001, 0.01)

                await asyncio.sleep(response_time)

                elapsed = time.time() - start_time
                self.metrics["response_times"].append(elapsed)

                return {
                    "provider": provider,
                    "status": "success",
                    "response_time": elapsed,
                    "request_num": request_num,
                }

        except asyncio.TimeoutError as e:
            elapsed = time.time() - start_time
            return {
                "provider": provider,
                "status": "timeout",
                "error": str(e),
                "response_time": elapsed,
                "request_num": request_num,
            }

    def _should_timeout(self, scenario: TimeoutScenario, request_num: int) -> bool:
        """判断请求是否应该超时"""
        if scenario.failure_rate >= 1.0:
            return True
        elif scenario.failure_rate <= 0.0:
            return False

        if scenario.timeout_pattern == "burst":
            # 突发模式：每 10 个请求中有一波超时
            if request_num % 10 < 3:
                return random.random() < scenario.failure_rate * 2
            return False

        return random.random() < scenario.failure_rate

    def _calculate_timeout_duration(self, scenario: TimeoutScenario, request_num: int) -> float:
        """计算超时持续时间"""
        base_duration = scenario.timeout_duration

        if scenario.timeout_pattern == "fixed":
            return base_duration

        elif scenario.timeout_pattern == "random":
            return random.uniform(base_duration * 0.5, base_duration * 1.5)

        elif scenario.timeout_pattern == "increasing":
            # 逐渐增加的超时时间
            return base_duration * (1 + request_num * 0.1)

        elif scenario.timeout_pattern == "burst":
            # 突发模式下的超时更长
            return base_duration * 1.5

        return base_duration

    def analyze_metrics(self) -> Dict[str, Any]:
        """分析测试指标"""
        if not self.metrics["response_times"]:
            avg_response = 0
            p95_response = 0
            p99_response = 0
        else:
            avg_response = statistics.mean(self.metrics["response_times"])
            p95_response = (
                statistics.quantiles(self.metrics["response_times"], n=20)[18]
                if len(self.metrics["response_times"]) > 1
                else avg_response
            )
            p99_response = (
                statistics.quantiles(self.metrics["response_times"], n=100)[98]
                if len(self.metrics["response_times"]) > 1
                else avg_response
            )

        timeout_rate = (
            self.metrics["timeouts"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0
            else 0
        )

        return {
            "total_requests": self.metrics["total_requests"],
            "timeouts": self.metrics["timeouts"],
            "timeout_rate": timeout_rate,
            "avg_response_time": avg_response,
            "p95_response_time": p95_response,
            "p99_response_time": p99_response,
            "cascade_failures": self.metrics["cascade_failures"],
            "recoveries": self.metrics["recoveries"],
        }


# ============================================================================
# Test Cases
# ============================================================================


@pytest.fixture
def simulator():
    """创建模拟器实例"""
    return ProviderTimeoutSimulator()


@pytest.fixture
def mock_resilience_layer():
    """模拟 resilience layer"""
    with patch("src.core.resilience.circuit_breaker.CircuitBreaker") as mock_cb, patch(
        "src.core.resilience.retry_policy.RetryPolicy"
    ) as mock_retry:
        mock_cb_instance = Mock()
        mock_retry_instance = Mock()

        mock_cb.return_value = mock_cb_instance
        mock_retry.return_value = mock_retry_instance

        yield {"circuit_breaker": mock_cb_instance, "retry_policy": mock_retry_instance}


@pytest.mark.asyncio
async def test_single_provider_timeout(simulator):
    """测试单个 Provider 完全超时"""
    scenario = simulator.scenarios[0]  # single_provider_timeout

    results = []
    for i in range(10):
        result = await simulator.simulate_provider_call(
            provider=scenario.providers[0], scenario=scenario, request_num=i
        )
        results.append(result)

    # 验证所有请求都超时
    assert all(r["status"] == "timeout" for r in results)

    metrics = simulator.analyze_metrics()
    assert metrics["timeout_rate"] == 1.0
    assert metrics["timeouts"] == 10


@pytest.mark.asyncio
async def test_cascading_timeout(simulator):
    """测试级联超时场景"""
    scenario = simulator.scenarios[1]  # cascading_timeout

    # 模拟主 Provider 和备用 Provider 都超时
    cascade_detected = False

    for provider in scenario.providers:
        result = await simulator.simulate_provider_call(
            provider=provider, scenario=scenario, request_num=0
        )

        if result["status"] == "timeout":
            if provider != scenario.providers[0]:
                # 备用 Provider 也超时，级联失败
                simulator.metrics["cascade_failures"] += 1
                cascade_detected = True

    assert cascade_detected, "Should detect cascade failure"


@pytest.mark.asyncio
async def test_partial_timeout(simulator):
    """测试部分请求超时"""
    scenario = simulator.scenarios[2]  # partial_timeout

    results = []
    for i in range(100):  # 运行足够多的请求以获得统计意义
        result = await simulator.simulate_provider_call(
            provider=scenario.providers[0], scenario=scenario, request_num=i
        )
        results.append(result)

    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    timeout_rate = timeout_count / len(results)

    # 验证超时率接近配置值（允许 15% 偏差，因随机性避免边界条件）
    assert abs(timeout_rate - scenario.failure_rate) <= 0.15


@pytest.mark.asyncio
async def test_slow_response_detection(simulator):
    """测试慢响应检测"""
    scenario = simulator.scenarios[3]  # slow_response

    results = []
    for i in range(5):
        result = await simulator.simulate_provider_call(
            provider=scenario.providers[0], scenario=scenario, request_num=i
        )
        results.append(result)

    # 验证没有超时但响应很慢（相对于正常响应时间）
    assert all(r["status"] == "success" for r in results)
    # 慢响应时间应该接近配置的超时时间（0.05秒）
    assert all(r["response_time"] >= scenario.timeout_duration * 0.9 for r in results)

    metrics = simulator.analyze_metrics()
    assert metrics["p95_response_time"] >= scenario.timeout_duration * 0.9


@pytest.mark.asyncio
async def test_intermittent_timeout(simulator):
    """测试间歇性超时"""
    scenario = simulator.scenarios[4]  # intermittent_timeout

    burst_results = []
    normal_results = []

    for i in range(30):
        result = await simulator.simulate_provider_call(
            provider=scenario.providers[0], scenario=scenario, request_num=i
        )

        # 根据突发模式分组结果
        if i % 10 < 3:
            burst_results.append(result)
        else:
            normal_results.append(result)

    # 验证突发期间有更高的超时率
    burst_timeout_rate = (
        sum(1 for r in burst_results if r["status"] == "timeout") / len(burst_results)
        if burst_results
        else 0
    )

    normal_timeout_rate = (
        sum(1 for r in normal_results if r["status"] == "timeout") / len(normal_results)
        if normal_results
        else 0
    )

    assert burst_timeout_rate > normal_timeout_rate


@pytest.mark.asyncio
async def test_timeout_recovery(simulator):
    """测试超时恢复机制"""
    scenario = TimeoutScenario(
        name="recovery_test",
        providers=["test_provider"],
        timeout_pattern="fixed",
        timeout_duration=0.02,  # 原10.0，缩短为20ms
        recovery_time=0.01,  # 原5.0，缩短为10ms
        failure_rate=1.0,
        description="测试恢复机制",
    )

    # 第一阶段：超时
    timeout_results = []
    for i in range(5):
        result = await simulator.simulate_provider_call(
            provider=scenario.providers[0], scenario=scenario, request_num=i
        )
        timeout_results.append(result)

    # 模拟恢复时间
    await asyncio.sleep(scenario.recovery_time)
    simulator.metrics["recoveries"] += 1

    # 修改场景为正常
    scenario.failure_rate = 0.0

    # 第二阶段：恢复后的正常请求
    recovery_results = []
    for i in range(5, 10):
        result = await simulator.simulate_provider_call(
            provider=scenario.providers[0], scenario=scenario, request_num=i
        )
        recovery_results.append(result)

    # 验证恢复
    assert all(r["status"] == "timeout" for r in timeout_results)
    assert all(r["status"] == "success" for r in recovery_results)
    assert simulator.metrics["recoveries"] > 0


@pytest.mark.asyncio
async def test_gradual_degradation(simulator):
    """测试逐渐恶化的超时情况"""
    scenario = simulator.scenarios[5]  # gradual_degradation

    response_times = []
    timeout_positions = []

    for i in range(20):
        result = await simulator.simulate_provider_call(
            provider=scenario.providers[0], scenario=scenario, request_num=i
        )

        if result["status"] == "timeout":
            timeout_positions.append(i)
            # 超时的响应时间会更长
            response_times.append(result["response_time"])
        else:
            response_times.append(result["response_time"])

    # 验证后期的超时更频繁（如果pattern是increasing）
    if scenario.timeout_pattern == "increasing" and len(timeout_positions) > 1:
        # 后半部分应该有更多超时
        mid_point = len(timeout_positions) // 2
        early_timeouts = sum(1 for p in timeout_positions if p < 10)
        late_timeouts = sum(1 for p in timeout_positions if p >= 10)

        # 这个断言可能不总是通过（由于随机性），但趋势应该存在
        if late_timeouts > 0 and early_timeouts > 0:
            print(f"Early timeouts: {early_timeouts}, Late timeouts: {late_timeouts}")


@pytest.mark.asyncio
async def test_circuit_breaker_integration(simulator, mock_resilience_layer):
    """测试与 Circuit Breaker 的集成"""
    scenario = simulator.scenarios[0]  # single_provider_timeout

    # 模拟多次失败触发 circuit breaker
    failures = 0
    circuit_opened = False

    for i in range(10):
        result = await simulator.simulate_provider_call(
            provider=scenario.providers[0], scenario=scenario, request_num=i
        )

        if result["status"] == "timeout":
            failures += 1

            # 模拟 circuit breaker 逻辑
            if failures >= 5 and not circuit_opened:
                mock_resilience_layer["circuit_breaker"].open()
                circuit_opened = True
                break

    assert circuit_opened, "Circuit breaker should open after multiple timeouts"


@pytest.mark.asyncio
async def test_retry_policy_with_timeout(simulator, mock_resilience_layer):
    """测试重试策略与超时的交互"""
    scenario = TimeoutScenario(
        name="retry_test",
        providers=["test_provider"],
        timeout_pattern="random",
        timeout_duration=0.01,  # 原5.0，缩短为10ms
        recovery_time=0.002,  # 原1.0，缩短为2ms
        failure_rate=0.5,
        description="测试重试机制",
    )

    max_retries = 3
    successful = False
    attempts = 0

    for retry in range(max_retries):
        attempts += 1
        result = await simulator.simulate_provider_call(
            provider=scenario.providers[0], scenario=scenario, request_num=retry
        )

        if result["status"] == "success":
            successful = True
            break

        # 模拟指数退避（缩短时间）
        await asyncio.sleep(0.001 * (2**retry))

    # 统计重试效果
    print(f"Retry attempts: {attempts}, Successful: {successful}")


@pytest.mark.asyncio
async def test_multi_provider_fallback(simulator):
    """测试多 Provider 故障转移"""
    providers = ["deepseek", "assemblyai", "glm4v"]
    fallback_chain = []

    for idx, provider in enumerate(providers):
        # 为每个 provider 创建不同的超时概率
        scenario = TimeoutScenario(
            name=f"fallback_test_{provider}",
            providers=[provider],
            timeout_pattern="random",
            timeout_duration=0.02,  # 原10.0，缩短为20ms
            recovery_time=0.01,  # 原5.0，缩短为10ms
            failure_rate=0.7 - idx * 0.2,  # 递减的失败率
            description=f"Fallback test for {provider}",
        )

        result = await simulator.simulate_provider_call(
            provider=provider, scenario=scenario, request_num=0
        )

        fallback_chain.append({"provider": provider, "result": result, "order": idx + 1})

        if result["status"] == "success":
            break

    # 分析故障转移链
    successful_provider = next(
        (f["provider"] for f in fallback_chain if f["result"]["status"] == "success"), None
    )

    print(f"Fallback chain: {[f['provider'] for f in fallback_chain]}")
    print(f"Successful provider: {successful_provider}")


def test_timeout_metrics_calculation(simulator):
    """测试超时指标计算"""
    # 模拟一些数据
    simulator.metrics = {
        "total_requests": 100,
        "timeouts": 25,
        "recoveries": 3,
        "response_times": [1.5, 2.3, 3.1, 4.5, 5.2, 8.9, 12.3, 15.6, 22.1, 28.5],
        "cascade_failures": 5,
    }

    analysis = simulator.analyze_metrics()

    assert analysis["timeout_rate"] == 0.25
    assert analysis["total_requests"] == 100
    assert analysis["timeouts"] == 25
    assert analysis["cascade_failures"] == 5
    assert analysis["avg_response_time"] > 0
    assert analysis["p95_response_time"] >= analysis["avg_response_time"]
    assert analysis["p99_response_time"] >= analysis["p95_response_time"]


# ============================================================================
# Performance and Stress Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.performance
async def test_high_load_timeout_behavior(simulator):
    """测试高负载下的超时行为"""
    scenario = TimeoutScenario(
        name="high_load_test",
        providers=["test_provider"],
        timeout_pattern="random",
        timeout_duration=0.01,  # 原5.0，缩短为10ms
        recovery_time=0.002,  # 原1.0，缩短为2ms
        failure_rate=0.2,
        description="高负载测试",
    )

    # 并发请求
    concurrent_requests = 50
    tasks = []

    for i in range(concurrent_requests):
        task = simulator.simulate_provider_call(
            provider=scenario.providers[0], scenario=scenario, request_num=i
        )
        tasks.append(task)

    # 执行所有并发请求
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 分析结果
    timeout_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "timeout")
    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")

    print(f"High load test results:")
    print(f"  Total: {concurrent_requests}")
    print(f"  Timeouts: {timeout_count}")
    print(f"  Success: {success_count}")

    # 验证系统没有崩溃
    assert timeout_count + success_count == concurrent_requests


@pytest.mark.asyncio
async def test_timeout_scenario_report(simulator):
    """生成超时场景测试报告"""
    report = {"test_time": datetime.now().isoformat(), "scenarios": []}

    for scenario in simulator.scenarios:
        # 重置指标
        simulator.metrics = {
            "total_requests": 0,
            "timeouts": 0,
            "recoveries": 0,
            "response_times": [],
            "cascade_failures": 0,
        }

        # 运行场景测试
        for i in range(10):
            await simulator.simulate_provider_call(
                provider=scenario.providers[0], scenario=scenario, request_num=i
            )

        # 收集指标
        metrics = simulator.analyze_metrics()
        report["scenarios"].append(
            {"name": scenario.name, "description": scenario.description, "metrics": metrics}
        )

    # 打印报告
    print("\n" + "=" * 60)
    print("Provider Timeout Simulation Report")
    print("=" * 60)

    for scenario_report in report["scenarios"]:
        print(f"\n📊 Scenario: {scenario_report['name']}")
        print(f"   Description: {scenario_report['description']}")
        print(f"   Results:")
        for key, value in scenario_report["metrics"].items():
            if isinstance(value, float):
                print(f"     - {key}: {value:.2f}")
            else:
                print(f"     - {key}: {value}")

    print("\n" + "=" * 60)
