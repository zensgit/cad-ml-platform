#!/usr/bin/env python3
"""
Chaos Injection Tool
混沌工程工具 - 注入故障以测试系统弹性
"""

import argparse
import asyncio
import json
import logging
import random
import signal
import sys
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChaosType(Enum):
    """混沌类型"""
    NETWORK_DELAY = "network_delay"
    NETWORK_PACKET_LOSS = "network_packet_loss"
    NETWORK_DISCONNECT = "network_disconnect"
    CPU_SPIKE = "cpu_spike"
    MEMORY_LEAK = "memory_leak"
    DISK_FULL = "disk_full"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_ERROR = "provider_error"
    RATE_LIMIT = "rate_limit"
    RANDOM_EXCEPTION = "random_exception"
    CIRCUIT_BREAKER_TRIP = "circuit_breaker_trip"
    SLOW_RESPONSE = "slow_response"


@dataclass
class ChaosScenario:
    """混沌场景"""
    name: str
    chaos_type: ChaosType
    duration: int  # 秒
    intensity: float  # 0.0 - 1.0
    target: Optional[str] = None  # 目标服务/组件
    params: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class ChaosResult:
    """混沌测试结果"""
    scenario: ChaosScenario
    success: bool
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    errors_captured: List[str]
    recovery_time: Optional[float] = None
    observations: List[str] = field(default_factory=list)


class ChaosInjector:
    """混沌注入器"""

    def __init__(self, target_url: str = "http://localhost:8000"):
        self.target_url = target_url
        self.scenarios: List[ChaosScenario] = []
        self.results: List[ChaosResult] = []
        self.is_running = False
        self._stop_event = threading.Event()

    def add_scenario(self, scenario: ChaosScenario):
        """添加混沌场景"""
        self.scenarios.append(scenario)
        logger.info(f"Added chaos scenario: {scenario.name} ({scenario.chaos_type.value})")

    async def inject_network_delay(self, scenario: ChaosScenario):
        """注入网络延迟"""
        delay_ms = int(scenario.intensity * 5000)  # 最大 5 秒延迟

        logger.info(f"Injecting network delay: {delay_ms}ms for {scenario.duration}s")

        # Linux tc 命令（需要 root 权限）
        if sys.platform == "linux":
            os.system(f"tc qdisc add dev lo root netem delay {delay_ms}ms")
            await asyncio.sleep(scenario.duration)
            os.system("tc qdisc del dev lo root netem")
        else:
            # 模拟延迟
            end_time = time.time() + scenario.duration
            while time.time() < end_time and not self._stop_event.is_set():
                await asyncio.sleep(0.1)
            logger.info("Network delay simulation completed")

    async def inject_packet_loss(self, scenario: ChaosScenario):
        """注入丢包"""
        loss_percent = int(scenario.intensity * 100)

        logger.info(f"Injecting packet loss: {loss_percent}% for {scenario.duration}s")

        if sys.platform == "linux":
            os.system(f"tc qdisc add dev lo root netem loss {loss_percent}%")
            await asyncio.sleep(scenario.duration)
            os.system("tc qdisc del dev lo root netem")
        else:
            # 模拟丢包
            await asyncio.sleep(scenario.duration)
            logger.info("Packet loss simulation completed")

    async def inject_cpu_spike(self, scenario: ChaosScenario):
        """注入 CPU 峰值"""
        logger.info(f"Injecting CPU spike: {scenario.intensity * 100}% for {scenario.duration}s")

        # CPU 密集型操作
        end_time = time.time() + scenario.duration
        while time.time() < end_time and not self._stop_event.is_set():
            # 执行一些 CPU 密集型计算
            for _ in range(int(scenario.intensity * 1000000)):
                _ = sum(i * i for i in range(100))
            await asyncio.sleep(0.01)

        logger.info("CPU spike completed")

    async def inject_memory_leak(self, scenario: ChaosScenario):
        """注入内存泄漏"""
        leak_size_mb = int(scenario.intensity * 500)  # 最大 500MB

        logger.info(f"Injecting memory leak: {leak_size_mb}MB for {scenario.duration}s")

        # 分配内存
        memory_blocks = []
        block_size = 1024 * 1024  # 1MB
        for _ in range(leak_size_mb):
            memory_blocks.append(bytearray(block_size))

        # 保持内存占用
        await asyncio.sleep(scenario.duration)

        # 释放内存
        memory_blocks.clear()
        logger.info("Memory leak cleaned up")

    async def inject_provider_timeout(self, scenario: ChaosScenario):
        """注入提供商超时"""
        provider = scenario.target or "all"
        timeout_probability = scenario.intensity

        logger.info(f"Injecting provider timeout: {provider} with {timeout_probability * 100}% probability")

        # 这里需要与实际的 provider 集成
        # 可以通过环境变量或配置文件来控制
        os.environ[f"CHAOS_PROVIDER_{provider.upper()}_TIMEOUT"] = str(timeout_probability)

        await asyncio.sleep(scenario.duration)

        # 清理环境变量
        os.environ.pop(f"CHAOS_PROVIDER_{provider.upper()}_TIMEOUT", None)
        logger.info("Provider timeout injection completed")

    async def inject_rate_limit(self, scenario: ChaosScenario):
        """注入限流"""
        rate_reduction = scenario.intensity  # 减少多少比例的限流

        logger.info(f"Injecting rate limit: {rate_reduction * 100}% reduction for {scenario.duration}s")

        # 设置环境变量供应用读取
        os.environ["CHAOS_RATE_LIMIT_REDUCTION"] = str(rate_reduction)

        await asyncio.sleep(scenario.duration)

        os.environ.pop("CHAOS_RATE_LIMIT_REDUCTION", None)
        logger.info("Rate limit injection completed")

    async def inject_random_exception(self, scenario: ChaosScenario):
        """注入随机异常"""
        exception_probability = scenario.intensity

        logger.info(f"Injecting random exceptions: {exception_probability * 100}% probability")

        os.environ["CHAOS_EXCEPTION_PROBABILITY"] = str(exception_probability)

        await asyncio.sleep(scenario.duration)

        os.environ.pop("CHAOS_EXCEPTION_PROBABILITY", None)
        logger.info("Random exception injection completed")

    async def inject_slow_response(self, scenario: ChaosScenario):
        """注入慢响应"""
        slowdown_factor = 1 + (scenario.intensity * 10)  # 最多慢 10 倍

        logger.info(f"Injecting slow response: {slowdown_factor}x slower for {scenario.duration}s")

        os.environ["CHAOS_SLOW_RESPONSE_FACTOR"] = str(slowdown_factor)

        await asyncio.sleep(scenario.duration)

        os.environ.pop("CHAOS_SLOW_RESPONSE_FACTOR", None)
        logger.info("Slow response injection completed")

    async def execute_scenario(self, scenario: ChaosScenario) -> ChaosResult:
        """执行混沌场景"""
        logger.info(f"Executing chaos scenario: {scenario.name}")

        # 收集执行前的指标
        metrics_before = await self.collect_metrics()

        # 记录开始时间
        scenario.start_time = datetime.now()
        scenario.status = "running"

        errors_captured = []
        try:
            # 根据类型执行注入
            if scenario.chaos_type == ChaosType.NETWORK_DELAY:
                await self.inject_network_delay(scenario)
            elif scenario.chaos_type == ChaosType.NETWORK_PACKET_LOSS:
                await self.inject_packet_loss(scenario)
            elif scenario.chaos_type == ChaosType.CPU_SPIKE:
                await self.inject_cpu_spike(scenario)
            elif scenario.chaos_type == ChaosType.MEMORY_LEAK:
                await self.inject_memory_leak(scenario)
            elif scenario.chaos_type == ChaosType.PROVIDER_TIMEOUT:
                await self.inject_provider_timeout(scenario)
            elif scenario.chaos_type == ChaosType.RATE_LIMIT:
                await self.inject_rate_limit(scenario)
            elif scenario.chaos_type == ChaosType.RANDOM_EXCEPTION:
                await self.inject_random_exception(scenario)
            elif scenario.chaos_type == ChaosType.SLOW_RESPONSE:
                await self.inject_slow_response(scenario)
            else:
                logger.warning(f"Unsupported chaos type: {scenario.chaos_type}")

            scenario.status = "completed"
            success = True

        except Exception as e:
            logger.error(f"Error executing scenario: {e}")
            errors_captured.append(str(e))
            scenario.status = "failed"
            success = False

        # 记录结束时间
        scenario.end_time = datetime.now()

        # 等待系统恢复
        await asyncio.sleep(5)

        # 收集执行后的指标
        metrics_after = await self.collect_metrics()

        # 计算恢复时间
        recovery_time = await self.measure_recovery_time()

        # 生成结果
        result = ChaosResult(
            scenario=scenario,
            success=success,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            errors_captured=errors_captured,
            recovery_time=recovery_time
        )

        # 添加观察结果
        self.analyze_result(result)

        return result

    async def collect_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        metrics = {}

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # 健康检查
                async with session.get(f"{self.target_url}/health") as resp:
                    if resp.status == 200:
                        health = await resp.json()
                        metrics["health"] = health

                # Prometheus 指标
                async with session.get(f"{self.target_url}/metrics") as resp:
                    if resp.status == 200:
                        metrics_text = await resp.text()
                        # 解析关键指标
                        metrics["error_rate"] = self._parse_metric(metrics_text, "error_rate")
                        metrics["response_time"] = self._parse_metric(metrics_text, "response_time")
                        metrics["throughput"] = self._parse_metric(metrics_text, "requests_total")

        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")

        return metrics

    def _parse_metric(self, metrics_text: str, metric_name: str) -> Optional[float]:
        """解析 Prometheus 格式的指标"""
        for line in metrics_text.split('\n'):
            if metric_name in line and not line.startswith('#'):
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[-1])
                except:
                    pass
        return None

    async def measure_recovery_time(self) -> float:
        """测量恢复时间"""
        start_time = time.time()
        max_wait = 60  # 最多等待 60 秒

        while time.time() - start_time < max_wait:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.target_url}/health") as resp:
                        if resp.status == 200:
                            health = await resp.json()
                            if health.get("status") == "healthy":
                                return time.time() - start_time
            except:
                pass

            await asyncio.sleep(1)

        return -1  # 未能恢复

    def analyze_result(self, result: ChaosResult):
        """分析测试结果"""
        # 比较前后指标
        before = result.metrics_before
        after = result.metrics_after

        # 错误率变化
        if "error_rate" in before and "error_rate" in after:
            error_increase = after["error_rate"] - before["error_rate"]
            if error_increase > 0.1:
                result.observations.append(f"Error rate increased by {error_increase * 100:.1f}%")

        # 响应时间变化
        if "response_time" in before and "response_time" in after:
            time_increase = after["response_time"] - before["response_time"]
            if time_increase > 100:
                result.observations.append(f"Response time increased by {time_increase:.0f}ms")

        # 恢复时间评估
        if result.recovery_time:
            if result.recovery_time < 10:
                result.observations.append("System recovered quickly (< 10s)")
            elif result.recovery_time < 30:
                result.observations.append("System recovered within acceptable time (< 30s)")
            elif result.recovery_time > 0:
                result.observations.append(f"System took {result.recovery_time:.1f}s to recover")
            else:
                result.observations.append("System did not recover within timeout")

    async def run_all_scenarios(self):
        """运行所有场景"""
        self.is_running = True

        for scenario in self.scenarios:
            if self._stop_event.is_set():
                break

            result = await self.execute_scenario(scenario)
            self.results.append(result)

            # 场景间休息
            await asyncio.sleep(10)

        self.is_running = False

    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        successful_scenarios = [r for r in self.results if r.success]
        failed_scenarios = [r for r in self.results if not r.success]

        report = {
            "summary": {
                "total_scenarios": len(self.results),
                "successful": len(successful_scenarios),
                "failed": len(failed_scenarios),
                "average_recovery_time": sum(
                    r.recovery_time for r in self.results if r.recovery_time and r.recovery_time > 0
                ) / len([r for r in self.results if r.recovery_time and r.recovery_time > 0])
                if any(r.recovery_time and r.recovery_time > 0 for r in self.results) else 0
            },
            "scenarios": [],
            "recommendations": []
        }

        # 详细场景结果
        for result in self.results:
            scenario_report = {
                "name": result.scenario.name,
                "type": result.scenario.chaos_type.value,
                "duration": result.scenario.duration,
                "intensity": result.scenario.intensity,
                "success": result.success,
                "recovery_time": result.recovery_time,
                "observations": result.observations,
                "errors": result.errors_captured
            }
            report["scenarios"].append(scenario_report)

        # 生成建议
        report["recommendations"] = self.generate_recommendations()

        return report

    def generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 分析恢复时间
        slow_recovery = [r for r in self.results if r.recovery_time and r.recovery_time > 30]
        if slow_recovery:
            recommendations.append(
                f"⚠️ {len(slow_recovery)} scenarios had slow recovery (>30s). "
                "Consider improving circuit breaker settings."
            )

        # 分析失败场景
        failed = [r for r in self.results if not r.success]
        if failed:
            recommendations.append(
                f"❌ {len(failed)} scenarios failed. Review error handling and resilience patterns."
            )

        # 网络相关
        network_scenarios = [
            r for r in self.results
            if r.scenario.chaos_type in [ChaosType.NETWORK_DELAY, ChaosType.NETWORK_PACKET_LOSS]
        ]
        if network_scenarios and any(r.recovery_time > 20 for r in network_scenarios):
            recommendations.append(
                "Network issues cause slow recovery. Consider implementing retry with exponential backoff."
            )

        # 资源相关
        resource_scenarios = [
            r for r in self.results
            if r.scenario.chaos_type in [ChaosType.CPU_SPIKE, ChaosType.MEMORY_LEAK]
        ]
        if resource_scenarios and any(not r.success for r in resource_scenarios):
            recommendations.append(
                "Resource constraints cause failures. Implement resource limits and monitoring."
            )

        # 通用建议
        recommendations.extend([
            "✅ Continue regular chaos testing to maintain resilience",
            "📊 Monitor key metrics during chaos tests for better insights",
            "🔄 Implement automated chaos testing in CI/CD pipeline",
            "📚 Document failure scenarios and recovery procedures"
        ])

        return recommendations

    def stop(self):
        """停止混沌注入"""
        self._stop_event.set()
        logger.info("Chaos injection stopped")


def create_standard_scenarios() -> List[ChaosScenario]:
    """创建标准测试场景"""
    scenarios = [
        ChaosScenario(
            name="Light Network Delay",
            chaos_type=ChaosType.NETWORK_DELAY,
            duration=30,
            intensity=0.2,
            params={"delay_ms": 100}
        ),
        ChaosScenario(
            name="Heavy Network Delay",
            chaos_type=ChaosType.NETWORK_DELAY,
            duration=30,
            intensity=0.8,
            params={"delay_ms": 2000}
        ),
        ChaosScenario(
            name="Packet Loss",
            chaos_type=ChaosType.NETWORK_PACKET_LOSS,
            duration=30,
            intensity=0.3,
            params={"loss_percent": 30}
        ),
        ChaosScenario(
            name="CPU Stress",
            chaos_type=ChaosType.CPU_SPIKE,
            duration=20,
            intensity=0.7
        ),
        ChaosScenario(
            name="Memory Pressure",
            chaos_type=ChaosType.MEMORY_LEAK,
            duration=30,
            intensity=0.5
        ),
        ChaosScenario(
            name="Provider Timeout",
            chaos_type=ChaosType.PROVIDER_TIMEOUT,
            duration=30,
            intensity=0.5,
            target="deepseek"
        ),
        ChaosScenario(
            name="Rate Limiting",
            chaos_type=ChaosType.RATE_LIMIT,
            duration=20,
            intensity=0.7
        ),
        ChaosScenario(
            name="Random Exceptions",
            chaos_type=ChaosType.RANDOM_EXCEPTION,
            duration=30,
            intensity=0.3
        ),
        ChaosScenario(
            name="Slow Responses",
            chaos_type=ChaosType.SLOW_RESPONSE,
            duration=30,
            intensity=0.5
        )
    ]

    return scenarios


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Chaos Engineering Tool")
    parser.add_argument("--target", default="http://localhost:8000",
                       help="Target service URL")
    parser.add_argument("--scenario", choices=[t.value for t in ChaosType],
                       help="Specific scenario to run")
    parser.add_argument("--duration", type=int, default=30,
                       help="Scenario duration in seconds")
    parser.add_argument("--intensity", type=float, default=0.5,
                       help="Chaos intensity (0.0-1.0)")
    parser.add_argument("--all", action="store_true",
                       help="Run all standard scenarios")
    parser.add_argument("--output", default="chaos-report.json",
                       help="Output file for report")

    args = parser.parse_args()

    # 创建注入器
    injector = ChaosInjector(target_url=args.target)

    # 设置信号处理
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal")
        injector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 添加场景
    if args.all:
        scenarios = create_standard_scenarios()
        for scenario in scenarios:
            injector.add_scenario(scenario)
    elif args.scenario:
        scenario = ChaosScenario(
            name=f"Custom {args.scenario}",
            chaos_type=ChaosType(args.scenario),
            duration=args.duration,
            intensity=args.intensity
        )
        injector.add_scenario(scenario)
    else:
        print("Please specify --scenario or --all")
        return 1

    # 运行场景
    logger.info("Starting chaos injection...")
    asyncio.run(injector.run_all_scenarios())

    # 生成报告
    report = injector.generate_report()

    # 保存报告
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Chaos test report saved to {args.output}")

    # 打印摘要
    print("\n=== Chaos Test Summary ===")
    print(f"Total Scenarios: {report['summary']['total_scenarios']}")
    print(f"Successful: {report['summary']['successful']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Avg Recovery Time: {report['summary']['average_recovery_time']:.1f}s")

    print("\n=== Recommendations ===")
    for rec in report['recommendations'][:5]:
        print(f"- {rec}")

    return 0 if report['summary']['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())