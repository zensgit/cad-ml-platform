#!/usr/bin/env python3
"""
Adaptive Rate Limiter Performance Benchmark
自适应限流器性能基准测试
"""

import time
import json
import statistics
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.resilience.adaptive_rate_limiter import (
    AdaptiveRateLimiter,
    AdaptiveConfig,
    AdaptivePhase
)
from src.core.resilience.adaptive_decorator import adaptive_rate_limit


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    scenario: str
    enabled: bool
    requests: int
    duration: float
    throughput: float
    latency_p50: float
    latency_p90: float
    latency_p95: float
    latency_p99: float
    overhead_percentage: float
    adjustments_made: int
    final_phase: str
    final_rate: float


class AdaptiveRateLimiterBenchmark:
    """自适应限流器基准测试"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_latencies: List[float] = []

    def benchmark_function(self, delay_ms: float = 10.0):
        """基准测试函数（模拟业务逻辑）"""
        time.sleep(delay_ms / 1000)
        return "success"

    def run_scenario(
        self,
        scenario_name: str,
        num_requests: int,
        num_threads: int,
        error_rate: float,
        use_adaptive: bool,
        config: AdaptiveConfig = None
    ) -> BenchmarkResult:
        """运行测试场景"""
        print(f"\n🔍 Running scenario: {scenario_name}")
        print(f"   Requests: {num_requests}, Threads: {num_threads}, Error rate: {error_rate:.2%}")
        print(f"   Adaptive: {'Enabled' if use_adaptive else 'Disabled'}")

        if use_adaptive and config:
            # 创建自适应限流器
            limiter = AdaptiveRateLimiter("benchmark", scenario_name, config)
            limiter.set_baseline(15.0)  # 基线15ms
        else:
            limiter = None

        latencies = []
        errors = 0
        success = 0
        start_time = time.time()

        def worker(request_id: int):
            nonlocal errors, success

            request_start = time.time()

            try:
                # 如果启用自适应限流
                if limiter:
                    if not limiter.acquire():
                        errors += 1
                        return None

                # 模拟错误
                if np.random.random() < error_rate:
                    if limiter:
                        limiter.record_error()
                    errors += 1
                    raise Exception("Simulated error")

                # 执行函数
                result = self.benchmark_function()

                if limiter:
                    limiter.record_success()
                success += 1

                return result

            finally:
                latency_ms = (time.time() - request_start) * 1000
                latencies.append(latency_ms)

                if limiter:
                    limiter.record_latency(latency_ms)

        # 使用线程池执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            futures = []
            for i in range(num_requests):
                future = executor.submit(worker, i)
                futures.append(future)

                # 定期评估（如果启用）
                if limiter and i % 100 == 0:
                    limiter.evaluate_and_adjust()

            # 等待完成
            concurrent.futures.wait(futures)

        duration = time.time() - start_time
        throughput = num_requests / duration

        # 计算延迟百分位
        sorted_latencies = sorted(latencies)
        p50 = np.percentile(sorted_latencies, 50)
        p90 = np.percentile(sorted_latencies, 90)
        p95 = np.percentile(sorted_latencies, 95)
        p99 = np.percentile(sorted_latencies, 99)

        # 计算开销
        if not use_adaptive:
            self.baseline_latencies = latencies.copy()
            overhead = 0.0
        else:
            if self.baseline_latencies:
                baseline_p95 = np.percentile(self.baseline_latencies, 95)
                overhead = ((p95 - baseline_p95) / baseline_p95) * 100
            else:
                overhead = 0.0

        # 获取最终状态
        if limiter:
            status = limiter.get_status()
            final_phase = status["phase"]
            final_rate = status["current_rate"]
            adjustments = len(status.get("recent_adjustments", []))
        else:
            final_phase = "N/A"
            final_rate = 0.0
            adjustments = 0

        result = BenchmarkResult(
            scenario=scenario_name,
            enabled=use_adaptive,
            requests=num_requests,
            duration=duration,
            throughput=throughput,
            latency_p50=p50,
            latency_p90=p90,
            latency_p95=p95,
            latency_p99=p99,
            overhead_percentage=overhead,
            adjustments_made=adjustments,
            final_phase=final_phase,
            final_rate=final_rate
        )

        self.results.append(result)

        # 打印结果
        print(f"   ✅ Completed in {duration:.2f}s")
        print(f"   Throughput: {throughput:.2f} req/s")
        print(f"   P95 Latency: {p95:.2f}ms (overhead: {overhead:+.2f}%)")
        if limiter:
            print(f"   Final phase: {final_phase}, Rate: {final_rate:.2f}")

        return result

    def run_all_scenarios(self):
        """运行所有测试场景"""
        print("=" * 60)
        print("🚀 Adaptive Rate Limiter Performance Benchmark")
        print("=" * 60)

        # 配置
        config = AdaptiveConfig(
            base_rate=1000.0,
            error_threshold=0.02,
            recover_threshold=0.008,
            latency_p95_threshold_multiplier=1.5,
            min_rate_ratio=0.15,
            adjust_min_interval_ms=1000,
            recover_step=0.1,
            error_alpha=0.25
        )

        scenarios = [
            # 场景1：正常负载基线（无自适应）
            ("Normal Load Baseline", 1000, 10, 0.01, False, None),

            # 场景2：正常负载（有自适应）
            ("Normal Load Adaptive", 1000, 10, 0.01, True, config),

            # 场景3：错误激增基线
            ("Error Spike Baseline", 1000, 10, 0.10, False, None),

            # 场景4：错误激增自适应
            ("Error Spike Adaptive", 1000, 10, 0.10, True, config),

            # 场景5：高并发基线
            ("High Concurrency Baseline", 2000, 50, 0.02, False, None),

            # 场景6：高并发自适应
            ("High Concurrency Adaptive", 2000, 50, 0.02, True, config),

            # 场景7：渐进错误（模拟系统退化）
            ("Gradual Degradation", 2000, 20, 0.05, True, config),
        ]

        for scenario in scenarios:
            self.run_scenario(*scenario)
            time.sleep(1)  # 场景间隔

        # 生成报告
        self.generate_report()

    def generate_report(self):
        """生成性能报告"""
        print("\n" + "=" * 60)
        print("📊 Performance Benchmark Report")
        print("=" * 60)

        # 按场景配对分析
        paired_results = {}
        for result in self.results:
            base_name = result.scenario.replace(" Baseline", "").replace(" Adaptive", "")
            if base_name not in paired_results:
                paired_results[base_name] = {}

            if "Baseline" in result.scenario:
                paired_results[base_name]["baseline"] = result
            else:
                paired_results[base_name]["adaptive"] = result

        print("\n### Overhead Analysis")
        print("-" * 40)

        overhead_list = []
        for scenario_name, pair in paired_results.items():
            if "baseline" in pair and "adaptive" in pair:
                baseline = pair["baseline"]
                adaptive = pair["adaptive"]

                overhead_p95 = ((adaptive.latency_p95 - baseline.latency_p95) / baseline.latency_p95) * 100
                overhead_p99 = ((adaptive.latency_p99 - baseline.latency_p99) / baseline.latency_p99) * 100

                overhead_list.append(overhead_p95)

                print(f"\n{scenario_name}:")
                print(f"  P95 Overhead: {overhead_p95:+.2f}%")
                print(f"  P99 Overhead: {overhead_p99:+.2f}%")
                print(f"  Throughput Impact: {(adaptive.throughput - baseline.throughput):+.2f} req/s")

                if adaptive.adjustments_made > 0:
                    print(f"  Adjustments Made: {adaptive.adjustments_made}")
                    print(f"  Final Phase: {adaptive.final_phase}")
                    print(f"  Final Rate: {adaptive.final_rate:.2f}")

        # 验证标准
        print("\n### Validation Against Requirements")
        print("-" * 40)

        avg_overhead = statistics.mean(overhead_list) if overhead_list else 0
        max_overhead = max(overhead_list) if overhead_list else 0

        print(f"Average P95 Overhead: {avg_overhead:.2f}%")
        print(f"Maximum P95 Overhead: {max_overhead:.2f}%")
        print(f"Target: ≤5%")

        if max_overhead <= 5.0:
            print("✅ PASSED: Overhead within acceptable range")
        else:
            print("❌ FAILED: Overhead exceeds 5% threshold")

        # 保存JSON报告
        self.save_json_report()

    def save_json_report(self):
        """保存JSON格式报告"""
        report_dir = Path(__file__).parent.parent / "reports" / "perf"
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / "adaptive_rate_limit_benchmark.json"

        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [asdict(r) for r in self.results],
            "summary": self._calculate_summary()
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"\n📁 Report saved to: {report_file}")

    def _calculate_summary(self) -> Dict[str, Any]:
        """计算汇总统计"""
        overhead_values = [r.overhead_percentage for r in self.results if r.enabled]

        return {
            "total_scenarios": len(self.results),
            "adaptive_scenarios": sum(1 for r in self.results if r.enabled),
            "avg_overhead_p95": statistics.mean(overhead_values) if overhead_values else 0,
            "max_overhead_p95": max(overhead_values) if overhead_values else 0,
            "min_overhead_p95": min(overhead_values) if overhead_values else 0,
            "meets_requirement": max(overhead_values) <= 5.0 if overhead_values else True
        }


def run_integration_test():
    """运行集成测试"""
    print("\n" + "=" * 60)
    print("🧪 Integration Test with Decorator")
    print("=" * 60)

    # 使用装饰器的函数
    @adaptive_rate_limit(service="test", endpoint="process", base_rate=100.0)
    def process_with_adaptive(data):
        time.sleep(0.01)  # 模拟处理
        if np.random.random() < 0.05:  # 5%错误率
            raise Exception("Processing error")
        return f"Processed: {data}"

    def process_without_adaptive(data):
        time.sleep(0.01)
        if np.random.random() < 0.05:
            raise Exception("Processing error")
        return f"Processed: {data}"

    # 测试有无自适应的差异
    print("\nTesting with adaptive rate limiting...")
    adaptive_times = []
    for i in range(100):
        start = time.time()
        try:
            process_with_adaptive(f"data_{i}")
        except:
            pass
        adaptive_times.append(time.time() - start)

    print("\nTesting without adaptive rate limiting...")
    normal_times = []
    for i in range(100):
        start = time.time()
        try:
            process_without_adaptive(f"data_{i}")
        except:
            pass
        normal_times.append(time.time() - start)

    # 分析结果
    adaptive_p95 = np.percentile(adaptive_times, 95) * 1000
    normal_p95 = np.percentile(normal_times, 95) * 1000
    overhead = ((adaptive_p95 - normal_p95) / normal_p95) * 100

    print(f"\n📊 Integration Test Results:")
    print(f"  Normal P95: {normal_p95:.2f}ms")
    print(f"  Adaptive P95: {adaptive_p95:.2f}ms")
    print(f"  Overhead: {overhead:+.2f}%")

    if overhead <= 5.0:
        print("  ✅ Integration test PASSED")
    else:
        print("  ❌ Integration test FAILED")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptive rate limiter performance benchmark"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (fewer requests)"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration test only"
    )

    args = parser.parse_args()

    if args.integration:
        run_integration_test()
    else:
        benchmark = AdaptiveRateLimiterBenchmark()

        if args.quick:
            # 快速测试
            config = AdaptiveConfig(base_rate=100.0)
            benchmark.run_scenario("Quick Test Baseline", 100, 5, 0.01, False, None)
            benchmark.run_scenario("Quick Test Adaptive", 100, 5, 0.01, True, config)
            benchmark.generate_report()
        else:
            # 完整基准测试
            benchmark.run_all_scenarios()

    return 0


if __name__ == "__main__":
    sys.exit(main())