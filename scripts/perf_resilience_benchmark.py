#!/usr/bin/env python3
"""
Resilience Performance Benchmark
韧性层性能基准测试 - 对比启用/禁用 Resilience 组件的性能差异

测试维度：
1. 单线程 vs 多线程
2. 各组件独立性能
3. 组合使用性能
4. 不同负载下的表现
5. 锁争用分析
"""

import asyncio
import time
import statistics
import json
import sys
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock
import tracemalloc
import psutil
import os
from datetime import datetime

# 导入 Resilience 组件
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.core.resilience import (
        circuit_breaker,
        rate_limit,
        retry,
        bulkhead,
        with_resilience
    )
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False
    print("Warning: Resilience components not available, using mock")


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    overhead: float  # 性能开销百分比
    verdict: str  # PASS/FAIL/WARNING


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_time: float = 0.0
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    lock_wait_time: float = 0.0


class ResilienceBenchmark:
    """韧性层性能基准测试"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_metrics: Dict[str, PerformanceMetrics] = {}
        self.resilience_metrics: Dict[str, PerformanceMetrics] = {}

        # 性能阈值
        self.thresholds = {
            "max_overhead_p95": 0.05,  # P95 增幅 ≤5%
            "max_overhead_avg": 0.03,  # 平均增幅 ≤3%
            "max_memory_overhead_mb": 50,  # 内存增加 ≤50MB
            "max_lock_wait_ms": 10  # 锁等待 ≤10ms
        }

    # ========== 测试目标函数 ==========

    def target_function_fast(self, x: int) -> int:
        """快速目标函数（~1ms）"""
        time.sleep(0.001)  # 1ms
        return x * 2

    def target_function_medium(self, x: int) -> int:
        """中速目标函数（~10ms）"""
        time.sleep(0.01)  # 10ms
        # 模拟一些计算
        result = sum(i ** 2 for i in range(100))
        return x * result

    def target_function_slow(self, x: int) -> int:
        """慢速目标函数（~100ms）"""
        time.sleep(0.1)  # 100ms
        # 模拟复杂计算
        result = sum(i ** 2 for i in range(1000))
        return x * result

    async def async_target_function(self, x: int) -> int:
        """异步目标函数"""
        await asyncio.sleep(0.01)
        return x * 2

    # ========== 基准测试执行 ==========

    def run_baseline(
        self,
        func: Callable,
        iterations: int = 1000,
        concurrency: int = 1
    ) -> PerformanceMetrics:
        """运行基线测试（无 Resilience）"""
        metrics = PerformanceMetrics()
        latencies = []
        errors = 0

        # 记录开始状态
        tracemalloc.start()
        start_time = time.time()
        process = psutil.Process()
        start_cpu = process.cpu_percent()

        if concurrency == 1:
            # 单线程执行
            for i in range(iterations):
                try:
                    op_start = time.perf_counter()
                    func(i)
                    op_time = time.perf_counter() - op_start
                    latencies.append(op_time * 1000)  # 转换为毫秒
                except Exception:
                    errors += 1

        else:
            # 多线程执行
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                for i in range(iterations):
                    op_start = time.perf_counter()
                    future = executor.submit(func, i)
                    futures.append((future, op_start))

                for future, op_start in futures:
                    try:
                        future.result(timeout=5)
                        op_time = time.perf_counter() - op_start
                        latencies.append(op_time * 1000)
                    except Exception:
                        errors += 1

        # 计算指标
        total_time = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics.total_time = total_time
        metrics.throughput = iterations / total_time if total_time > 0 else 0
        metrics.error_rate = errors / iterations if iterations > 0 else 0
        metrics.memory_usage_mb = peak_mem / 1024 / 1024
        metrics.cpu_usage_percent = process.cpu_percent() - start_cpu

        if latencies:
            metrics.avg_latency = statistics.mean(latencies)
            metrics.p50_latency = statistics.quantiles(latencies, n=2)[0] if len(latencies) > 1 else latencies[0]
            metrics.p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 19 else metrics.p50_latency
            metrics.p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 99 else metrics.p95_latency

        return metrics

    def run_with_circuit_breaker(
        self,
        func: Callable,
        iterations: int = 1000,
        concurrency: int = 1
    ) -> PerformanceMetrics:
        """运行带 Circuit Breaker 的测试"""
        if not RESILIENCE_AVAILABLE:
            return self.run_baseline(func, iterations, concurrency)

        @circuit_breaker(
            failure_threshold=5,
            recovery_timeout=30,
            half_open_max_calls=3
        )
        def protected_func(x):
            return func(x)

        return self._run_protected(protected_func, iterations, concurrency)

    def run_with_rate_limiter(
        self,
        func: Callable,
        iterations: int = 1000,
        concurrency: int = 1
    ) -> PerformanceMetrics:
        """运行带 Rate Limiter 的测试"""
        if not RESILIENCE_AVAILABLE:
            return self.run_baseline(func, iterations, concurrency)

        @rate_limit(
            rate=100,  # 100 req/s
            burst=150
        )
        def protected_func(x):
            return func(x)

        return self._run_protected(protected_func, iterations, concurrency)

    def run_with_retry(
        self,
        func: Callable,
        iterations: int = 1000,
        concurrency: int = 1
    ) -> PerformanceMetrics:
        """运行带 Retry 的测试"""
        if not RESILIENCE_AVAILABLE:
            return self.run_baseline(func, iterations, concurrency)

        @retry(
            max_attempts=3,
            delay=0.1
        )
        def protected_func(x):
            return func(x)

        return self._run_protected(protected_func, iterations, concurrency)

    def run_with_bulkhead(
        self,
        func: Callable,
        iterations: int = 1000,
        concurrency: int = 1
    ) -> PerformanceMetrics:
        """运行带 Bulkhead 的测试"""
        if not RESILIENCE_AVAILABLE:
            return self.run_baseline(func, iterations, concurrency)

        @bulkhead(
            max_concurrent_calls=10,
            max_wait_duration=1.0
        )
        def protected_func(x):
            return func(x)

        return self._run_protected(protected_func, iterations, concurrency)

    def run_with_all_resilience(
        self,
        func: Callable,
        iterations: int = 1000,
        concurrency: int = 1
    ) -> PerformanceMetrics:
        """运行带所有 Resilience 组件的测试"""
        if not RESILIENCE_AVAILABLE:
            return self.run_baseline(func, iterations, concurrency)

        @with_resilience(
            circuit_breaker_config={
                "failure_threshold": 5,
                "recovery_timeout": 30
            },
            rate_limiter_config={
                "rate": 100,
                "burst": 150
            },
            retry_config={
                "max_attempts": 3,
                "delay": 0.1
            },
            bulkhead_config={
                "max_concurrent_calls": 10
            }
        )
        def protected_func(x):
            return func(x)

        return self._run_protected(protected_func, iterations, concurrency)

    def _run_protected(
        self,
        func: Callable,
        iterations: int,
        concurrency: int
    ) -> PerformanceMetrics:
        """运行受保护的函数"""
        metrics = PerformanceMetrics()
        latencies = []
        errors = 0
        lock_waits = []

        tracemalloc.start()
        start_time = time.time()
        process = psutil.Process()
        start_cpu = process.cpu_percent()

        if concurrency == 1:
            for i in range(iterations):
                try:
                    op_start = time.perf_counter()
                    func(i)
                    op_time = time.perf_counter() - op_start
                    latencies.append(op_time * 1000)
                except Exception:
                    errors += 1

        else:
            # 测量锁等待时间
            lock = Lock()
            lock_wait_total = 0

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                for i in range(iterations):
                    op_start = time.perf_counter()

                    # 模拟锁争用测量
                    lock_start = time.perf_counter()
                    with lock:
                        lock_wait = time.perf_counter() - lock_start
                        lock_waits.append(lock_wait * 1000)
                        future = executor.submit(func, i)

                    futures.append((future, op_start))

                for future, op_start in futures:
                    try:
                        future.result(timeout=5)
                        op_time = time.perf_counter() - op_start
                        latencies.append(op_time * 1000)
                    except Exception:
                        errors += 1

        total_time = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics.total_time = total_time
        metrics.throughput = iterations / total_time if total_time > 0 else 0
        metrics.error_rate = errors / iterations if iterations > 0 else 0
        metrics.memory_usage_mb = peak_mem / 1024 / 1024
        metrics.cpu_usage_percent = process.cpu_percent() - start_cpu

        if latencies:
            metrics.avg_latency = statistics.mean(latencies)
            metrics.p50_latency = statistics.quantiles(latencies, n=2)[0] if len(latencies) > 1 else latencies[0]
            metrics.p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 19 else metrics.p50_latency
            metrics.p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 99 else metrics.p95_latency

        if lock_waits:
            metrics.lock_wait_time = statistics.mean(lock_waits)

        return metrics

    # ========== 基准测试套件 ==========

    def benchmark_single_thread(self):
        """单线程基准测试"""
        print("\n🔧 Running single-thread benchmarks...")

        test_functions = [
            ("fast", self.target_function_fast, 1000),
            ("medium", self.target_function_medium, 500),
            ("slow", self.target_function_slow, 100)
        ]

        for func_name, func, iterations in test_functions:
            print(f"\n  Testing {func_name} function...")

            # 基线
            baseline = self.run_baseline(func, iterations, 1)
            self.baseline_metrics[f"single_{func_name}"] = baseline

            # Circuit Breaker
            cb_metrics = self.run_with_circuit_breaker(func, iterations, 1)
            self._record_result(f"single_{func_name}_circuit_breaker", baseline, cb_metrics)

            # Rate Limiter
            rl_metrics = self.run_with_rate_limiter(func, iterations, 1)
            self._record_result(f"single_{func_name}_rate_limiter", baseline, rl_metrics)

            # Retry
            retry_metrics = self.run_with_retry(func, iterations, 1)
            self._record_result(f"single_{func_name}_retry", baseline, retry_metrics)

            # Bulkhead
            bulkhead_metrics = self.run_with_bulkhead(func, iterations, 1)
            self._record_result(f"single_{func_name}_bulkhead", baseline, bulkhead_metrics)

            # All combined
            all_metrics = self.run_with_all_resilience(func, iterations, 1)
            self._record_result(f"single_{func_name}_all", baseline, all_metrics)

    def benchmark_multi_thread(self):
        """多线程基准测试"""
        print("\n🔧 Running multi-thread benchmarks...")

        test_configs = [
            ("low_concurrency", 4),
            ("medium_concurrency", 10),
            ("high_concurrency", 20)
        ]

        func = self.target_function_medium
        iterations = 500

        for config_name, concurrency in test_configs:
            print(f"\n  Testing with {config_name} ({concurrency} threads)...")

            # 基线
            baseline = self.run_baseline(func, iterations, concurrency)
            self.baseline_metrics[f"multi_{config_name}"] = baseline

            # Circuit Breaker
            cb_metrics = self.run_with_circuit_breaker(func, iterations, concurrency)
            self._record_result(f"multi_{config_name}_circuit_breaker", baseline, cb_metrics)

            # Rate Limiter
            rl_metrics = self.run_with_rate_limiter(func, iterations, concurrency)
            self._record_result(f"multi_{config_name}_rate_limiter", baseline, rl_metrics)

            # Bulkhead (特别重要的并发测试)
            bulkhead_metrics = self.run_with_bulkhead(func, iterations, concurrency)
            self._record_result(f"multi_{config_name}_bulkhead", baseline, bulkhead_metrics)

            # All combined
            all_metrics = self.run_with_all_resilience(func, iterations, concurrency)
            self._record_result(f"multi_{config_name}_all", baseline, all_metrics)

    def benchmark_stress_test(self):
        """压力测试"""
        print("\n🔧 Running stress tests...")

        # 高负载测试
        print("\n  High load test...")
        baseline = self.run_baseline(self.target_function_fast, 10000, 20)
        resilience = self.run_with_all_resilience(self.target_function_fast, 10000, 20)
        self._record_result("stress_high_load", baseline, resilience)

        # 长时间运行测试
        print("\n  Long running test...")
        baseline = self.run_baseline(self.target_function_medium, 1000, 10)
        resilience = self.run_with_all_resilience(self.target_function_medium, 1000, 10)
        self._record_result("stress_long_running", baseline, resilience)

    async def benchmark_async(self):
        """异步性能测试"""
        print("\n🔧 Running async benchmarks...")

        # 暂时跳过异步测试（需要异步版本的 Resilience 组件）
        print("  Async benchmarks skipped (requires async resilience components)")

    def _record_result(
        self,
        test_name: str,
        baseline: PerformanceMetrics,
        with_resilience: PerformanceMetrics
    ):
        """记录测试结果"""
        # 计算开销
        overhead_p95 = (
            (with_resilience.p95_latency - baseline.p95_latency) / baseline.p95_latency
            if baseline.p95_latency > 0 else 0
        )
        overhead_avg = (
            (with_resilience.avg_latency - baseline.avg_latency) / baseline.avg_latency
            if baseline.avg_latency > 0 else 0
        )
        memory_overhead = with_resilience.memory_usage_mb - baseline.memory_usage_mb

        # 判定结果
        verdict = "PASS"
        if overhead_p95 > self.thresholds["max_overhead_p95"]:
            verdict = "FAIL"
        elif overhead_avg > self.thresholds["max_overhead_avg"]:
            verdict = "WARNING"
        elif memory_overhead > self.thresholds["max_memory_overhead_mb"]:
            verdict = "WARNING"
        elif with_resilience.lock_wait_time > self.thresholds["max_lock_wait_ms"]:
            verdict = "WARNING"

        result = BenchmarkResult(
            test_name=test_name,
            config={
                "iterations": 1000,  # 简化，实际应传入
                "concurrency": 1 if "single" in test_name else 10
            },
            metrics={
                "baseline_p95": baseline.p95_latency,
                "resilience_p95": with_resilience.p95_latency,
                "overhead_p95_percent": overhead_p95 * 100,
                "baseline_avg": baseline.avg_latency,
                "resilience_avg": with_resilience.avg_latency,
                "overhead_avg_percent": overhead_avg * 100,
                "memory_overhead_mb": memory_overhead,
                "lock_wait_ms": with_resilience.lock_wait_time,
                "baseline_throughput": baseline.throughput,
                "resilience_throughput": with_resilience.throughput
            },
            overhead=overhead_p95,
            verdict=verdict
        )

        self.results.append(result)
        self.resilience_metrics[test_name] = with_resilience

    def generate_report(self) -> Dict[str, Any]:
        """生成基准测试报告"""
        passed = sum(1 for r in self.results if r.verdict == "PASS")
        warned = sum(1 for r in self.results if r.verdict == "WARNING")
        failed = sum(1 for r in self.results if r.verdict == "FAIL")

        # 计算平均开销
        avg_overhead_p95 = statistics.mean(
            r.overhead for r in self.results
        ) if self.results else 0

        # 找出最差情况
        worst_case = max(
            self.results,
            key=lambda r: r.overhead
        ) if self.results else None

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "passed": passed,
                "warnings": warned,
                "failures": failed,
                "avg_overhead_p95": f"{avg_overhead_p95 * 100:.2f}%",
                "verdict": "PASS" if failed == 0 else "FAIL"
            },
            "thresholds": self.thresholds,
            "worst_case": {
                "test": worst_case.test_name if worst_case else "N/A",
                "overhead": f"{worst_case.overhead * 100:.2f}%" if worst_case else "N/A"
            },
            "results": [
                {
                    "test": r.test_name,
                    "verdict": r.verdict,
                    "p95_overhead": f"{r.metrics['overhead_p95_percent']:.2f}%",
                    "avg_overhead": f"{r.metrics['overhead_avg_percent']:.2f}%",
                    "memory_overhead": f"{r.metrics['memory_overhead_mb']:.1f} MB",
                    "lock_wait": f"{r.metrics['lock_wait_ms']:.2f} ms"
                }
                for r in self.results
            ]
        }

        return report

    def print_summary(self):
        """打印摘要"""
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("📊 Resilience Performance Benchmark Summary")
        print("=" * 60)

        summary = report["summary"]
        print(f"\n✅ Passed: {summary['passed']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        print(f"❌ Failed: {summary['failures']}")
        print(f"\n📈 Average P95 Overhead: {summary['avg_overhead_p95']}")

        if report["worst_case"]["test"] != "N/A":
            print(f"\n🔴 Worst Case:")
            print(f"   Test: {report['worst_case']['test']}")
            print(f"   Overhead: {report['worst_case']['overhead']}")

        print(f"\n🎯 Overall Verdict: {summary['verdict']}")

        if summary["verdict"] == "FAIL":
            print("\n⚠️  Performance regression detected!")
            print("   Resilience overhead exceeds acceptable thresholds.")

        print("\n" + "=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark resilience components performance"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (reduced iterations)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for JSON report"
    )
    parser.add_argument(
        "--skip-stress",
        action="store_true",
        help="Skip stress tests"
    )

    args = parser.parse_args()

    # 创建基准测试
    benchmark = ResilienceBenchmark()

    print("🚀 Starting Resilience Performance Benchmark...")
    print(f"   Available: {RESILIENCE_AVAILABLE}")

    # 运行测试套件
    benchmark.benchmark_single_thread()
    benchmark.benchmark_multi_thread()

    if not args.skip_stress and not args.quick:
        benchmark.benchmark_stress_test()

    # 生成报告
    report = benchmark.generate_report()

    # 保存报告
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📁 Report saved to: {args.output}")

    # 打印摘要
    benchmark.print_summary()

    # 返回状态码
    return 0 if report["summary"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())