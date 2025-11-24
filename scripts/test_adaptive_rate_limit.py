#!/usr/bin/env python3
"""
自适应限流系统测试套件
测试所有限流组件的功能和集成
"""

import os
import sys
import json
import time
import random
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import subprocess
import threading
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
try:
    from scripts.adaptive_rate_limiter import (
        AdaptiveRateLimiter,
        RateLimitConfig,
        SystemMetrics,
        Decision,
        Request,
        TokenBucketAlgorithm,
        LeakyBucketAlgorithm,
        SlidingWindowAlgorithm,
    )
    from scripts.rate_limit_analyzer import (
        RateLimitAnalyzer,
        PatternType,
        TrafficMetrics,
        UserProfile,
        AnomalyDetector,
        PatternDetector,
    )
    from scripts.auto_calibrator import (
        AutoCalibrator,
        OptimizationGoal,
        Parameters,
        PerformanceScore,
        TestStatus,
    )
    from scripts.performance_monitor import (
        PerformanceMonitor,
        SLAConfig,
        ComplianceStatus,
        ImpactLevel,
        AlertSeverity,
        MetricSnapshot,
    )
except ImportError:
    from adaptive_rate_limiter import (
        AdaptiveRateLimiter,
        RateLimitConfig,
        SystemMetrics,
        Decision,
        Request,
        TokenBucketAlgorithm,
        LeakyBucketAlgorithm,
        SlidingWindowAlgorithm,
    )
    from rate_limit_analyzer import (
        RateLimitAnalyzer,
        PatternType,
        TrafficMetrics,
        UserProfile,
        AnomalyDetector,
        PatternDetector,
    )
    from auto_calibrator import (
        AutoCalibrator,
        OptimizationGoal,
        Parameters,
        PerformanceScore,
        TestStatus,
    )
    from performance_monitor import (
        PerformanceMonitor,
        SLAConfig,
        ComplianceStatus,
        ImpactLevel,
        AlertSeverity,
        MetricSnapshot,
    )


def run_command(cmd: str, check: bool = False) -> tuple:
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


class TestAdaptiveRateLimiter:
    """测试自适应限流器"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def test_token_bucket(self):
        """测试令牌桶算法"""
        print("\n🔍 测试令牌桶算法...")

        config = RateLimitConfig(
            algorithm="token_bucket",
            rate_limit=10,  # 10 req/s
            burst_size=5
        )
        limiter = AdaptiveRateLimiter(config)

        # 测试突发请求
        allowed = 0
        for i in range(10):
            request = Request(
                id=f"req_{i}",
                user_id="test_user",
                ip="127.0.0.1",
                path="/api/test",
                timestamp=datetime.now()
            )
            metrics = SystemMetrics(
                cpu_usage=0.5,
                memory_usage=0.6,
                latency_p50=50,
                latency_p95=95,
                latency_p99=150,
                error_rate=0.01,
                request_rate=8,
                active_connections=100
            )
            decision = limiter.should_allow_request(request, metrics)
            if decision.allowed:
                allowed += 1

        # 前5个应该被允许（burst size）
        if allowed >= 5:
            self.passed += 1
            print(f"  ✅ 令牌桶突发处理正确: {allowed}/10 请求被允许")
            return True
        else:
            self.failed += 1
            print(f"  ❌ 令牌桶突发处理失败: 只有 {allowed}/10 请求被允许")
            return False

    def test_sliding_window(self):
        """测试滑动窗口算法"""
        print("\n🔍 测试滑动窗口算法...")

        config = RateLimitConfig(
            algorithm="sliding_window",
            rate_limit=100,
            window_size=1.0  # 1秒窗口
        )
        limiter = AdaptiveRateLimiter(config)

        # 测试窗口内请求计数
        metrics = SystemMetrics(
            cpu_usage=0.5, memory_usage=0.6,
            latency_p50=50, latency_p95=95, latency_p99=150,
            error_rate=0.01, request_rate=80, active_connections=100
        )

        # 发送请求直到被限流
        allowed_count = 0
        for i in range(150):
            request = Request(
                id=f"req_{i}",
                user_id="test_user",
                ip="127.0.0.1",
                path="/api/test",
                timestamp=datetime.now()
            )
            decision = limiter.should_allow_request(request, metrics)
            if decision.allowed:
                allowed_count += 1
            time.sleep(0.001)  # 小延迟

        # 应该接近但不超过rate_limit
        if 90 <= allowed_count <= 110:
            self.passed += 1
            print(f"  ✅ 滑动窗口限流正确: {allowed_count} 请求被允许")
            return True
        else:
            self.failed += 1
            print(f"  ❌ 滑动窗口限流失败: {allowed_count} 请求被允许（期望约100）")
            return False

    def test_adaptation(self):
        """测试自适应调整"""
        print("\n🔍 测试自适应调整...")

        config = RateLimitConfig(
            algorithm="adaptive_window",
            rate_limit=100,
            enable_adaptation=True
        )
        limiter = AdaptiveRateLimiter(config)

        # 模拟高负载
        high_load_metrics = SystemMetrics(
            cpu_usage=0.9,
            memory_usage=0.85,
            latency_p50=100,
            latency_p95=200,
            latency_p99=500,
            error_rate=0.05,
            request_rate=150,
            active_connections=500
        )

        # 触发自适应
        limiter.adapt_thresholds(high_load_metrics)
        new_threshold = limiter.get_effective_rate_limit()

        # 高负载时应该降低阈值
        if new_threshold < 100:
            self.passed += 1
            print(f"  ✅ 高负载自适应正确: 阈值降至 {new_threshold}")

            # 测试低负载恢复
            low_load_metrics = SystemMetrics(
                cpu_usage=0.3,
                memory_usage=0.4,
                latency_p50=30,
                latency_p95=60,
                latency_p99=90,
                error_rate=0.001,
                request_rate=50,
                active_connections=100
            )
            limiter.adapt_thresholds(low_load_metrics)
            recovered_threshold = limiter.get_effective_rate_limit()

            if recovered_threshold > new_threshold:
                print(f"  ✅ 低负载恢复正确: 阈值恢复至 {recovered_threshold}")
                return True

        self.failed += 1
        print(f"  ❌ 自适应调整失败")
        return False

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("\n" + "="*60)
        print("📊 测试自适应限流器")
        print("="*60)

        test_methods = [
            self.test_token_bucket,
            self.test_sliding_window,
            self.test_adaptation,
        ]

        for test in test_methods:
            try:
                test()
            except Exception as e:
                self.failed += 1
                print(f"  ❌ 测试异常: {e}")

        return {
            'component': 'AdaptiveRateLimiter',
            'passed': self.passed,
            'failed': self.failed,
            'total': self.passed + self.failed
        }


class TestRateLimitAnalyzer:
    """测试流量分析器"""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def test_pattern_detection(self):
        """测试流量模式检测"""
        print("\n🔍 测试流量模式检测...")

        analyzer = RateLimitAnalyzer()

        # 测试正常流量
        normal_metrics = TrafficMetrics(
            request_count=1000,
            unique_ips=100,
            avg_request_rate=100,
            peak_request_rate=150,
            error_rate=0.01,
            avg_response_time=50,
            bandwidth_usage=10 * 1024 * 1024,
            time_window=60
        )
        pattern = analyzer.classify_traffic_pattern(normal_metrics)

        if pattern == PatternType.NORMAL:
            self.passed += 1
            print(f"  ✅ 正常流量识别正确")
        else:
            self.failed += 1
            print(f"  ❌ 正常流量识别失败: {pattern}")

        # 测试DDoS模式
        ddos_metrics = TrafficMetrics(
            request_count=100000,
            unique_ips=10,
            avg_request_rate=5000,
            peak_request_rate=10000,
            error_rate=0.5,
            avg_response_time=500,
            bandwidth_usage=1000 * 1024 * 1024,
            time_window=60
        )
        pattern = analyzer.classify_traffic_pattern(ddos_metrics)

        if pattern == PatternType.DDOS:
            self.passed += 1
            print(f"  ✅ DDoS攻击识别正确")
            return True
        else:
            self.failed += 1
            print(f"  ❌ DDoS攻击识别失败: {pattern}")
            return False

    def test_anomaly_detection(self):
        """测试异常检测"""
        print("\n🔍 测试异常检测...")

        detector = AnomalyDetector()

        # 添加正常数据建立基线
        for _ in range(100):
            detector.add_sample(random.uniform(90, 110))

        # 测试正常值
        normal_score = detector.detect_anomaly(100)
        if normal_score < 0.5:
            self.passed += 1
            print(f"  ✅ 正常值检测正确: 异常分数 {normal_score:.2f}")
        else:
            self.failed += 1
            print(f"  ❌ 正常值检测失败: 异常分数 {normal_score:.2f}")

        # 测试异常值
        anomaly_score = detector.detect_anomaly(500)
        if anomaly_score > 0.7:
            self.passed += 1
            print(f"  ✅ 异常值检测正确: 异常分数 {anomaly_score:.2f}")
            return True
        else:
            self.failed += 1
            print(f"  ❌ 异常值检测失败: 异常分数 {anomaly_score:.2f}")
            return False

    def test_user_profiling(self):
        """测试用户行为画像"""
        print("\n🔍 测试用户行为画像...")

        analyzer = RateLimitAnalyzer()

        # 添加用户请求历史
        user_id = "test_user"
        for i in range(100):
            analyzer.record_request(
                user_id=user_id,
                timestamp=datetime.now() - timedelta(seconds=i),
                path="/api/test",
                response_time=random.uniform(40, 60),
                status_code=200
            )

        # 生成用户画像
        profile = analyzer.generate_user_profile(user_id)

        if profile and profile.reputation_score > 0:
            self.passed += 1
            print(f"  ✅ 用户画像生成成功: 信誉分 {profile.reputation_score:.2f}")
            return True
        else:
            self.failed += 1
            print(f"  ❌ 用户画像生成失败")
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("\n" + "="*60)
        print("📊 测试流量分析器")
        print("="*60)

        test_methods = [
            self.test_pattern_detection,
            self.test_anomaly_detection,
            self.test_user_profiling,
        ]

        for test in test_methods:
            try:
                test()
            except Exception as e:
                self.failed += 1
                print(f"  ❌ 测试异常: {e}")

        return {
            'component': 'RateLimitAnalyzer',
            'passed': self.passed,
            'failed': self.failed,
            'total': self.passed + self.failed
        }


class TestAutoCalibrator:
    """测试参数校准器"""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def test_parameter_optimization(self):
        """测试参数优化"""
        print("\n🔍 测试参数优化...")

        calibrator = AutoCalibrator(
            optimization_goal=OptimizationGoal.BALANCE_PERFORMANCE
        )

        # 初始参数
        params = Parameters(
            rate_limit=1000,
            burst_size=100,
            window_size=60,
            max_retries=3,
            backoff_factor=2.0,
            cpu_threshold=0.8,
            memory_threshold=0.85,
            latency_threshold=100,
            error_threshold=0.01
        )

        # 系统指标
        metrics = SystemMetrics(
            cpu_usage=0.7,
            memory_usage=0.75,
            latency_p50=60,
            latency_p95=120,
            latency_p99=200,
            error_rate=0.02,
            request_rate=800,
            active_connections=500
        )

        # 评估性能
        score = calibrator.evaluate_performance(params, metrics)

        if score.overall_score > 0:
            self.passed += 1
            print(f"  ✅ 性能评估成功: 综合评分 {score.overall_score:.2f}")
            return True
        else:
            self.failed += 1
            print(f"  ❌ 性能评估失败")
            return False

    def test_ab_testing(self):
        """测试A/B测试功能"""
        print("\n🔍 测试A/B测试...")

        calibrator = AutoCalibrator()

        params_a = Parameters(
            rate_limit=1000, burst_size=100, window_size=60,
            max_retries=3, backoff_factor=2.0,
            cpu_threshold=0.8, memory_threshold=0.85,
            latency_threshold=100, error_threshold=0.01
        )

        params_b = params_a.mutate(0.2)  # 20%变异

        # 运行短时间测试
        result = calibrator.run_ab_test(
            variant_a=params_a,
            variant_b=params_b,
            duration=1,  # 1秒快速测试
            traffic_split=0.5
        )

        if result.status == TestStatus.COMPLETED:
            self.passed += 1
            print(f"  ✅ A/B测试完成: 获胜者 {result.winner}, P值 {result.p_value:.4f}")
            return True
        else:
            self.failed += 1
            print(f"  ❌ A/B测试失败: {result.status.value}")
            return False

    def test_recommendation(self):
        """测试参数建议"""
        print("\n🔍 测试参数建议...")

        calibrator = AutoCalibrator()
        analyzer = RateLimitAnalyzer()

        # 创建流量分析
        from scripts.rate_limit_analyzer import TrafficAnalysis
        traffic_analysis = TrafficAnalysis(
            pattern=PatternType.SPIKE,
            confidence=0.8,
            anomaly_score=0.2,
            metrics=TrafficMetrics(
                request_count=10000,
                unique_ips=500,
                avg_request_rate=1500,
                peak_request_rate=3000,
                error_rate=0.02,
                avg_response_time=80,
                bandwidth_usage=100 * 1024 * 1024,
                time_window=60
            ),
            recommendations=[]
        )

        # 获取建议
        recommendations = calibrator.get_recommendation(traffic_analysis)

        if recommendations and 'suggestions' in recommendations:
            self.passed += 1
            print(f"  ✅ 建议生成成功: {len(recommendations['suggestions'])} 条建议")
            for suggestion in recommendations['suggestions'][:3]:
                print(f"    - {suggestion}")
            return True
        else:
            self.failed += 1
            print(f"  ❌ 建议生成失败")
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("\n" + "="*60)
        print("📊 测试参数校准器")
        print("="*60)

        test_methods = [
            self.test_parameter_optimization,
            self.test_ab_testing,
            self.test_recommendation,
        ]

        for test in test_methods:
            try:
                test()
            except Exception as e:
                self.failed += 1
                print(f"  ❌ 测试异常: {e}")

        return {
            'component': 'AutoCalibrator',
            'passed': self.passed,
            'failed': self.failed,
            'total': self.passed + self.failed
        }


class TestPerformanceMonitor:
    """测试性能监控器"""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def test_sla_compliance(self):
        """测试SLA合规性检查"""
        print("\n🔍 测试SLA合规性检查...")

        sla_config = SLAConfig(
            availability_target=0.999,
            latency_p95_target=100,
            error_rate_target=0.001
        )
        monitor = PerformanceMonitor(sla_config)

        # 添加正常指标
        for i in range(10):
            snapshot = MetricSnapshot(
                timestamp=datetime.now() - timedelta(seconds=i),
                throughput=1000,
                latency_p50=40,
                latency_p95=80,
                latency_p99=120,
                error_rate=0.0005,
                cpu_usage=0.6,
                memory_usage=0.65,
                rate_limited_requests=10,
                total_requests=1000,
                active_connections=500
            )
            monitor.add_metric_snapshot(snapshot)

        compliance = monitor.check_sla_compliance()

        if compliance == ComplianceStatus.COMPLIANT:
            self.passed += 1
            print(f"  ✅ SLA合规检查正确: {compliance.value}")
            return True
        else:
            self.failed += 1
            print(f"  ❌ SLA合规检查失败: {compliance.value}")
            return False

    def test_impact_monitoring(self):
        """测试影响监控"""
        print("\n🔍 测试影响监控...")

        monitor = PerformanceMonitor()

        # 添加退化指标
        for i in range(10):
            snapshot = MetricSnapshot(
                timestamp=datetime.now() - timedelta(seconds=i),
                throughput=500,  # 低吞吐量
                latency_p50=100,
                latency_p95=200,  # 高延迟
                latency_p99=500,
                error_rate=0.05,  # 高错误率
                cpu_usage=0.9,
                memory_usage=0.85,
                rate_limited_requests=300,  # 大量限流
                total_requests=1000,
                active_connections=500
            )
            monitor.add_metric_snapshot(snapshot)

        # 监控影响
        report = monitor.monitor_impact()

        if report.impact_level in [ImpactLevel.HIGH, ImpactLevel.SEVERE]:
            self.passed += 1
            print(f"  ✅ 影响评估正确: {report.impact_level.value}")
            if report.recommendations:
                print(f"    建议: {report.recommendations[0]}")
            return True
        else:
            self.failed += 1
            print(f"  ❌ 影响评估失败: {report.impact_level.value}")
            return False

    def test_alert_generation(self):
        """测试告警生成"""
        print("\n🔍 测试告警生成...")

        monitor = PerformanceMonitor()

        # 添加触发告警的指标
        for i in range(5):
            snapshot = MetricSnapshot(
                timestamp=datetime.now() - timedelta(seconds=i),
                throughput=100,
                latency_p50=200,
                latency_p95=500,
                latency_p99=1000,
                error_rate=0.1,
                cpu_usage=0.95,
                memory_usage=0.92,
                rate_limited_requests=500,
                total_requests=1000,
                active_connections=500
            )
            monitor.add_metric_snapshot(snapshot)

        alerts = monitor.generate_alerts()

        if alerts and any(a.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] for a in alerts):
            self.passed += 1
            print(f"  ✅ 告警生成成功: {len(alerts)} 个告警")
            for alert in alerts[:3]:
                print(f"    [{alert.severity.value}] {alert.message}")
            return True
        else:
            self.failed += 1
            print(f"  ❌ 告警生成失败")
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("\n" + "="*60)
        print("📊 测试性能监控器")
        print("="*60)

        test_methods = [
            self.test_sla_compliance,
            self.test_impact_monitoring,
            self.test_alert_generation,
        ]

        for test in test_methods:
            try:
                test()
            except Exception as e:
                self.failed += 1
                print(f"  ❌ 测试异常: {e}")

        return {
            'component': 'PerformanceMonitor',
            'passed': self.passed,
            'failed': self.failed,
            'total': self.passed + self.failed
        }


class TestIntegration:
    """集成测试"""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def test_end_to_end(self):
        """端到端测试"""
        print("\n🔍 运行端到端集成测试...")

        # 创建所有组件
        config = RateLimitConfig(
            algorithm="adaptive_window",
            rate_limit=1000,
            enable_adaptation=True
        )
        limiter = AdaptiveRateLimiter(config)
        analyzer = RateLimitAnalyzer()
        calibrator = AutoCalibrator()
        monitor = PerformanceMonitor()

        # 模拟流量
        print("  模拟流量处理...")
        for i in range(100):
            # 创建请求
            request = Request(
                id=f"req_{i}",
                user_id=f"user_{i % 10}",
                ip=f"192.168.1.{i % 256}",
                path="/api/test",
                timestamp=datetime.now()
            )

            # 获取系统指标
            metrics = SystemMetrics(
                cpu_usage=random.uniform(0.5, 0.8),
                memory_usage=random.uniform(0.6, 0.8),
                latency_p50=random.uniform(40, 60),
                latency_p95=random.uniform(80, 120),
                latency_p99=random.uniform(150, 200),
                error_rate=random.uniform(0.001, 0.01),
                request_rate=random.uniform(800, 1200),
                active_connections=random.randint(400, 600)
            )

            # 限流决策
            decision = limiter.should_allow_request(request, metrics)

            # 记录指标
            snapshot = MetricSnapshot(
                timestamp=datetime.now(),
                throughput=metrics.request_rate,
                latency_p50=metrics.latency_p50,
                latency_p95=metrics.latency_p95,
                latency_p99=metrics.latency_p99,
                error_rate=metrics.error_rate,
                cpu_usage=metrics.cpu_usage,
                memory_usage=metrics.memory_usage,
                rate_limited_requests=0 if decision.allowed else 1,
                total_requests=1,
                active_connections=metrics.active_connections
            )
            monitor.add_metric_snapshot(snapshot)

            time.sleep(0.01)

        # 分析流量
        print("  分析流量模式...")
        analysis = analyzer.analyze_traffic(60)

        # 检查SLA
        print("  检查SLA合规性...")
        compliance = monitor.check_sla_compliance()

        # 生成影响报告
        print("  生成影响报告...")
        impact = monitor.monitor_impact()

        # 验证结果
        if analysis and compliance and impact:
            self.passed += 1
            print(f"  ✅ 端到端测试成功")
            print(f"    - 流量模式: {analysis.pattern.value}")
            print(f"    - SLA状态: {compliance.value}")
            print(f"    - 影响级别: {impact.impact_level.value}")
            return True
        else:
            self.failed += 1
            print(f"  ❌ 端到端测试失败")
            return False

    def test_workflow(self):
        """测试CI/CD工作流"""
        print("\n🔍 测试CI/CD工作流...")

        workflow_file = ".github/workflows/adaptive-rate-limit-monitor.yml"

        if Path(workflow_file).exists():
            with open(workflow_file, 'r') as f:
                content = f.read()

            required_jobs = [
                'analyze-traffic-patterns',
                'check-performance-impact',
                'calibrate-parameters',
                'load-test-validation',
                'monitor-dashboard',
                'rollback-on-failure'
            ]

            missing_jobs = []
            for job in required_jobs:
                if job not in content:
                    missing_jobs.append(job)

            if not missing_jobs:
                self.passed += 1
                print(f"  ✅ CI/CD工作流配置正确")
                return True
            else:
                self.failed += 1
                print(f"  ❌ CI/CD工作流缺少job: {', '.join(missing_jobs)}")
                return False
        else:
            self.failed += 1
            print(f"  ❌ CI/CD工作流文件不存在")
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("\n" + "="*60)
        print("📊 集成测试")
        print("="*60)

        test_methods = [
            self.test_end_to_end,
            self.test_workflow,
        ]

        for test in test_methods:
            try:
                test()
            except Exception as e:
                self.failed += 1
                print(f"  ❌ 测试异常: {e}")

        return {
            'component': 'Integration',
            'passed': self.passed,
            'failed': self.failed,
            'total': self.passed + self.failed
        }


def test_command_line():
    """测试命令行工具"""
    print("\n" + "="*60)
    print("📊 测试命令行工具")
    print("="*60)

    commands = [
        ("python3 scripts/adaptive_rate_limiter.py --help", "限流器帮助"),
        ("python3 scripts/rate_limit_analyzer.py --help", "分析器帮助"),
        ("python3 scripts/auto_calibrator.py --help", "校准器帮助"),
        ("python3 scripts/performance_monitor.py --help", "监控器帮助"),
    ]

    passed = 0
    failed = 0

    for cmd, description in commands:
        success, stdout, stderr = run_command(cmd)
        if success:
            passed += 1
            print(f"  ✅ {description}: 命令执行成功")
        else:
            failed += 1
            print(f"  ❌ {description}: 命令执行失败")
            print(f"    错误: {stderr[:100]}")

    return {
        'component': 'CommandLine',
        'passed': passed,
        'failed': failed,
        'total': passed + failed
    }


def generate_test_report(results: List[Dict[str, Any]]):
    """生成测试报告"""
    print("\n" + "="*60)
    print("📊 测试总结报告")
    print("="*60)

    total_passed = sum(r['passed'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    total_tests = sum(r['total'] for r in results)

    print(f"\n总测试数: {total_tests}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_failed}")
    print(f"通过率: {(total_passed/total_tests*100):.1f}%")

    print("\n组件测试结果:")
    for result in results:
        status = "✅" if result['failed'] == 0 else "❌"
        print(f"  {status} {result['component']}: {result['passed']}/{result['total']} 通过")

    # 生成JSON报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'pass_rate': total_passed / total_tests * 100
        },
        'components': results
    }

    with open('test_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("\n测试报告已保存到 test_report.json")

    if total_failed == 0:
        print("\n🎉 所有测试通过！自适应限流系统正常工作！")
        return True
    else:
        print(f"\n⚠️ {total_failed} 个测试失败，请检查相关功能。")
        return False


def main():
    """主测试函数"""
    print("🚀 开始测试 Day 8-10 自适应限流系统...")
    print("="*60)

    # 检查Python版本
    if sys.version_info < (3, 8):
        print("⚠️ 警告: Python版本低于3.8，某些功能可能不兼容")

    results = []

    # 运行组件测试
    testers = [
        TestAdaptiveRateLimiter(),
        TestRateLimitAnalyzer(),
        TestAutoCalibrator(),
        TestPerformanceMonitor(),
        TestIntegration(),
    ]

    for tester in testers:
        result = tester.run_all_tests()
        results.append(result)

    # 测试命令行工具
    cmd_result = test_command_line()
    results.append(cmd_result)

    # 生成报告
    success = generate_test_report(results)

    # 生成使用建议
    print("\n💡 使用建议:")
    print("1. 启动监控:")
    print("   python3 scripts/performance_monitor.py --monitor")
    print("\n2. 分析流量:")
    print("   python3 scripts/rate_limit_analyzer.py --time-window 300")
    print("\n3. 运行校准:")
    print("   python3 scripts/auto_calibrator.py --goal balance")
    print("\n4. 启用CI/CD:")
    print("   在GitHub仓库启用 adaptive-rate-limit-monitor.yml 工作流")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()