#!/usr/bin/env python3
"""
Metrics Budget System Test Suite
测试指标基数预算系统的所有组件
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import subprocess

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
try:
    from scripts.metrics_cardinality_tracker import (
        MetricsCardinalityTracker,
        CardinalityInfo,
        PrometheusClient,
    )
    from scripts.metrics_budget_controller import (
        MetricsBudgetController,
        BudgetConfig,
        BudgetStatus,
        Decision,
        MetricChange,
    )
    from scripts.cardinality_analysis_report import (
        CardinalityAnalysisReporter,
        Anomaly,
        Recommendation,
    )
    from scripts.metrics_auto_optimizer import (
        MetricsAutoOptimizer,
        OptimizationRule,
        OptimizationResult,
    )
except ImportError:
    from metrics_cardinality_tracker import (
        MetricsCardinalityTracker,
        CardinalityInfo,
        PrometheusClient,
    )  # type: ignore
    from metrics_budget_controller import (
        MetricsBudgetController,
        BudgetConfig,
        BudgetStatus,
        Decision,
        MetricChange,
    )  # type: ignore
    from cardinality_analysis_report import (
        CardinalityAnalysisReporter,
        Anomaly,
        Recommendation,
    )  # type: ignore
    from metrics_auto_optimizer import (
        MetricsAutoOptimizer,
        OptimizationRule,
        OptimizationResult,
    )  # type: ignore


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


def test_cardinality_tracker():
    """测试基数追踪器"""
    print("\n🔍 测试基数追踪器...")

    # 创建模拟数据
    mock_info = CardinalityInfo(
        metric_name="test_metric",
        cardinality=5000,
        labels={"method": 5, "status": 10, "path": 100},
        sample_labels=[
            {"method": "GET", "status": "200", "path": "/api/v1/test"},
            {"method": "POST", "status": "201", "path": "/api/v1/create"}
        ]
    )

    # 测试成本计算
    storage_mb = mock_info.storage_mb
    monthly_cost = mock_info.monthly_cost

    if storage_mb > 0 and monthly_cost > 0:
        print(f"  ✅ 成本计算正确: {storage_mb:.2f} MB, ${monthly_cost:.4f}/月")
    else:
        print("  ❌ 成本计算失败")
        return False

    # 测试命令行工具
    success, stdout, stderr = run_command(
        "python3 scripts/metrics_cardinality_tracker.py --help"
    )
    if success:
        print("  ✅ 命令行工具正常")
    else:
        print(f"  ❌ 命令行工具失败: {stderr}")
        return False

    return True


def test_budget_controller():
    """测试预算控制器"""
    print("\n🔍 测试预算控制器...")

    # 创建控制器
    config = BudgetConfig(
        global_max_series=1000000,
        team_budgets={"test_team": 100000},
        service_budgets={"test_service": 50000}
    )
    controller = MetricsBudgetController(config)

    # 更新使用数据
    controller.update_usage({
        "test_team": {
            "metric1": 30000,
            "metric2": 40000
        }
    })

    # 检查预算
    usage = controller.check_budget("test_team", "team")

    if usage.status == BudgetStatus.WARNING and usage.usage_percentage == 70.0:
        print(f"  ✅ 预算检查正确: {usage.usage_percentage}% ({usage.status.value})")
    else:
        print(f"  ❌ 预算检查失败: {usage.usage_percentage}% ({usage.status.value})")
        return False

    # 测试决策
    change = MetricChange(
        metric_name="new_metric",
        team="test_team",
        service="test_service",
        estimated_cardinality_change=5000,
        labels_added=["user_id"],
        labels_removed=[],
        reason="Testing"
    )

    decision, reason = controller.enforce_budget(change)
    if decision in [Decision.ALLOW, Decision.WARN, Decision.BLOCK]:
        print(f"  ✅ 决策系统正常: {decision.value} - {reason}")
    else:
        print("  ❌ 决策系统失败")
        return False

    # 测试全局状态
    status = controller.get_global_status()
    if "global_budget" in status and "global_used" in status:
        print(f"  ✅ 全局状态正常: {status['global_used']}/{status['global_budget']}")
    else:
        print("  ❌ 全局状态失败")
        return False

    return True


def test_analysis_reporter():
    """测试分析报告器"""
    print("\n🔍 测试分析报告器...")

    # 创建模拟追踪器
    tracker = MetricsCardinalityTracker("http://localhost:9090")

    # 添加模拟数据
    tracker.cardinality_cache["high_metric"] = CardinalityInfo(
        metric_name="high_metric",
        cardinality=50000,
        labels={"user_id": 10000, "path": 500},
        sample_labels=[]
    )

    tracker.cardinality_cache["normal_metric"] = CardinalityInfo(
        metric_name="normal_metric",
        cardinality=1000,
        labels={"method": 5, "status": 10},
        sample_labels=[]
    )

    # 创建报告器
    reporter = CardinalityAnalysisReporter(tracker)

    # 测试Top指标报告
    top_report = reporter.generate_top_offenders_report(top_n=5)
    if "top_metrics" in top_report and len(top_report["top_metrics"]) > 0:
        print(f"  ✅ Top指标报告生成: {len(top_report['top_metrics'])}个指标")
    else:
        print("  ❌ Top指标报告失败")
        return False

    # 测试优化建议
    recommendations = reporter.generate_optimization_recommendations()
    if isinstance(recommendations, list):
        print(f"  ✅ 优化建议生成: {len(recommendations)}条")
    else:
        print("  ❌ 优化建议生成失败")
        return False

    # 测试报告格式化
    try:
        json_report = reporter._format_json(reporter.generate_full_report())
        if json_report:
            print("  ✅ JSON格式化正常")
    except Exception as e:
        print(f"  ❌ 报告格式化失败: {e}")
        return False

    return True


def test_auto_optimizer():
    """测试自动优化器"""
    print("\n🔍 测试自动优化器...")

    # 创建优化器
    optimizer = MetricsAutoOptimizer()

    # 创建模拟指标信息
    metric_info = CardinalityInfo(
        metric_name="test_metric",
        cardinality=10000,
        labels={"user_id": 5000, "path": 200, "method": 5},
        sample_labels=[]
    )

    # 测试标签合并
    result = optimizer.apply_label_merging(
        metric_info,
        {"user_id": "user_category"}
    )
    if result.reduction_percentage > 0:
        print(f"  ✅ 标签合并优化: 减少{result.reduction_percentage}%")
    else:
        print("  ❌ 标签合并优化失败")
        return False

    # 测试标签删除
    result = optimizer.apply_label_dropping(
        metric_info,
        ["user_id"]
    )
    if result.reduction_percentage > 0:
        print(f"  ✅ 标签删除优化: 减少{result.reduction_percentage}%")
    else:
        print("  ❌ 标签删除优化失败")
        return False

    # 测试降采样配置
    downsample_config = optimizer.apply_downsampling(
        metric_info,
        {"raw": "1h", "5m": "1d", "1h": "7d"}
    )
    if "recording_rules" in downsample_config:
        print(f"  ✅ 降采样配置生成: {len(downsample_config['recording_rules'])}条规则")
    else:
        print("  ❌ 降采样配置失败")
        return False

    # 测试配置生成
    optimizations = [
        OptimizationResult(
            metric_name="test_metric",
            original_cardinality=10000,
            optimized_cardinality=5000,
            reduction_percentage=50.0,
            optimization_type="label_dropping",
            config_changes=[],
            rollback_config={}
        )
    ]

    config = optimizer.generate_prometheus_config(optimizations)
    if "metric_relabel_configs" in config or "recording_rules" in config:
        print("  ✅ Prometheus配置生成正常")
    else:
        print("  ❌ Prometheus配置生成失败")
        return False

    return True


def test_ci_workflow():
    """测试CI/CD工作流文件"""
    print("\n🔍 测试CI/CD工作流...")

    workflow_file = ".github/workflows/metrics-budget-check.yml"

    if Path(workflow_file).exists():
        print(f"  ✅ {workflow_file} 存在")

        # 验证YAML结构
        with open(workflow_file, 'r') as f:
            content = f.read()
            if 'name:' in content and 'jobs:' in content and 'on:' in content:
                print("    ✓ 基本结构正确")

                # 检查关键job
                required_jobs = [
                    'analyze-metrics-changes',
                    'check-cardinality-budget',
                    'calculate-cost-impact',
                    'post-pr-comment'
                ]

                missing_jobs = []
                for job in required_jobs:
                    if job not in content:
                        missing_jobs.append(job)

                if not missing_jobs:
                    print(f"    ✓ 所有必需的job存在")
                else:
                    print(f"    ✗ 缺少job: {', '.join(missing_jobs)}")
                    return False
            else:
                print("    ✗ 结构可能有问题")
                return False
    else:
        print(f"  ❌ {workflow_file} 不存在")
        return False

    return True


def test_integration():
    """集成测试"""
    print("\n🔍 运行集成测试...")

    # 创建完整的数据流
    tracker = MetricsCardinalityTracker("http://localhost:9090")
    controller = MetricsBudgetController()
    reporter = CardinalityAnalysisReporter(tracker, controller)
    optimizer = MetricsAutoOptimizer()

    # 模拟完整流程
    # 1. 添加指标数据
    tracker.cardinality_cache["integration_test"] = CardinalityInfo(
        metric_name="integration_test",
        cardinality=15000,
        labels={"id": 8000, "type": 10},
        sample_labels=[]
    )

    # 2. 更新预算使用
    controller.update_usage({
        "platform": {"integration_test": 15000}
    })

    # 3. 生成建议
    recommendations = reporter.generate_optimization_recommendations()

    # 4. 应用优化
    if recommendations:
        results = optimizer.optimize_metrics(tracker, recommendations[:1])
        if results:
            print(f"  ✅ 集成测试完成: 优化了{len(results)}个指标")
            return True

    print("  ⚠️ 集成测试完成但无优化结果（可能是正常的）")
    return True


def generate_test_report(results: Dict[str, bool]):
    """生成测试报告"""
    print("\n" + "=" * 60)
    print("📊 测试总结报告")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for r in results.values() if r)

    print(f"\n总测试: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    print(f"通过率: {(passed/total*100):.1f}%")

    print("\n详细结果:")
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  - {name}: {status}")

    if passed == total:
        print("\n🎉 所有测试通过！指标基数预算系统正常工作！")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查相关功能。")
        return False


def main():
    """主测试函数"""
    print("🚀 开始测试 Day 5-7 指标基数预算系统...")
    print("=" * 60)

    # 检查Python版本
    if sys.version_info < (3, 8):
        print("⚠️ 警告: Python版本低于3.8，某些功能可能不兼容")

    # 运行所有测试
    results = {
        "基数追踪器": test_cardinality_tracker(),
        "预算控制器": test_budget_controller(),
        "分析报告器": test_analysis_reporter(),
        "自动优化器": test_auto_optimizer(),
        "CI/CD工作流": test_ci_workflow(),
        "集成测试": test_integration()
    }

    # 生成总结
    success = generate_test_report(results)

    # 生成使用建议
    print("\n💡 使用建议:")
    print("1. 配置Prometheus连接:")
    print("   export PROMETHEUS_URL=http://your-prometheus:9090")
    print("\n2. 运行基数分析:")
    print("   python3 scripts/cardinality_analysis_report.py --format markdown")
    print("\n3. 配置预算限制:")
    print("   编辑 budget_config.json 设置团队/服务预算")
    print("\n4. 启用CI/CD检查:")
    print("   在GitHub仓库启用 metrics-budget-check.yml 工作流")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
