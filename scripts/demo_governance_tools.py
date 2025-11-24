#!/usr/bin/env python3
"""
Day 1-4 治理工具演示脚本
Demonstrates the governance tools in action
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd):
    """运行命令并返回输出"""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stdout, result.stderr


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(f"🔧 {title}")
    print("=" * 60)


def demo_release_risk_scorer():
    """演示发布风险评分器"""
    print_section("发布风险评分器演示")

    print("\n1. 评估当前分支风险...")
    cmd = "python3 scripts/release_risk_scorer.py --base-branch main --output-format markdown"
    success, stdout, stderr = run_command(cmd)

    if success:
        print(stdout)
    else:
        print(f"❌ 错误: {stderr}")

    print("\n2. JSON格式输出...")
    cmd = "python3 scripts/release_risk_scorer.py --base-branch main --output-format json"
    success, stdout, stderr = run_command(cmd)

    if success:
        data = json.loads(stdout)
        print(f"📊 风险分数: {data['score']}/100")
        print(f"📈 风险等级: {data['level']}")
        print(f"🚦 是否阻断: {'是' if data['blocking'] else '否'}")

        print("\n各维度评分:")
        for dim, score in data['parts'].items():
            percentage = score * 100
            bar = "█" * int(percentage / 10) + "░" * (10 - int(percentage / 10))
            print(f"  - {dim:<15} {bar} {percentage:.1f}%")


def demo_error_code_scanner():
    """演示错误码扫描器"""
    print_section("错误码扫描器演示")

    print("\n扫描项目中的错误码...")
    cmd = "python3 scripts/error_code_scanner.py --format json"
    success, stdout, stderr = run_command(cmd)

    if success:
        try:
            data = json.loads(stdout)
            summary = data.get('summary', {})

            print(f"\n📊 扫描结果:")
            print(f"  - 总错误码数: {summary.get('total_codes', 0)}")
            print(f"  - 活跃错误码: {summary.get('active_codes', 0)}")
            print(f"  - 未使用错误码: {summary.get('unused_codes', 0)}")
            print(f"  - 重复错误码: {summary.get('duplicate_codes', 0)}")
            print(f"  - 活跃率: {summary.get('active_rate', 0):.1f}%")

            # 显示分类
            if 'error_codes' in data:
                print("\n🏷️ 错误码分类:")
                categories = {}
                for code_info in data['error_codes']:
                    status = code_info.get('status', 'UNKNOWN')
                    categories[status] = categories.get(status, 0) + 1

                for status, count in sorted(categories.items()):
                    emoji = {
                        'ACTIVE': '✅',
                        'RARE': '⚠️',
                        'UNUSED': '❌',
                        'DEPRECATED': '🔄',
                        'DUPLICATE': '👥',
                        'ORPHAN': '👻',
                        'ZOMBIE': '💀'
                    }.get(status, '❓')
                    print(f"  {emoji} {status}: {count}个")

        except json.JSONDecodeError:
            print("⚠️ 输出不是有效的JSON")
    else:
        print("⚠️ 错误码扫描未找到定义（可能是正常的）")


def demo_error_code_lifecycle():
    """演示错误码生命周期管理"""
    print_section("错误码生命周期管理演示")

    print("\n生成错误码清理计划...")
    cmd = "python3 scripts/error_code_lifecycle.py --plan --format json"
    success, stdout, stderr = run_command(cmd)

    if success:
        try:
            data = json.loads(stdout)
            plan = data.get('cleanup_plan', {})

            print("\n🧹 清理计划:")
            print(f"  - 立即删除: {len(plan.get('immediate_removal', []))}个")
            print(f"  - 标记弃用: {len(plan.get('deprecation', []))}个")
            print(f"  - 合并重复: {len(plan.get('consolidation', []))}个")
            print(f"  - 调查孤立: {len(plan.get('investigation', []))}个")

            total = (len(plan.get('immediate_removal', [])) +
                    len(plan.get('deprecation', [])) +
                    len(plan.get('consolidation', [])))

            if total > 0:
                print(f"\n💡 建议: 可以清理 {total} 个错误码，提高代码质量")
            else:
                print("\n✅ 错误码管理良好，无需清理")

        except json.JSONDecodeError:
            print("⚠️ 输出不是有效的JSON")
    else:
        print("⚠️ 生命周期分析未完成")


def demo_data_collector():
    """演示数据收集器"""
    print_section("发布数据收集器演示")

    print("\n收集发布相关数据...")
    cmd = "python3 scripts/release_data_collector.py --base-branch main --output -"
    success, stdout, stderr = run_command(cmd)

    if success:
        data = json.loads(stdout)

        print("\n📊 Git统计:")
        git_data = data.get('git', {})
        print(f"  - 文件变更: {git_data.get('files_changed', 0)}")
        print(f"  - 代码新增: +{git_data.get('lines_added', 0)}")
        print(f"  - 代码删除: -{git_data.get('lines_deleted', 0)}")

        areas = git_data.get('by_area', {})
        if areas:
            print("\n🗂️ 按区域分布:")
            for area, count in areas.items():
                if count > 0:
                    print(f"  - {area}: {count}个文件")

        print("\n📦 依赖变化:")
        deps = data.get('deps', {})
        print(f"  - 新增依赖: {deps.get('added', 0)}")
        print(f"  - 删除依赖: {deps.get('removed', 0)}")

        print("\n🔧 错误码变化:")
        errors = data.get('errors', {})
        print(f"  - 新增错误码: {errors.get('added', 0)}")
        print(f"  - 删除错误码: {errors.get('removed', 0)}")

        print("\n📈 指标变化:")
        metrics = data.get('metrics', {})
        print(f"  - 新增指标: {metrics.get('added', 0)}")
        print(f"  - 删除指标: {metrics.get('removed', 0)}")


def demo_ci_integration():
    """演示CI/CD集成"""
    print_section("CI/CD集成说明")

    print("\n📋 GitHub Actions工作流:")

    workflows = [
        (".github/workflows/release-risk-check.yml", "发布风险检查"),
        (".github/workflows/error-code-cleanup.yml", "错误码月度清理")
    ]

    for workflow, desc in workflows:
        if Path(workflow).exists():
            print(f"\n✅ {desc}: {workflow}")
            with open(workflow, 'r') as f:
                lines = f.readlines()[:10]
                for line in lines:
                    if 'name:' in line:
                        print(f"   名称: {line.split(':', 1)[1].strip()}")
                    elif 'on:' in line:
                        print("   触发条件:")
                    elif 'schedule:' in line:
                        print("     - 定时执行")
                    elif 'pull_request:' in line:
                        print("     - Pull Request")
                    elif 'workflow_dispatch:' in line:
                        print("     - 手动触发")
        else:
            print(f"\n❌ {desc}: 文件不存在")

    print("\n💡 集成建议:")
    print("1. 在PR中自动运行风险评分，高风险自动阻断")
    print("2. 每月1号自动扫描和清理未使用的错误码")
    print("3. 使用风险评分作为发布决策的重要参考")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🚀 CAD ML Platform - Day 1-4 治理工具演示")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 演示各个工具
    demo_release_risk_scorer()
    demo_error_code_scanner()
    demo_error_code_lifecycle()
    demo_data_collector()
    demo_ci_integration()

    # 总结
    print("\n" + "=" * 60)
    print("🎯 演示完成！")
    print("=" * 60)
    print("\n核心价值:")
    print("1. 🎯 发布风险量化: 8维度评分，科学决策")
    print("2. 🧹 错误码自动治理: 生命周期管理，保持整洁")
    print("3. 🤖 CI/CD集成: 自动化执行，持续改进")
    print("4. 📊 数据驱动: 基于实际数据的治理决策")

    print("\n下一步行动:")
    print("1. 配置GitHub Actions工作流")
    print("2. 根据项目特点调整风险权重")
    print("3. 设置错误码清理策略")
    print("4. 监控治理效果，持续优化")


if __name__ == "__main__":
    main()