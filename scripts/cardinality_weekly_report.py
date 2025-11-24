#!/usr/bin/env python3
"""
Cardinality Weekly Report Generator
指标基数周报生成器 - 自动生成基数分析报告和优化建议

功能：
1. 周度基数增长分析
2. 自动阈值计算和建议
3. 标签组合分析
4. 优化建议生成
5. 历史趋势对比
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import requests
import statistics
from collections import defaultdict


class ActionType(Enum):
    """建议动作类型"""
    MERGE = "merge"           # 合并标签
    PRUNE = "label-prune"     # 裁剪标签
    KEEP = "keep"             # 保持现状
    WATCH = "watch"           # 持续观察
    URGENT = "urgent"         # 紧急处理


@dataclass
class MetricCardinality:
    """指标基数信息"""
    metric_name: str
    cardinality: int
    label_dimensions: List[str]
    growth_rate: float = 0.0
    weekly_change: int = 0
    monthly_trend: str = ""
    action: ActionType = ActionType.KEEP
    recommendation: str = ""
    label_combinations: Dict[str, int] = field(default_factory=dict)


@dataclass
class CardinalityThreshold:
    """基数阈值配置"""
    warning: int = 100
    critical: int = 1000
    growth_rate_warning: float = 0.1  # 10%
    growth_rate_critical: float = 0.2  # 20%
    max_label_dimensions: int = 3
    max_combinations: int = 500


class CardinalityWeeklyReporter:
    """基数周报生成器"""

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.history_file = Path(".cardinality_history.json")
        self.history = self.load_history()
        self.thresholds = CardinalityThreshold()

        # 白名单组合（允许的标签组合）
        self.allowed_combinations = [
            ["provider", "status"],
            ["provider", "error_code"],
            ["endpoint", "method"],
            ["stage", "status"]
        ]

        # 禁止的组合（容易导致基数爆炸）
        self.forbidden_combinations = [
            ["provider", "error_code", "stage", "severity"],
            ["user_id", "endpoint", "timestamp"],
            ["request_id", "any"]
        ]

    def load_history(self) -> Dict[str, Any]:
        """加载历史数据"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {
            "weeks": [],
            "metrics": {},
            "trends": {}
        }

    def save_history(self):
        """保存历史数据"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def query_prometheus(self, query: str) -> List[Dict[str, Any]]:
        """查询 Prometheus"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query}
            )
            response.raise_for_status()
            data = response.json()

            if data["status"] == "success":
                return data["data"]["result"]
            return []
        except Exception as e:
            print(f"Error querying Prometheus: {e}")
            return []

    def get_metric_cardinality(self) -> Dict[str, MetricCardinality]:
        """获取所有指标的基数"""
        metrics = {}

        # 查询所有指标
        all_metrics_query = "group by(__name__)({__name__=~'.+'})"
        results = self.query_prometheus(all_metrics_query)

        for result in results:
            metric_name = result["metric"].get("__name__", "unknown")

            # 查询该指标的基数
            cardinality_query = f"count(count by(__name__, {{{','.join(['instance', 'job'])}}})({metric_name}))"
            card_results = self.query_prometheus(cardinality_query)

            if card_results:
                cardinality = int(float(card_results[0]["value"][1]))

                # 获取标签维度
                labels_query = f"group by({{{','.join(['instance', 'job'])}}})({metric_name})"
                label_results = self.query_prometheus(labels_query)

                label_dimensions = []
                if label_results and label_results[0]["metric"]:
                    label_dimensions = list(label_results[0]["metric"].keys())
                    label_dimensions = [l for l in label_dimensions if l != "__name__"]

                metrics[metric_name] = MetricCardinality(
                    metric_name=metric_name,
                    cardinality=cardinality,
                    label_dimensions=label_dimensions
                )

        return metrics

    def calculate_growth_rates(self, current: Dict[str, MetricCardinality]):
        """计算增长率"""
        if not self.history["weeks"]:
            return

        # 获取上周数据
        last_week = self.history["weeks"][-1] if self.history["weeks"] else {}
        last_week_metrics = last_week.get("metrics", {})

        for metric_name, metric_info in current.items():
            if metric_name in last_week_metrics:
                last_cardinality = last_week_metrics[metric_name]["cardinality"]
                metric_info.weekly_change = metric_info.cardinality - last_cardinality

                if last_cardinality > 0:
                    metric_info.growth_rate = (
                        (metric_info.cardinality - last_cardinality) / last_cardinality
                    )

            # 计算月度趋势
            metric_info.monthly_trend = self._calculate_monthly_trend(metric_name)

    def _calculate_monthly_trend(self, metric_name: str) -> str:
        """计算月度趋势"""
        if metric_name not in self.history["trends"]:
            return "new"

        trend_data = self.history["trends"][metric_name]
        if len(trend_data) < 4:
            return "insufficient_data"

        # 获取最近4周的数据
        recent_values = trend_data[-4:]
        avg_value = statistics.mean(recent_values)

        # 计算斜率
        x = list(range(len(recent_values)))
        y = recent_values

        n = len(x)
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / \
                (n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2)

        if abs(slope / avg_value) < 0.05:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def analyze_label_combinations(self, metrics: Dict[str, MetricCardinality]):
        """分析标签组合"""
        for metric_name, metric_info in metrics.items():
            if len(metric_info.label_dimensions) > 1:
                # 分析标签组合
                combinations = self._get_label_combinations(
                    metric_name,
                    metric_info.label_dimensions
                )
                metric_info.label_combinations = combinations

                # 检查禁止的组合
                for forbidden in self.forbidden_combinations:
                    if self._matches_combination(metric_info.label_dimensions, forbidden):
                        metric_info.action = ActionType.URGENT
                        metric_info.recommendation = f"Forbidden label combination detected: {forbidden}"
                        break

    def _get_label_combinations(
        self,
        metric_name: str,
        labels: List[str]
    ) -> Dict[str, int]:
        """获取标签组合统计"""
        combinations = {}

        # 生成所有可能的标签组合
        from itertools import combinations as iter_combinations

        for r in range(2, min(len(labels) + 1, 4)):  # 最多考虑3个标签的组合
            for combo in iter_combinations(labels, r):
                combo_str = "+".join(sorted(combo))

                # 查询该组合的基数
                query = f"count(count by({','.join(combo)})({metric_name}))"
                results = self.query_prometheus(query)

                if results:
                    cardinality = int(float(results[0]["value"][1]))
                    combinations[combo_str] = cardinality

        return combinations

    def _matches_combination(self, labels: List[str], pattern: List[str]) -> bool:
        """检查标签是否匹配模式"""
        if "any" in pattern:
            return any(label in labels for label in pattern if label != "any")

        return all(label in labels for label in pattern)

    def calculate_automatic_thresholds(
        self,
        metrics: Dict[str, MetricCardinality]
    ) -> CardinalityThreshold:
        """自动计算阈值建议"""
        if not metrics:
            return self.thresholds

        cardinalities = [m.cardinality for m in metrics.values()]

        # 计算统计值
        p50 = statistics.quantiles(cardinalities, n=2)[0] if len(cardinalities) > 1 else cardinalities[0]
        p75 = statistics.quantiles(cardinalities, n=4)[2] if len(cardinalities) > 3 else p50
        p95 = statistics.quantiles(cardinalities, n=20)[18] if len(cardinalities) > 19 else p75
        p99 = statistics.quantiles(cardinalities, n=100)[98] if len(cardinalities) > 99 else p95

        # 建议阈值
        suggested = CardinalityThreshold()
        suggested.warning = int(p75 * 1.5)
        suggested.critical = int(p95 * 2)

        # 限制最小值
        suggested.warning = max(suggested.warning, 100)
        suggested.critical = max(suggested.critical, 500)

        # 增长率阈值基于历史数据
        if self.history["weeks"]:
            growth_rates = []
            for metric in metrics.values():
                if metric.growth_rate != 0:
                    growth_rates.append(abs(metric.growth_rate))

            if growth_rates:
                avg_growth = statistics.mean(growth_rates)
                suggested.growth_rate_warning = max(avg_growth * 2, 0.1)
                suggested.growth_rate_critical = max(avg_growth * 3, 0.2)

        return suggested

    def generate_recommendations(
        self,
        metrics: Dict[str, MetricCardinality]
    ) -> List[Tuple[MetricCardinality, ActionType, str]]:
        """生成优化建议"""
        recommendations = []

        for metric in metrics.values():
            # 基数过高
            if metric.cardinality > self.thresholds.critical:
                metric.action = ActionType.URGENT
                metric.recommendation = f"Critical cardinality ({metric.cardinality}), needs immediate optimization"
                recommendations.append((metric, ActionType.URGENT, metric.recommendation))

            # 增长过快
            elif metric.growth_rate > self.thresholds.growth_rate_critical:
                metric.action = ActionType.PRUNE
                metric.recommendation = f"Rapid growth ({metric.growth_rate:.1%}), consider pruning labels"
                recommendations.append((metric, ActionType.PRUNE, metric.recommendation))

            # 标签维度过多
            elif len(metric.label_dimensions) > self.thresholds.max_label_dimensions:
                metric.action = ActionType.MERGE
                metric.recommendation = f"Too many dimensions ({len(metric.label_dimensions)}), consider merging"
                recommendations.append((metric, ActionType.MERGE, metric.recommendation))

            # 警告级别
            elif metric.cardinality > self.thresholds.warning:
                metric.action = ActionType.WATCH
                metric.recommendation = f"Approaching warning threshold, monitor closely"
                recommendations.append((metric, ActionType.WATCH, metric.recommendation))

            # 稳定增长
            elif metric.monthly_trend == "increasing" and metric.growth_rate > 0.05:
                metric.action = ActionType.WATCH
                metric.recommendation = "Steady growth detected, continue monitoring"
                recommendations.append((metric, ActionType.WATCH, metric.recommendation))

        return sorted(recommendations, key=lambda x: x[1].value)

    def generate_report(self, output_format: str = "markdown") -> str:
        """生成周报"""
        print("📊 Collecting metrics cardinality...")
        metrics = self.get_metric_cardinality()

        print("📈 Calculating growth rates...")
        self.calculate_growth_rates(metrics)

        print("🔍 Analyzing label combinations...")
        self.analyze_label_combinations(metrics)

        print("🎯 Calculating automatic thresholds...")
        suggested_thresholds = self.calculate_automatic_thresholds(metrics)

        print("💡 Generating recommendations...")
        recommendations = self.generate_recommendations(metrics)

        # 更新历史
        week_data = {
            "week": datetime.now().strftime("%Y-W%U"),
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                name: {
                    "cardinality": m.cardinality,
                    "labels": m.label_dimensions,
                    "growth_rate": m.growth_rate
                }
                for name, m in metrics.items()
            },
            "total_cardinality": sum(m.cardinality for m in metrics.values())
        }

        self.history["weeks"].append(week_data)

        # 更新趋势数据
        for name, metric in metrics.items():
            if name not in self.history["trends"]:
                self.history["trends"][name] = []
            self.history["trends"][name].append(metric.cardinality)
            # 保留最近12周
            self.history["trends"][name] = self.history["trends"][name][-12:]

        self.save_history()

        # 生成报告
        report = {
            "week": week_data["week"],
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_metrics": len(metrics),
                "total_cardinality": week_data["total_cardinality"],
                "high_cardinality_count": sum(1 for m in metrics.values() if m.cardinality > self.thresholds.warning),
                "urgent_actions": sum(1 for m in metrics.values() if m.action == ActionType.URGENT)
            },
            "thresholds": {
                "current": {
                    "warning": self.thresholds.warning,
                    "critical": self.thresholds.critical,
                    "growth_rate_warning": f"{self.thresholds.growth_rate_warning:.1%}",
                    "growth_rate_critical": f"{self.thresholds.growth_rate_critical:.1%}"
                },
                "suggested": {
                    "warning": suggested_thresholds.warning,
                    "critical": suggested_thresholds.critical,
                    "growth_rate_warning": f"{suggested_thresholds.growth_rate_warning:.1%}",
                    "growth_rate_critical": f"{suggested_thresholds.growth_rate_critical:.1%}"
                }
            },
            "top_metrics": sorted(metrics.values(), key=lambda x: x.cardinality, reverse=True)[:10],
            "recommendations": recommendations[:10],
            "growth_alerts": [m for m in metrics.values() if abs(m.growth_rate) > self.thresholds.growth_rate_warning]
        }

        if output_format == "json":
            return json.dumps(report, indent=2, default=str)
        else:
            return self._format_markdown(report)

    def _format_markdown(self, report: Dict[str, Any]) -> str:
        """格式化为 Markdown"""
        lines = []
        lines.append("# 📊 Cardinality Weekly Report")
        lines.append(f"\n**Week**: {report['week']}")
        lines.append(f"**Generated**: {report['generated_at'][:19]}\n")

        # 摘要
        summary = report["summary"]
        lines.append("## 📈 Summary\n")
        lines.append(f"- **Total Metrics**: {summary['total_metrics']}")
        lines.append(f"- **Total Cardinality**: {summary['total_cardinality']:,}")
        lines.append(f"- **High Cardinality Metrics**: {summary['high_cardinality_count']}")
        lines.append(f"- **Urgent Actions Required**: {summary['urgent_actions']}\n")

        # 阈值比较
        lines.append("## 🎯 Threshold Analysis\n")
        lines.append("| Type | Current | Suggested | Change |")
        lines.append("|------|---------|-----------|--------|")

        current = report["thresholds"]["current"]
        suggested = report["thresholds"]["suggested"]

        for key in ["warning", "critical"]:
            curr_val = current[key]
            sugg_val = suggested[key]
            change = "➡️" if curr_val == sugg_val else "⬆️" if sugg_val > curr_val else "⬇️"
            lines.append(f"| {key.title()} | {curr_val} | {sugg_val} | {change} |")
        lines.append("")

        # Top 指标
        if report["top_metrics"]:
            lines.append("## 🔝 Top Cardinality Metrics\n")
            lines.append("| Metric | Cardinality | Labels | Growth | Action |")
            lines.append("|--------|-------------|--------|--------|--------|")

            for metric in report["top_metrics"][:5]:
                labels = ", ".join(metric.label_dimensions[:3])
                if len(metric.label_dimensions) > 3:
                    labels += f" +{len(metric.label_dimensions) - 3}"

                growth = f"{metric.growth_rate:+.1%}" if metric.growth_rate else "—"

                action_emoji = {
                    ActionType.URGENT: "🚨",
                    ActionType.PRUNE: "✂️",
                    ActionType.MERGE: "🔀",
                    ActionType.WATCH: "👁️",
                    ActionType.KEEP: "✅"
                }.get(metric.action, "❓")

                lines.append(
                    f"| {metric.metric_name[:30]} | {metric.cardinality:,} | "
                    f"{labels} | {growth} | {action_emoji} {metric.action.value} |"
                )
            lines.append("")

        # 增长告警
        growth_alerts = report.get("growth_alerts", [])
        if growth_alerts:
            lines.append("## ⚠️ Growth Alerts\n")
            lines.append("Metrics with significant growth:\n")
            for metric in growth_alerts[:5]:
                emoji = "📈" if metric.growth_rate > 0 else "📉"
                lines.append(
                    f"- {emoji} **{metric.metric_name}**: "
                    f"{metric.growth_rate:+.1%} "
                    f"({metric.cardinality - metric.weekly_change} → {metric.cardinality})"
                )
            lines.append("")

        # 建议
        if report["recommendations"]:
            lines.append("## 💡 Recommendations\n")
            for metric, action, recommendation in report["recommendations"][:5]:
                action_emoji = {
                    ActionType.URGENT: "🚨",
                    ActionType.PRUNE: "✂️",
                    ActionType.MERGE: "🔀",
                    ActionType.WATCH: "👁️",
                    ActionType.KEEP: "✅"
                }.get(action, "❓")

                lines.append(f"### {action_emoji} {metric.metric_name}")
                lines.append(f"- **Action**: {action.value}")
                lines.append(f"- **Reason**: {recommendation}")
                lines.append(f"- **Current Cardinality**: {metric.cardinality:,}")
                lines.append("")

        # 历史趋势
        lines.append("## 📉 Historical Trend\n")
        if len(self.history["weeks"]) > 1:
            lines.append("| Week | Total Cardinality | Change |")
            lines.append("|------|-------------------|--------|")

            for i in range(max(0, len(self.history["weeks"]) - 4), len(self.history["weeks"])):
                week = self.history["weeks"][i]
                total = week["total_cardinality"]
                change = "—"
                if i > 0:
                    prev_total = self.history["weeks"][i - 1]["total_cardinality"]
                    change_val = total - prev_total
                    change_pct = (change_val / prev_total * 100) if prev_total > 0 else 0
                    change = f"{change_val:+,} ({change_pct:+.1f}%)"

                lines.append(f"| {week['week']} | {total:,} | {change} |")
        else:
            lines.append("*First week of monitoring, no historical data available.*")

        lines.append("\n---\n")
        lines.append("*Automatic thresholds are calculated based on statistical analysis of current metrics.*")
        lines.append("*Consider adjusting thresholds if suggested values differ significantly from current settings.*")

        return "\n".join(lines)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate weekly cardinality report"
    )
    parser.add_argument(
        "--prometheus-url",
        default="http://localhost:9090",
        help="Prometheus server URL"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--warning-threshold",
        type=int,
        default=100,
        help="Warning threshold for cardinality"
    )
    parser.add_argument(
        "--critical-threshold",
        type=int,
        default=1000,
        help="Critical threshold for cardinality"
    )

    args = parser.parse_args()

    # 创建报告生成器
    reporter = CardinalityWeeklyReporter(args.prometheus_url)

    # 设置阈值
    if args.warning_threshold:
        reporter.thresholds.warning = args.warning_threshold
    if args.critical_threshold:
        reporter.thresholds.critical = args.critical_threshold

    # 生成报告
    print("🚀 Generating cardinality weekly report...")
    report = reporter.generate_report(args.format)

    # 输出结果
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"✅ Report saved to: {args.output}")
    else:
        print("\n" + report)

    return 0


if __name__ == "__main__":
    sys.exit(main())