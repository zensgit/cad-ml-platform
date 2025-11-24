#!/usr/bin/env python3
"""
Cardinality Analysis Reporter
生成指标基数分析报告，识别问题并提供优化建议
"""

import json
import logging
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import math

# 导入相关模块
try:
    from .metrics_cardinality_tracker import (
        MetricsCardinalityTracker,
        CardinalityInfo,
        LabelAnalysis,
    )
    from .metrics_budget_controller import MetricsBudgetController, BudgetStatus
except ImportError:  # fallback when executed as standalone script without package context
    from metrics_cardinality_tracker import (
        MetricsCardinalityTracker,
        CardinalityInfo,
        LabelAnalysis,
    )  # type: ignore
    from metrics_budget_controller import MetricsBudgetController, BudgetStatus  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """基数异常"""
    metric_name: str
    anomaly_type: str  # "sudden_spike", "gradual_growth", "label_explosion"
    severity: str  # "low", "medium", "high", "critical"
    current_value: int
    expected_value: int
    deviation_percentage: float
    detected_at: str
    description: str


@dataclass
class Recommendation:
    """优化建议"""
    metric_name: str
    priority: int  # 1-10, 10最高
    action: str  # 具体操作
    expected_savings: int  # 预计节省的序列数
    expected_cost_reduction: float  # 预计成本降低($)
    implementation_difficulty: str  # "easy", "medium", "hard"
    risk_level: str  # "low", "medium", "high"


@dataclass
class Report:
    """分析报告"""
    generated_at: str
    summary: Dict[str, Any]
    top_offenders: List[Dict[str, Any]]
    anomalies: List[Anomaly]
    recommendations: List[Recommendation]
    cost_analysis: Dict[str, Any]
    trend_analysis: Dict[str, Any]


class CardinalityAnalysisReporter:
    """基数分析报告生成器"""

    def __init__(self, tracker: MetricsCardinalityTracker,
                 controller: MetricsBudgetController = None):
        self.tracker = tracker
        self.controller = controller
        self.anomaly_threshold = 2.0  # 标准差倍数，用于异常检测

    def generate_top_offenders_report(self, top_n: int = 10) -> Dict[str, Any]:
        """
        生成高基数指标Top列表

        Args:
            top_n: 返回前N个高基数指标

        Returns:
            Top指标报告
        """
        top_metrics = []

        # 收集所有指标的基数信息
        for metric_name, info in self.tracker.cardinality_cache.items():
            top_metrics.append({
                'metric': metric_name,
                'cardinality': info.cardinality,
                'labels_count': len(info.labels),
                'max_label_cardinality': max(info.labels.values()) if info.labels else 0,
                'storage_mb': round(info.storage_mb, 2),
                'monthly_cost': round(info.monthly_cost, 4),
                'cost_per_series': round(info.monthly_cost / max(info.cardinality, 1), 6)
            })

        # 按成本排序
        top_metrics.sort(key=lambda x: x['monthly_cost'], reverse=True)

        # 计算累计成本
        total_cost = sum(m['monthly_cost'] for m in top_metrics)
        cumulative_cost = 0

        for metric in top_metrics[:top_n]:
            cumulative_cost += metric['monthly_cost']
            metric['cost_percentage'] = round((metric['monthly_cost'] / total_cost * 100), 2)
            metric['cumulative_percentage'] = round((cumulative_cost / total_cost * 100), 2)

            # 添加优化潜力评分
            metric['optimization_potential'] = self._calculate_optimization_potential(metric)

        return {
            'timestamp': datetime.now().isoformat(),
            'total_metrics_analyzed': len(top_metrics),
            'total_monthly_cost': round(total_cost, 2),
            'top_metrics': top_metrics[:top_n],
            'pareto_analysis': {
                'top_20_percent_metrics': len([m for m in top_metrics if m['cumulative_percentage'] <= 80]),
                'cost_concentration': f"Top {top_n} metrics account for "
                                     f"{top_metrics[min(top_n-1, len(top_metrics)-1)]['cumulative_percentage']:.1f}% of cost"
            }
        }

    def detect_anomalies(self) -> List[Anomaly]:
        """
        检测基数异常

        Returns:
            异常列表
        """
        anomalies = []

        # 收集历史数据用于统计分析
        for metric_name in self.tracker.history:
            history = self.tracker.history[metric_name]

            if len(history) < 3:  # 需要至少3个数据点
                continue

            # 提取基数历史
            cardinalities = [h['cardinality'] for h in history]

            # 计算统计指标
            mean_card = statistics.mean(cardinalities)
            stdev_card = statistics.stdev(cardinalities) if len(cardinalities) > 1 else 0
            current_card = cardinalities[-1]

            # 检测突发增长
            if stdev_card > 0:
                z_score = (current_card - mean_card) / stdev_card

                if abs(z_score) > self.anomaly_threshold:
                    anomaly = self._create_anomaly(
                        metric_name, "sudden_spike" if z_score > 0 else "sudden_drop",
                        current_card, int(mean_card), z_score
                    )
                    anomalies.append(anomaly)

            # 检测持续增长
            if len(cardinalities) >= 5:
                recent_growth = (cardinalities[-1] - cardinalities[-5]) / max(cardinalities[-5], 1)
                if recent_growth > 0.5:  # 50%增长
                    anomaly = self._create_anomaly(
                        metric_name, "gradual_growth",
                        current_card, cardinalities[-5], recent_growth
                    )
                    anomalies.append(anomaly)

            # 检测label爆炸
            info = self.tracker.cardinality_cache.get(metric_name)
            if info:
                for label_name, unique_count in info.labels.items():
                    if unique_count > 1000:  # 超过1000个唯一值
                        anomaly = Anomaly(
                            metric_name=metric_name,
                            anomaly_type="label_explosion",
                            severity=self._calculate_severity(unique_count, 1000),
                            current_value=unique_count,
                            expected_value=100,  # 理想值
                            deviation_percentage=((unique_count - 100) / 100 * 100),
                            detected_at=datetime.now().isoformat(),
                            description=f"Label '{label_name}' has {unique_count} unique values"
                        )
                        anomalies.append(anomaly)

        # 按严重程度排序
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(key=lambda x: severity_order.get(x.severity, 4))

        return anomalies

    def generate_optimization_recommendations(self) -> List[Recommendation]:
        """
        生成优化建议

        Returns:
            优化建议列表
        """
        recommendations = []

        # 分析每个高基数指标
        for metric_name, info in self.tracker.cardinality_cache.items():
            if info.cardinality < 1000:  # 跳过低基数指标
                continue

            # 分析labels
            label_analyses = self.tracker.identify_high_cardinality_labels(metric_name)

            for analysis in label_analyses:
                if analysis.unique_values > 100:
                    # 生成针对高基数label的建议
                    rec = self._create_label_recommendation(
                        metric_name, info, analysis
                    )
                    if rec:
                        recommendations.append(rec)

            # 检查是否可以降采样
            if info.cardinality > 5000:
                rec = self._create_downsampling_recommendation(metric_name, info)
                if rec:
                    recommendations.append(rec)

            # 检查是否可以使用recording rules
            if info.cardinality > 10000:
                rec = self._create_recording_rule_recommendation(metric_name, info)
                if rec:
                    recommendations.append(rec)

        # 添加全局优化建议
        global_recs = self._create_global_recommendations()
        recommendations.extend(global_recs)

        # 按优先级排序
        recommendations.sort(key=lambda x: x.priority, reverse=True)

        return recommendations[:20]  # 返回前20个建议

    def _create_anomaly(self, metric_name: str, anomaly_type: str,
                       current: int, expected: int, deviation: float) -> Anomaly:
        """创建异常对象"""
        severity = self._calculate_severity(abs(deviation), self.anomaly_threshold)

        descriptions = {
            "sudden_spike": f"Sudden increase from {expected:,} to {current:,} series",
            "sudden_drop": f"Sudden decrease from {expected:,} to {current:,} series",
            "gradual_growth": f"Gradual growth of {deviation*100:.1f}% detected"
        }

        return Anomaly(
            metric_name=metric_name,
            anomaly_type=anomaly_type,
            severity=severity,
            current_value=current,
            expected_value=expected,
            deviation_percentage=round(deviation * 100, 2),
            detected_at=datetime.now().isoformat(),
            description=descriptions.get(anomaly_type, "Anomaly detected")
        )

    def _calculate_severity(self, value: float, threshold: float) -> str:
        """计算严重程度"""
        ratio = value / threshold
        if ratio >= 5:
            return "critical"
        elif ratio >= 3:
            return "high"
        elif ratio >= 2:
            return "medium"
        else:
            return "low"

    def _calculate_optimization_potential(self, metric: Dict) -> int:
        """计算优化潜力评分(1-10)"""
        score = 0

        # 基于成本
        if metric['monthly_cost'] > 1.0:
            score += 3
        elif metric['monthly_cost'] > 0.5:
            score += 2
        elif metric['monthly_cost'] > 0.1:
            score += 1

        # 基于基数
        if metric['cardinality'] > 10000:
            score += 3
        elif metric['cardinality'] > 5000:
            score += 2
        elif metric['cardinality'] > 1000:
            score += 1

        # 基于label复杂度
        if metric['max_label_cardinality'] > 1000:
            score += 2
        elif metric['max_label_cardinality'] > 100:
            score += 1

        # 基于成本效率
        if metric['cost_per_series'] > 0.0001:
            score += 1

        return min(score, 10)

    def _create_label_recommendation(self, metric_name: str,
                                    info: CardinalityInfo,
                                    analysis: LabelAnalysis) -> Optional[Recommendation]:
        """创建label优化建议"""
        if 'id' in analysis.label_name.lower() or 'uuid' in analysis.label_name.lower():
            # ID类label，建议删除
            expected_savings = int(info.cardinality * 0.8)  # 预计减少80%
            return Recommendation(
                metric_name=metric_name,
                priority=9,
                action=f"Remove ID label '{analysis.label_name}' or use recording rules",
                expected_savings=expected_savings,
                expected_cost_reduction=expected_savings * 0.001,
                implementation_difficulty="easy",
                risk_level="low"
            )
        elif analysis.unique_values > 1000:
            # 高基数label，建议减少粒度
            expected_savings = int(info.cardinality * 0.3)
            return Recommendation(
                metric_name=metric_name,
                priority=7,
                action=f"Reduce granularity of label '{analysis.label_name}' "
                       f"(currently {analysis.unique_values} unique values)",
                expected_savings=expected_savings,
                expected_cost_reduction=expected_savings * 0.001,
                implementation_difficulty="medium",
                risk_level="medium"
            )
        elif analysis.entropy > 0.9:
            # 高熵label，建议聚合
            expected_savings = int(info.cardinality * 0.4)
            return Recommendation(
                metric_name=metric_name,
                priority=6,
                action=f"Aggregate label '{analysis.label_name}' values into categories",
                expected_savings=expected_savings,
                expected_cost_reduction=expected_savings * 0.001,
                implementation_difficulty="medium",
                risk_level="low"
            )

        return None

    def _create_downsampling_recommendation(self, metric_name: str,
                                           info: CardinalityInfo) -> Recommendation:
        """创建降采样建议"""
        expected_savings = int(info.cardinality * 0.5)  # 预计减少50%
        return Recommendation(
            metric_name=metric_name,
            priority=5,
            action=f"Apply downsampling for {metric_name} (5m:1h, 1h:1d retention)",
            expected_savings=expected_savings,
            expected_cost_reduction=expected_savings * 0.001,
            implementation_difficulty="easy",
            risk_level="low"
        )

    def _create_recording_rule_recommendation(self, metric_name: str,
                                             info: CardinalityInfo) -> Recommendation:
        """创建recording rule建议"""
        expected_savings = int(info.cardinality * 0.7)  # 预计减少70%
        return Recommendation(
            metric_name=metric_name,
            priority=8,
            action=f"Create recording rules for common {metric_name} queries",
            expected_savings=expected_savings,
            expected_cost_reduction=expected_savings * 0.001,
            implementation_difficulty="medium",
            risk_level="low"
        )

    def _create_global_recommendations(self) -> List[Recommendation]:
        """创建全局优化建议"""
        recommendations = []

        # 基于总体成本分析
        cost_summary = self.tracker.estimate_total_cost()

        if cost_summary['total_monthly_cost'] > 100:
            recommendations.append(Recommendation(
                metric_name="GLOBAL",
                priority=10,
                action="Implement global cardinality limits and budget controls",
                expected_savings=int(cost_summary['total_cardinality'] * 0.2),
                expected_cost_reduction=cost_summary['total_monthly_cost'] * 0.2,
                implementation_difficulty="hard",
                risk_level="medium"
            ))

        if cost_summary['average_cardinality'] > 5000:
            recommendations.append(Recommendation(
                metric_name="GLOBAL",
                priority=8,
                action="Review and optimize high-cardinality metrics across all services",
                expected_savings=int(cost_summary['total_cardinality'] * 0.3),
                expected_cost_reduction=cost_summary['total_monthly_cost'] * 0.3,
                implementation_difficulty="medium",
                risk_level="low"
            ))

        return recommendations

    def generate_full_report(self, output_format: str = "json") -> str:
        """
        生成完整分析报告

        Args:
            output_format: 输出格式 ("json", "markdown", "html")

        Returns:
            格式化的报告字符串
        """
        # 收集各部分数据
        top_offenders = self.generate_top_offenders_report()
        anomalies = self.detect_anomalies()
        recommendations = self.generate_optimization_recommendations()

        # 成本分析
        cost_analysis = self.tracker.estimate_total_cost()

        # 趋势分析
        trends = self.tracker.track_cardinality_trends()
        trend_analysis = {
            'growing_metrics': len([t for t in trends if t.is_growing]),
            'exploding_metrics': len([t for t in trends if t.is_exploding]),
            'top_growing': [
                {
                    'metric': t.metric_name,
                    'growth_rate': t.growth_rate,
                    'growth_absolute': t.growth_absolute
                }
                for t in trends[:5]
            ]
        }

        # 汇总
        summary = {
            'total_metrics': cost_analysis['total_metrics'],
            'total_cardinality': cost_analysis['total_cardinality'],
            'total_monthly_cost': cost_analysis['total_monthly_cost'],
            'anomalies_detected': len(anomalies),
            'recommendations_count': len(recommendations),
            'potential_savings': sum(r.expected_cost_reduction for r in recommendations)
        }

        # 创建报告对象
        report = Report(
            generated_at=datetime.now().isoformat(),
            summary=summary,
            top_offenders=top_offenders['top_metrics'],
            anomalies=anomalies,
            recommendations=recommendations,
            cost_analysis=cost_analysis,
            trend_analysis=trend_analysis
        )

        # 格式化输出
        if output_format == "json":
            return self._format_json(report)
        elif output_format == "markdown":
            return self._format_markdown(report)
        elif output_format == "html":
            return self._format_html(report)
        else:
            return self._format_json(report)

    def _format_json(self, report: Report) -> str:
        """格式化为JSON"""
        return json.dumps(asdict(report), indent=2, ensure_ascii=False)

    def _format_markdown(self, report: Report) -> str:
        """格式化为Markdown"""
        md = []

        md.append("# 📊 Prometheus Metrics Cardinality Analysis Report")
        md.append(f"\n**Generated**: {report.generated_at}")
        md.append("\n---\n")

        # 摘要
        md.append("## 📈 Executive Summary\n")
        md.append(f"- **Total Metrics**: {report.summary['total_metrics']:,}")
        md.append(f"- **Total Time Series**: {report.summary['total_cardinality']:,}")
        md.append(f"- **Monthly Cost**: ${report.summary['total_monthly_cost']:.2f}")
        md.append(f"- **Anomalies Detected**: {report.summary['anomalies_detected']}")
        md.append(f"- **Optimization Opportunities**: {report.summary['recommendations_count']}")
        md.append(f"- **Potential Monthly Savings**: ${report.summary['potential_savings']:.2f}")

        # Top指标
        md.append("\n## 💰 Top Cost Metrics\n")
        md.append("| Metric | Cardinality | Monthly Cost | Cost % | Optimization Potential |")
        md.append("|--------|-------------|--------------|--------|------------------------|")
        for metric in report.top_offenders[:10]:
            md.append(f"| {metric['metric']} | {metric['cardinality']:,} | "
                     f"${metric['monthly_cost']:.4f} | {metric['cost_percentage']}% | "
                     f"{metric['optimization_potential']}/10 |")

        # 异常
        if report.anomalies:
            md.append("\n## ⚠️ Detected Anomalies\n")
            for anomaly in report.anomalies[:5]:
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(anomaly.severity, "⚪")
                md.append(f"\n### {severity_emoji} {anomaly.metric_name}")
                md.append(f"- **Type**: {anomaly.anomaly_type}")
                md.append(f"- **Severity**: {anomaly.severity}")
                md.append(f"- **Description**: {anomaly.description}")

        # 建议
        if report.recommendations:
            md.append("\n## 💡 Optimization Recommendations\n")
            for i, rec in enumerate(report.recommendations[:10], 1):
                difficulty_emoji = {
                    "easy": "✅",
                    "medium": "⚠️",
                    "hard": "🔴"
                }.get(rec.implementation_difficulty, "❓")
                md.append(f"\n### {i}. {rec.metric_name} (Priority: {rec.priority}/10)")
                md.append(f"- **Action**: {rec.action}")
                md.append(f"- **Expected Savings**: {rec.expected_savings:,} series "
                         f"(${rec.expected_cost_reduction:.2f}/month)")
                md.append(f"- **Difficulty**: {difficulty_emoji} {rec.implementation_difficulty}")
                md.append(f"- **Risk**: {rec.risk_level}")

        return "\n".join(md)

    def _format_html(self, report: Report) -> str:
        """格式化为HTML"""
        # 简化的HTML格式
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Metrics Cardinality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .critical {{ color: red; }}
                .high {{ color: orange; }}
                .medium {{ color: yellow; }}
                .low {{ color: green; }}
            </style>
        </head>
        <body>
            <h1>Metrics Cardinality Analysis Report</h1>
            <p>Generated: {report.generated_at}</p>

            <h2>Summary</h2>
            <ul>
                <li>Total Metrics: {report.summary['total_metrics']:,}</li>
                <li>Total Cardinality: {report.summary['total_cardinality']:,}</li>
                <li>Monthly Cost: ${report.summary['total_monthly_cost']:.2f}</li>
                <li>Potential Savings: ${report.summary['potential_savings']:.2f}</li>
            </ul>

            <p>Full report available in JSON/Markdown format.</p>
        </body>
        </html>
        """
        return html


def main():
    """主函数 - 用于测试"""
    import argparse

    parser = argparse.ArgumentParser(description='Cardinality Analysis Reporter')
    parser.add_argument('--prometheus-url', default='http://localhost:9090')
    parser.add_argument('--format', choices=['json', 'markdown', 'html'], default='markdown')
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--top', type=int, default=10, help='Number of top metrics')

    args = parser.parse_args()

    # 创建组件
    tracker = MetricsCardinalityTracker(args.prometheus_url)
    controller = MetricsBudgetController()
    reporter = CardinalityAnalysisReporter(tracker, controller)

    # 模拟一些数据用于测试
    print("🔍 Analyzing metrics...")

    # 这里应该先让tracker收集一些数据
    # tracker.get_metric_cardinality("sample_metric")

    # 生成报告
    if args.format == 'json':
        # 生成JSON格式的Top指标报告
        report = reporter.generate_top_offenders_report(args.top)
        report_str = json.dumps(report, indent=2)
    else:
        # 生成完整报告
        report_str = reporter.generate_full_report(args.format)

    # 输出
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"✅ Report saved to: {args.output}")
    else:
        print(report_str)


if __name__ == "__main__":
    main()
