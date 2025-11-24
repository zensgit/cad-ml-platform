#!/usr/bin/env python3
"""
Cardinality Audit Script
指标基数审计脚本 - 监控和审计 Prometheus 指标的标签维度增长
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any
import urllib.request
import urllib.error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MetricInfo:
    """指标信息"""
    name: str
    labels: Set[str]
    unique_values: Dict[str, Set[str]] = field(default_factory=dict)
    sample_count: int = 0
    cardinality: int = 0
    growth_rate: float = 0.0


@dataclass
class CardinalityReport:
    """基数报告"""
    timestamp: datetime
    total_metrics: int
    total_samples: int
    high_cardinality_metrics: List[Dict[str, Any]]
    label_analysis: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    warnings: List[str]


class CardinalityAuditor:
    """指标基数审计器"""

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        thresholds: Optional[Dict[str, int]] = None
    ):
        """
        初始化审计器

        Args:
            prometheus_url: Prometheus 服务器地址
            thresholds: 基数阈值配置
        """
        self.prometheus_url = prometheus_url.rstrip('/')
        self.thresholds = thresholds or {
            "warning": 100,      # 警告阈值
            "critical": 1000,    # 严重阈值
            "label_values": 50,  # 单标签值数量阈值
        }
        self.metrics: Dict[str, MetricInfo] = {}
        self.historical_data: List[CardinalityReport] = []

    def fetch_metrics(self) -> str:
        """从 Prometheus 获取指标"""
        url = f"{self.prometheus_url}/api/v1/label/__name__/values"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                return data.get("data", [])
        except urllib.error.URLError as e:
            logger.error(f"Failed to fetch metrics from Prometheus: {e}")
            raise

    def fetch_metric_metadata(self, metric_name: str) -> Dict[str, Any]:
        """获取指标元数据"""
        url = f"{self.prometheus_url}/api/v1/metadata"
        params = f"?metric={metric_name}"
        try:
            with urllib.request.urlopen(url + params) as response:
                data = json.loads(response.read().decode())
                return data.get("data", {}).get(metric_name, [])
        except urllib.error.URLError as e:
            logger.warning(f"Failed to fetch metadata for {metric_name}: {e}")
            return []

    def fetch_series(self, metric_name: str) -> List[Dict[str, str]]:
        """获取指标序列"""
        url = f"{self.prometheus_url}/api/v1/series"
        params = f'?match[]={metric_name}'
        try:
            req = urllib.request.Request(url + params)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                return data.get("data", [])
        except urllib.error.URLError as e:
            logger.warning(f"Failed to fetch series for {metric_name}: {e}")
            return []

    def analyze_metric(self, metric_name: str) -> MetricInfo:
        """分析单个指标"""
        series = self.fetch_series(metric_name)

        metric_info = MetricInfo(name=metric_name, labels=set())

        for serie in series:
            # 收集所有标签
            for label, value in serie.items():
                if label != "__name__":
                    metric_info.labels.add(label)
                    if label not in metric_info.unique_values:
                        metric_info.unique_values[label] = set()
                    metric_info.unique_values[label].add(value)

        metric_info.sample_count = len(series)

        # 计算基数（所有标签组合的唯一数量）
        metric_info.cardinality = len(series)

        return metric_info

    def analyze_all_metrics(self) -> None:
        """分析所有指标"""
        logger.info("Fetching metric names from Prometheus...")
        metric_names = self.fetch_metrics()

        logger.info(f"Found {len(metric_names)} metrics, analyzing...")

        for metric_name in metric_names:
            # 跳过内部指标
            if metric_name.startswith("prometheus_") or metric_name.startswith("go_"):
                continue

            try:
                metric_info = self.analyze_metric(metric_name)
                self.metrics[metric_name] = metric_info

                if metric_info.cardinality > self.thresholds["warning"]:
                    logger.warning(
                        f"High cardinality detected: {metric_name} "
                        f"has {metric_info.cardinality} series"
                    )
            except Exception as e:
                logger.error(f"Error analyzing metric {metric_name}: {e}")

    def identify_high_cardinality_labels(self) -> Dict[str, List[str]]:
        """识别高基数标签"""
        high_cardinality_labels = defaultdict(list)

        for metric_name, metric_info in self.metrics.items():
            for label, values in metric_info.unique_values.items():
                if len(values) > self.thresholds["label_values"]:
                    high_cardinality_labels[label].append({
                        "metric": metric_name,
                        "value_count": len(values),
                        "sample_values": list(values)[:5]  # 前5个示例值
                    })

        return dict(high_cardinality_labels)

    def calculate_growth_rates(self) -> None:
        """计算增长率"""
        if len(self.historical_data) < 2:
            return

        previous = self.historical_data[-2]
        current_metrics = {m["name"]: m["cardinality"]
                         for m in self.generate_report().high_cardinality_metrics}
        previous_metrics = {m["name"]: m["cardinality"]
                          for m in previous.high_cardinality_metrics}

        for metric_name in self.metrics:
            if metric_name in previous_metrics:
                old_card = previous_metrics.get(metric_name, 0)
                new_card = current_metrics.get(metric_name, 0)
                if old_card > 0:
                    growth_rate = ((new_card - old_card) / old_card) * 100
                    self.metrics[metric_name].growth_rate = growth_rate

    def generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 检查高基数指标
        for metric_name, metric_info in self.metrics.items():
            if metric_info.cardinality > self.thresholds["critical"]:
                recommendations.append(
                    f"CRITICAL: Metric '{metric_name}' has extremely high cardinality "
                    f"({metric_info.cardinality} series). Consider reducing label dimensions."
                )

            # 检查快速增长
            if metric_info.growth_rate > 20:  # 20% 增长
                recommendations.append(
                    f"WARNING: Metric '{metric_name}' cardinality growing rapidly "
                    f"({metric_info.growth_rate:.1f}% growth). Monitor closely."
                )

        # 检查问题标签
        high_card_labels = self.identify_high_cardinality_labels()
        for label, metrics in high_card_labels.items():
            if len(metrics) > 3:  # 多个指标使用同一高基数标签
                recommendations.append(
                    f"Label '{label}' has high cardinality across {len(metrics)} metrics. "
                    f"Consider using recording rules or removing this label."
                )

        # 通用建议
        total_series = sum(m.cardinality for m in self.metrics.values())
        if total_series > 100000:
            recommendations.append(
                "Total series count exceeds 100k. Consider implementing: "
                "1) Recording rules for frequently queried metrics, "
                "2) Metric relabeling to drop unnecessary labels, "
                "3) Shorter retention periods for high-cardinality metrics."
            )

        return recommendations

    def generate_report(self) -> CardinalityReport:
        """生成审计报告"""
        high_cardinality_metrics = []

        for metric_name, metric_info in self.metrics.items():
            if metric_info.cardinality > self.thresholds["warning"]:
                high_cardinality_metrics.append({
                    "name": metric_name,
                    "cardinality": metric_info.cardinality,
                    "labels": list(metric_info.labels),
                    "label_cardinalities": {
                        label: len(values)
                        for label, values in metric_info.unique_values.items()
                    },
                    "growth_rate": metric_info.growth_rate
                })

        # 按基数排序
        high_cardinality_metrics.sort(key=lambda x: x["cardinality"], reverse=True)

        # 标签分析
        label_analysis = {}
        label_usage = defaultdict(int)
        label_cardinality = defaultdict(list)

        for metric_info in self.metrics.values():
            for label, values in metric_info.unique_values.items():
                label_usage[label] += 1
                label_cardinality[label].append(len(values))

        for label in label_usage:
            cardinalities = label_cardinality[label]
            label_analysis[label] = {
                "usage_count": label_usage[label],
                "avg_cardinality": sum(cardinalities) / len(cardinalities),
                "max_cardinality": max(cardinalities),
                "total_unique_values": sum(cardinalities)
            }

        # 生成警告
        warnings = []
        for metric_name, metric_info in self.metrics.items():
            if metric_info.cardinality > self.thresholds["critical"]:
                warnings.append(
                    f"Metric '{metric_name}' exceeds critical threshold "
                    f"({metric_info.cardinality} > {self.thresholds['critical']})"
                )

        report = CardinalityReport(
            timestamp=datetime.now(),
            total_metrics=len(self.metrics),
            total_samples=sum(m.sample_count for m in self.metrics.values()),
            high_cardinality_metrics=high_cardinality_metrics[:20],  # Top 20
            label_analysis=label_analysis,
            recommendations=self.generate_recommendations(),
            warnings=warnings
        )

        # 保存到历史
        self.historical_data.append(report)
        if len(self.historical_data) > 100:  # 保留最近100次
            self.historical_data = self.historical_data[-100:]

        return report

    def export_report(self, report: CardinalityReport, format: str = "json") -> str:
        """导出报告"""
        if format == "json":
            return json.dumps({
                "timestamp": report.timestamp.isoformat(),
                "summary": {
                    "total_metrics": report.total_metrics,
                    "total_samples": report.total_samples,
                    "high_cardinality_count": len(report.high_cardinality_metrics),
                    "warning_count": len(report.warnings)
                },
                "high_cardinality_metrics": report.high_cardinality_metrics,
                "label_analysis": report.label_analysis,
                "recommendations": report.recommendations,
                "warnings": report.warnings
            }, indent=2)

        elif format == "markdown":
            lines = [
                "# Cardinality Audit Report",
                f"\n**Generated**: {report.timestamp.isoformat()}",
                f"\n## Summary",
                f"- Total Metrics: {report.total_metrics}",
                f"- Total Samples: {report.total_samples}",
                f"- High Cardinality Metrics: {len(report.high_cardinality_metrics)}",
                f"\n## High Cardinality Metrics",
            ]

            for metric in report.high_cardinality_metrics[:10]:
                lines.append(
                    f"\n### {metric['name']}",
                )
                lines.append(f"- Cardinality: {metric['cardinality']}")
                lines.append(f"- Labels: {', '.join(metric['labels'])}")
                if metric['growth_rate'] > 0:
                    lines.append(f"- Growth Rate: {metric['growth_rate']:.1f}%")

            if report.warnings:
                lines.append("\n## Warnings")
                for warning in report.warnings:
                    lines.append(f"- ⚠️ {warning}")

            if report.recommendations:
                lines.append("\n## Recommendations")
                for rec in report.recommendations:
                    lines.append(f"- {rec}")

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def run_audit(self) -> CardinalityReport:
        """执行完整审计"""
        logger.info("Starting cardinality audit...")

        # 分析所有指标
        self.analyze_all_metrics()

        # 计算增长率
        self.calculate_growth_rates()

        # 生成报告
        report = self.generate_report()

        logger.info(
            f"Audit complete: {report.total_metrics} metrics analyzed, "
            f"{len(report.high_cardinality_metrics)} high cardinality metrics found"
        )

        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Audit Prometheus metrics cardinality"
    )
    parser.add_argument(
        "--prometheus-url",
        default="http://localhost:9090",
        help="Prometheus server URL"
    )
    parser.add_argument(
        "--warning-threshold",
        type=int,
        default=100,
        help="Warning threshold for metric cardinality"
    )
    parser.add_argument(
        "--critical-threshold",
        type=int,
        default=1000,
        help="Critical threshold for metric cardinality"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--output",
        help="Output file (default: stdout)"
    )

    args = parser.parse_args()

    # 创建审计器
    auditor = CardinalityAuditor(
        prometheus_url=args.prometheus_url,
        thresholds={
            "warning": args.warning_threshold,
            "critical": args.critical_threshold,
            "label_values": 50,
        }
    )

    try:
        # 执行审计
        report = auditor.run_audit()

        # 导出报告
        output = auditor.export_report(report, format=args.format)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"Report saved to {args.output}")
        else:
            print(output)

        # 返回退出码
        if report.warnings:
            return 1  # 有警告
        return 0  # 正常

    except Exception as e:
        logger.error(f"Audit failed: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())