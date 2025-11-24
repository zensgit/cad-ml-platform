#!/usr/bin/env python3
"""
Metrics Drift Detection Tool
指标漂移检测工具 - 检测指标偏离基线的情况
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from enum import Enum

# 引入基线管理器
sys.path.append(str(Path(__file__).parent))
from metrics_baseline_snapshot import MetricsBaselineManager, BaselineSnapshot


class DriftSeverity(str, Enum):
    """漂移严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftDetection:
    """漂移检测结果"""
    metric_name: str
    drift_type: str  # cardinality, distribution, labels, missing, new
    severity: DriftSeverity
    details: Dict[str, Any]
    baseline_value: Any
    current_value: Any
    drift_score: float  # 0-100
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DriftReport:
    """漂移报告"""
    timestamp: datetime
    baseline_info: Dict[str, Any]
    current_info: Dict[str, Any]
    detections: List[DriftDetection]
    summary: Dict[str, Any]
    risk_score: float  # 0-100


class MetricsDriftDetector:
    """指标漂移检测器"""

    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline_manager = MetricsBaselineManager()
        self.baseline: Optional[BaselineSnapshot] = None
        self.current_metrics: Dict[str, Any] = {}
        self.detections: List[DriftDetection] = []

        # 漂移阈值配置
        self.thresholds = {
            "cardinality": {
                "low": 1.2,      # 20% increase
                "medium": 1.5,   # 50% increase
                "high": 2.0,     # 100% increase
                "critical": 3.0  # 200% increase
            },
            "distribution": {
                "low": 0.3,      # KS statistic > 0.3
                "medium": 0.5,   # KS statistic > 0.5
                "high": 0.7,     # KS statistic > 0.7
                "critical": 0.9  # KS statistic > 0.9
            },
            "value_change": {
                "low": 1.5,      # 50% change in mean
                "medium": 2.0,   # 100% change
                "high": 3.0,     # 200% change
                "critical": 5.0  # 400% change
            }
        }

        if baseline_file:
            self.load_baseline(baseline_file)

    def load_baseline(self, filename: str):
        """加载基线"""
        self.baseline = self.baseline_manager.load_baseline(filename)
        print(f"✅ Loaded baseline: {filename}")
        print(f"   Environment: {self.baseline.environment}")
        print(f"   Timestamp: {self.baseline.timestamp}")
        print(f"   Metrics: {len(self.baseline.metrics)}")

    def detect_drift(self, time_range: str = "1h") -> DriftReport:
        """执行漂移检测"""
        if not self.baseline:
            raise ValueError("No baseline loaded. Use load_baseline() first.")

        print(f"\n🔍 Detecting drift against baseline...")

        # 获取当前指标
        self.current_metrics = self.baseline_manager.fetch_current_metrics(time_range)

        # 清空之前的检测结果
        self.detections.clear()

        # 检测各种类型的漂移
        self._detect_missing_metrics()
        self._detect_new_metrics()
        self._detect_cardinality_drift()
        self._detect_label_drift()
        self._detect_distribution_drift()

        # 生成报告
        report = self._generate_report()

        return report

    def _detect_missing_metrics(self):
        """检测消失的指标"""
        baseline_metrics = set(self.baseline.metrics.keys())
        current_metrics = set(self.current_metrics.keys())

        missing = baseline_metrics - current_metrics

        for metric_name in missing:
            detection = DriftDetection(
                metric_name=metric_name,
                drift_type="missing",
                severity=DriftSeverity.HIGH,
                details={"reason": "Metric not found in current data"},
                baseline_value="present",
                current_value="missing",
                drift_score=80.0,
                recommendation="Investigate why metric is no longer being collected"
            )
            self.detections.append(detection)

    def _detect_new_metrics(self):
        """检测新增的指标"""
        baseline_metrics = set(self.baseline.metrics.keys())
        current_metrics = set(self.current_metrics.keys())

        new_metrics = current_metrics - baseline_metrics

        for metric_name in new_metrics:
            current = self.current_metrics[metric_name]

            detection = DriftDetection(
                metric_name=metric_name,
                drift_type="new",
                severity=DriftSeverity.MEDIUM,
                details={
                    "labels": current.get("labels", []),
                    "cardinality": current.get("cardinality", 0)
                },
                baseline_value="missing",
                current_value="present",
                drift_score=50.0,
                recommendation="Review new metric and update baseline if legitimate"
            )
            self.detections.append(detection)

    def _detect_cardinality_drift(self):
        """检测基数漂移"""
        for metric_name in set(self.baseline.metrics.keys()) & set(self.current_metrics.keys()):
            baseline_metric = self.baseline.metrics[metric_name]
            current_metric = self.current_metrics[metric_name]

            baseline_card = baseline_metric.cardinality
            current_card = current_metric.get("cardinality", 0)

            if baseline_card == 0:
                continue

            # 计算变化率
            change_rate = current_card / baseline_card if baseline_card > 0 else float('inf')

            # 确定严重程度
            severity = None
            for level in ["critical", "high", "medium", "low"]:
                if change_rate >= self.thresholds["cardinality"][level]:
                    severity = DriftSeverity(level)
                    break

            if severity:
                drift_score = min(100, (change_rate - 1) * 100)

                detection = DriftDetection(
                    metric_name=metric_name,
                    drift_type="cardinality",
                    severity=severity,
                    details={
                        "change_rate": f"{change_rate:.2f}x",
                        "percentage": f"{(change_rate - 1) * 100:.1f}%"
                    },
                    baseline_value=baseline_card,
                    current_value=current_card,
                    drift_score=drift_score,
                    recommendation=self._get_cardinality_recommendation(change_rate)
                )
                self.detections.append(detection)

    def _detect_label_drift(self):
        """检测标签漂移"""
        for metric_name in set(self.baseline.metrics.keys()) & set(self.current_metrics.keys()):
            baseline_metric = self.baseline.metrics[metric_name]
            current_metric = self.current_metrics[metric_name]

            baseline_labels = set(baseline_metric.labels)
            current_labels = set(current_metric.get("labels", []))

            # 检查标签集变化
            new_labels = current_labels - baseline_labels
            removed_labels = baseline_labels - current_labels

            if new_labels or removed_labels:
                severity = DriftSeverity.HIGH if removed_labels else DriftSeverity.MEDIUM

                detection = DriftDetection(
                    metric_name=metric_name,
                    drift_type="labels",
                    severity=severity,
                    details={
                        "new_labels": list(new_labels),
                        "removed_labels": list(removed_labels)
                    },
                    baseline_value=list(baseline_labels),
                    current_value=list(current_labels),
                    drift_score=60.0 if new_labels or removed_labels else 0,
                    recommendation="Label schema changed - review and update monitoring"
                )
                self.detections.append(detection)

            # 检查标签值漂移
            for label in baseline_labels & current_labels:
                baseline_values = baseline_metric.label_values.get(label, set())
                current_values = set(current_metric.get("label_values", {}).get(label, []))

                new_values = current_values - baseline_values
                if len(new_values) > len(baseline_values) * 0.5:  # 50%以上的新值
                    detection = DriftDetection(
                        metric_name=metric_name,
                        drift_type="label_values",
                        severity=DriftSeverity.MEDIUM,
                        details={
                            "label": label,
                            "new_values_count": len(new_values),
                            "baseline_count": len(baseline_values),
                            "sample_new_values": list(new_values)[:5]
                        },
                        baseline_value=len(baseline_values),
                        current_value=len(current_values),
                        drift_score=40.0,
                        recommendation=f"Label '{label}' has many new values - check for unbounded growth"
                    )
                    self.detections.append(detection)

    def _detect_distribution_drift(self):
        """检测值分布漂移"""
        for metric_name in set(self.baseline.metrics.keys()) & set(self.current_metrics.keys()):
            baseline_metric = self.baseline.metrics[metric_name]
            current_metric = self.current_metrics[metric_name]

            baseline_stats = baseline_metric.value_stats
            current_values = current_metric.get("values", [])

            if not current_values:
                continue

            # 计算当前统计
            current_stats = {
                "mean": float(np.mean(current_values)),
                "std": float(np.std(current_values)),
                "p50": float(np.percentile(current_values, 50)),
                "p90": float(np.percentile(current_values, 90)),
                "p99": float(np.percentile(current_values, 99))
            }

            # 检测均值漂移
            if baseline_stats["mean"] > 0:
                mean_change = abs(current_stats["mean"] - baseline_stats["mean"]) / baseline_stats["mean"]

                severity = None
                for level in ["critical", "high", "medium", "low"]:
                    if mean_change >= self.thresholds["value_change"][level]:
                        severity = DriftSeverity(level)
                        break

                if severity:
                    drift_score = min(100, mean_change * 100)

                    detection = DriftDetection(
                        metric_name=metric_name,
                        drift_type="distribution",
                        severity=severity,
                        details={
                            "metric": "mean",
                            "change_percentage": f"{mean_change * 100:.1f}%",
                            "baseline_stats": baseline_stats,
                            "current_stats": current_stats
                        },
                        baseline_value=baseline_stats["mean"],
                        current_value=current_stats["mean"],
                        drift_score=drift_score,
                        recommendation=self._get_distribution_recommendation(mean_change)
                    )
                    self.detections.append(detection)

            # 检测P99漂移（尾部延迟）
            if baseline_stats["p99"] > 0:
                p99_change = (current_stats["p99"] - baseline_stats["p99"]) / baseline_stats["p99"]

                if p99_change > 0.5:  # P99增加50%以上
                    detection = DriftDetection(
                        metric_name=metric_name,
                        drift_type="tail_latency",
                        severity=DriftSeverity.HIGH if p99_change > 1.0 else DriftSeverity.MEDIUM,
                        details={
                            "metric": "p99",
                            "change_percentage": f"{p99_change * 100:.1f}%"
                        },
                        baseline_value=baseline_stats["p99"],
                        current_value=current_stats["p99"],
                        drift_score=min(100, p99_change * 50),
                        recommendation="Tail latency increased significantly - investigate performance issues"
                    )
                    self.detections.append(detection)

    def _get_cardinality_recommendation(self, change_rate: float) -> str:
        """获取基数漂移建议"""
        if change_rate >= 3.0:
            return "CRITICAL: Cardinality explosion detected! Implement immediate controls"
        elif change_rate >= 2.0:
            return "HIGH: Cardinality doubled - review label usage and implement limits"
        elif change_rate >= 1.5:
            return "MEDIUM: Significant cardinality growth - monitor closely"
        else:
            return "LOW: Minor cardinality increase - continue monitoring"

    def _get_distribution_recommendation(self, change_rate: float) -> str:
        """获取分布漂移建议"""
        if change_rate >= 5.0:
            return "CRITICAL: Extreme value shift - investigate system changes immediately"
        elif change_rate >= 3.0:
            return "HIGH: Major distribution change - review recent deployments"
        elif change_rate >= 2.0:
            return "MEDIUM: Significant value change - analyze root cause"
        else:
            return "LOW: Notable drift - monitor trend"

    def _generate_report(self) -> DriftReport:
        """生成漂移报告"""
        # 计算风险分数
        risk_score = self._calculate_risk_score()

        # 按严重程度统计
        severity_counts = defaultdict(int)
        for detection in self.detections:
            severity_counts[detection.severity] += 1

        report = DriftReport(
            timestamp=datetime.now(),
            baseline_info={
                "timestamp": self.baseline.timestamp.isoformat(),
                "environment": self.baseline.environment,
                "metrics_count": len(self.baseline.metrics)
            },
            current_info={
                "timestamp": datetime.now().isoformat(),
                "metrics_count": len(self.current_metrics)
            },
            detections=self.detections,
            summary={
                "total_drifts": len(self.detections),
                "critical": severity_counts[DriftSeverity.CRITICAL],
                "high": severity_counts[DriftSeverity.HIGH],
                "medium": severity_counts[DriftSeverity.MEDIUM],
                "low": severity_counts[DriftSeverity.LOW],
                "drift_types": self._summarize_drift_types()
            },
            risk_score=risk_score
        )

        return report

    def _calculate_risk_score(self) -> float:
        """计算整体风险分数"""
        if not self.detections:
            return 0.0

        weights = {
            DriftSeverity.CRITICAL: 10,
            DriftSeverity.HIGH: 5,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.LOW: 1
        }

        total_weight = sum(weights[d.severity] for d in self.detections)
        max_possible = len(self.detections) * weights[DriftSeverity.CRITICAL]

        return min(100.0, (total_weight / max_possible * 100) if max_possible > 0 else 0)

    def _summarize_drift_types(self) -> Dict[str, int]:
        """汇总漂移类型"""
        type_counts = defaultdict(int)
        for detection in self.detections:
            type_counts[detection.drift_type] += 1
        return dict(type_counts)

    def generate_markdown_report(self, report: DriftReport) -> str:
        """生成Markdown格式报告"""
        lines = []
        lines.append("# Metrics Drift Detection Report")
        lines.append(f"\n**Generated**: {report.timestamp.isoformat()}")
        lines.append(f"**Risk Score**: {report.risk_score:.1f}/100 ")

        # 风险等级
        if report.risk_score >= 80:
            lines.append("**Risk Level**: 🔴 CRITICAL")
        elif report.risk_score >= 60:
            lines.append("**Risk Level**: 🟠 HIGH")
        elif report.risk_score >= 40:
            lines.append("**Risk Level**: 🟡 MEDIUM")
        elif report.risk_score >= 20:
            lines.append("**Risk Level**: 🟢 LOW")
        else:
            lines.append("**Risk Level**: ✅ MINIMAL")

        lines.append(f"\n## 📊 Summary\n")
        summary = report.summary
        lines.append(f"- **Total Drifts Detected**: {summary['total_drifts']}")
        lines.append(f"- **Critical**: {summary['critical']} | **High**: {summary['high']} | "
                    f"**Medium**: {summary['medium']} | **Low**: {summary['low']}")

        # 漂移类型分布
        lines.append(f"\n### Drift Types:")
        for drift_type, count in summary['drift_types'].items():
            lines.append(f"- {drift_type}: {count}")

        # 基线信息
        lines.append(f"\n## 🎯 Baseline Information\n")
        lines.append(f"- **Environment**: {report.baseline_info['environment']}")
        lines.append(f"- **Created**: {report.baseline_info['timestamp']}")
        lines.append(f"- **Metrics**: {report.baseline_info['metrics_count']}")

        # 严重漂移Top 10
        critical_drifts = [d for d in report.detections
                          if d.severity in [DriftSeverity.CRITICAL, DriftSeverity.HIGH]]

        if critical_drifts:
            lines.append(f"\n## 🚨 Critical Drifts\n")

            for drift in sorted(critical_drifts, key=lambda x: x.drift_score, reverse=True)[:10]:
                severity_emoji = {
                    DriftSeverity.CRITICAL: "🔴",
                    DriftSeverity.HIGH: "🟠"
                }.get(drift.severity, "⚪")

                lines.append(f"### {severity_emoji} {drift.metric_name}")
                lines.append(f"- **Type**: {drift.drift_type}")
                lines.append(f"- **Drift Score**: {drift.drift_score:.1f}/100")
                lines.append(f"- **Baseline → Current**: {drift.baseline_value} → {drift.current_value}")

                # 详情
                if drift.drift_type == "cardinality":
                    lines.append(f"- **Change**: {drift.details.get('change_rate', 'N/A')} ({drift.details.get('percentage', 'N/A')})")
                elif drift.drift_type == "distribution":
                    lines.append(f"- **Metric**: {drift.details.get('metric', 'N/A')} changed {drift.details.get('change_percentage', 'N/A')}")

                lines.append(f"- **Recommendation**: {drift.recommendation}\n")

        # 所有漂移列表
        if len(report.detections) > len(critical_drifts):
            lines.append(f"\n## 📋 All Drifts\n")
            lines.append("| Severity | Metric | Type | Score | Action |")
            lines.append("|----------|--------|------|-------|--------|")

            for drift in sorted(report.detections, key=lambda x: x.drift_score, reverse=True):
                severity_emoji = {
                    DriftSeverity.CRITICAL: "🔴",
                    DriftSeverity.HIGH: "🟠",
                    DriftSeverity.MEDIUM: "🟡",
                    DriftSeverity.LOW: "🟢"
                }.get(drift.severity, "⚪")

                lines.append(f"| {severity_emoji} {drift.severity} | {drift.metric_name} | "
                           f"{drift.drift_type} | {drift.drift_score:.1f} | Monitor |")

        # 行动建议
        lines.append(f"\n## 💡 Action Items\n")

        if report.risk_score >= 80:
            lines.append("1. **IMMEDIATE**: Investigate critical drifts immediately")
            lines.append("2. **ROLLBACK**: Consider rolling back recent changes if drift is deployment-related")
            lines.append("3. **ALERT**: Notify on-call team of critical metric changes")
        elif report.risk_score >= 60:
            lines.append("1. **INVESTIGATE**: Review high-severity drifts within 24 hours")
            lines.append("2. **MONITOR**: Set up alerts for continuing drift")
            lines.append("3. **UPDATE**: Update baseline after confirming changes are expected")
        elif report.risk_score >= 40:
            lines.append("1. **REVIEW**: Analyze medium-severity drifts in next sprint")
            lines.append("2. **DOCUMENT**: Document any expected changes")
            lines.append("3. **BASELINE**: Consider updating baseline if changes are permanent")
        else:
            lines.append("1. **MONITOR**: Continue regular monitoring")
            lines.append("2. **BASELINE**: Update baseline quarterly or after major releases")

        return "\n".join(lines)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect metrics drift from baseline"
    )
    parser.add_argument(
        "baseline",
        help="Baseline file to compare against"
    )
    parser.add_argument(
        "--time-range",
        default="1h",
        help="Time range for current metrics (default: 1h)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Risk score threshold for alerts (default: 60)"
    )

    args = parser.parse_args()

    # 创建检测器
    detector = MetricsDriftDetector(args.baseline)

    # 执行检测
    report = detector.detect_drift(args.time_range)

    # 生成输出
    if args.format == "markdown":
        output = detector.generate_markdown_report(report)
    else:
        # JSON格式
        output_dict = {
            "timestamp": report.timestamp.isoformat(),
            "risk_score": report.risk_score,
            "baseline_info": report.baseline_info,
            "current_info": report.current_info,
            "summary": report.summary,
            "detections": [
                {
                    "metric": d.metric_name,
                    "type": d.drift_type,
                    "severity": d.severity,
                    "score": d.drift_score,
                    "baseline": str(d.baseline_value),
                    "current": str(d.current_value),
                    "details": d.details,
                    "recommendation": d.recommendation
                }
                for d in report.detections
            ]
        }
        output = json.dumps(output_dict, indent=2)

    # 保存或打印
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"✅ Report saved to: {args.output}")
    else:
        print(output)

    # 根据风险分数返回退出码
    if report.risk_score >= args.threshold:
        print(f"\n⚠️ Risk score ({report.risk_score:.1f}) exceeds threshold ({args.threshold})")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())