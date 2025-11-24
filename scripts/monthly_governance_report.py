#!/usr/bin/env python3
"""
Monthly Governance Report Generator
月度治理报告生成器 - 汇总所有治理指标和合规状态
"""

import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import calendar


@dataclass
class GovernanceMetric:
    """治理指标"""
    category: str
    metric_name: str
    current_value: Any
    target_value: Any
    status: str  # pass, warning, fail
    trend: str  # improving, stable, degrading
    details: str


@dataclass
class ComplianceSection:
    """合规部分"""
    name: str
    score: float  # 0-100
    violations: List[str]
    recommendations: List[str]
    metrics: List[GovernanceMetric]


@dataclass
class MonthlyGovernanceReport:
    """月度治理报告"""
    report_month: str
    generation_date: datetime
    overall_score: float
    executive_summary: str
    sections: List[ComplianceSection]
    achievements: List[str]
    issues: List[str]
    action_items: List[Dict[str, str]]
    trends: Dict[str, Any]


class GovernanceReportGenerator:
    """治理报告生成器"""

    def __init__(self, month: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent
        self.reports_dir = self.project_root / "reports" / "governance"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 设置报告月份
        if month:
            self.report_month = month
        else:
            # 默认为上个月
            today = datetime.now()
            if today.month == 1:
                self.report_month = f"{today.year - 1}-12"
            else:
                self.report_month = f"{today.year}-{today.month - 1:02d}"

        self.sections: List[ComplianceSection] = []
        self.achievements: List[str] = []
        self.issues: List[str] = []
        self.action_items: List[Dict[str, str]] = []

    def generate_report(self) -> MonthlyGovernanceReport:
        """生成完整报告"""
        print(f"📊 Generating Monthly Governance Report for {self.report_month}")
        print("=" * 60)

        # 收集各个部分的数据
        self._collect_error_code_metrics()
        self._collect_metrics_compliance()
        self._collect_resilience_metrics()
        self._collect_cardinality_metrics()
        self._collect_drift_metrics()
        self._collect_adaptive_metrics()

        # 计算总体分数
        overall_score = self._calculate_overall_score()

        # 生成执行摘要
        executive_summary = self._generate_executive_summary(overall_score)

        # 分析趋势
        trends = self._analyze_trends()

        # 创建报告
        report = MonthlyGovernanceReport(
            report_month=self.report_month,
            generation_date=datetime.now(),
            overall_score=overall_score,
            executive_summary=executive_summary,
            sections=self.sections,
            achievements=self.achievements,
            issues=self.issues,
            action_items=self.action_items,
            trends=trends
        )

        return report

    def _collect_error_code_metrics(self):
        """收集错误码生命周期指标"""
        print("📝 Collecting error code lifecycle metrics...")

        metrics = []

        # 模拟数据（实际应从审计工具获取）
        total_codes = 85
        active_codes = 62
        deprecated_codes = 8
        unused_codes = 15

        metrics.append(GovernanceMetric(
            category="Error Codes",
            metric_name="Total Error Codes",
            current_value=total_codes,
            target_value=100,
            status="pass" if total_codes <= 100 else "fail",
            trend="stable",
            details=f"{total_codes} error codes defined"
        ))

        metrics.append(GovernanceMetric(
            category="Error Codes",
            metric_name="Active Usage Rate",
            current_value=f"{active_codes/total_codes*100:.1f}%",
            target_value=">70%",
            status="pass" if active_codes/total_codes > 0.7 else "warning",
            trend="improving",
            details=f"{active_codes} codes actively used"
        ))

        metrics.append(GovernanceMetric(
            category="Error Codes",
            metric_name="Deprecation Candidates",
            current_value=deprecated_codes + unused_codes,
            target_value="<20",
            status="warning" if deprecated_codes + unused_codes < 20 else "fail",
            trend="stable",
            details=f"{deprecated_codes} deprecated, {unused_codes} unused"
        ))

        # 计算部分分数
        score = 80.0 if active_codes/total_codes > 0.7 else 60.0

        violations = []
        if unused_codes > 10:
            violations.append(f"{unused_codes} unused error codes detected")

        recommendations = []
        if deprecated_codes > 0:
            recommendations.append(f"Remove {deprecated_codes} deprecated error codes")
        if unused_codes > 0:
            recommendations.append(f"Review and remove {unused_codes} unused codes")

        section = ComplianceSection(
            name="Error Code Lifecycle",
            score=score,
            violations=violations,
            recommendations=recommendations,
            metrics=metrics
        )

        self.sections.append(section)

        if score >= 80:
            self.achievements.append("Error code usage rate above 70%")
        else:
            self.issues.append("High number of unused error codes")

    def _collect_metrics_compliance(self):
        """收集指标合规性数据"""
        print("📏 Collecting metrics compliance data...")

        metrics = []

        # 模拟数据
        total_metrics = 45
        compliant_metrics = 38
        label_violations = 3
        cardinality_warnings = 4

        compliance_rate = compliant_metrics / total_metrics * 100

        metrics.append(GovernanceMetric(
            category="Metrics",
            metric_name="Schema Compliance",
            current_value=f"{compliance_rate:.1f}%",
            target_value=">90%",
            status="pass" if compliance_rate > 90 else "warning",
            trend="improving",
            details=f"{compliant_metrics}/{total_metrics} metrics compliant"
        ))

        metrics.append(GovernanceMetric(
            category="Metrics",
            metric_name="Label Violations",
            current_value=label_violations,
            target_value=0,
            status="warning" if label_violations > 0 else "pass",
            trend="improving",
            details=f"{label_violations} metrics with forbidden labels"
        ))

        score = max(0, 100 - label_violations * 5 - cardinality_warnings * 2)

        violations = []
        if label_violations > 0:
            violations.append(f"{label_violations} metrics using forbidden labels")

        recommendations = []
        if label_violations > 0:
            recommendations.append("Remove sensitive labels (user_id, session_id)")
        if cardinality_warnings > 0:
            recommendations.append("Review high cardinality metrics")

        section = ComplianceSection(
            name="Metrics Compliance",
            score=score,
            violations=violations,
            recommendations=recommendations,
            metrics=metrics
        )

        self.sections.append(section)

    def _collect_resilience_metrics(self):
        """收集韧性层指标"""
        print("🛡️ Collecting resilience metrics...")

        metrics = []

        # 模拟数据
        circuit_breaker_coverage = 85
        rate_limiter_coverage = 78
        avg_recovery_time = 8.5  # seconds
        failed_requests_prevented = 15420

        metrics.append(GovernanceMetric(
            category="Resilience",
            metric_name="Circuit Breaker Coverage",
            current_value=f"{circuit_breaker_coverage}%",
            target_value=">80%",
            status="pass",
            trend="stable",
            details="85% of critical paths protected"
        ))

        metrics.append(GovernanceMetric(
            category="Resilience",
            metric_name="Rate Limiter Coverage",
            current_value=f"{rate_limiter_coverage}%",
            target_value=">75%",
            status="pass",
            trend="improving",
            details="78% of endpoints rate limited"
        ))

        metrics.append(GovernanceMetric(
            category="Resilience",
            metric_name="Avg Recovery Time",
            current_value=f"{avg_recovery_time}s",
            target_value="<10s",
            status="pass",
            trend="improving",
            details="Average circuit recovery in 8.5 seconds"
        ))

        score = (circuit_breaker_coverage + rate_limiter_coverage) / 2

        section = ComplianceSection(
            name="Resilience Layer",
            score=score,
            violations=[],
            recommendations=["Increase rate limiter coverage to 85%"],
            metrics=metrics
        )

        self.sections.append(section)
        self.achievements.append(f"Prevented {failed_requests_prevented:,} cascading failures")

    def _collect_cardinality_metrics(self):
        """收集基数控制指标"""
        print("📈 Collecting cardinality metrics...")

        metrics = []

        # 模拟数据
        total_series = 125000
        max_series_limit = 1000000
        high_cardinality_metrics = 5
        growth_rate = 3.2  # % per week

        metrics.append(GovernanceMetric(
            category="Cardinality",
            metric_name="Total Series",
            current_value=f"{total_series:,}",
            target_value=f"<{max_series_limit:,}",
            status="pass",
            trend="stable",
            details=f"Using {total_series/max_series_limit*100:.1f}% of limit"
        ))

        metrics.append(GovernanceMetric(
            category="Cardinality",
            metric_name="Growth Rate",
            current_value=f"{growth_rate}%/week",
            target_value="<5%/week",
            status="pass",
            trend="stable",
            details="Controlled growth rate"
        ))

        metrics.append(GovernanceMetric(
            category="Cardinality",
            metric_name="High Cardinality Metrics",
            current_value=high_cardinality_metrics,
            target_value="<10",
            status="pass",
            trend="stable",
            details=f"{high_cardinality_metrics} metrics exceed threshold"
        ))

        score = 90 if high_cardinality_metrics < 10 else 70

        section = ComplianceSection(
            name="Cardinality Control",
            score=score,
            violations=[],
            recommendations=["Monitor top 5 high cardinality metrics"],
            metrics=metrics
        )

        self.sections.append(section)

    def _collect_drift_metrics(self):
        """收集漂移检测指标"""
        print("🔄 Collecting drift detection metrics...")

        metrics = []

        # 模拟数据
        drifted_metrics = 8
        critical_drifts = 1
        baseline_age_days = 14

        metrics.append(GovernanceMetric(
            category="Drift",
            metric_name="Drifted Metrics",
            current_value=drifted_metrics,
            target_value="<15",
            status="pass",
            trend="stable",
            details=f"{drifted_metrics} metrics drifted from baseline"
        ))

        metrics.append(GovernanceMetric(
            category="Drift",
            metric_name="Critical Drifts",
            current_value=critical_drifts,
            target_value=0,
            status="warning" if critical_drifts > 0 else "pass",
            trend="improving",
            details=f"{critical_drifts} critical drift detected"
        ))

        score = 85 if critical_drifts == 0 else 70

        recommendations = []
        if baseline_age_days > 30:
            recommendations.append("Update baseline (current age: 14 days)")
        if critical_drifts > 0:
            recommendations.append("Investigate critical drift immediately")

        section = ComplianceSection(
            name="Drift Detection",
            score=score,
            violations=[] if critical_drifts == 0 else [f"{critical_drifts} critical drift"],
            recommendations=recommendations,
            metrics=metrics
        )

        self.sections.append(section)

    def _collect_adaptive_metrics(self):
        """收集自适应限流指标"""
        print("🎯 Collecting adaptive rate limiting metrics...")

        metrics = []

        # 模拟数据
        adaptive_enabled_pct = 92
        avg_adjustment_time = 1.8  # seconds
        false_positive_rate = 2.1  # %
        performance_overhead = 3.5  # %

        metrics.append(GovernanceMetric(
            category="Adaptive",
            metric_name="Coverage",
            current_value=f"{adaptive_enabled_pct}%",
            target_value=">90%",
            status="pass",
            trend="stable",
            details=f"{adaptive_enabled_pct}% endpoints adaptive-enabled"
        ))

        metrics.append(GovernanceMetric(
            category="Adaptive",
            metric_name="Response Time",
            current_value=f"{avg_adjustment_time}s",
            target_value="<2s",
            status="pass",
            trend="improving",
            details="Fast adaptation to load changes"
        ))

        metrics.append(GovernanceMetric(
            category="Adaptive",
            metric_name="P95 Overhead",
            current_value=f"{performance_overhead}%",
            target_value="<5%",
            status="pass",
            trend="stable",
            details="Within performance budget"
        ))

        score = 95 if performance_overhead < 5 else 80

        section = ComplianceSection(
            name="Adaptive Rate Limiting",
            score=score,
            violations=[],
            recommendations=[],
            metrics=metrics
        )

        self.sections.append(section)
        self.achievements.append(f"Adaptive limiting overhead only {performance_overhead}%")

    def _calculate_overall_score(self) -> float:
        """计算总体治理分数"""
        if not self.sections:
            return 0.0

        # 加权平均
        weights = {
            "Error Code Lifecycle": 0.15,
            "Metrics Compliance": 0.20,
            "Resilience Layer": 0.25,
            "Cardinality Control": 0.15,
            "Drift Detection": 0.10,
            "Adaptive Rate Limiting": 0.15
        }

        total_score = 0.0
        total_weight = 0.0

        for section in self.sections:
            weight = weights.get(section.name, 0.1)
            total_score += section.score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _generate_executive_summary(self, overall_score: float) -> str:
        """生成执行摘要"""
        year, month = self.report_month.split('-')
        month_name = calendar.month_name[int(month)]

        summary = f"The governance score for {month_name} {year} is {overall_score:.1f}/100. "

        if overall_score >= 85:
            summary += "The platform demonstrates excellent governance with all key metrics within targets. "
        elif overall_score >= 70:
            summary += "The platform shows good governance with most metrics meeting targets. "
        else:
            summary += "The platform requires attention in several governance areas. "

        # 添加关键成就
        if self.achievements:
            summary += f"Key achievement: {self.achievements[0]}. "

        # 添加主要问题
        if self.issues:
            summary += f"Main concern: {self.issues[0]}. "

        # 添加建议
        critical_sections = [s for s in self.sections if s.score < 70]
        if critical_sections:
            summary += f"Priority focus area: {critical_sections[0].name}."

        return summary

    def _analyze_trends(self) -> Dict[str, Any]:
        """分析趋势"""
        # 模拟历史数据
        trends = {
            "overall_score_trend": [78.5, 80.2, 82.1, 83.5, 85.0],  # 最近5个月
            "error_codes_trend": "decreasing",
            "cardinality_trend": "stable",
            "resilience_trend": "improving",
            "improvements": [
                "15% reduction in unused error codes",
                "20% increase in resilience coverage",
                "Adaptive rate limiting fully deployed"
            ],
            "concerns": [
                "Metrics cardinality growing 3.2% weekly",
                "3 metrics still using forbidden labels"
            ]
        }

        return trends

    def generate_markdown_report(self, report: MonthlyGovernanceReport) -> str:
        """生成Markdown格式报告"""
        lines = []

        year, month = report.report_month.split('-')
        month_name = calendar.month_name[int(month)]

        lines.append(f"# Monthly Governance Report - {month_name} {year}")
        lines.append(f"\n**Generated**: {report.generation_date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Overall Score**: {report.overall_score:.1f}/100")

        # 评级
        if report.overall_score >= 85:
            lines.append("**Rating**: ⭐⭐⭐⭐⭐ EXCELLENT")
        elif report.overall_score >= 75:
            lines.append("**Rating**: ⭐⭐⭐⭐ GOOD")
        elif report.overall_score >= 65:
            lines.append("**Rating**: ⭐⭐⭐ SATISFACTORY")
        else:
            lines.append("**Rating**: ⭐⭐ NEEDS IMPROVEMENT")

        lines.append(f"\n## 📋 Executive Summary\n")
        lines.append(report.executive_summary)

        # 分数卡
        lines.append("\n## 🎯 Governance Scorecard\n")
        lines.append("| Category | Score | Status | Trend |")
        lines.append("|----------|-------|--------|-------|")

        for section in report.sections:
            status_emoji = "✅" if section.score >= 80 else "⚠️" if section.score >= 60 else "❌"
            trend_emoji = "📈" if any(m.trend == "improving" for m in section.metrics) else "➡️"
            lines.append(f"| {section.name} | {section.score:.0f} | {status_emoji} | {trend_emoji} |")

        # 成就
        if report.achievements:
            lines.append("\n## 🏆 Achievements\n")
            for achievement in report.achievements:
                lines.append(f"- ✨ {achievement}")

        # 问题
        if report.issues:
            lines.append("\n## ⚠️ Issues Identified\n")
            for issue in report.issues:
                lines.append(f"- 🔴 {issue}")

        # 详细指标
        lines.append("\n## 📊 Detailed Metrics\n")

        for section in report.sections:
            lines.append(f"### {section.name} (Score: {section.score:.0f}/100)\n")

            if section.metrics:
                lines.append("| Metric | Current | Target | Status |")
                lines.append("|--------|---------|--------|--------|")

                for metric in section.metrics:
                    status_icon = "✅" if metric.status == "pass" else "⚠️" if metric.status == "warning" else "❌"
                    lines.append(f"| {metric.metric_name} | {metric.current_value} | {metric.target_value} | {status_icon} |")

            if section.violations:
                lines.append("\n**Violations:**")
                for violation in section.violations:
                    lines.append(f"- ❌ {violation}")

            if section.recommendations:
                lines.append("\n**Recommendations:**")
                for rec in section.recommendations:
                    lines.append(f"- 💡 {rec}")

            lines.append("")

        # 趋势分析
        lines.append("## 📈 Trend Analysis\n")

        if report.trends.get("overall_score_trend"):
            scores = report.trends["overall_score_trend"]
            lines.append(f"**Overall Score Trend** (Last 5 months): {' → '.join(str(s) for s in scores)}")

        improvements = report.trends.get("improvements", [])
        if improvements:
            lines.append("\n**Improvements:**")
            for imp in improvements:
                lines.append(f"- 📈 {imp}")

        concerns = report.trends.get("concerns", [])
        if concerns:
            lines.append("\n**Concerns:**")
            for concern in concerns:
                lines.append(f"- 📉 {concern}")

        # 行动项
        lines.append("\n## 🎬 Action Items\n")

        # 生成行动项
        priority_items = []

        for section in report.sections:
            if section.score < 70:
                priority_items.append({
                    "priority": "HIGH",
                    "action": f"Improve {section.name}",
                    "owner": "platform-team",
                    "due": "Next sprint"
                })

        if not priority_items:
            priority_items.append({
                "priority": "MEDIUM",
                "action": "Maintain current governance standards",
                "owner": "platform-team",
                "due": "Ongoing"
            })

        lines.append("| Priority | Action | Owner | Due Date |")
        lines.append("|----------|--------|-------|----------|")

        for item in priority_items:
            priority_emoji = "🔴" if item["priority"] == "HIGH" else "🟡" if item["priority"] == "MEDIUM" else "🟢"
            lines.append(f"| {priority_emoji} {item['priority']} | {item['action']} | {item['owner']} | {item['due']} |")

        # 下月重点
        lines.append("\n## 🎯 Next Month Focus Areas\n")
        lines.append("1. **Error Code Cleanup**: Remove deprecated and unused codes")
        lines.append("2. **Metrics Compliance**: Fix remaining label violations")
        lines.append("3. **Baseline Update**: Refresh drift detection baseline")
        lines.append("4. **Cardinality Review**: Analyze top 10 high cardinality metrics")

        # 签名
        lines.append("\n---")
        lines.append("\n*This report was automatically generated by the Governance Report System.*")
        lines.append(f"*For questions, contact the Platform Team.*")

        return "\n".join(lines)

    def save_report(self, report: MonthlyGovernanceReport, format: str = "markdown"):
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"governance_report_{report.report_month}_{timestamp}"

        if format == "markdown":
            content = self.generate_markdown_report(report)
            filepath = self.reports_dir / f"{filename}.md"
            filepath.write_text(content)
        else:  # json
            report_dict = {
                "report_month": report.report_month,
                "generation_date": report.generation_date.isoformat(),
                "overall_score": report.overall_score,
                "executive_summary": report.executive_summary,
                "sections": [
                    {
                        "name": s.name,
                        "score": s.score,
                        "violations": s.violations,
                        "recommendations": s.recommendations,
                        "metrics": [asdict(m) for m in s.metrics]
                    }
                    for s in report.sections
                ],
                "achievements": report.achievements,
                "issues": report.issues,
                "trends": report.trends
            }
            filepath = self.reports_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)

        print(f"✅ Report saved to: {filepath}")
        return str(filepath)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate monthly governance report"
    )

    parser.add_argument(
        "--month",
        help="Report month (YYYY-MM format)",
        default=None
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (optional)"
    )

    args = parser.parse_args()

    # 生成报告
    generator = GovernanceReportGenerator(args.month)
    report = generator.generate_report()

    # 保存或输出
    if args.output:
        # 保存到指定路径
        if args.format == "markdown":
            content = generator.generate_markdown_report(report)
        else:
            content = json.dumps(asdict(report), indent=2, default=str)

        with open(args.output, 'w') as f:
            f.write(content)
        print(f"\n✅ Report saved to: {args.output}")
    else:
        # 保存到默认位置并打印
        filepath = generator.save_report(report, args.format)

        # 如果是markdown，也打印到控制台
        if args.format == "markdown":
            print("\n" + "=" * 60)
            print(generator.generate_markdown_report(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())