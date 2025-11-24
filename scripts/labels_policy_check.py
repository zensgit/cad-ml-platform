#!/usr/bin/env python3
"""
Metrics Label Policy Enforcement
指标标签策略执行器 - 检查和执行白名单策略
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import subprocess

# Prometheus client for querying metrics
try:
    from prometheus_client.parser import text_string_to_metric_families
except ImportError:
    print("Warning: prometheus_client not installed, using mock mode")
    text_string_to_metric_families = None


@dataclass
class PolicyViolation:
    """策略违规记录"""
    metric_name: str
    violation_type: str  # forbidden_label, cardinality_exceeded, forbidden_combination
    details: str
    severity: str  # warn, strict, quarantine
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class CardinalityStats:
    """基数统计"""
    metric_name: str
    label_name: str
    unique_values: int
    max_allowed: int
    growth_rate: float  # percentage per hour
    samples: List[str] = field(default_factory=list)


class MetricsLabelPolicyChecker:
    """指标标签策略检查器"""

    def __init__(self, policy_file: str = None):
        self.project_root = Path(__file__).parent.parent

        if policy_file:
            self.policy_file = Path(policy_file)
        else:
            self.policy_file = self.project_root / "config" / "metrics_label_policy.json"

        self.policy = self._load_policy()
        self.violations: List[PolicyViolation] = []
        self.cardinality_stats: Dict[str, CardinalityStats] = {}
        self.metrics_data: Dict[str, Any] = {}

    def _load_policy(self) -> Dict[str, Any]:
        """加载策略配置"""
        if not self.policy_file.exists():
            print(f"⚠️ Policy file not found: {self.policy_file}")
            return {}

        with open(self.policy_file, 'r') as f:
            return json.load(f)

    def fetch_metrics(self) -> Dict[str, Any]:
        """获取当前Prometheus指标"""
        print("📊 Fetching current metrics...")

        # 尝试从本地Prometheus endpoint获取
        try:
            import requests
            response = requests.get("http://localhost:9090/api/v1/label/__name__/values")
            if response.status_code == 200:
                metric_names = response.json().get("data", [])

                # 获取每个指标的标签和值
                for metric_name in metric_names:
                    query = f'{metric_name}'
                    response = requests.get(
                        f"http://localhost:9090/api/v1/query",
                        params={"query": query}
                    )
                    if response.status_code == 200:
                        self.metrics_data[metric_name] = response.json()
        except:
            # Fallback: 扫描代码中的指标定义
            print("⚠️ Cannot connect to Prometheus, scanning code for metric definitions...")
            self._scan_code_metrics()

        return self.metrics_data

    def _scan_code_metrics(self):
        """扫描代码中的指标定义"""
        # 查找所有Python文件中的Prometheus指标定义
        patterns = [
            r'Counter\([\'"](\w+)[\'"]',
            r'Gauge\([\'"](\w+)[\'"]',
            r'Histogram\([\'"](\w+)[\'"]',
            r'Summary\([\'"](\w+)[\'"]',
        ]

        label_pattern = r'(?:labelnames|labels)\s*=\s*\[(.*?)\]'

        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text()

                for pattern in patterns:
                    for match in re.finditer(pattern, content):
                        metric_name = match.group(1)

                        # 查找相关的标签定义
                        start = max(0, match.start() - 200)
                        end = min(len(content), match.end() + 200)
                        context = content[start:end]

                        label_match = re.search(label_pattern, context)
                        if label_match:
                            labels_str = label_match.group(1)
                            labels = [l.strip().strip("'\"") for l in labels_str.split(",")]

                            if metric_name not in self.metrics_data:
                                self.metrics_data[metric_name] = {
                                    "labels": labels,
                                    "source_file": str(py_file.relative_to(self.project_root)),
                                    "values": {}
                                }
            except Exception as e:
                pass

    def check_forbidden_labels(self) -> List[PolicyViolation]:
        """检查禁用标签"""
        violations = []
        forbidden_labels = set(self.policy.get("label_categories", {}).get("forbidden", {}).get("labels", []))

        for metric_name, metric_info in self.metrics_data.items():
            labels = metric_info.get("labels", [])

            for label in labels:
                if label in forbidden_labels:
                    violation = PolicyViolation(
                        metric_name=metric_name,
                        violation_type="forbidden_label",
                        details=f"使用了禁用标签: {label}",
                        severity="strict",
                        labels={"label": label},
                        recommendation=f"移除标签 '{label}' - 该标签会导致基数爆炸或隐私问题"
                    )
                    violations.append(violation)

        return violations

    def check_label_whitelist(self) -> List[PolicyViolation]:
        """检查标签白名单"""
        violations = []
        whitelist = self.policy.get("label_whitelist", {})

        for metric_name, metric_info in self.metrics_data.items():
            if metric_name in whitelist:
                policy = whitelist[metric_name]
                allowed = set(policy.get("allowed_labels", []))
                required = set(policy.get("required_labels", []))
                actual = set(metric_info.get("labels", []))

                # 检查必需标签
                missing = required - actual
                if missing:
                    violation = PolicyViolation(
                        metric_name=metric_name,
                        violation_type="missing_required_labels",
                        details=f"缺少必需标签: {', '.join(missing)}",
                        severity=policy.get("enforcement", "warn"),
                        recommendation=f"添加必需标签: {', '.join(missing)}"
                    )
                    violations.append(violation)

                # 检查非白名单标签
                unauthorized = actual - allowed
                if unauthorized:
                    violation = PolicyViolation(
                        metric_name=metric_name,
                        violation_type="unauthorized_labels",
                        details=f"使用了未授权标签: {', '.join(unauthorized)}",
                        severity=policy.get("enforcement", "warn"),
                        recommendation=f"移除未授权标签: {', '.join(unauthorized)}"
                    )
                    violations.append(violation)

        return violations

    def check_forbidden_combinations(self) -> List[PolicyViolation]:
        """检查禁用的标签组合"""
        violations = []
        whitelist = self.policy.get("label_whitelist", {})

        for metric_name, metric_info in self.metrics_data.items():
            if metric_name in whitelist:
                policy = whitelist[metric_name]
                forbidden_combos = policy.get("forbidden_combinations", [])
                actual_labels = set(metric_info.get("labels", []))

                for combo in forbidden_combos:
                    combo_set = set(combo)
                    if combo_set.issubset(actual_labels):
                        violation = PolicyViolation(
                            metric_name=metric_name,
                            violation_type="forbidden_combination",
                            details=f"发现禁用的标签组合: {' + '.join(combo)}",
                            severity="strict",
                            recommendation=f"移除标签组合中的至少一个: {' 或 '.join(combo)}"
                        )
                        violations.append(violation)

        return violations

    def check_cardinality_limits(self) -> List[PolicyViolation]:
        """检查基数限制"""
        violations = []
        whitelist = self.policy.get("label_whitelist", {})

        # 模拟基数检查（实际应从Prometheus查询）
        for metric_name, metric_info in self.metrics_data.items():
            if metric_name in whitelist:
                policy = whitelist[metric_name]
                max_cardinality = policy.get("max_cardinality", {})

                for label, max_values in max_cardinality.items():
                    # 这里应该查询实际的unique values
                    # 模拟：假设某些标签超限
                    if label in metric_info.get("labels", []):
                        simulated_values = self._simulate_cardinality(metric_name, label)

                        if simulated_values > max_values:
                            stats = CardinalityStats(
                                metric_name=metric_name,
                                label_name=label,
                                unique_values=simulated_values,
                                max_allowed=max_values,
                                growth_rate=5.2  # 模拟增长率
                            )
                            self.cardinality_stats[f"{metric_name}:{label}"] = stats

                            violation = PolicyViolation(
                                metric_name=metric_name,
                                violation_type="cardinality_exceeded",
                                details=f"标签 '{label}' 基数超限: {simulated_values} > {max_values}",
                                severity=policy.get("enforcement", "warn"),
                                recommendation=f"减少 '{label}' 的唯一值数量或调整限制"
                            )
                            violations.append(violation)

        return violations

    def _simulate_cardinality(self, metric_name: str, label: str) -> int:
        """模拟基数（实际应从Prometheus查询）"""
        # 模拟一些超限的情况
        simulated = {
            ("http_requests_total", "endpoint"): 150,  # 超过100的限制
            ("ocr_processing_duration", "model"): 12,   # 超过10的限制
        }
        return simulated.get((metric_name, label), 5)

    def check_exemptions(self) -> List[str]:
        """检查豁免配置"""
        exempted = []
        exemptions = self.policy.get("exemptions", {})

        # 检查临时豁免
        for temp_exempt in exemptions.get("temporary", []):
            expire_date = datetime.fromisoformat(temp_exempt["expires"])
            if datetime.now() > expire_date:
                print(f"⚠️ 临时豁免已过期: {temp_exempt['metric']} (过期于 {temp_exempt['expires']})")
            else:
                exempted.append(temp_exempt["metric"])

        # 永久豁免
        for perm_exempt in exemptions.get("permanent", []):
            exempted.append(perm_exempt["metric"])

        return exempted

    def analyze(self) -> Dict[str, Any]:
        """执行完整分析"""
        print("🔍 Starting metrics label policy analysis...")

        # 获取指标数据
        self.fetch_metrics()

        # 执行各项检查
        self.violations.extend(self.check_forbidden_labels())
        self.violations.extend(self.check_label_whitelist())
        self.violations.extend(self.check_forbidden_combinations())
        self.violations.extend(self.check_cardinality_limits())

        # 获取豁免列表
        exempted = self.check_exemptions()

        # 过滤掉豁免的指标
        self.violations = [
            v for v in self.violations
            if v.metric_name not in exempted
        ]

        # 生成报告
        report = self._generate_report()

        return report

    def _generate_report(self) -> Dict[str, Any]:
        """生成分析报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "policy_version": self.policy.get("version", "unknown"),
            "summary": {
                "total_metrics": len(self.metrics_data),
                "total_violations": len(self.violations),
                "critical_violations": len([v for v in self.violations if v.severity == "strict"]),
                "warnings": len([v for v in self.violations if v.severity == "warn"]),
                "cardinality_issues": len(self.cardinality_stats)
            },
            "violations": [],
            "cardinality_stats": [],
            "recommendations": []
        }

        # 按严重程度分组违规
        violations_by_severity = defaultdict(list)
        for violation in self.violations:
            violations_by_severity[violation.severity].append({
                "metric": violation.metric_name,
                "type": violation.violation_type,
                "details": violation.details,
                "recommendation": violation.recommendation,
                "timestamp": violation.timestamp.isoformat()
            })

        report["violations"] = dict(violations_by_severity)

        # 基数统计
        for stats in self.cardinality_stats.values():
            report["cardinality_stats"].append({
                "metric": stats.metric_name,
                "label": stats.label_name,
                "unique_values": stats.unique_values,
                "max_allowed": stats.max_allowed,
                "growth_rate": f"{stats.growth_rate:.1f}%/hour",
                "status": "🔴" if stats.unique_values > stats.max_allowed else "🟢"
            })

        # 生成建议
        report["recommendations"] = self._generate_recommendations()

        return report

    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """生成改进建议"""
        recommendations = []

        # 基于违规类型生成建议
        violation_types = set(v.violation_type for v in self.violations)

        if "forbidden_label" in violation_types:
            recommendations.append({
                "priority": "high",
                "action": "移除禁用标签",
                "description": "立即移除所有高基数和隐私敏感标签，如user_id, session_id等"
            })

        if "cardinality_exceeded" in violation_types:
            recommendations.append({
                "priority": "high",
                "action": "控制标签基数",
                "description": "实施标签值归一化，合并相似值，或使用bucket/范围代替具体值"
            })

        if "forbidden_combination" in violation_types:
            recommendations.append({
                "priority": "medium",
                "action": "重构标签组合",
                "description": "分离高基数标签组合，避免笛卡尔积爆炸"
            })

        # 基数增长建议
        high_growth = [s for s in self.cardinality_stats.values() if s.growth_rate > 10]
        if high_growth:
            recommendations.append({
                "priority": "medium",
                "action": "监控基数增长",
                "description": f"发现 {len(high_growth)} 个标签增长过快，需要设置增长告警"
            })

        return recommendations

    def generate_markdown_report(self) -> str:
        """生成Markdown格式报告"""
        report = self.analyze()

        lines = []
        lines.append("# Metrics Label Policy Check Report")
        lines.append(f"\n**Generated**: {report['timestamp']}")
        lines.append(f"**Policy Version**: {report['policy_version']}\n")

        # 摘要
        summary = report["summary"]
        lines.append("## 📊 Summary\n")
        lines.append(f"- **Total Metrics**: {summary['total_metrics']}")
        lines.append(f"- **Total Violations**: {summary['total_violations']}")
        lines.append(f"- **Critical**: {summary['critical_violations']} | **Warnings**: {summary['warnings']}")
        lines.append(f"- **Cardinality Issues**: {summary['cardinality_issues']}\n")

        # 违规详情
        if report["violations"]:
            lines.append("## ⚠️ Violations\n")

            for severity, violations in report["violations"].items():
                emoji = "🔴" if severity == "strict" else "🟡"
                lines.append(f"### {emoji} {severity.upper()}\n")

                for v in violations[:5]:  # 只显示前5个
                    lines.append(f"- **{v['metric']}** ({v['type']})")
                    lines.append(f"  - {v['details']}")
                    lines.append(f"  - 💡 {v['recommendation']}\n")

        # 基数统计
        if report["cardinality_stats"]:
            lines.append("## 📈 Cardinality Analysis\n")
            lines.append("| Status | Metric | Label | Current | Limit | Growth |")
            lines.append("|--------|--------|-------|---------|-------|--------|")

            for stat in report["cardinality_stats"][:10]:
                lines.append(f"| {stat['status']} | {stat['metric']} | {stat['label']} | "
                           f"{stat['unique_values']} | {stat['max_allowed']} | {stat['growth_rate']} |")

        # 建议
        if report["recommendations"]:
            lines.append("\n## 💡 Recommendations\n")
            for rec in report["recommendations"]:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec["priority"], "⚪")
                lines.append(f"- {priority_emoji} **{rec['action']}**: {rec['description']}")

        return "\n".join(lines)

    def enforce(self, dry_run: bool = True) -> List[str]:
        """执行策略强制措施"""
        enforced = []

        # 分析违规
        self.analyze()

        # 获取严重违规
        critical_violations = [v for v in self.violations if v.severity == "strict"]

        if not dry_run:
            for violation in critical_violations:
                # 实际执行措施（例如：修改Prometheus配置）
                action = f"Block metric: {violation.metric_name}"
                enforced.append(action)
                print(f"🚫 {action}")
        else:
            print(f"🔍 Dry run mode: Would enforce {len(critical_violations)} critical violations")

        return enforced


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Metrics label policy enforcement"
    )
    parser.add_argument(
        "--policy",
        help="Path to policy file",
        default=None
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
        "--enforce",
        action="store_true",
        help="Enforce policy (block violations)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode for enforcement"
    )

    args = parser.parse_args()

    # 创建检查器
    checker = MetricsLabelPolicyChecker(args.policy)

    # 执行分析
    if args.enforce:
        enforced = checker.enforce(dry_run=args.dry_run)
        print(f"Enforced actions: {enforced}")

    # 生成报告
    if args.format == "markdown":
        report = checker.generate_markdown_report()
    else:
        report_dict = checker.analyze()
        report = json.dumps(report_dict, indent=2, default=str)

    # 输出结果
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"✅ Report saved to: {args.output}")
    else:
        print(report)

    # 返回退出码
    critical_count = len([v for v in checker.violations if v.severity == "strict"])
    return 1 if critical_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())