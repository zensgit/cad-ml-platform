#!/usr/bin/env python3
"""
Prometheus Rules Diff Tool
Prometheus规则差异工具 - 比较recording rules和alert rules的变更
"""

import yaml
import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import difflib
import re


class ChangeType(str, Enum):
    """变更类型"""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class RuleChange:
    """规则变更"""
    rule_name: str
    rule_type: str  # recording_rule or alert
    change_type: ChangeType
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    details: List[str] = field(default_factory=list)
    severity: str = "info"  # info, warning, critical


@dataclass
class RuleDiffReport:
    """差异报告"""
    timestamp: datetime
    old_file: str
    new_file: str
    total_changes: int
    changes_by_type: Dict[str, int]
    rule_changes: List[RuleChange]
    breaking_changes: List[RuleChange]
    summary: Dict[str, Any]


class PrometheusRulesDiff:
    """Prometheus规则差异分析工具"""

    def __init__(self):
        self.old_rules: Dict[str, Any] = {}
        self.new_rules: Dict[str, Any] = {}
        self.changes: List[RuleChange] = []
        self.breaking_changes: List[RuleChange] = []

    def load_rules(self, file_path: str) -> Dict[str, Any]:
        """加载规则文件"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Rules file not found: {file_path}")

        with open(path, 'r') as f:
            content = yaml.safe_load(f)

        # 解析规则
        rules_dict = {
            'recording_rules': {},
            'alert_rules': {},
            'groups': content.get('groups', [])
        }

        for group in content.get('groups', []):
            group_name = group.get('name', 'unnamed')

            for rule in group.get('rules', []):
                if 'record' in rule:
                    # Recording rule
                    rule_name = rule['record']
                    rules_dict['recording_rules'][rule_name] = {
                        'group': group_name,
                        'expr': rule.get('expr', ''),
                        'labels': rule.get('labels', {})
                    }
                elif 'alert' in rule:
                    # Alert rule
                    rule_name = rule['alert']
                    rules_dict['alert_rules'][rule_name] = {
                        'group': group_name,
                        'expr': rule.get('expr', ''),
                        'for': rule.get('for', '0s'),
                        'labels': rule.get('labels', {}),
                        'annotations': rule.get('annotations', {})
                    }

        return rules_dict

    def diff_rules(self, old_file: str, new_file: str) -> RuleDiffReport:
        """比较两个规则文件"""
        print(f"🔍 Comparing rules files:")
        print(f"   Old: {old_file}")
        print(f"   New: {new_file}")

        # 加载规则
        self.old_rules = self.load_rules(old_file)
        self.new_rules = self.load_rules(new_file)

        # 清空之前的变更
        self.changes.clear()
        self.breaking_changes.clear()

        # 比较recording rules
        self._diff_recording_rules()

        # 比较alert rules
        self._diff_alert_rules()

        # 识别破坏性变更
        self._identify_breaking_changes()

        # 生成报告
        report = self._generate_report(old_file, new_file)

        return report

    def _diff_recording_rules(self):
        """比较recording rules"""
        old_recording = self.old_rules.get('recording_rules', {})
        new_recording = self.new_rules.get('recording_rules', {})

        all_rules = set(old_recording.keys()) | set(new_recording.keys())

        for rule_name in all_rules:
            old_rule = old_recording.get(rule_name)
            new_rule = new_recording.get(rule_name)

            if old_rule is None and new_rule is not None:
                # 新增规则
                change = RuleChange(
                    rule_name=rule_name,
                    rule_type="recording_rule",
                    change_type=ChangeType.ADDED,
                    new_value=new_rule,
                    details=[f"New recording rule added in group '{new_rule['group']}'"],
                    severity="info"
                )
                self.changes.append(change)

            elif old_rule is not None and new_rule is None:
                # 删除规则
                change = RuleChange(
                    rule_name=rule_name,
                    rule_type="recording_rule",
                    change_type=ChangeType.REMOVED,
                    old_value=old_rule,
                    details=[f"Recording rule removed from group '{old_rule['group']}'"],
                    severity="warning"
                )
                self.changes.append(change)

            elif old_rule != new_rule:
                # 修改规则
                details = self._compare_rule_details(old_rule, new_rule)

                severity = "info"
                if old_rule.get('expr') != new_rule.get('expr'):
                    severity = "warning"  # 表达式变更是重要的

                change = RuleChange(
                    rule_name=rule_name,
                    rule_type="recording_rule",
                    change_type=ChangeType.MODIFIED,
                    old_value=old_rule,
                    new_value=new_rule,
                    details=details,
                    severity=severity
                )
                self.changes.append(change)

    def _diff_alert_rules(self):
        """比较alert rules"""
        old_alerts = self.old_rules.get('alert_rules', {})
        new_alerts = self.new_rules.get('alert_rules', {})

        all_alerts = set(old_alerts.keys()) | set(new_alerts.keys())

        for alert_name in all_alerts:
            old_alert = old_alerts.get(alert_name)
            new_alert = new_alerts.get(alert_name)

            if old_alert is None and new_alert is not None:
                # 新增告警
                change = RuleChange(
                    rule_name=alert_name,
                    rule_type="alert",
                    change_type=ChangeType.ADDED,
                    new_value=new_alert,
                    details=[f"New alert added in group '{new_alert['group']}'"],
                    severity="info"
                )
                self.changes.append(change)

            elif old_alert is not None and new_alert is None:
                # 删除告警
                change = RuleChange(
                    rule_name=alert_name,
                    rule_type="alert",
                    change_type=ChangeType.REMOVED,
                    old_value=old_alert,
                    details=[f"Alert removed from group '{old_alert['group']}'"],
                    severity="critical"  # 删除告警是严重的
                )
                self.changes.append(change)

            elif old_alert != new_alert:
                # 修改告警
                details = self._compare_alert_details(old_alert, new_alert)

                # 判断严重程度
                severity = "info"
                if old_alert.get('expr') != new_alert.get('expr'):
                    severity = "warning"
                if old_alert.get('for') != new_alert.get('for'):
                    severity = "warning"
                if old_alert.get('labels', {}).get('severity') != new_alert.get('labels', {}).get('severity'):
                    severity = "critical"

                change = RuleChange(
                    rule_name=alert_name,
                    rule_type="alert",
                    change_type=ChangeType.MODIFIED,
                    old_value=old_alert,
                    new_value=new_alert,
                    details=details,
                    severity=severity
                )
                self.changes.append(change)

    def _compare_rule_details(self, old_rule: Dict, new_rule: Dict) -> List[str]:
        """比较规则细节"""
        details = []

        # 比较表达式
        if old_rule.get('expr') != new_rule.get('expr'):
            details.append("Expression changed:")
            details.append(f"  OLD: {old_rule.get('expr')}")
            details.append(f"  NEW: {new_rule.get('expr')}")

        # 比较标签
        old_labels = old_rule.get('labels', {})
        new_labels = new_rule.get('labels', {})

        if old_labels != new_labels:
            added_labels = set(new_labels.keys()) - set(old_labels.keys())
            removed_labels = set(old_labels.keys()) - set(new_labels.keys())
            changed_labels = {k for k in old_labels.keys() & new_labels.keys()
                            if old_labels[k] != new_labels[k]}

            if added_labels:
                details.append(f"Labels added: {', '.join(added_labels)}")
            if removed_labels:
                details.append(f"Labels removed: {', '.join(removed_labels)}")
            if changed_labels:
                for label in changed_labels:
                    details.append(f"Label '{label}' changed: {old_labels[label]} → {new_labels[label]}")

        # 比较组
        if old_rule.get('group') != new_rule.get('group'):
            details.append(f"Moved from group '{old_rule.get('group')}' to '{new_rule.get('group')}'")

        return details

    def _compare_alert_details(self, old_alert: Dict, new_alert: Dict) -> List[str]:
        """比较告警细节"""
        details = []

        # 基础比较（继承自规则比较）
        details.extend(self._compare_rule_details(old_alert, new_alert))

        # 比较for duration
        if old_alert.get('for') != new_alert.get('for'):
            details.append(f"Duration changed: {old_alert.get('for')} → {new_alert.get('for')}")

        # 比较annotations
        old_annotations = old_alert.get('annotations', {})
        new_annotations = new_alert.get('annotations', {})

        if old_annotations != new_annotations:
            added_annotations = set(new_annotations.keys()) - set(old_annotations.keys())
            removed_annotations = set(old_annotations.keys()) - set(new_annotations.keys())
            changed_annotations = {k for k in old_annotations.keys() & new_annotations.keys()
                                 if old_annotations[k] != new_annotations[k]}

            if added_annotations:
                details.append(f"Annotations added: {', '.join(added_annotations)}")
            if removed_annotations:
                details.append(f"Annotations removed: {', '.join(removed_annotations)}")
            if changed_annotations:
                for annotation in changed_annotations:
                    details.append(f"Annotation '{annotation}' changed")

        return details

    def _identify_breaking_changes(self):
        """识别破坏性变更"""
        for change in self.changes:
            is_breaking = False

            # 删除规则是破坏性的
            if change.change_type == ChangeType.REMOVED:
                is_breaking = True

            # 修改表达式是潜在破坏性的
            if change.change_type == ChangeType.MODIFIED:
                if change.old_value and change.new_value:
                    if change.old_value.get('expr') != change.new_value.get('expr'):
                        # 检查是否只是格式化变化
                        old_expr = self._normalize_expr(change.old_value.get('expr', ''))
                        new_expr = self._normalize_expr(change.new_value.get('expr', ''))

                        if old_expr != new_expr:
                            is_breaking = True

            # 告警severity变更是破坏性的
            if change.rule_type == "alert" and change.change_type == ChangeType.MODIFIED:
                old_severity = change.old_value.get('labels', {}).get('severity') if change.old_value else None
                new_severity = change.new_value.get('labels', {}).get('severity') if change.new_value else None

                if old_severity != new_severity:
                    is_breaking = True

            if is_breaking:
                self.breaking_changes.append(change)

    def _normalize_expr(self, expr: str) -> str:
        """标准化表达式（去除空白等）"""
        # 去除多余空白
        expr = re.sub(r'\s+', ' ', expr)
        # 去除括号周围的空白
        expr = re.sub(r'\s*\(\s*', '(', expr)
        expr = re.sub(r'\s*\)\s*', ')', expr)
        # 去除操作符周围的空白
        expr = re.sub(r'\s*([+\-*/=<>!]+)\s*', r'\1', expr)

        return expr.strip()

    def _generate_report(self, old_file: str, new_file: str) -> RuleDiffReport:
        """生成差异报告"""
        changes_by_type = {
            ChangeType.ADDED: 0,
            ChangeType.REMOVED: 0,
            ChangeType.MODIFIED: 0,
            ChangeType.UNCHANGED: 0
        }

        for change in self.changes:
            changes_by_type[change.change_type] += 1

        # 统计未变更的规则
        total_old_rules = (len(self.old_rules.get('recording_rules', {})) +
                          len(self.old_rules.get('alert_rules', {})))
        total_new_rules = (len(self.new_rules.get('recording_rules', {})) +
                          len(self.new_rules.get('alert_rules', {})))

        total_changes = len(self.changes)

        report = RuleDiffReport(
            timestamp=datetime.now(),
            old_file=old_file,
            new_file=new_file,
            total_changes=total_changes,
            changes_by_type=changes_by_type,
            rule_changes=self.changes,
            breaking_changes=self.breaking_changes,
            summary={
                'total_old_rules': total_old_rules,
                'total_new_rules': total_new_rules,
                'recording_rules_changed': sum(1 for c in self.changes if c.rule_type == "recording_rule"),
                'alert_rules_changed': sum(1 for c in self.changes if c.rule_type == "alert"),
                'breaking_changes_count': len(self.breaking_changes),
                'severity_breakdown': self._get_severity_breakdown()
            }
        )

        return report

    def _get_severity_breakdown(self) -> Dict[str, int]:
        """获取严重程度分布"""
        breakdown = {'info': 0, 'warning': 0, 'critical': 0}

        for change in self.changes:
            breakdown[change.severity] += 1

        return breakdown

    def generate_markdown_report(self, report: RuleDiffReport) -> str:
        """生成Markdown格式报告"""
        lines = []
        lines.append("# Prometheus Rules Diff Report")
        lines.append(f"\n**Generated**: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Old File**: `{Path(report.old_file).name}`")
        lines.append(f"**New File**: `{Path(report.new_file).name}`\n")

        # 摘要
        lines.append("## 📊 Summary\n")
        lines.append(f"- **Total Changes**: {report.total_changes}")
        lines.append(f"- **Added**: {report.changes_by_type[ChangeType.ADDED]} ✨")
        lines.append(f"- **Modified**: {report.changes_by_type[ChangeType.MODIFIED]} 📝")
        lines.append(f"- **Removed**: {report.changes_by_type[ChangeType.REMOVED]} 🗑️")
        lines.append(f"- **Breaking Changes**: {report.summary['breaking_changes_count']} ⚠️\n")

        # 破坏性变更
        if report.breaking_changes:
            lines.append("## ⚠️ Breaking Changes\n")
            lines.append("The following changes may break existing functionality:\n")

            for change in report.breaking_changes:
                emoji = {"critical": "🔴", "warning": "🟠", "info": "🟡"}.get(change.severity, "⚪")
                lines.append(f"### {emoji} {change.rule_name} ({change.rule_type})")
                lines.append(f"**Change Type**: {change.change_type.value}\n")

                for detail in change.details:
                    lines.append(f"- {detail}")
                lines.append("")

        # 所有变更详情
        lines.append("## 📋 All Changes\n")

        # 按类型分组
        for change_type in [ChangeType.ADDED, ChangeType.MODIFIED, ChangeType.REMOVED]:
            type_changes = [c for c in report.rule_changes if c.change_type == change_type]

            if type_changes:
                type_emoji = {
                    ChangeType.ADDED: "✨",
                    ChangeType.MODIFIED: "📝",
                    ChangeType.REMOVED: "🗑️"
                }.get(change_type, "")

                lines.append(f"### {type_emoji} {change_type.value.title()}\n")

                for change in type_changes:
                    severity_emoji = {
                        "critical": "🔴",
                        "warning": "🟠",
                        "info": "🟢"
                    }.get(change.severity, "⚪")

                    lines.append(f"#### {severity_emoji} `{change.rule_name}`")
                    lines.append(f"- **Type**: {change.rule_type}")

                    if change.details:
                        lines.append("- **Details**:")
                        for detail in change.details:
                            # 缩进细节
                            if detail.startswith("  "):
                                lines.append(f"  {detail}")
                            else:
                                lines.append(f"  - {detail}")
                    lines.append("")

        # 建议
        lines.append("## 💡 Recommendations\n")

        if report.summary['breaking_changes_count'] > 0:
            lines.append("### ⚠️ Breaking Changes Detected")
            lines.append("1. Review all breaking changes carefully")
            lines.append("2. Update dependent dashboards and alerts")
            lines.append("3. Notify teams about removed metrics/alerts")
            lines.append("4. Test in staging environment first\n")

        if report.changes_by_type[ChangeType.REMOVED] > 0:
            lines.append("### 🗑️ Rules Removed")
            lines.append("1. Verify that removed rules are no longer needed")
            lines.append("2. Check for any dependencies on removed metrics")
            lines.append("3. Archive old rule definitions for reference\n")

        if report.summary['alert_rules_changed'] > 0:
            lines.append("### 🚨 Alert Changes")
            lines.append("1. Update alert routing if severity changed")
            lines.append("2. Review alert thresholds and durations")
            lines.append("3. Update runbooks for modified alerts\n")

        # 验证检查
        lines.append("## ✅ Validation Checklist\n")
        lines.append("- [ ] All expressions are syntactically valid")
        lines.append("- [ ] No duplicate rule names")
        lines.append("- [ ] Labels follow naming conventions")
        lines.append("- [ ] Alert severity levels are appropriate")
        lines.append("- [ ] Recording rules have meaningful names")
        lines.append("- [ ] Annotations contain required fields")
        lines.append("- [ ] Groups are logically organized")

        return "\n".join(lines)

    def generate_json_report(self, report: RuleDiffReport) -> str:
        """生成JSON格式报告"""
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "old_file": report.old_file,
            "new_file": report.new_file,
            "summary": report.summary,
            "changes": [
                {
                    "rule_name": change.rule_name,
                    "rule_type": change.rule_type,
                    "change_type": change.change_type.value,
                    "severity": change.severity,
                    "details": change.details,
                    "old_value": change.old_value,
                    "new_value": change.new_value
                }
                for change in report.rule_changes
            ],
            "breaking_changes": [
                {
                    "rule_name": change.rule_name,
                    "rule_type": change.rule_type,
                    "reason": change.details[0] if change.details else "Unknown"
                }
                for change in report.breaking_changes
            ]
        }

        return json.dumps(report_dict, indent=2, default=str)

    def validate_rules(self, file_path: str) -> List[str]:
        """验证规则文件的有效性"""
        errors = []

        try:
            rules = self.load_rules(file_path)
        except Exception as e:
            errors.append(f"Failed to load rules: {e}")
            return errors

        # 检查recording rules
        for rule_name, rule_data in rules.get('recording_rules', {}).items():
            # 检查命名规范
            if not re.match(r'^[a-z_][a-z0-9_]*(?::[a-z_][a-z0-9_]*)*$', rule_name):
                errors.append(f"Recording rule '{rule_name}' does not follow naming convention")

            # 检查表达式
            if not rule_data.get('expr'):
                errors.append(f"Recording rule '{rule_name}' has empty expression")

        # 检查alert rules
        for alert_name, alert_data in rules.get('alert_rules', {}).items():
            # 检查命名
            if not re.match(r'^[A-Z][A-Za-z0-9]*$', alert_name):
                errors.append(f"Alert '{alert_name}' does not follow naming convention (PascalCase)")

            # 检查必需字段
            if not alert_data.get('expr'):
                errors.append(f"Alert '{alert_name}' has empty expression")

            # 检查severity标签
            severity = alert_data.get('labels', {}).get('severity')
            if severity and severity not in ['critical', 'warning', 'info']:
                errors.append(f"Alert '{alert_name}' has invalid severity: {severity}")

            # 检查annotations
            annotations = alert_data.get('annotations', {})
            if not annotations.get('summary'):
                errors.append(f"Alert '{alert_name}' missing required annotation: summary")

        return errors


def create_sample_rules():
    """创建示例规则文件（用于测试）"""
    old_rules = """
groups:
  - name: example_recording
    interval: 30s
    rules:
      - record: job:http_requests:rate5m
        expr: sum(rate(http_requests_total[5m])) by (job)
        labels:
          team: platform

      - record: instance:memory_usage:percentage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

  - name: example_alerts
    rules:
      - alert: HighErrorRate
        expr: job:http_requests:rate5m{status=~"5.."} > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High error rate detected
          description: "Error rate is {{ $value }}"

      - alert: MemoryPressure
        expr: instance:memory_usage:percentage > 80
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: High memory usage
"""

    new_rules = """
groups:
  - name: example_recording
    interval: 30s
    rules:
      - record: job:http_requests:rate5m
        expr: sum(rate(http_requests_total[5m])) by (job, method)  # Added method label
        labels:
          team: platform
          env: production  # New label

      - record: instance:cpu_usage:percentage  # New metric
        expr: 100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

  - name: example_alerts
    rules:
      - alert: HighErrorRate
        expr: job:http_requests:rate5m{status=~"5.."} > 0.1  # Threshold changed
        for: 2m  # Duration changed
        labels:
          severity: critical  # Severity changed
        annotations:
          summary: High error rate detected
          description: "Error rate is {{ $value }} for job {{ $labels.job }}"
          runbook: https://wiki.example.com/runbooks/high-error-rate  # New annotation

      - alert: DiskSpaceLow  # New alert
        expr: node_filesystem_avail_bytes / node_filesystem_size_bytes < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: Low disk space
"""

    # 保存示例文件
    Path("old_rules.yml").write_text(old_rules)
    Path("new_rules.yml").write_text(new_rules)

    print("✅ Sample rules files created: old_rules.yml, new_rules.yml")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prometheus rules diff tool"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # diff命令
    diff_parser = subparsers.add_parser("diff", help="Compare two rules files")
    diff_parser.add_argument("old_file", help="Old rules file")
    diff_parser.add_argument("new_file", help="New rules file")
    diff_parser.add_argument("--output", "-o", help="Output file path")
    diff_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")

    # validate命令
    validate_parser = subparsers.add_parser("validate", help="Validate rules file")
    validate_parser.add_argument("file", help="Rules file to validate")

    # sample命令
    sample_parser = subparsers.add_parser("sample", help="Create sample rules files")

    args = parser.parse_args()

    if args.command == "diff":
        differ = PrometheusRulesDiff()

        try:
            report = differ.diff_rules(args.old_file, args.new_file)
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1

        # 生成输出
        if args.format == "markdown":
            output = differ.generate_markdown_report(report)
        else:
            output = differ.generate_json_report(report)

        # 保存或打印
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"✅ Report saved to: {args.output}")
        else:
            print(output)

        # 返回退出码
        if report.breaking_changes:
            return 2  # 有破坏性变更
        elif report.total_changes > 0:
            return 1  # 有变更
        else:
            return 0  # 无变更

    elif args.command == "validate":
        differ = PrometheusRulesDiff()
        errors = differ.validate_rules(args.file)

        if errors:
            print("❌ Validation errors found:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print("✅ Rules file is valid")
            return 0

    elif args.command == "sample":
        create_sample_rules()
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())