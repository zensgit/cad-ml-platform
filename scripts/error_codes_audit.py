#!/usr/bin/env python3
"""
Error Codes Audit Tool
错误码使用频率审计工具 - 统计分析错误码使用情况并标注生命周期状态

功能：
1. 统计错误码使用频率
2. 识别未使用的错误码
3. 标注 deprecated 候选
4. 生成生命周期报告
5. 提供优化建议
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import re
import ast
from collections import defaultdict, Counter


class ErrorCodeLifecycle(Enum):
    """错误码生命周期状态"""
    ACTIVE = "active"           # 活跃使用
    DEPRECATED = "deprecated"    # 已弃用
    CANDIDATE = "candidate"      # 弃用候选
    UNUSED = "unused"           # 从未使用
    LEGACY = "legacy"           # 遗留代码


@dataclass
class ErrorCodeUsage:
    """错误码使用信息"""
    code: str
    count: int = 0
    last_seen: Optional[datetime] = None
    first_seen: Optional[datetime] = None
    locations: List[str] = field(default_factory=list)
    severity: str = ""
    source: str = ""
    lifecycle: ErrorCodeLifecycle = ErrorCodeLifecycle.ACTIVE
    recommendation: str = ""


class ErrorCodesAuditor:
    """错误码审计器"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.test_dir = self.project_root / "tests"
        self.logs_dir = self.project_root / "logs"

        # 错误码定义文件
        self.error_codes_file = self.src_dir / "core" / "errors_extended.py"
        self.usage_history_file = self.project_root / ".error_codes_history.json"

        self.defined_codes: Dict[str, ErrorCodeUsage] = {}
        self.usage_stats: Dict[str, ErrorCodeUsage] = {}
        self.history: Dict[str, Any] = {}

    def load_history(self):
        """加载历史使用记录"""
        if self.usage_history_file.exists():
            with open(self.usage_history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                "audits": [],
                "usage_trends": {}
            }

    def save_history(self):
        """保存历史记录"""
        with open(self.usage_history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def extract_defined_codes(self) -> Dict[str, ErrorCodeUsage]:
        """提取所有定义的错误码"""
        defined = {}

        if not self.error_codes_file.exists():
            print(f"Warning: Error codes file not found: {self.error_codes_file}")
            return defined

        with open(self.error_codes_file, 'r') as f:
            content = f.read()

        # 解析 ErrorCode 枚举
        error_code_pattern = re.compile(r'([A-Z_]+)\s*=\s*["\']([^"\']+)["\']')
        matches = error_code_pattern.findall(content)

        for name, value in matches:
            if not name.startswith('_'):  # 跳过私有属性
                defined[value] = ErrorCodeUsage(
                    code=value,
                    lifecycle=ErrorCodeLifecycle.UNUSED
                )

        # 提取源和严重性映射
        self._extract_metadata(content, defined)

        return defined

    def _extract_metadata(self, content: str, codes: Dict[str, ErrorCodeUsage]):
        """提取错误码元数据（源、严重性）"""
        # 解析 ERROR_SOURCE_MAP
        source_map_match = re.search(
            r'ERROR_SOURCE_MAP\s*=\s*\{([^}]+)\}',
            content,
            re.DOTALL
        )
        if source_map_match:
            source_map_str = "{" + source_map_match.group(1) + "}"
            try:
                # 简单的字符串替换来评估
                source_map_str = re.sub(r'ErrorCode\.(\w+)', r'"\1"', source_map_str)
                source_map_str = re.sub(r'ErrorSource\.(\w+)', r'"\1"', source_map_str)
                # 安全评估
                source_map = ast.literal_eval(source_map_str)
                for code, source in source_map.items():
                    if code in codes:
                        codes[code].source = source
            except:
                pass

        # 解析 ERROR_SEVERITY_MAP
        severity_map_match = re.search(
            r'ERROR_SEVERITY_MAP\s*=\s*\{([^}]+)\}',
            content,
            re.DOTALL
        )
        if severity_map_match:
            severity_map_str = "{" + severity_map_match.group(1) + "}"
            try:
                severity_map_str = re.sub(r'ErrorCode\.(\w+)', r'"\1"', severity_map_str)
                severity_map_str = re.sub(r'ErrorSeverity\.(\w+)', r'"\1"', severity_map_str)
                severity_map = ast.literal_eval(severity_map_str)
                for code, severity in severity_map.items():
                    if code in codes:
                        codes[code].severity = severity
            except:
                pass

    def scan_usage(self) -> Dict[str, ErrorCodeUsage]:
        """扫描代码库中的错误码使用"""
        usage = defaultdict(lambda: ErrorCodeUsage(code=""))

        # 扫描源代码和测试
        for directory in [self.src_dir, self.test_dir]:
            if not directory.exists():
                continue

            for py_file in directory.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 查找错误码使用
                    # 模式1: ErrorCode.XXX
                    pattern1 = re.compile(r'ErrorCode\.([A-Z_]+)')
                    # 模式2: "error_code": "xxx"
                    pattern2 = re.compile(r'["\']error_code["\']\s*:\s*["\']([^"\']+)["\']')
                    # 模式3: error_code="xxx"
                    pattern3 = re.compile(r'error_code\s*=\s*["\']([^"\']+)["\']')

                    matches = []
                    matches.extend(pattern1.findall(content))
                    matches.extend(pattern2.findall(content))
                    matches.extend(pattern3.findall(content))

                    for match in matches:
                        code = match
                        if code not in usage:
                            usage[code] = ErrorCodeUsage(code=code)

                        usage[code].count += 1
                        relative_path = str(py_file.relative_to(self.project_root))
                        if relative_path not in usage[code].locations:
                            usage[code].locations.append(relative_path)

                        usage[code].last_seen = datetime.now()
                        if usage[code].first_seen is None:
                            usage[code].first_seen = datetime.now()

                except Exception as e:
                    print(f"Error scanning {py_file}: {e}")

        # 扫描日志文件（如果存在）
        if self.logs_dir.exists():
            self._scan_logs(usage)

        return dict(usage)

    def _scan_logs(self, usage: Dict[str, ErrorCodeUsage]):
        """扫描日志文件中的错误码"""
        for log_file in self.logs_dir.glob("*.log"):
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # 查找日志中的错误码
                pattern = re.compile(r'error_code["\']?\s*:\s*["\']?([A-Z_]+)')
                matches = pattern.findall(content)

                for code in matches:
                    if code not in usage:
                        usage[code] = ErrorCodeUsage(code=code)
                    usage[code].count += 1

            except Exception as e:
                print(f"Error scanning log {log_file}: {e}")

    def classify_lifecycle(self):
        """分类错误码生命周期状态"""
        current_time = datetime.now()

        for code, info in self.defined_codes.items():
            usage = self.usage_stats.get(code)

            if not usage or usage.count == 0:
                # 从未使用
                info.lifecycle = ErrorCodeLifecycle.UNUSED
                info.recommendation = "Consider removing this unused error code"

            elif usage.count < 5:
                # 很少使用
                if usage.last_seen and (current_time - usage.last_seen).days > 30:
                    info.lifecycle = ErrorCodeLifecycle.CANDIDATE
                    info.recommendation = "Low usage, candidate for deprecation"
                else:
                    info.lifecycle = ErrorCodeLifecycle.ACTIVE
                    info.recommendation = "Low usage but recently active"

            elif usage.last_seen and (current_time - usage.last_seen).days > 60:
                # 长期未使用
                info.lifecycle = ErrorCodeLifecycle.CANDIDATE
                info.recommendation = "Not used in 60+ days, consider deprecation"

            else:
                # 活跃使用
                info.lifecycle = ErrorCodeLifecycle.ACTIVE
                info.recommendation = "Actively used"

            # 更新使用信息
            if usage:
                info.count = usage.count
                info.locations = usage.locations[:5]  # 保留前5个位置
                info.last_seen = usage.last_seen
                info.first_seen = usage.first_seen

    def calculate_metrics(self) -> Dict[str, Any]:
        """计算审计指标"""
        total = len(self.defined_codes)
        if total == 0:
            return {}

        lifecycle_counts = Counter(
            code.lifecycle for code in self.defined_codes.values()
        )

        # 计算使用率
        used_codes = sum(
            1 for code in self.defined_codes.values()
            if code.lifecycle != ErrorCodeLifecycle.UNUSED
        )

        # 计算集中度（Top 3 错误码占比）
        sorted_codes = sorted(
            [c for c in self.defined_codes.values() if c.count > 0],
            key=lambda x: x.count,
            reverse=True
        )

        top3_count = sum(c.count for c in sorted_codes[:3])
        total_count = sum(c.count for c in sorted_codes)
        concentration = top3_count / total_count if total_count > 0 else 0

        return {
            "total_defined": total,
            "total_used": used_codes,
            "usage_rate": used_codes / total,
            "lifecycle_distribution": dict(lifecycle_counts),
            "concentration_top3": concentration,
            "unused_count": lifecycle_counts[ErrorCodeLifecycle.UNUSED],
            "candidate_count": lifecycle_counts[ErrorCodeLifecycle.CANDIDATE],
            "active_count": lifecycle_counts[ErrorCodeLifecycle.ACTIVE]
        }

    def generate_recommendations(self) -> List[Dict[str, str]]:
        """生成优化建议"""
        recommendations = []
        metrics = self.calculate_metrics()

        # 未使用错误码过多
        if metrics.get("unused_count", 0) > 10:
            recommendations.append({
                "severity": "high",
                "category": "cleanup",
                "message": f"Found {metrics['unused_count']} unused error codes. Consider removing them.",
                "action": "Review and remove unused error codes"
            })

        # 集中度过高
        if metrics.get("concentration_top3", 0) > 0.7:
            recommendations.append({
                "severity": "medium",
                "category": "distribution",
                "message": f"Top 3 error codes account for {metrics['concentration_top3']:.1%} of usage",
                "action": "Review if errors are properly categorized"
            })

        # 弃用候选
        if metrics.get("candidate_count", 0) > 5:
            recommendations.append({
                "severity": "medium",
                "category": "lifecycle",
                "message": f"Found {metrics['candidate_count']} deprecation candidates",
                "action": "Mark these codes as deprecated in next release"
            })

        # 使用率低
        if metrics.get("usage_rate", 1) < 0.5:
            recommendations.append({
                "severity": "high",
                "category": "efficiency",
                "message": f"Only {metrics['usage_rate']:.1%} of defined codes are used",
                "action": "Consolidate error codes or improve error handling"
            })

        return recommendations

    def generate_report(self, output_format: str = "json") -> str:
        """生成审计报告"""
        metrics = self.calculate_metrics()
        recommendations = self.generate_recommendations()

        # 按生命周期分组
        grouped = defaultdict(list)
        for code, info in self.defined_codes.items():
            grouped[info.lifecycle.value].append({
                "code": code,
                "count": info.count,
                "severity": info.severity,
                "source": info.source,
                "last_seen": info.last_seen.isoformat() if info.last_seen else None,
                "locations": info.locations[:3],  # 前3个位置
                "recommendation": info.recommendation
            })

        # 排序
        for lifecycle in grouped:
            grouped[lifecycle].sort(key=lambda x: x["count"], reverse=True)

        report = {
            "audit_time": datetime.now().isoformat(),
            "metrics": metrics,
            "lifecycle_groups": dict(grouped),
            "recommendations": recommendations,
            "top_used": [
                {
                    "code": code.code,
                    "count": code.count,
                    "locations": code.locations[:3]
                }
                for code in sorted(
                    [c for c in self.defined_codes.values() if c.count > 0],
                    key=lambda x: x.count,
                    reverse=True
                )[:10]
            ]
        }

        # 更新历史
        self.history["audits"].append({
            "timestamp": report["audit_time"],
            "metrics": metrics
        })

        # 跟踪趋势
        for code, info in self.defined_codes.items():
            if code not in self.history["usage_trends"]:
                self.history["usage_trends"][code] = []
            self.history["usage_trends"][code].append({
                "timestamp": datetime.now().isoformat(),
                "count": info.count,
                "lifecycle": info.lifecycle.value
            })

        self.save_history()

        if output_format == "json":
            return json.dumps(report, indent=2)
        else:
            return self._format_markdown(report)

    def _format_markdown(self, report: Dict[str, Any]) -> str:
        """格式化为 Markdown 报告"""
        lines = []
        lines.append("# Error Codes Audit Report")
        lines.append(f"\n**Generated**: {report['audit_time']}\n")

        # 指标摘要
        metrics = report["metrics"]
        lines.append("## 📊 Summary Metrics\n")
        lines.append(f"- **Total Defined**: {metrics.get('total_defined', 0)}")
        lines.append(f"- **Total Used**: {metrics.get('total_used', 0)}")
        lines.append(f"- **Usage Rate**: {metrics.get('usage_rate', 0):.1%}")
        lines.append(f"- **Top 3 Concentration**: {metrics.get('concentration_top3', 0):.1%}")
        lines.append("")

        # 生命周期分布
        lines.append("## 🔄 Lifecycle Distribution\n")
        for lifecycle, count in metrics.get("lifecycle_distribution", {}).items():
            emoji = {
                "active": "✅",
                "unused": "❌",
                "candidate": "⚠️",
                "deprecated": "🚫",
                "legacy": "📦"
            }.get(lifecycle, "❓")
            lines.append(f"- {emoji} **{lifecycle.upper()}**: {count} codes")
        lines.append("")

        # Top 使用的错误码
        if report.get("top_used"):
            lines.append("## 🔝 Top Used Error Codes\n")
            lines.append("| Code | Count | Locations |")
            lines.append("|------|-------|-----------|")
            for item in report["top_used"][:5]:
                locations = ", ".join(item["locations"][:2])
                lines.append(f"| {item['code']} | {item['count']} | {locations} |")
            lines.append("")

        # 未使用的错误码
        unused = report["lifecycle_groups"].get("unused", [])
        if unused:
            lines.append("## ❌ Unused Error Codes\n")
            lines.append("The following codes are defined but never used:")
            for item in unused[:10]:
                lines.append(f"- `{item['code']}` - {item['recommendation']}")
            if len(unused) > 10:
                lines.append(f"- ... and {len(unused) - 10} more")
            lines.append("")

        # 弃用候选
        candidates = report["lifecycle_groups"].get("candidate", [])
        if candidates:
            lines.append("## ⚠️ Deprecation Candidates\n")
            lines.append("| Code | Last Usage | Count | Recommendation |")
            lines.append("|------|------------|-------|----------------|")
            for item in candidates[:10]:
                last_seen = item['last_seen'][:10] if item['last_seen'] else "Never"
                lines.append(f"| {item['code']} | {last_seen} | {item['count']} | {item['recommendation']} |")
            lines.append("")

        # 建议
        if report.get("recommendations"):
            lines.append("## 💡 Recommendations\n")
            for rec in report["recommendations"]:
                severity_emoji = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(rec["severity"], "⚪")
                lines.append(f"### {severity_emoji} {rec['category'].upper()}")
                lines.append(f"- **Issue**: {rec['message']}")
                lines.append(f"- **Action**: {rec['action']}\n")

        return "\n".join(lines)

    def run_audit(self, output_format: str = "json") -> str:
        """运行完整审计"""
        print("🔍 Starting error codes audit...")

        # 加载历史
        self.load_history()

        # 提取定义的错误码
        print("📋 Extracting defined error codes...")
        self.defined_codes = self.extract_defined_codes()
        print(f"   Found {len(self.defined_codes)} defined codes")

        # 扫描使用情况
        print("🔎 Scanning code usage...")
        self.usage_stats = self.scan_usage()
        print(f"   Found {len(self.usage_stats)} codes in use")

        # 分类生命周期
        print("🏷️ Classifying lifecycle states...")
        self.classify_lifecycle()

        # 生成报告
        print("📊 Generating report...")
        report = self.generate_report(output_format)

        print("✅ Audit complete!")
        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit error codes usage and lifecycle"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path",
        default=None
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "markdown"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory"
    )

    args = parser.parse_args()

    # 运行审计
    auditor = ErrorCodesAuditor(args.project_root)
    report = auditor.run_audit(args.format)

    # 输出结果
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print("\n" + report)

    return 0


if __name__ == "__main__":
    sys.exit(main())