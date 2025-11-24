#!/usr/bin/env python3
"""
Enhanced Error Codes Lifecycle Management
增强的错误码生命周期管理 - 自动标注和淘汰候选
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import subprocess

# 引入原有的审计工具
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.error_codes_audit import ErrorCodesAuditor, ErrorCodeLifecycle


@dataclass
class DeprecationCandidate:
    """弃用候选信息"""
    code: str
    reason: str
    last_used: Optional[datetime]
    usage_count_7d: int
    usage_count_14d: int
    usage_count_30d: int
    recommendation: str
    priority: str  # high, medium, low


class EnhancedErrorCodesManager:
    """增强的错误码管理器"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.auditor = ErrorCodesAuditor(project_root)
        self.governance_dir = self.project_root / "reports" / "governance"
        self.governance_dir.mkdir(parents=True, exist_ok=True)

        # 生命周期配置
        self.lifecycle_config = {
            "active_threshold_7d": 5,      # 7天内使用>5次为活跃
            "candidate_threshold_14d": 5,   # 14天内使用<5次为候选
            "deprecated_threshold_21d": 0,  # 21天内使用=0为弃用
            "removal_threshold_30d": 0,     # 30天后自动移除
            "max_active_codes": 80,         # 最大活跃错误码数
            "max_total_codes": 100          # 最大总错误码数
        }

        self.deprecation_candidates: List[DeprecationCandidate] = []

    def analyze_lifecycle(self) -> Dict[str, Any]:
        """分析错误码生命周期"""
        print("🔍 Analyzing error codes lifecycle...")

        # 运行基础审计
        self.auditor.load_history()
        defined_codes = self.auditor.extract_defined_codes()
        usage_stats = self.auditor.scan_usage()

        # 增强的生命周期分析
        lifecycle_analysis = self._enhanced_lifecycle_analysis(
            defined_codes,
            usage_stats
        )

        # 识别弃用候选
        self._identify_deprecation_candidates(lifecycle_analysis)

        # 检查是否超限
        violations = self._check_limit_violations(lifecycle_analysis)

        # 生成治理报告
        report = self._generate_governance_report(
            lifecycle_analysis,
            violations
        )

        return report

    def _enhanced_lifecycle_analysis(
        self,
        defined_codes: Dict[str, Any],
        usage_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """增强的生命周期分析"""
        current_time = datetime.now()
        analysis = {
            "timestamp": current_time.isoformat(),
            "codes": {},
            "statistics": {
                "total": len(defined_codes),
                "active": 0,
                "candidate": 0,
                "deprecated": 0,
                "unused": 0,
                "legacy": 0
            }
        }

        # 加载历史数据计算不同时间窗口的使用情况
        history = self.auditor.history.get("usage_trends", {})

        for code, info in defined_codes.items():
            usage = usage_stats.get(code)

            # 计算不同时间窗口的使用次数
            usage_7d = self._calculate_usage_window(history, code, 7)
            usage_14d = self._calculate_usage_window(history, code, 14)
            usage_21d = self._calculate_usage_window(history, code, 21)
            usage_30d = self._calculate_usage_window(history, code, 30)

            # 确定生命周期状态
            if usage_7d >= self.lifecycle_config["active_threshold_7d"]:
                lifecycle = ErrorCodeLifecycle.ACTIVE
            elif usage_14d < self.lifecycle_config["candidate_threshold_14d"]:
                if usage_21d == 0:
                    lifecycle = ErrorCodeLifecycle.DEPRECATED
                else:
                    lifecycle = ErrorCodeLifecycle.CANDIDATE
            elif not usage or usage.count == 0:
                lifecycle = ErrorCodeLifecycle.UNUSED
            else:
                lifecycle = ErrorCodeLifecycle.ACTIVE

            # 记录分析结果
            analysis["codes"][code] = {
                "lifecycle": lifecycle.value,
                "usage_7d": usage_7d,
                "usage_14d": usage_14d,
                "usage_21d": usage_21d,
                "usage_30d": usage_30d,
                "last_seen": usage.last_seen.isoformat() if usage and usage.last_seen else None,
                "locations": usage.locations[:3] if usage else [],
                "severity": info.severity if hasattr(info, 'severity') else "",
                "source": info.source if hasattr(info, 'source') else ""
            }

            # 更新统计
            analysis["statistics"][lifecycle.value.lower()] += 1

        return analysis

    def _calculate_usage_window(
        self,
        history: Dict[str, List[Dict]],
        code: str,
        days: int
    ) -> int:
        """计算指定时间窗口内的使用次数"""
        if code not in history:
            return 0

        cutoff_date = datetime.now() - timedelta(days=days)
        total_usage = 0

        for entry in history[code]:
            if "timestamp" in entry:
                try:
                    timestamp = datetime.fromisoformat(entry["timestamp"])
                    if timestamp >= cutoff_date:
                        total_usage += entry.get("count", 0)
                except:
                    pass

        return total_usage

    def _identify_deprecation_candidates(self, analysis: Dict[str, Any]):
        """识别弃用候选"""
        self.deprecation_candidates.clear()

        for code, info in analysis["codes"].items():
            if info["lifecycle"] in ["candidate", "deprecated", "unused"]:
                # 确定优先级
                if info["lifecycle"] == "deprecated":
                    priority = "high"
                    reason = "No usage in 21+ days"
                    recommendation = "Remove immediately"
                elif info["lifecycle"] == "unused":
                    priority = "high"
                    reason = "Never used since definition"
                    recommendation = "Remove in next release"
                else:  # candidate
                    priority = "medium"
                    reason = f"Low usage ({info['usage_14d']} times in 14d)"
                    recommendation = "Monitor for 1 more week"

                candidate = DeprecationCandidate(
                    code=code,
                    reason=reason,
                    last_used=datetime.fromisoformat(info["last_seen"]) if info["last_seen"] else None,
                    usage_count_7d=info["usage_7d"],
                    usage_count_14d=info["usage_14d"],
                    usage_count_30d=info["usage_30d"],
                    recommendation=recommendation,
                    priority=priority
                )
                self.deprecation_candidates.append(candidate)

    def _check_limit_violations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查是否超出限制"""
        violations = []
        stats = analysis["statistics"]

        # 检查活跃错误码数量
        if stats["active"] > self.lifecycle_config["max_active_codes"]:
            violations.append({
                "type": "active_limit_exceeded",
                "severity": "warning",
                "current": stats["active"],
                "limit": self.lifecycle_config["max_active_codes"],
                "message": f"Active error codes ({stats['active']}) exceed limit ({self.lifecycle_config['max_active_codes']})",
                "action": "Review and consolidate error codes"
            })

        # 检查总错误码数量
        if stats["total"] > self.lifecycle_config["max_total_codes"]:
            violations.append({
                "type": "total_limit_exceeded",
                "severity": "error",
                "current": stats["total"],
                "limit": self.lifecycle_config["max_total_codes"],
                "message": f"Total error codes ({stats['total']}) exceed limit ({self.lifecycle_config['max_total_codes']})",
                "action": "Immediate cleanup required"
            })

        # 检查弃用候选比例
        deprecation_ratio = (stats["candidate"] + stats["deprecated"] + stats["unused"]) / stats["total"] if stats["total"] > 0 else 0
        if deprecation_ratio > 0.3:  # 30%以上为弃用候选
            violations.append({
                "type": "high_deprecation_ratio",
                "severity": "warning",
                "ratio": deprecation_ratio,
                "message": f"{deprecation_ratio:.1%} of codes are deprecation candidates",
                "action": "Execute cleanup to reduce technical debt"
            })

        return violations

    def _generate_governance_report(
        self,
        analysis: Dict[str, Any],
        violations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成治理报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_codes": analysis["statistics"]["total"],
                "lifecycle_distribution": analysis["statistics"],
                "violations": len(violations),
                "deprecation_candidates": len(self.deprecation_candidates)
            },
            "violations": violations,
            "deprecation_candidates": [
                asdict(c) for c in sorted(
                    self.deprecation_candidates,
                    key=lambda x: (x.priority, x.usage_count_30d)
                )
            ],
            "lifecycle_details": analysis["codes"],
            "recommendations": self._generate_recommendations(analysis, violations)
        }

        # 保存报告
        report_path = self.governance_dir / f"error_codes_status_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📁 Report saved to: {report_path}")

        return report

    def _generate_recommendations(
        self,
        analysis: Dict[str, Any],
        violations: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """生成建议"""
        recommendations = []

        # 基于违规情况的建议
        if any(v["type"] == "total_limit_exceeded" for v in violations):
            recommendations.append({
                "priority": "critical",
                "action": "Execute immediate cleanup",
                "description": "Total error codes exceed limit. Remove all deprecated and unused codes immediately."
            })

        # 基于弃用候选的建议
        high_priority_candidates = [c for c in self.deprecation_candidates if c.priority == "high"]
        if len(high_priority_candidates) > 5:
            recommendations.append({
                "priority": "high",
                "action": f"Remove {len(high_priority_candidates)} unused error codes",
                "description": "Multiple error codes have been unused for extended periods."
            })

        # 基于使用模式的建议
        stats = analysis["statistics"]
        if stats["active"] < stats["total"] * 0.5:
            recommendations.append({
                "priority": "medium",
                "action": "Consolidate error codes",
                "description": "Less than 50% of defined error codes are actively used."
            })

        return recommendations

    def auto_deprecate(self, dry_run: bool = True) -> List[str]:
        """自动执行弃用流程"""
        deprecated_codes = []

        for candidate in self.deprecation_candidates:
            if candidate.priority == "high" and candidate.usage_count_30d == 0:
                deprecated_codes.append(candidate.code)

                if not dry_run:
                    # 实际执行弃用（标记或删除）
                    self._mark_as_deprecated(candidate.code)

        if dry_run:
            print(f"🔍 Dry run: Would deprecate {len(deprecated_codes)} codes")
        else:
            print(f"✅ Deprecated {len(deprecated_codes)} codes")

        return deprecated_codes

    def _mark_as_deprecated(self, code: str):
        """标记错误码为弃用"""
        # 这里实现实际的代码修改逻辑
        # 例如：添加 @deprecated 装饰器或注释
        pass

    def generate_markdown_report(self) -> str:
        """生成 Markdown 格式报告"""
        report = self.analyze_lifecycle()

        lines = []
        lines.append("# Error Codes Lifecycle Governance Report")
        lines.append(f"\n**Generated**: {report['timestamp']}\n")

        # 摘要
        summary = report["summary"]
        lines.append("## 📊 Summary\n")
        lines.append(f"- **Total Codes**: {summary['total_codes']}")
        dist = summary["lifecycle_distribution"]
        lines.append(f"- **Active**: {dist['active']} | **Candidate**: {dist['candidate']} | **Deprecated**: {dist['deprecated']} | **Unused**: {dist['unused']}")
        lines.append(f"- **Violations**: {summary['violations']}")
        lines.append(f"- **Deprecation Candidates**: {summary['deprecation_candidates']}\n")

        # 违规情况
        if report["violations"]:
            lines.append("## ⚠️ Violations\n")
            for violation in report["violations"]:
                emoji = "🔴" if violation["severity"] == "error" else "🟡"
                lines.append(f"{emoji} **{violation['type']}**")
                lines.append(f"   - {violation['message']}")
                lines.append(f"   - Action: {violation['action']}\n")

        # 弃用候选
        if report["deprecation_candidates"]:
            lines.append("## 📝 Deprecation Candidates\n")
            lines.append("| Priority | Code | Reason | Last Used | 7d/14d/30d Usage | Action |")
            lines.append("|----------|------|--------|-----------|------------------|--------|")

            for candidate in report["deprecation_candidates"][:10]:
                last_used = candidate["last_used"][:10] if candidate["last_used"] else "Never"
                usage = f"{candidate['usage_count_7d']}/{candidate['usage_count_14d']}/{candidate['usage_count_30d']}"
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(candidate["priority"], "⚪")
                lines.append(f"| {priority_emoji} {candidate['priority']} | {candidate['code']} | {candidate['reason']} | {last_used} | {usage} | {candidate['recommendation']} |")

        # 建议
        if report["recommendations"]:
            lines.append("\n## 💡 Recommendations\n")
            for rec in report["recommendations"]:
                priority_emoji = {"critical": "🔴", "high": "🟡", "medium": "🟢"}.get(rec["priority"], "⚪")
                lines.append(f"- {priority_emoji} **{rec['action']}**: {rec['description']}")

        return "\n".join(lines)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced error codes lifecycle management"
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
        "--auto-deprecate",
        action="store_true",
        help="Automatically deprecate unused codes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run for auto-deprecation"
    )

    args = parser.parse_args()

    # 创建管理器
    manager = EnhancedErrorCodesManager()

    # 分析生命周期
    if args.auto_deprecate:
        deprecated = manager.auto_deprecate(dry_run=args.dry_run)
        print(f"Deprecated codes: {deprecated}")

    # 生成报告
    if args.format == "markdown":
        report = manager.generate_markdown_report()
    else:
        report_dict = manager.analyze_lifecycle()
        report = json.dumps(report_dict, indent=2, default=str)

    # 输出结果
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"✅ Report saved to: {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())