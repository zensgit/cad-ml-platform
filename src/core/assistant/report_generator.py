"""
Analysis Report Generator.

Runs multiple tools in parallel and formats the results into a
structured Markdown report -- no LLM call required.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from .tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class AnalysisReportGenerator:
    """Generate a full analysis report for a CAD drawing by running all tools."""

    def __init__(self) -> None:
        self._tools = dict(TOOL_REGISTRY)

    async def generate_full_report(self, file_id: str) -> str:
        """Run classify, feature, process, cost, and quality tools in parallel
        and format the combined results as Markdown.

        Args:
            file_id: The file identifier for the CAD drawing.

        Returns:
            A Markdown-formatted report string.
        """
        classify_task = self._tools["classify_part"].execute({"file_id": file_id})
        feature_task = self._tools["extract_features"].execute({"file_id": file_id})
        process_task = self._tools["recommend_process"].execute({"file_id": file_id})
        cost_task = self._tools["estimate_cost"].execute({"file_id": file_id})
        quality_task = self._tools["assess_quality"].execute({"file_id": file_id})

        classify, feature, process, cost, quality = await asyncio.gather(
            classify_task, feature_task, process_task, cost_task, quality_task,
        )

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return self._format_report(file_id, now, classify, feature, process, cost, quality)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_report(
        file_id: str,
        timestamp: str,
        classify: Dict[str, Any],
        feature: Dict[str, Any],
        process: Dict[str, Any],
        cost: Dict[str, Any],
        quality: Dict[str, Any],
    ) -> str:
        lines = [
            f"# CAD 图纸分析报告",
            "",
            f"- **文件ID**: `{file_id}`",
            f"- **生成时间**: {timestamp}",
            "",
            "---",
            "",
            "## 概要",
            "",
            f"零件类型 **{classify.get('label', 'N/A')}**，"
            f"分类置信度 {classify.get('confidence', 0):.0%}。"
            f"特征维度 {feature.get('dimension', 'N/A')} (版本 {feature.get('version', 'N/A')})。"
            f"预估单件成本 {cost.get('total', 'N/A')} {cost.get('currency', 'CNY')}。",
            "",
            "---",
            "",
            "## 分类结果",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 类别 | {classify.get('label', 'N/A')} |",
            f"| 置信度 | {classify.get('confidence', 0):.2%} |",
        ]

        contributions = classify.get("source_contributions", {})
        if contributions:
            lines.append(f"| 来源贡献 | {contributions} |")

        lines.extend([
            "",
            "---",
            "",
            "## 几何特征",
            "",
            f"- **特征维度**: {feature.get('dimension', 'N/A')}",
            f"- **版本**: {feature.get('version', 'N/A')}",
        ])
        summary = feature.get("summary", {})
        if summary:
            lines.append(f"- **实体数量**: {summary.get('entity_count', 'N/A')}")
            lines.append(f"- **复杂度**: {summary.get('complexity', 'N/A')}")

        lines.extend([
            "",
            "---",
            "",
            "## 推荐工艺",
            "",
            f"- **主要工艺**: {process.get('primary_process', 'N/A')}",
        ])
        alternatives = process.get("alternatives", [])
        if alternatives:
            lines.append(f"- **备选方案**: {', '.join(alternatives)}")
        reasoning = process.get("reasoning", "")
        if reasoning:
            lines.append(f"- **推荐理由**: {reasoning}")

        lines.extend([
            "",
            "---",
            "",
            "## 成本估算",
            "",
            f"| 项目 | 金额 ({cost.get('currency', 'CNY')}) |",
            f"|------|------|",
            f"| 材料费 | {cost.get('material_cost', 'N/A')} |",
            f"| 加工费 | {cost.get('machining_cost', 'N/A')} |",
            f"| 装夹费 | {cost.get('setup_cost', 'N/A')} |",
            f"| 管理费 | {cost.get('overhead', 'N/A')} |",
            f"| **合计** | **{cost.get('total', 'N/A')}** |",
        ])
        route = cost.get("process_route", [])
        if route:
            lines.append(f"\n工艺路线: {' -> '.join(route)}")

        lines.extend([
            "",
            "---",
            "",
            "## 质量评估",
            "",
            f"- **综合评分**: {quality.get('overall_score', 'N/A')}",
        ])
        issues = quality.get("issues", [])
        if issues:
            lines.append("\n### 发现问题")
            for issue in issues:
                lines.append(f"- {issue}")

        suggestions = quality.get("suggestions", [])
        if suggestions:
            lines.append("\n### 改进建议")
            for s in suggestions:
                lines.append(f"- {s}")

        lines.extend([
            "",
            "---",
            "",
            "## 改进建议",
            "",
        ])
        # Aggregate suggestions from quality and process
        all_suggestions = list(suggestions)
        if reasoning:
            all_suggestions.append(f"工艺建议: {reasoning}")
        if not all_suggestions:
            all_suggestions.append("暂无特别建议，图纸质量良好。")
        for idx, s in enumerate(all_suggestions, 1):
            lines.append(f"{idx}. {s}")

        lines.append("")
        return "\n".join(lines)
