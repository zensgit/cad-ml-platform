"""
QualityTool -- assess the quality of a CAD drawing.
"""

import logging
from typing import Any, Dict, List

from .base import BaseTool

logger = logging.getLogger(__name__)


class QualityTool(BaseTool):
    """Assess the quality of a CAD drawing (annotations, dimensions, layers)."""

    name = "assess_quality"
    description = "评估图纸质量（标注完整性、尺寸一致性、图层规范性等）"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "图纸文件ID",
            },
        },
        "required": ["file_id"],
    }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        file_id = params["file_id"]
        logger.info("assess_quality called: file_id=%s", file_id)

        try:
            from src.core.assistant.quality_evaluation import ResponseQualityEvaluator

            evaluator = ResponseQualityEvaluator()
            result = evaluator.evaluate_drawing(file_id)
            return {
                "overall_score": round(result.overall_score, 2),
                "issues": result.issues,
                "suggestions": result.suggestions,
            }
        except Exception as exc:
            logger.warning("assess_quality fallback for %s: %s", file_id, exc)
            issues: List[str] = [
                "无法连接图纸质量评估服务，以下为通用检查项",
                "请确认标注完整性（尺寸标注、公差标注、表面粗糙度）",
                "请确认图层规范性（0图层应为空）",
            ]
            suggestions: List[str] = [
                "建议检查是否有缺失的关键尺寸标注",
                "建议统一图层命名规范",
                "建议添加标题栏和技术要求",
            ]
            return {
                "overall_score": 0.0,
                "issues": issues,
                "suggestions": suggestions,
                "note": f"质量评估服务暂不可用，返回通用建议。原因: {exc}",
            }
