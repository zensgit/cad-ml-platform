"""
QualityTool -- assess the quality of a CAD drawing.
"""

import logging
from typing import Any, Dict, List

from .base import BaseTool
from src.core.assistant.tool_status import (
    STATUS_FAILED,
    STATUS_UNAVAILABLE,
    failure_result,
)

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
            logger.warning("tool fallback: %s", type(exc).__name__)
            return failure_result(
                STATUS_UNAVAILABLE,
                "quality_service_unavailable",
                file_id=file_id,
            )
