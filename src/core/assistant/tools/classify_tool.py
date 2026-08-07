"""
ClassifyTool -- CAD part classification via the hybrid classifier.
"""

import logging
from typing import Any, Dict

from .base import BaseTool
from src.core.assistant.tool_status import (
    STATUS_FAILED,
    STATUS_UNAVAILABLE,
    failure_result,
)

logger = logging.getLogger(__name__)


class ClassifyTool(BaseTool):
    """Classify a CAD drawing into one of the standard part families."""

    name = "classify_part"
    description = "对 CAD 图纸进行零件分类，识别零件类型（法兰盘、轴、壳体、支架等8类）"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "待分类图纸的文件ID",
            },
            "use_hybrid": {
                "type": "boolean",
                "description": "是否使用混合分类器（结合文件名、Graph2D、标题栏等多源信息）",
                "default": True,
            },
        },
        "required": ["file_id"],
    }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        file_id = params["file_id"]
        use_hybrid = params.get("use_hybrid", True)
        logger.info("classify_part called: file_id=%s use_hybrid=%s", file_id, use_hybrid)

        try:
            if use_hybrid:
                from src.ml.hybrid_classifier import HybridClassifier

                classifier = HybridClassifier()
                result = classifier.classify(file_id)
                return {
                    "label": result.label,
                    "confidence": round(result.confidence, 4),
                    "source_contributions": {
                        k: round(v, 4) for k, v in getattr(result, "source_contributions", {}).items()
                    },
                }
            else:
                from src.ml.part_classifier import PartClassifier

                classifier = PartClassifier()
                result = classifier.predict(file_id)
                return {
                    "label": result.get("label", "unknown"),
                    "confidence": round(result.get("confidence", 0.0), 4),
                    "source_contributions": {},
                }
        except Exception as exc:
            logger.warning("classify_part fallback for %s: %s", file_id, type(exc).__name__)
            return failure_result(
                STATUS_UNAVAILABLE,
                "classify_service_unavailable",
                label=None,
                confidence=0.0,
                source_contributions={},
            )
