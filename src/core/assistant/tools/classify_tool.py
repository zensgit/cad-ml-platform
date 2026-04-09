"""
ClassifyTool -- CAD part classification via the hybrid classifier.
"""

import logging
from typing import Any, Dict

from .base import BaseTool

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
            logger.warning("classify_part fallback for %s: %s", file_id, exc)
            return {
                "label": "unknown",
                "confidence": 0.0,
                "source_contributions": {},
                "note": f"分类服务暂不可用，已返回默认结果。原因: {exc}",
            }
