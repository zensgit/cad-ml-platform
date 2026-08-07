"""
ProcessTool -- recommend machining processes for a given part.
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


class ProcessTool(BaseTool):
    """Recommend a manufacturing process route for a CAD part."""

    name = "recommend_process"
    description = "根据零件特征推荐加工工艺路线"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "图纸文件ID",
            },
            "material": {
                "type": "string",
                "description": "材料类型",
                "default": "steel",
            },
            "batch_size": {
                "type": "integer",
                "description": "批量大小",
                "default": 1,
            },
        },
        "required": ["file_id"],
    }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        file_id = params["file_id"]
        material = params.get("material", "steel")
        batch_size = params.get("batch_size", 1)
        logger.info(
            "recommend_process called: file_id=%s material=%s batch=%d",
            file_id, material, batch_size,
        )

        try:
            from src.ml.process_classifier import ProcessClassifier

            classifier = ProcessClassifier()
            result = classifier.predict_process_route(file_id, material=material, batch_size=batch_size)
            return {
                "primary_process": result.get("primary", "machining"),
                "alternatives": result.get("alternatives", []),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as exc:
            logger.warning("tool fallback: %s", type(exc).__name__)
            return failure_result(
                STATUS_UNAVAILABLE,
                "process_service_unavailable",
                file_id=file_id,
            )
