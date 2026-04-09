"""
ProcessTool -- recommend machining processes for a given part.
"""

import logging
from typing import Any, Dict

from .base import BaseTool

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
            logger.warning("recommend_process fallback for %s: %s", file_id, exc)
            process_map = {
                "steel": ("CNC machining", ["turning", "milling", "grinding"]),
                "aluminum": ("CNC machining", ["milling", "turning", "anodizing"]),
                "stainless_steel": ("CNC machining", ["turning", "milling", "polishing"]),
                "cast_iron": ("casting + machining", ["sand casting", "CNC finishing"]),
                "plastic": ("injection molding", ["3D printing", "CNC machining"]),
            }
            primary, alts = process_map.get(material, ("CNC machining", ["turning", "milling"]))
            if batch_size > 100:
                primary = "批量加工 - " + primary
            return {
                "primary_process": primary,
                "alternatives": alts,
                "reasoning": f"基于材料({material})和批量({batch_size})的默认推荐。工艺推荐服务暂不可用。",
            }
