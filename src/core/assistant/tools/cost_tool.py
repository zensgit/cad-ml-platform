"""
CostTool -- manufacturing cost estimation.
"""

import logging
from typing import Any, Dict

from .base import BaseTool

logger = logging.getLogger(__name__)


class CostTool(BaseTool):
    """Estimate the manufacturing cost of a part."""

    name = "estimate_cost"
    description = "估算零件的制造成本，包括材料费、加工费、管理费"
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
            "tolerance_grade": {
                "type": "string",
                "description": "公差等级 (IT6-IT12)",
                "default": "IT8",
            },
        },
        "required": ["file_id"],
    }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        file_id = params["file_id"]
        material = params.get("material", "steel")
        batch_size = params.get("batch_size", 1)
        tolerance_grade = params.get("tolerance_grade", "IT8")
        logger.info(
            "estimate_cost called: file_id=%s material=%s batch=%d tol=%s",
            file_id, material, batch_size, tolerance_grade,
        )

        try:
            from src.ml.cost import CostEstimator, CostEstimateRequest

            estimator = CostEstimator()
            request = CostEstimateRequest(
                material=material,
                batch_size=batch_size,
                tolerance_grade=tolerance_grade,
            )
            response = estimator.estimate(request)
            return {
                "material_cost": response.estimate.material_cost,
                "machining_cost": response.estimate.machining_cost,
                "setup_cost": response.estimate.setup_cost,
                "overhead": response.estimate.overhead,
                "total": response.estimate.total,
                "currency": response.estimate.currency,
                "confidence": response.confidence,
                "process_route": response.process_route,
            }
        except Exception as exc:
            logger.warning("estimate_cost fallback for %s: %s", file_id, exc)
            # Deterministic mock for testing / offline
            base = {"steel": 45.0, "aluminum": 38.0, "stainless_steel": 62.0}.get(material, 50.0)
            tol_factor = {"IT6": 1.5, "IT7": 1.3, "IT8": 1.0, "IT9": 0.9, "IT10": 0.8}.get(tolerance_grade, 1.0)
            material_cost = round(base * tol_factor, 2)
            machining_cost = round(material_cost * 1.8, 2)
            setup_cost = round(200.0 / max(batch_size, 1), 2)
            overhead = round((material_cost + machining_cost) * 0.15, 2)
            total = round(material_cost + machining_cost + setup_cost + overhead, 2)
            return {
                "material_cost": material_cost,
                "machining_cost": machining_cost,
                "setup_cost": setup_cost,
                "overhead": overhead,
                "total": total,
                "currency": "CNY",
                "confidence": 0.5,
                "process_route": ["blanking", "machining", "inspection"],
                "note": f"成本估算服务暂不可用，返回估算值。原因: {exc}",
            }
