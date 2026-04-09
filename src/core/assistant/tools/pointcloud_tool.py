"""
PointCloudTool -- analyze 3D point cloud / mesh files (STL, OBJ, PLY, XYZ).
"""

import logging
from typing import Any, Dict

from .base import BaseTool

logger = logging.getLogger(__name__)


class PointCloudTool(BaseTool):
    """Analyze 3D mesh/point cloud files for classification and feature extraction."""

    name = "analyze_3d"
    description = (
        "分析3D模型文件（STL/OBJ/PLY/XYZ），支持点云分类、"
        "特征提取和相似零件搜索"
    )
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "已上传的3D文件ID",
            },
            "action": {
                "type": "string",
                "description": "分析类型",
                "enum": ["classify", "features", "similar"],
                "default": "classify",
            },
            "top_k": {
                "type": "integer",
                "description": "相似搜索返回数量",
                "default": 5,
            },
        },
        "required": ["file_id"],
    }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        file_id = params["file_id"]
        action = params.get("action", "classify")
        logger.info("analyze_3d called: file_id=%s action=%s", file_id, action)

        try:
            from src.ml.pointnet.inference import PointNet3DAnalyzer

            analyzer = PointNet3DAnalyzer()

            if action == "classify":
                result = analyzer.classify(file_id)
                return {
                    "action": "classify",
                    "file_id": file_id,
                    **result,
                }
            elif action == "features":
                result = analyzer.extract_features(file_id)
                return {
                    "action": "features",
                    "file_id": file_id,
                    **result,
                }
            elif action == "similar":
                top_k = params.get("top_k", 5)
                result = analyzer.find_similar(file_id, top_k=top_k)
                return {
                    "action": "similar",
                    "file_id": file_id,
                    **result,
                }
            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as exc:
            logger.warning("analyze_3d fallback: %s", exc)
            return {
                "action": action,
                "file_id": file_id,
                "status": "model_unavailable",
                "supported_formats": [".stl", ".obj", ".ply", ".xyz"],
                "note": str(exc),
            }
