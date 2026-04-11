"""
FeatureTool -- extract geometric feature vectors from CAD drawings.
"""

import logging
from typing import Any, Dict

from .base import BaseTool

logger = logging.getLogger(__name__)


class FeatureTool(BaseTool):
    """Extract the geometric feature vector of a CAD drawing."""

    name = "extract_features"
    description = "提取 CAD 图纸的几何特征向量"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "图纸文件ID",
            },
            "version": {
                "type": "string",
                "description": "特征版本",
                "enum": ["v3", "v4"],
                "default": "v3",
            },
        },
        "required": ["file_id"],
    }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        file_id = params["file_id"]
        version = params.get("version", "v3")
        logger.info("extract_features called: file_id=%s version=%s", file_id, version)

        try:
            from src.core.feature_extractor import FeatureExtractor

            extractor = FeatureExtractor(feature_version=version)
            vector = extractor.extract(file_id)
            dimension = len(vector) if vector is not None else 0
            entity_count = int(vector[0]) if vector is not None and len(vector) > 0 else 0
            complexity = float(vector[-1]) if vector is not None and len(vector) > 1 else 0.0
            return {
                "dimension": dimension,
                "version": version,
                "summary": {
                    "entity_count": entity_count,
                    "complexity": round(complexity, 4),
                },
            }
        except Exception as exc:
            logger.warning("extract_features fallback for %s: %s", file_id, exc)
            dim = {"v3": 17, "v4": 22}.get(version, 17)
            return {
                "dimension": dim,
                "version": version,
                "summary": {
                    "entity_count": 0,
                    "complexity": 0.0,
                },
                "note": f"特征提取服务暂不可用，返回默认维度。原因: {exc}",
            }
