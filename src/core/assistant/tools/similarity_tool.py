"""
SimilarityTool -- search for similar parts in the vector store.
"""

import logging
from typing import Any, Dict

from .base import BaseTool

logger = logging.getLogger(__name__)


class SimilarityTool(BaseTool):
    """Search for similar CAD parts using vector similarity."""

    name = "search_similar"
    description = "在向量库中搜索与指定图纸相似的零件"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "查询图纸的文件ID",
            },
            "top_k": {
                "type": "integer",
                "description": "返回最相似的零件数量",
                "default": 5,
            },
            "min_similarity": {
                "type": "number",
                "description": "最低相似度阈值 (0-1)",
                "default": 0.7,
            },
            "material_filter": {
                "type": "string",
                "description": "按材料过滤（可选，如 steel、aluminum）",
            },
        },
        "required": ["file_id"],
    }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        file_id = params["file_id"]
        top_k = params.get("top_k", 5)
        min_similarity = params.get("min_similarity", 0.7)
        material_filter = params.get("material_filter")
        logger.info(
            "search_similar called: file_id=%s top_k=%d min_sim=%.2f material=%s",
            file_id, top_k, min_similarity, material_filter,
        )

        try:
            from src.core.feature_extractor import FeatureExtractor
            from src.core.assistant.semantic_retrieval import create_semantic_retriever

            extractor = FeatureExtractor(feature_version="v3")
            retriever = create_semantic_retriever()

            results_raw = retriever.search(
                query=file_id,
                top_k=top_k,
            )

            results = []
            for item in results_raw:
                sim = getattr(item, "similarity", getattr(item, "score", 0.0))
                if sim < min_similarity:
                    continue
                label = getattr(item, "label", getattr(item, "metadata", {}).get("label", "unknown"))
                mat = getattr(item, "material", getattr(item, "metadata", {}).get("material", ""))
                if material_filter and mat and material_filter.lower() not in mat.lower():
                    continue
                results.append({
                    "id": getattr(item, "id", str(item)),
                    "similarity": round(float(sim), 4),
                    "label": label,
                })
            return {"results": results[:top_k], "count": len(results)}

        except Exception as exc:
            logger.warning("search_similar fallback for %s: %s", file_id, exc)
            return {
                "results": [],
                "count": 0,
                "note": f"相似性搜索服务暂不可用。原因: {exc}",
            }
