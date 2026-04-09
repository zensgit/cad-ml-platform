"""
KnowledgeTool -- query the manufacturing knowledge base.
"""

import logging
from typing import Any, Dict, List

from .base import BaseTool

logger = logging.getLogger(__name__)


class KnowledgeTool(BaseTool):
    """Query the manufacturing knowledge base (materials, welding, GD&T, etc.)."""

    name = "query_knowledge"
    description = "查询制造业知识库（材料属性、焊接参数、GD&T规则等）"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "自然语言查询",
            },
            "category": {
                "type": "string",
                "description": "知识类别",
                "enum": ["materials", "welding", "gdt", "standards", "all"],
                "default": "all",
            },
        },
        "required": ["query"],
    }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params["query"]
        category = params.get("category", "all")
        logger.info("query_knowledge called: query=%r category=%s", query, category)

        try:
            from src.core.assistant.knowledge_retriever import KnowledgeRetriever
            from src.core.assistant.query_analyzer import QueryAnalyzer

            analyzer = QueryAnalyzer()
            analyzed = analyzer.analyze(query)
            retriever = KnowledgeRetriever()
            raw_results = retriever.retrieve(analyzed, max_results=10)

            results: List[Dict[str, Any]] = []
            for item in raw_results:
                source = item.source.value if hasattr(item.source, "value") else str(item.source)
                if category != "all":
                    category_map = {
                        "materials": ["materials"],
                        "welding": ["welding"],
                        "gdt": ["gdt", "tolerance"],
                        "standards": ["design_standards", "standards"],
                    }
                    allowed = category_map.get(category, [])
                    if source not in allowed:
                        continue
                results.append({
                    "source": source,
                    "summary": item.summary,
                    "relevance": round(float(item.relevance), 4),
                    "metadata": getattr(item, "metadata", {}),
                })

            return {"results": results, "count": len(results)}

        except Exception as exc:
            logger.warning("query_knowledge fallback: %s", exc)
            fallback_results: List[Dict[str, Any]] = []
            if category in ("materials", "all") and any(kw in query for kw in ("钢", "steel", "铝", "aluminum", "304", "Q235")):
                fallback_results.append({
                    "source": "materials",
                    "summary": "请参考 GB/T 标准查询具体材料属性参数",
                    "relevance": 0.5,
                    "metadata": {},
                })
            if category in ("gdt", "all") and any(kw in query for kw in ("公差", "tolerance", "GD&T", "形位")):
                fallback_results.append({
                    "source": "gdt",
                    "summary": "GD&T 规则参考 ASME Y14.5 / GB/T 1182",
                    "relevance": 0.5,
                    "metadata": {},
                })
            if not fallback_results:
                fallback_results.append({
                    "source": "general",
                    "summary": "知识库查询服务暂不可用，请稍后重试",
                    "relevance": 0.0,
                    "metadata": {},
                })
            return {
                "results": fallback_results,
                "count": len(fallback_results),
                "note": f"知识库服务暂不可用。原因: {exc}",
            }
