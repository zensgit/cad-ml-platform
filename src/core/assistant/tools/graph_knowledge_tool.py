"""
GraphKnowledgeTool -- query the manufacturing knowledge graph for structured reasoning.
"""

import logging
from typing import Any, Dict

from .base import BaseTool

logger = logging.getLogger(__name__)


class GraphKnowledgeTool(BaseTool):
    """Query the manufacturing knowledge graph for multi-hop reasoning."""

    name = "query_graph"
    description = (
        "查询制造业知识图谱，支持多跳推理（如：SUS304适合什么工艺？"
        "法兰盘用什么材料？CNC车削能达到什么精度？推荐替代材料等）"
    )
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "自然语言问题",
            },
            "action": {
                "type": "string",
                "description": "查询类型",
                "enum": [
                    "query",
                    "optimal_process",
                    "alternative_materials",
                    "explain_relationship",
                ],
                "default": "query",
            },
            "part_type": {
                "type": "string",
                "description": "零件类型（用于工艺推荐）",
            },
            "material": {
                "type": "string",
                "description": "材料（用于工艺推荐或替代材料查找）",
            },
            "tolerance": {
                "type": "string",
                "description": "公差等级（如 IT7）",
            },
            "surface_finish": {
                "type": "string",
                "description": "表面粗糙度（如 Ra1.6）",
            },
            "entity_a": {
                "type": "string",
                "description": "实体A（用于关系解释）",
            },
            "entity_b": {
                "type": "string",
                "description": "实体B（用于关系解释）",
            },
        },
        "required": ["question"],
    }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        question = params["question"]
        action = params.get("action", "query")
        logger.info("query_graph called: question=%r action=%s", question, action)

        try:
            from src.ml.knowledge import ManufacturingKnowledgeGraph, GraphQueryEngine

            graph = ManufacturingKnowledgeGraph()
            graph.build_default_graph()
            engine = GraphQueryEngine(graph)

            if action == "optimal_process":
                part_type = params.get("part_type", "")
                material = params.get("material", "")
                tolerance = params.get("tolerance")
                surface = params.get("surface_finish")
                results = engine.find_optimal_process(
                    part_type=part_type,
                    material=material,
                    tolerance=tolerance,
                    surface=surface,
                )
                return {
                    "action": "optimal_process",
                    "part_type": part_type,
                    "material": material,
                    "recommendations": results[:5],
                    "count": len(results),
                }

            elif action == "alternative_materials":
                material = params.get("material", "")
                requirements = {}
                if params.get("tolerance"):
                    requirements["tolerance"] = params["tolerance"]
                if params.get("surface_finish"):
                    requirements["surface_finish"] = params["surface_finish"]
                results = engine.find_alternative_materials(
                    current_material=material,
                    requirements=requirements if requirements else None,
                )
                return {
                    "action": "alternative_materials",
                    "current_material": material,
                    "alternatives": results[:5],
                    "count": len(results),
                }

            elif action == "explain_relationship":
                entity_a = params.get("entity_a", "")
                entity_b = params.get("entity_b", "")
                explanation = engine.explain_relationship(entity_a, entity_b)
                return {
                    "action": "explain_relationship",
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "explanation": explanation,
                }

            else:
                result = engine.query(question)
                return {
                    "action": "query",
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "entities": result.entities,
                }

        except Exception as exc:
            logger.warning("query_graph fallback: %s", exc)
            return {
                "action": action,
                "answer": f"知识图谱查询暂不可用: {question}",
                "confidence": 0.0,
                "note": "graph_unavailable",
            }
