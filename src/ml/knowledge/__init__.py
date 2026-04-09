"""Manufacturing knowledge graph module.

Provides a lightweight in-memory knowledge graph for multi-hop reasoning
over manufacturing entities (materials, processes, part types, properties)
and a natural-language query engine for intelligent process recommendation.
"""

from src.ml.knowledge.graph import (
    KnowledgeEdge,
    KnowledgeNode,
    ManufacturingKnowledgeGraph,
)
from src.ml.knowledge.query_engine import GraphQueryEngine, QueryResult

__all__ = [
    "KnowledgeEdge",
    "KnowledgeNode",
    "ManufacturingKnowledgeGraph",
    "GraphQueryEngine",
    "QueryResult",
]
