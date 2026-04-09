"""
Function-calling tool definitions for the CAD Copilot assistant.

Exports all concrete tool classes and a ``TOOL_REGISTRY`` dict that maps
tool names to instantiated tool objects.
"""

from .base import BaseTool
from .classify_tool import ClassifyTool
from .similarity_tool import SimilarityTool
from .cost_tool import CostTool
from .feature_tool import FeatureTool
from .process_tool import ProcessTool
from .quality_tool import QualityTool
from .knowledge_tool import KnowledgeTool
from .graph_knowledge_tool import GraphKnowledgeTool
from .pointcloud_tool import PointCloudTool

# Instantiate one of each tool and index by name.
TOOL_REGISTRY = {
    tool.name: tool
    for tool in [
        ClassifyTool(),
        SimilarityTool(),
        CostTool(),
        FeatureTool(),
        ProcessTool(),
        QualityTool(),
        KnowledgeTool(),
        GraphKnowledgeTool(),
        PointCloudTool(),
    ]
}

__all__ = [
    "BaseTool",
    "ClassifyTool",
    "SimilarityTool",
    "CostTool",
    "FeatureTool",
    "ProcessTool",
    "QualityTool",
    "KnowledgeTool",
    "GraphKnowledgeTool",
    "PointCloudTool",
    "TOOL_REGISTRY",
]
