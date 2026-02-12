"""
Geometry analysis modules for CAD files.

Provides:
- Geometric feature extraction
- Topological analysis
- Spatial indexing
"""

from src.core.cad.geometry.features import (
    GeometryType,
    BoundingBox,
    GeometricFeatures,
    GeometryExtractor,
    DrawingStatistics,
    DrawingAnalyzer,
)

from src.core.cad.geometry.topology import (
    ConnectionType,
    TopologicalNode,
    TopologicalEdge,
    ConnectedComponent,
    TopologyAnalysis,
    TopologyGraph,
    TopologyAnalyzer,
)

from src.core.cad.geometry.spatial import (
    SpatialBounds,
    SpatialEntry,
    SpatialIndex,
    GridIndex,
    RTreeIndex,
    SpatialQuery,
)

__all__ = [
    # Features
    "GeometryType",
    "BoundingBox",
    "GeometricFeatures",
    "GeometryExtractor",
    "DrawingStatistics",
    "DrawingAnalyzer",
    # Topology
    "ConnectionType",
    "TopologicalNode",
    "TopologicalEdge",
    "ConnectedComponent",
    "TopologyAnalysis",
    "TopologyGraph",
    "TopologyAnalyzer",
    # Spatial
    "SpatialBounds",
    "SpatialEntry",
    "SpatialIndex",
    "GridIndex",
    "RTreeIndex",
    "SpatialQuery",
]
