"""
Topological analysis for CAD geometry.

Analyzes connectivity and relationships between geometric entities.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConnectionType(str, Enum):
    """Types of connections between entities."""
    ENDPOINT = "endpoint"  # Share an endpoint
    INTERSECTION = "intersection"  # Cross each other
    TANGENT = "tangent"  # Touch tangentially
    OVERLAP = "overlap"  # Partially overlap
    CONTAINMENT = "containment"  # One contains the other
    PROXIMITY = "proximity"  # Close but not touching


@dataclass
class TopologicalNode:
    """A node in the topological graph."""
    entity_id: str
    entity_type: str
    position: Optional[Tuple[float, float]] = None
    neighbors: Set[str] = field(default_factory=set)
    connection_types: Dict[str, ConnectionType] = field(default_factory=dict)
    degree: int = 0
    cluster_id: int = -1


@dataclass
class TopologicalEdge:
    """An edge in the topological graph."""
    source_id: str
    target_id: str
    connection_type: ConnectionType
    connection_point: Optional[Tuple[float, float]] = None
    distance: float = 0.0


@dataclass
class ConnectedComponent:
    """A connected component of entities."""
    component_id: int
    node_ids: Set[str]
    edge_count: int
    bbox: Optional[Tuple[float, float, float, float]] = None  # min_x, min_y, max_x, max_y

    @property
    def size(self) -> int:
        return len(self.node_ids)


@dataclass
class TopologyAnalysis:
    """Complete topological analysis results."""
    node_count: int
    edge_count: int
    connected_components: List[ConnectedComponent]
    isolated_nodes: int
    max_degree: int
    mean_degree: float
    clustering_coefficient: float
    density: float

    # Entity statistics
    junction_points: List[Tuple[float, float]]  # Points where >2 entities meet
    endpoint_pairs: List[Tuple[str, str]]  # Pairs sharing endpoints
    chains: List[List[str]]  # Linear sequences of connected entities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "component_count": len(self.connected_components),
            "isolated_nodes": self.isolated_nodes,
            "max_degree": self.max_degree,
            "mean_degree": self.mean_degree,
            "clustering_coefficient": self.clustering_coefficient,
            "density": self.density,
            "junction_count": len(self.junction_points),
            "chain_count": len(self.chains),
        }


class TopologyGraph:
    """
    Graph representation of geometric topology.

    Nodes are entities, edges represent connections.
    """

    def __init__(self):
        self._nodes: Dict[str, TopologicalNode] = {}
        self._edges: List[TopologicalEdge] = []
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)

    def add_node(
        self,
        entity_id: str,
        entity_type: str,
        position: Optional[Tuple[float, float]] = None,
    ) -> TopologicalNode:
        """Add a node to the graph."""
        if entity_id not in self._nodes:
            node = TopologicalNode(
                entity_id=entity_id,
                entity_type=entity_type,
                position=position,
            )
            self._nodes[entity_id] = node
        return self._nodes[entity_id]

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        connection_type: ConnectionType,
        connection_point: Optional[Tuple[float, float]] = None,
        distance: float = 0.0,
    ) -> TopologicalEdge:
        """Add an edge to the graph."""
        edge = TopologicalEdge(
            source_id=source_id,
            target_id=target_id,
            connection_type=connection_type,
            connection_point=connection_point,
            distance=distance,
        )
        self._edges.append(edge)

        # Update adjacency
        self._adjacency[source_id].add(target_id)
        self._adjacency[target_id].add(source_id)

        # Update nodes
        if source_id in self._nodes:
            self._nodes[source_id].neighbors.add(target_id)
            self._nodes[source_id].connection_types[target_id] = connection_type
            self._nodes[source_id].degree = len(self._nodes[source_id].neighbors)

        if target_id in self._nodes:
            self._nodes[target_id].neighbors.add(source_id)
            self._nodes[target_id].connection_types[source_id] = connection_type
            self._nodes[target_id].degree = len(self._nodes[target_id].neighbors)

        return edge

    def get_node(self, entity_id: str) -> Optional[TopologicalNode]:
        return self._nodes.get(entity_id)

    def get_neighbors(self, entity_id: str) -> Set[str]:
        return self._adjacency.get(entity_id, set())

    def find_connected_components(self) -> List[ConnectedComponent]:
        """Find all connected components using BFS."""
        visited = set()
        components = []
        component_id = 0

        for node_id in self._nodes:
            if node_id in visited:
                continue

            # BFS to find component
            component_nodes = set()
            queue = [node_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                component_nodes.add(current)

                for neighbor in self._adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            # Count edges in component
            edge_count = sum(
                1 for e in self._edges
                if e.source_id in component_nodes and e.target_id in component_nodes
            )

            components.append(ConnectedComponent(
                component_id=component_id,
                node_ids=component_nodes,
                edge_count=edge_count,
            ))

            # Update cluster IDs
            for nid in component_nodes:
                self._nodes[nid].cluster_id = component_id

            component_id += 1

        return components

    def find_chains(self) -> List[List[str]]:
        """Find linear chains of connected entities (degree-2 sequences)."""
        chains = []
        visited_edges = set()

        for node_id, node in self._nodes.items():
            if node.degree != 1:  # Start from endpoints
                continue

            # Follow the chain
            chain = [node_id]
            current = node_id
            visited_edges.add((current, list(node.neighbors)[0]))

            while True:
                neighbors = [n for n in self._adjacency[current] if n not in chain]
                if not neighbors:
                    break

                next_node = neighbors[0]
                chain.append(next_node)

                edge_key = (min(current, next_node), max(current, next_node))
                if edge_key in visited_edges:
                    break
                visited_edges.add(edge_key)

                current = next_node
                if self._nodes[current].degree != 2:
                    break

            if len(chain) > 1:
                chains.append(chain)

        return chains

    def compute_clustering_coefficient(self) -> float:
        """Compute global clustering coefficient."""
        if not self._nodes:
            return 0.0

        total_triangles = 0
        total_triplets = 0

        for node_id, node in self._nodes.items():
            neighbors = list(node.neighbors)
            k = len(neighbors)

            if k < 2:
                continue

            # Count triangles (neighbors that are connected to each other)
            triangles = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in self._adjacency[neighbors[i]]:
                        triangles += 1

            total_triangles += triangles
            total_triplets += k * (k - 1) / 2

        if total_triplets == 0:
            return 0.0

        return total_triangles / total_triplets

    @property
    def nodes(self) -> Dict[str, TopologicalNode]:
        return self._nodes

    @property
    def edges(self) -> List[TopologicalEdge]:
        return self._edges


class TopologyAnalyzer:
    """
    Analyzes topological relationships between CAD entities.
    """

    def __init__(
        self,
        tolerance: float = 1e-4,
        proximity_threshold: float = 1.0,
    ):
        self.tolerance = tolerance
        self.proximity_threshold = proximity_threshold

    def analyze(self, entities: List[Any]) -> TopologyAnalysis:
        """Analyze topology of entities."""
        graph = self._build_graph(entities)
        components = graph.find_connected_components()
        chains = graph.find_chains()

        # Find junction points
        junction_points = []
        for node in graph.nodes.values():
            if node.degree > 2 and node.position:
                junction_points.append(node.position)

        # Find endpoint pairs
        endpoint_pairs = [
            (e.source_id, e.target_id)
            for e in graph.edges
            if e.connection_type == ConnectionType.ENDPOINT
        ]

        # Compute statistics
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)
        degrees = [n.degree for n in graph.nodes.values()]
        max_degree = max(degrees) if degrees else 0
        mean_degree = np.mean(degrees) if degrees else 0.0
        isolated = sum(1 for d in degrees if d == 0)

        # Graph density
        max_edges = node_count * (node_count - 1) / 2
        density = edge_count / max_edges if max_edges > 0 else 0.0

        clustering = graph.compute_clustering_coefficient()

        return TopologyAnalysis(
            node_count=node_count,
            edge_count=edge_count,
            connected_components=components,
            isolated_nodes=isolated,
            max_degree=max_degree,
            mean_degree=float(mean_degree),
            clustering_coefficient=clustering,
            density=density,
            junction_points=junction_points,
            endpoint_pairs=endpoint_pairs,
            chains=chains,
        )

    def _build_graph(self, entities: List[Any]) -> TopologyGraph:
        """Build topology graph from entities."""
        graph = TopologyGraph()

        # Extract endpoints for each entity
        entity_endpoints: Dict[str, List[Tuple[float, float]]] = {}

        for entity in entities:
            entity_id = self._get_entity_id(entity)
            entity_type = entity.dxftype()
            endpoints = self._get_endpoints(entity)

            # Add node
            centroid = self._get_centroid(endpoints) if endpoints else None
            graph.add_node(entity_id, entity_type, centroid)
            entity_endpoints[entity_id] = endpoints

        # Find connections
        entity_ids = list(entity_endpoints.keys())
        for i, id1 in enumerate(entity_ids):
            for j, id2 in enumerate(entity_ids):
                if i >= j:
                    continue

                pts1 = entity_endpoints[id1]
                pts2 = entity_endpoints[id2]

                connection = self._find_connection(pts1, pts2)
                if connection:
                    conn_type, conn_point = connection
                    graph.add_edge(id1, id2, conn_type, conn_point)

        return graph

    def _get_entity_id(self, entity: Any) -> str:
        """Get unique ID for entity."""
        if hasattr(entity.dxf, "handle"):
            return str(entity.dxf.handle)
        return str(id(entity))

    def _get_endpoints(self, entity: Any) -> List[Tuple[float, float]]:
        """Get endpoints of an entity."""
        dxf_type = entity.dxftype()
        points = []

        try:
            if dxf_type == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                points = [(start.x, start.y), (end.x, end.y)]

            elif dxf_type in ("POLYLINE", "LWPOLYLINE"):
                if hasattr(entity, 'get_points'):
                    pts = list(entity.get_points())
                    if pts:
                        points = [(pts[0][0], pts[0][1]), (pts[-1][0], pts[-1][1])]

            elif dxf_type == "ARC":
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_angle = np.radians(entity.dxf.start_angle)
                end_angle = np.radians(entity.dxf.end_angle)

                start_pt = (
                    center.x + radius * np.cos(start_angle),
                    center.y + radius * np.sin(start_angle),
                )
                end_pt = (
                    center.x + radius * np.cos(end_angle),
                    center.y + radius * np.sin(end_angle),
                )
                points = [start_pt, end_pt]

            elif dxf_type == "CIRCLE":
                # Circle has no endpoints, use center
                center = entity.dxf.center
                points = [(center.x, center.y)]

            elif dxf_type == "POINT":
                loc = entity.dxf.location
                points = [(loc.x, loc.y)]

        except Exception as e:
            logger.debug(f"Could not get endpoints for {dxf_type}: {e}")

        return points

    def _get_centroid(self, points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Compute centroid of points."""
        if not points:
            return None
        xs, ys = zip(*points)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _find_connection(
        self,
        pts1: List[Tuple[float, float]],
        pts2: List[Tuple[float, float]],
    ) -> Optional[Tuple[ConnectionType, Optional[Tuple[float, float]]]]:
        """Find connection between two sets of points."""
        if not pts1 or not pts2:
            return None

        # Check endpoint connections
        for p1 in pts1:
            for p2 in pts2:
                dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

                if dist < self.tolerance:
                    # Exact endpoint match
                    return (ConnectionType.ENDPOINT, p1)

                elif dist < self.proximity_threshold:
                    # Proximity
                    return (ConnectionType.PROXIMITY, ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))

        return None
