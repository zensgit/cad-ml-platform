"""
Manufacturing Knowledge Graph.

Lightweight in-memory graph that stores structured relationships between
manufacturing entities -- materials, processes, part types, and properties.
Enables multi-hop reasoning such as:

    SUS304 -> suitable_for -> CNC车削 -> produces -> Ra1.6

No external graph database is required; everything lives in Python dicts.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    id: str  # e.g. "material:sus304"
    type: str  # e.g. "material", "process", "part_type", "property"
    name: str  # e.g. "SUS304不锈钢"
    aliases: List[str] = field(default_factory=list)
    properties: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "aliases": list(self.aliases),
            "properties": dict(self.properties),
        }


@dataclass
class KnowledgeEdge:
    """A relationship between two nodes."""

    source: str  # node id
    target: str  # node id
    relation: str  # e.g. "suitable_for", "produces", "requires"
    weight: float = 1.0  # confidence / strength 0-1
    properties: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "properties": dict(self.properties),
        }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class ManufacturingKnowledgeGraph:
    """In-memory manufacturing knowledge graph."""

    def __init__(self) -> None:
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._edges: List[KnowledgeEdge] = []
        self._adjacency: Dict[str, List[KnowledgeEdge]] = defaultdict(list)
        # Reverse adjacency for efficient backward traversal
        self._reverse_adjacency: Dict[str, List[KnowledgeEdge]] = defaultdict(list)
        # Alias lookup  (lowered alias -> node id)
        self._alias_index: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: KnowledgeNode) -> None:
        """Add or replace a node and index its aliases."""
        self._nodes[node.id] = node
        for alias in node.aliases:
            self._alias_index[alias.lower()] = node.id
        # Also index the canonical name
        self._alias_index[node.name.lower()] = node.id

    def add_edge(self, edge: KnowledgeEdge) -> None:
        """Add an edge and update adjacency maps."""
        self._edges.append(edge)
        self._adjacency[edge.source].append(edge)
        self._reverse_adjacency[edge.target].append(edge)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        return self._nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
    ) -> List[Tuple[KnowledgeNode, KnowledgeEdge]]:
        """Return (neighbor_node, edge) pairs reachable from *node_id*.

        If *relation* is given, only edges with that relation type are
        returned.
        """
        results: List[Tuple[KnowledgeNode, KnowledgeEdge]] = []
        for edge in self._adjacency.get(node_id, []):
            if relation and edge.relation != relation:
                continue
            target_node = self._nodes.get(edge.target)
            if target_node:
                results.append((target_node, edge))
        return results

    def get_reverse_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
    ) -> List[Tuple[KnowledgeNode, KnowledgeEdge]]:
        """Return nodes that have an edge *pointing to* *node_id*."""
        results: List[Tuple[KnowledgeNode, KnowledgeEdge]] = []
        for edge in self._reverse_adjacency.get(node_id, []):
            if relation and edge.relation != relation:
                continue
            source_node = self._nodes.get(edge.source)
            if source_node:
                results.append((source_node, edge))
        return results

    def find_nodes_by_type(self, node_type: str) -> List[KnowledgeNode]:
        return [n for n in self._nodes.values() if n.type == node_type]

    def find_node_by_alias(self, alias: str) -> Optional[KnowledgeNode]:
        node_id = self._alias_index.get(alias.lower())
        if node_id:
            return self._nodes.get(node_id)
        return None

    # ------------------------------------------------------------------
    # Path finding (BFS)
    # ------------------------------------------------------------------

    def query_path(
        self,
        from_id: str,
        to_id: str,
        max_hops: int = 3,
    ) -> List[List[str]]:
        """Find all paths between *from_id* and *to_id* via BFS.

        Returns a list of paths, where each path is a list of node ids.
        Only paths with length <= *max_hops* edges are returned.
        """
        if from_id not in self._nodes or to_id not in self._nodes:
            return []

        all_paths: List[List[str]] = []
        # BFS queue: each element is a path so far
        queue: deque[List[str]] = deque([[from_id]])

        while queue:
            path = queue.popleft()
            current = path[-1]

            if current == to_id:
                all_paths.append(path)
                continue

            if len(path) - 1 >= max_hops:
                continue

            for edge in self._adjacency.get(current, []):
                if edge.target not in path:  # avoid cycles
                    queue.append(path + [edge.target])

        return all_paths

    # ------------------------------------------------------------------
    # Stats / serialization
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        nodes_by_type: Dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            nodes_by_type[node.type] += 1
        edges_by_relation: Dict[str, int] = defaultdict(int)
        for edge in self._edges:
            edges_by_relation[edge.relation] += 1
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "nodes_by_type": dict(nodes_by_type),
            "edges_by_relation": dict(edges_by_relation),
        }

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            "stats": self.get_stats(),
        }

    # ==================================================================
    # Default graph construction with real manufacturing knowledge
    # ==================================================================

    def build_default_graph(self) -> None:  # noqa: C901  (intentionally large)
        """Populate the graph with real Chinese manufacturing knowledge."""

        self._build_materials()
        self._build_part_types()
        self._build_processes()
        self._build_properties()
        self._build_material_process_edges()
        self._build_part_material_edges()
        self._build_process_property_edges()
        self._build_part_process_edges()
        self._build_material_property_edges()

    # ---- Materials (>= 10) -------------------------------------------

    def _build_materials(self) -> None:
        materials = [
            KnowledgeNode(
                id="material:q235",
                type="material",
                name="Q235碳钢",
                aliases=["Q235", "Q235A", "Q235B", "普碳钢", "A3钢"],
                properties={
                    "density": 7850,
                    "price_per_kg": 5.0,
                    "tensile_strength_mpa": 370,
                    "machinability": "good",
                    "weldability": "excellent",
                    "corrosion_resistance": "low",
                },
            ),
            KnowledgeNode(
                id="material:45steel",
                type="material",
                name="45#钢",
                aliases=["45号钢", "45钢", "S45C", "AISI 1045", "C45"],
                properties={
                    "density": 7850,
                    "price_per_kg": 6.5,
                    "tensile_strength_mpa": 600,
                    "machinability": "good",
                    "weldability": "moderate",
                    "corrosion_resistance": "low",
                    "heat_treatable": True,
                },
            ),
            KnowledgeNode(
                id="material:sus304",
                type="material",
                name="SUS304不锈钢",
                aliases=["304不锈钢", "SUS304", "AISI 304", "0Cr18Ni9", "06Cr19Ni10"],
                properties={
                    "density": 7930,
                    "price_per_kg": 22.0,
                    "tensile_strength_mpa": 520,
                    "machinability": "moderate",
                    "weldability": "good",
                    "corrosion_resistance": "high",
                },
            ),
            KnowledgeNode(
                id="material:sus316",
                type="material",
                name="SUS316不锈钢",
                aliases=["316不锈钢", "SUS316", "AISI 316", "0Cr17Ni12Mo2", "316L"],
                properties={
                    "density": 7980,
                    "price_per_kg": 35.0,
                    "tensile_strength_mpa": 485,
                    "machinability": "moderate",
                    "weldability": "good",
                    "corrosion_resistance": "very_high",
                },
            ),
            KnowledgeNode(
                id="material:al6061",
                type="material",
                name="6061铝合金",
                aliases=["6061", "6061-T6", "铝合金6061", "AL6061"],
                properties={
                    "density": 2700,
                    "price_per_kg": 25.0,
                    "tensile_strength_mpa": 310,
                    "machinability": "excellent",
                    "weldability": "good",
                    "corrosion_resistance": "moderate",
                },
            ),
            KnowledgeNode(
                id="material:al7075",
                type="material",
                name="7075铝合金",
                aliases=["7075", "7075-T6", "铝合金7075", "AL7075", "超硬铝"],
                properties={
                    "density": 2810,
                    "price_per_kg": 45.0,
                    "tensile_strength_mpa": 572,
                    "machinability": "good",
                    "weldability": "poor",
                    "corrosion_resistance": "moderate",
                },
            ),
            KnowledgeNode(
                id="material:tc4",
                type="material",
                name="TC4钛合金",
                aliases=["TC4", "Ti-6Al-4V", "钛合金TC4", "GR5钛合金"],
                properties={
                    "density": 4430,
                    "price_per_kg": 180.0,
                    "tensile_strength_mpa": 895,
                    "machinability": "poor",
                    "weldability": "moderate",
                    "corrosion_resistance": "very_high",
                },
            ),
            KnowledgeNode(
                id="material:abs",
                type="material",
                name="ABS塑料",
                aliases=["ABS", "丙烯腈-丁二烯-苯乙烯"],
                properties={
                    "density": 1050,
                    "price_per_kg": 15.0,
                    "tensile_strength_mpa": 45,
                    "machinability": "excellent",
                    "weldability": "none",
                    "corrosion_resistance": "high",
                },
            ),
            KnowledgeNode(
                id="material:pa66",
                type="material",
                name="PA66尼龙",
                aliases=["PA66", "尼龙66", "聚酰胺66", "Nylon 66"],
                properties={
                    "density": 1140,
                    "price_per_kg": 28.0,
                    "tensile_strength_mpa": 82,
                    "machinability": "good",
                    "weldability": "none",
                    "corrosion_resistance": "high",
                    "self_lubricating": True,
                },
            ),
            KnowledgeNode(
                id="material:h62",
                type="material",
                name="黄铜H62",
                aliases=["H62", "黄铜", "CuZn40", "C28000"],
                properties={
                    "density": 8430,
                    "price_per_kg": 42.0,
                    "tensile_strength_mpa": 390,
                    "machinability": "excellent",
                    "weldability": "moderate",
                    "corrosion_resistance": "moderate",
                    "conductivity": "high",
                },
            ),
            KnowledgeNode(
                id="material:40cr",
                type="material",
                name="40Cr合金钢",
                aliases=["40Cr", "SCr440", "AISI 5140", "40铬钢"],
                properties={
                    "density": 7850,
                    "price_per_kg": 8.0,
                    "tensile_strength_mpa": 980,
                    "machinability": "moderate",
                    "weldability": "poor",
                    "corrosion_resistance": "low",
                    "heat_treatable": True,
                },
            ),
        ]
        for m in materials:
            self.add_node(m)

    # ---- Part types (8) ----------------------------------------------

    def _build_part_types(self) -> None:
        part_types = [
            KnowledgeNode(
                id="part:flange",
                type="part_type",
                name="法兰盘",
                aliases=["法兰", "flange", "法兰片"],
                properties={"geometry": "disk_with_holes", "complexity": "medium"},
            ),
            KnowledgeNode(
                id="part:shaft",
                type="part_type",
                name="轴",
                aliases=["shaft", "转轴", "主轴", "传动轴"],
                properties={"geometry": "cylindrical", "complexity": "medium"},
            ),
            KnowledgeNode(
                id="part:housing",
                type="part_type",
                name="壳体",
                aliases=["housing", "箱体", "外壳", "机壳"],
                properties={"geometry": "box_with_cavities", "complexity": "high"},
            ),
            KnowledgeNode(
                id="part:bracket",
                type="part_type",
                name="支架",
                aliases=["bracket", "托架", "安装架"],
                properties={"geometry": "l_shape_or_plate", "complexity": "low"},
            ),
            KnowledgeNode(
                id="part:gear",
                type="part_type",
                name="齿轮",
                aliases=["gear", "齿轮组", "直齿轮", "斜齿轮"],
                properties={"geometry": "disk_with_teeth", "complexity": "high"},
            ),
            KnowledgeNode(
                id="part:connector",
                type="part_type",
                name="连接件",
                aliases=["connector", "接头", "管接头", "连接器"],
                properties={"geometry": "cylindrical_with_threads", "complexity": "medium"},
            ),
            KnowledgeNode(
                id="part:seal",
                type="part_type",
                name="密封件",
                aliases=["seal", "密封圈", "O型圈", "密封垫"],
                properties={"geometry": "ring_or_gasket", "complexity": "low"},
            ),
            KnowledgeNode(
                id="part:other",
                type="part_type",
                name="其他",
                aliases=["other", "杂件", "定制件"],
                properties={"geometry": "varies", "complexity": "varies"},
            ),
        ]
        for p in part_types:
            self.add_node(p)

    # ---- Processes (>= 8) --------------------------------------------

    def _build_processes(self) -> None:
        processes = [
            KnowledgeNode(
                id="process:cnc_turning",
                type="process",
                name="CNC车削",
                aliases=["CNC turning", "数控车削", "车床加工", "车削"],
                properties={
                    "cost_per_hour": 50.0,
                    "setup_time_min": 30,
                    "suitable_geometry": "rotational",
                    "max_diameter_mm": 500,
                },
            ),
            KnowledgeNode(
                id="process:cnc_milling",
                type="process",
                name="CNC铣削",
                aliases=["CNC milling", "数控铣削", "铣床加工", "铣削"],
                properties={
                    "cost_per_hour": 60.0,
                    "setup_time_min": 45,
                    "suitable_geometry": "prismatic",
                    "max_size_mm": 1000,
                },
            ),
            KnowledgeNode(
                id="process:5axis",
                type="process",
                name="5轴加工",
                aliases=["五轴加工", "5-axis machining", "五轴CNC", "5轴CNC"],
                properties={
                    "cost_per_hour": 120.0,
                    "setup_time_min": 60,
                    "suitable_geometry": "complex_freeform",
                    "max_size_mm": 800,
                },
            ),
            KnowledgeNode(
                id="process:wire_edm",
                type="process",
                name="线切割",
                aliases=["wire EDM", "电火花线切割", "WEDM", "慢走丝"],
                properties={
                    "cost_per_hour": 40.0,
                    "setup_time_min": 60,
                    "suitable_geometry": "2d_profile",
                    "max_thickness_mm": 300,
                },
            ),
            KnowledgeNode(
                id="process:grinding",
                type="process",
                name="磨削",
                aliases=["grinding", "研磨", "外圆磨", "平面磨"],
                properties={
                    "cost_per_hour": 55.0,
                    "setup_time_min": 30,
                    "suitable_geometry": "cylindrical_or_flat",
                },
            ),
            KnowledgeNode(
                id="process:casting",
                type="process",
                name="铸造",
                aliases=["casting", "翻砂铸造", "精密铸造", "压铸"],
                properties={
                    "cost_per_hour": 30.0,
                    "setup_time_min": 0,
                    "suitable_geometry": "complex_hollow",
                    "min_batch": 50,
                },
            ),
            KnowledgeNode(
                id="process:forging",
                type="process",
                name="锻造",
                aliases=["forging", "热锻", "冷锻", "模锻"],
                properties={
                    "cost_per_hour": 45.0,
                    "setup_time_min": 0,
                    "suitable_geometry": "solid_simple",
                    "min_batch": 100,
                },
            ),
            KnowledgeNode(
                id="process:3d_printing",
                type="process",
                name="3D打印",
                aliases=["3D printing", "增材制造", "SLM", "SLS", "FDM"],
                properties={
                    "cost_per_hour": 40.0,
                    "setup_time_min": 15,
                    "suitable_geometry": "complex_any",
                    "min_batch": 1,
                },
            ),
            KnowledgeNode(
                id="process:drilling",
                type="process",
                name="钻孔",
                aliases=["drilling", "钻削", "打孔"],
                properties={
                    "cost_per_hour": 35.0,
                    "setup_time_min": 15,
                    "suitable_geometry": "holes",
                },
            ),
            KnowledgeNode(
                id="process:sheet_metal",
                type="process",
                name="钣金加工",
                aliases=["sheet metal", "折弯", "冲压", "激光切割"],
                properties={
                    "cost_per_hour": 35.0,
                    "setup_time_min": 20,
                    "suitable_geometry": "flat_bent",
                    "max_thickness_mm": 6,
                },
            ),
        ]
        for p in processes:
            self.add_node(p)

    # ---- Properties (tolerance grades, surface finishes, etc.) --------

    def _build_properties(self) -> None:
        # Tolerance grades
        tolerance_grades = [
            ("IT6", 0.016),
            ("IT7", 0.025),
            ("IT8", 0.039),
            ("IT9", 0.062),
            ("IT10", 0.100),
            ("IT11", 0.160),
            ("IT12", 0.250),
        ]
        for grade, value_mm in tolerance_grades:
            self.add_node(KnowledgeNode(
                id=f"property:{grade.lower()}",
                type="property",
                name=f"{grade}公差等级",
                aliases=[grade, grade.lower()],
                properties={"tolerance_mm_for_25mm": value_mm, "category": "tolerance"},
            ))

        # Surface finishes
        surface_finishes = [
            ("Ra0.8", 0.8, "mirror-like"),
            ("Ra1.6", 1.6, "smooth"),
            ("Ra3.2", 3.2, "fine"),
            ("Ra6.3", 6.3, "medium"),
            ("Ra12.5", 12.5, "rough"),
        ]
        for name, value, desc in surface_finishes:
            self.add_node(KnowledgeNode(
                id=f"property:{name.lower()}",
                type="property",
                name=f"{name}表面粗糙度",
                aliases=[name, name.lower(), f"表面粗糙度{name}"],
                properties={
                    "ra_um": value,
                    "description": desc,
                    "category": "surface_finish",
                },
            ))

        # Complexity levels
        for level in ("high", "medium", "low"):
            self.add_node(KnowledgeNode(
                id=f"property:complexity_{level}",
                type="property",
                name=f"{level}复杂度",
                aliases=[f"{level} complexity", f"{level}复杂度"],
                properties={"category": "complexity", "level": level},
            ))

    # ==================================================================
    # Edge construction
    # ==================================================================

    def _build_material_process_edges(self) -> None:
        """material -> suitable_for -> process"""
        # Each tuple: (material_id, process_id, weight, conditions)
        mappings = [
            # Q235 carbon steel
            ("material:q235", "process:cnc_turning", 0.9, {}),
            ("material:q235", "process:cnc_milling", 0.9, {}),
            ("material:q235", "process:drilling", 0.9, {}),
            ("material:q235", "process:casting", 0.7, {}),
            ("material:q235", "process:forging", 0.8, {}),
            ("material:q235", "process:sheet_metal", 0.8, {}),
            ("material:q235", "process:wire_edm", 0.6, {}),
            # 45# steel
            ("material:45steel", "process:cnc_turning", 0.9, {}),
            ("material:45steel", "process:cnc_milling", 0.9, {}),
            ("material:45steel", "process:grinding", 0.8, {"note": "after heat treatment"}),
            ("material:45steel", "process:forging", 0.9, {}),
            ("material:45steel", "process:wire_edm", 0.7, {}),
            ("material:45steel", "process:drilling", 0.8, {}),
            # SUS304 stainless
            ("material:sus304", "process:cnc_turning", 0.8, {"note": "use low speed, high feed"}),
            ("material:sus304", "process:cnc_milling", 0.8, {}),
            ("material:sus304", "process:5axis", 0.7, {}),
            ("material:sus304", "process:wire_edm", 0.7, {}),
            ("material:sus304", "process:grinding", 0.6, {}),
            ("material:sus304", "process:casting", 0.6, {"note": "investment casting"}),
            ("material:sus304", "process:drilling", 0.7, {}),
            # SUS316 stainless
            ("material:sus316", "process:cnc_turning", 0.7, {}),
            ("material:sus316", "process:cnc_milling", 0.7, {}),
            ("material:sus316", "process:5axis", 0.7, {}),
            ("material:sus316", "process:wire_edm", 0.6, {}),
            ("material:sus316", "process:casting", 0.6, {}),
            # 6061 aluminum
            ("material:al6061", "process:cnc_turning", 0.95, {}),
            ("material:al6061", "process:cnc_milling", 0.95, {}),
            ("material:al6061", "process:5axis", 0.9, {}),
            ("material:al6061", "process:drilling", 0.9, {}),
            ("material:al6061", "process:wire_edm", 0.5, {}),
            ("material:al6061", "process:casting", 0.7, {"note": "die casting"}),
            ("material:al6061", "process:3d_printing", 0.6, {"note": "SLM"}),
            # 7075 aluminum
            ("material:al7075", "process:cnc_turning", 0.9, {}),
            ("material:al7075", "process:cnc_milling", 0.9, {}),
            ("material:al7075", "process:5axis", 0.9, {}),
            ("material:al7075", "process:grinding", 0.7, {}),
            ("material:al7075", "process:wire_edm", 0.5, {}),
            # TC4 titanium
            ("material:tc4", "process:cnc_turning", 0.6, {"note": "requires rigid setup, low speed"}),
            ("material:tc4", "process:cnc_milling", 0.6, {}),
            ("material:tc4", "process:5axis", 0.8, {"note": "preferred for complex Ti parts"}),
            ("material:tc4", "process:wire_edm", 0.7, {}),
            ("material:tc4", "process:grinding", 0.7, {}),
            ("material:tc4", "process:3d_printing", 0.8, {"note": "SLM/EBM popular for Ti"}),
            ("material:tc4", "process:forging", 0.7, {}),
            # ABS plastic
            ("material:abs", "process:cnc_milling", 0.8, {}),
            ("material:abs", "process:3d_printing", 0.95, {"note": "FDM primary material"}),
            ("material:abs", "process:cnc_turning", 0.6, {}),
            # PA66 nylon
            ("material:pa66", "process:cnc_turning", 0.7, {}),
            ("material:pa66", "process:cnc_milling", 0.7, {}),
            ("material:pa66", "process:3d_printing", 0.9, {"note": "SLS popular for nylon"}),
            # Brass H62
            ("material:h62", "process:cnc_turning", 0.95, {}),
            ("material:h62", "process:cnc_milling", 0.9, {}),
            ("material:h62", "process:drilling", 0.9, {}),
            ("material:h62", "process:wire_edm", 0.8, {}),
            ("material:h62", "process:casting", 0.7, {}),
            # 40Cr alloy steel
            ("material:40cr", "process:cnc_turning", 0.85, {}),
            ("material:40cr", "process:cnc_milling", 0.8, {}),
            ("material:40cr", "process:grinding", 0.9, {"note": "after quenching"}),
            ("material:40cr", "process:forging", 0.9, {}),
            ("material:40cr", "process:wire_edm", 0.7, {}),
        ]
        for mat_id, proc_id, weight, props in mappings:
            self.add_edge(KnowledgeEdge(
                source=mat_id,
                target=proc_id,
                relation="suitable_for",
                weight=weight,
                properties=props,
            ))

    def _build_part_material_edges(self) -> None:
        """part_type -> commonly_made_from -> material"""
        mappings = [
            # Flange
            ("part:flange", "material:q235", 0.9, {}),
            ("part:flange", "material:45steel", 0.8, {}),
            ("part:flange", "material:sus304", 0.8, {"note": "corrosion resistant flange"}),
            ("part:flange", "material:sus316", 0.7, {"note": "chemical industry"}),
            # Shaft
            ("part:shaft", "material:45steel", 0.95, {}),
            ("part:shaft", "material:40cr", 0.9, {"note": "high-load shafts"}),
            ("part:shaft", "material:sus304", 0.6, {"note": "corrosion environment"}),
            # Housing
            ("part:housing", "material:al6061", 0.9, {}),
            ("part:housing", "material:q235", 0.7, {}),
            ("part:housing", "material:abs", 0.7, {"note": "prototype housings"}),
            ("part:housing", "material:al7075", 0.7, {}),
            # Bracket
            ("part:bracket", "material:q235", 0.9, {}),
            ("part:bracket", "material:al6061", 0.8, {}),
            ("part:bracket", "material:sus304", 0.6, {}),
            # Gear
            ("part:gear", "material:45steel", 0.9, {}),
            ("part:gear", "material:40cr", 0.95, {}),
            ("part:gear", "material:pa66", 0.6, {"note": "low-load plastic gears"}),
            # Connector
            ("part:connector", "material:sus304", 0.85, {}),
            ("part:connector", "material:h62", 0.8, {}),
            ("part:connector", "material:al6061", 0.6, {}),
            # Seal
            ("part:seal", "material:pa66", 0.7, {}),
            ("part:seal", "material:h62", 0.5, {}),
            ("part:seal", "material:abs", 0.4, {}),
        ]
        for part_id, mat_id, weight, props in mappings:
            self.add_edge(KnowledgeEdge(
                source=part_id,
                target=mat_id,
                relation="commonly_made_from",
                weight=weight,
                properties=props,
            ))

    def _build_process_property_edges(self) -> None:
        """process -> produces -> surface_finish and process -> achieves -> tolerance"""
        # Surface finish capabilities
        surface_map = [
            ("process:cnc_turning", ["ra1.6", "ra3.2", "ra6.3"]),
            ("process:cnc_milling", ["ra1.6", "ra3.2", "ra6.3"]),
            ("process:5axis", ["ra0.8", "ra1.6", "ra3.2"]),
            ("process:grinding", ["ra0.8", "ra1.6"]),
            ("process:wire_edm", ["ra0.8", "ra1.6", "ra3.2"]),
            ("process:casting", ["ra6.3", "ra12.5"]),
            ("process:forging", ["ra6.3", "ra12.5"]),
            ("process:3d_printing", ["ra6.3", "ra12.5"]),
            ("process:drilling", ["ra3.2", "ra6.3"]),
            ("process:sheet_metal", ["ra3.2", "ra6.3", "ra12.5"]),
        ]
        for proc_id, finishes in surface_map:
            for f in finishes:
                # Lower Ra value = better finish, assign higher weight
                ra_val = float(f.replace("ra", ""))
                weight = max(0.3, 1.0 - ra_val / 20.0)
                self.add_edge(KnowledgeEdge(
                    source=proc_id,
                    target=f"property:{f}",
                    relation="produces",
                    weight=round(weight, 2),
                ))

        # Tolerance capabilities
        tolerance_map = [
            ("process:cnc_turning", ["it7", "it8", "it9"]),
            ("process:cnc_milling", ["it7", "it8", "it9"]),
            ("process:5axis", ["it6", "it7", "it8"]),
            ("process:grinding", ["it6", "it7"]),
            ("process:wire_edm", ["it7", "it8"]),
            ("process:casting", ["it10", "it11", "it12"]),
            ("process:forging", ["it9", "it10", "it11"]),
            ("process:3d_printing", ["it9", "it10", "it11"]),
            ("process:drilling", ["it9", "it10"]),
            ("process:sheet_metal", ["it10", "it11", "it12"]),
        ]
        for proc_id, grades in tolerance_map:
            for g in grades:
                # Lower IT number = tighter tolerance = higher weight
                it_num = int(g.replace("it", ""))
                weight = max(0.3, 1.0 - (it_num - 6) / 10.0)
                self.add_edge(KnowledgeEdge(
                    source=proc_id,
                    target=f"property:{g}",
                    relation="achieves",
                    weight=round(weight, 2),
                ))

    def _build_part_process_edges(self) -> None:
        """part_type -> typically_requires -> process"""
        mappings = [
            ("part:flange", "process:cnc_turning", 0.9, {}),
            ("part:flange", "process:cnc_milling", 0.7, {"note": "bolt holes"}),
            ("part:flange", "process:drilling", 0.8, {}),
            ("part:shaft", "process:cnc_turning", 0.95, {}),
            ("part:shaft", "process:grinding", 0.8, {"note": "bearing seats"}),
            ("part:housing", "process:cnc_milling", 0.9, {}),
            ("part:housing", "process:5axis", 0.7, {"note": "complex cavities"}),
            ("part:housing", "process:casting", 0.6, {"note": "large batches"}),
            ("part:bracket", "process:cnc_milling", 0.8, {}),
            ("part:bracket", "process:sheet_metal", 0.8, {}),
            ("part:gear", "process:cnc_milling", 0.7, {}),
            ("part:gear", "process:wire_edm", 0.8, {"note": "gear tooth profile"}),
            ("part:gear", "process:grinding", 0.9, {"note": "gear finishing"}),
            ("part:connector", "process:cnc_turning", 0.9, {}),
            ("part:connector", "process:drilling", 0.7, {}),
            ("part:seal", "process:cnc_turning", 0.7, {}),
            ("part:seal", "process:3d_printing", 0.6, {"note": "prototype seals"}),
        ]
        for part_id, proc_id, weight, props in mappings:
            self.add_edge(KnowledgeEdge(
                source=part_id,
                target=proc_id,
                relation="typically_requires",
                weight=weight,
                properties=props,
            ))

    def _build_material_property_edges(self) -> None:
        """material -> has_property -> property (machinability, cost tier, etc.)"""
        # Map machinability to a complexity-like property
        machinability_map = {
            "excellent": "property:complexity_low",
            "good": "property:complexity_low",
            "moderate": "property:complexity_medium",
            "poor": "property:complexity_high",
        }
        for node in self.find_nodes_by_type("material"):
            mac = node.properties.get("machinability", "moderate")
            prop_id = machinability_map.get(mac, "property:complexity_medium")
            weight = {"excellent": 0.95, "good": 0.85, "moderate": 0.6, "poor": 0.4}.get(mac, 0.5)
            self.add_edge(KnowledgeEdge(
                source=node.id,
                target=prop_id,
                relation="has_property",
                weight=weight,
                properties={"aspect": "machinability_complexity"},
            ))
