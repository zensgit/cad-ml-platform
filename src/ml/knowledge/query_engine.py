"""
Graph Query Engine for manufacturing knowledge graph.

Provides natural-language query parsing and multi-hop graph traversal to
answer questions like:

    "SUS304适合什么加工工艺？"
    "哪些材料适合做法兰盘？"
    "法兰盘用SUS304做，推荐什么工艺？"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.ml.knowledge.graph import ManufacturingKnowledgeGraph, KnowledgeNode


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Structured result from a graph query."""

    answer: str  # Natural language answer (Chinese)
    entities: List[dict] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------

class GraphQueryEngine:
    """Natural language query engine over the manufacturing knowledge graph."""

    def __init__(self, graph: ManufacturingKnowledgeGraph) -> None:
        self.graph = graph

    # ==================================================================
    # Public API
    # ==================================================================

    def query(self, question: str) -> QueryResult:
        """Answer a natural-language question using graph traversal.

        Supported patterns
        ------------------
        1. "<material>适合什么加工工艺？"
        2. "哪些材料适合做<part_type>？"
        3. "<process>能达到什么精度/表面粗糙度？"
        4. "<property>需要什么加工方式？"
        5. "<part_type>用<material>做，推荐什么工艺？"  (multi-hop)
        """
        q = question.strip()

        # Pattern 5: multi-hop  "法兰盘用SUS304做，推荐什么工艺"
        match = re.search(r"(.+?)用(.+?)做.*(?:推荐|什么).*(?:工艺|加工)", q)
        if match:
            part_str, mat_str = match.group(1).strip(), match.group(2).strip()
            result = self.find_optimal_process(part_type=part_str, material=mat_str)
            if result:
                names = [r["process"] for r in result]
                return QueryResult(
                    answer=f"{part_str}使用{mat_str}制造，推荐工艺：{'、'.join(names)}。",
                    entities=[r for r in result],
                    confidence=max((r.get("score", 0) for r in result), default=0),
                    reasoning=[
                        f"查找材料 {mat_str} 适合的加工工艺",
                        f"查找零件 {part_str} 常用的加工工艺",
                        "取交集并按综合得分排序",
                    ],
                )

        # Pattern 1: material -> processes
        match = re.search(r"(.+?)适合什么.*(?:工艺|加工)", q)
        if match:
            entity_str = match.group(1).strip()
            node = self._resolve_node(entity_str)
            if node and node.type == "material":
                neighbors = self.graph.get_neighbors(node.id, relation="suitable_for")
                neighbors.sort(key=lambda x: x[1].weight, reverse=True)
                names = [n.name for n, _ in neighbors]
                return QueryResult(
                    answer=f"{node.name}适合的加工工艺有：{'、'.join(names)}。",
                    entities=[{"id": n.id, "name": n.name, "weight": e.weight} for n, e in neighbors],
                    confidence=neighbors[0][1].weight if neighbors else 0,
                    reasoning=[f"从 {node.name} 出发沿 suitable_for 边查找工艺节点"],
                )

        # Pattern 2: "哪些材料适合做<part>" or "<part>常用什么材料"
        match = re.search(r"(?:哪些|什么)材料.*(?:适合|可以).*(?:做|制造|加工)(.+)", q)
        if not match:
            match = re.search(r"(.+?)(?:常用|通常用|一般用|适合用|需要|用)(?:什么|哪些|哪种)材料", q)
        if match:
            part_str = match.group(1).strip().rstrip("？?。")
            node = self._resolve_node(part_str)
            if node and node.type == "part_type":
                neighbors = self.graph.get_neighbors(node.id, relation="commonly_made_from")
                neighbors.sort(key=lambda x: x[1].weight, reverse=True)
                names = [n.name for n, _ in neighbors]
                return QueryResult(
                    answer=f"适合制造{node.name}的材料有：{'、'.join(names)}。",
                    entities=[{"id": n.id, "name": n.name, "weight": e.weight} for n, e in neighbors],
                    confidence=neighbors[0][1].weight if neighbors else 0,
                    reasoning=[f"从零件 {node.name} 沿 commonly_made_from 边查找材料"],
                )

        # Pattern 2b: "<part>用什么材料做" / "<part>材料"
        match = re.search(r"(.+?)(?:用什么|选什么|选哪种)(?:材料|材质)", q)
        if not match:
            match = re.search(r"(.+?)的(?:常用)?材料(?:是什么|有哪些)?", q)
        if match:
            part_str = match.group(1).strip().rstrip("？?。")
            node = self._resolve_node(part_str)
            if node and node.type == "part_type":
                neighbors = self.graph.get_neighbors(node.id, relation="commonly_made_from")
                neighbors.sort(key=lambda x: x[1].weight, reverse=True)
                names = [n.name for n, _ in neighbors]
                if names:
                    return QueryResult(
                        answer=f"适合制造{node.name}的材料有：{'、'.join(names)}。",
                        entities=[{"id": n.id, "name": n.name, "weight": e.weight} for n, e in neighbors],
                        confidence=neighbors[0][1].weight if neighbors else 0,
                        reasoning=[f"从零件 {node.name} 沿 commonly_made_from 边查找材料"],
                    )

        # Pattern 1b: "<material>能做什么" / "<material>适合加工什么零件"
        match = re.search(r"(.+?)(?:能做|适合做|可以做|适合加工)(?:什么|哪些)(?:零件|产品|工件)?", q)
        if match:
            mat_str = match.group(1).strip()
            node = self._resolve_node(mat_str)
            if node and node.type == "material":
                rev = self.graph.get_reverse_neighbors(node.id)
                parts = [(n, e) for n, e in rev if n.type == "part_type"]
                parts.sort(key=lambda x: x[1].weight, reverse=True)
                names = [n.name for n, _ in parts]
                if names:
                    return QueryResult(
                        answer=f"{node.name}适合制造的零件类型有：{'、'.join(names)}。",
                        entities=[{"id": n.id, "name": n.name} for n, _ in parts],
                        confidence=0.7,
                        reasoning=[f"反向查找使用 {node.name} 的零件类型"],
                    )

        # Pattern 3: "<process>能达到什么精度/粗糙度"
        match = re.search(r"(.+?)(?:能|可以)达到.*(?:精度|公差|粗糙度|表面)", q)
        if match:
            proc_str = match.group(1).strip()
            node = self._resolve_node(proc_str)
            if node and node.type == "process":
                achieves = self.graph.get_neighbors(node.id, relation="achieves")
                produces = self.graph.get_neighbors(node.id, relation="produces")
                tol_names = [n.name for n, _ in sorted(achieves, key=lambda x: x[1].weight, reverse=True)]
                surf_names = [n.name for n, _ in sorted(produces, key=lambda x: x[1].weight, reverse=True)]
                answer_parts = []
                if tol_names:
                    answer_parts.append(f"可达到的公差等级：{'、'.join(tol_names)}")
                if surf_names:
                    answer_parts.append(f"可达到的表面粗糙度：{'、'.join(surf_names)}")
                return QueryResult(
                    answer=f"{node.name}{'；'.join(answer_parts)}。",
                    entities=[{"id": n.id, "name": n.name} for n, _ in achieves + produces],
                    confidence=0.8,
                    reasoning=[f"从工艺 {node.name} 沿 achieves/produces 边查找属性"],
                )

        # Pattern 4: "<property>需要什么加工"
        match = re.search(r"(.+?)(?:需要|要求).*(?:什么|哪些).*(?:加工|工艺)", q)
        if match:
            prop_str = match.group(1).strip()
            node = self._resolve_node(prop_str)
            if node and node.type == "property":
                rev = self.graph.get_reverse_neighbors(node.id)
                processes = [(n, e) for n, e in rev if n.type == "process"]
                processes.sort(key=lambda x: x[1].weight, reverse=True)
                names = [n.name for n, _ in processes]
                return QueryResult(
                    answer=f"能达到{node.name}的加工工艺有：{'、'.join(names)}。",
                    entities=[{"id": n.id, "name": n.name} for n, _ in processes],
                    confidence=0.7,
                    reasoning=[f"反向查找指向 {node.name} 的工艺节点"],
                )

        # Fallback: try to resolve any entity and show its neighbors
        node = self._resolve_node(q.rstrip("？?。"))
        if node:
            all_neighbors = self.graph.get_neighbors(node.id)
            if all_neighbors:
                lines = [f"{node.name} 的关联信息："]
                for n, e in all_neighbors:
                    lines.append(f"  - ({e.relation}) → {n.name}")
                return QueryResult(
                    answer="\n".join(lines),
                    entities=[{"id": n.id, "name": n.name, "relation": e.relation} for n, e in all_neighbors],
                    confidence=0.5,
                    reasoning=["未匹配特定问题模式，返回实体的所有邻居节点"],
                )

        return QueryResult(
            answer="抱歉，无法理解该问题。请尝试更具体的制造相关问题。",
            confidence=0.0,
            reasoning=["未找到匹配的实体或问题模式"],
        )

    # ------------------------------------------------------------------
    # find_optimal_process
    # ------------------------------------------------------------------

    def find_optimal_process(
        self,
        part_type: str,
        material: str,
        tolerance: Optional[str] = None,
        surface: Optional[str] = None,
    ) -> List[dict]:
        """Multi-constraint process recommendation using graph traversal.

        1. Find material node -> get suitable processes
        2. Find part_type node -> get common processes
        3. Intersect -> filter by tolerance/surface capability
        4. Rank by cost efficiency
        """
        mat_node = self._resolve_node(material)
        part_node = self._resolve_node(part_type)

        if not mat_node:
            return []

        # Step 1: processes the material is suitable for
        mat_procs = self.graph.get_neighbors(mat_node.id, relation="suitable_for")
        mat_proc_map: Dict[str, float] = {n.id: e.weight for n, e in mat_procs}

        # Step 2: processes the part type typically requires
        part_proc_map: Dict[str, float] = {}
        if part_node:
            part_procs = self.graph.get_neighbors(part_node.id, relation="typically_requires")
            part_proc_map = {n.id: e.weight for n, e in part_procs}

        # Step 3: intersect (or just use material processes if no part match)
        if part_proc_map:
            candidate_ids = set(mat_proc_map) & set(part_proc_map)
            if not candidate_ids:
                # Relax: use material processes ranked lower
                candidate_ids = set(mat_proc_map)
        else:
            candidate_ids = set(mat_proc_map)

        # Step 4: filter by tolerance / surface if specified
        if tolerance:
            tol_node = self._resolve_node(tolerance)
            if tol_node:
                capable = {n.id for n, _ in self.graph.get_reverse_neighbors(tol_node.id, relation="achieves")}
                filtered = candidate_ids & capable
                if filtered:
                    candidate_ids = filtered

        if surface:
            surf_node = self._resolve_node(surface)
            if surf_node:
                capable = {n.id for n, _ in self.graph.get_reverse_neighbors(surf_node.id, relation="produces")}
                filtered = candidate_ids & capable
                if filtered:
                    candidate_ids = filtered

        # Build ranked result
        results: List[dict] = []
        for proc_id in candidate_ids:
            proc_node = self.graph.get_node(proc_id)
            if not proc_node:
                continue
            mat_score = mat_proc_map.get(proc_id, 0)
            part_score = part_proc_map.get(proc_id, 0)
            combined = mat_score * 0.5 + part_score * 0.5 if part_proc_map else mat_score
            cost = proc_node.properties.get("cost_per_hour", 50.0)
            results.append({
                "process": proc_node.name,
                "process_id": proc_id,
                "material_suitability": mat_score,
                "part_suitability": part_score,
                "score": round(combined, 3),
                "cost_per_hour": cost,
            })

        results.sort(key=lambda r: (-r["score"], r["cost_per_hour"]))
        return results

    # ------------------------------------------------------------------
    # find_alternative_materials
    # ------------------------------------------------------------------

    def find_alternative_materials(
        self,
        current_material: str,
        requirements: Optional[dict] = None,
    ) -> List[dict]:
        """Find alternative materials that satisfy similar requirements.

        Parameters
        ----------
        current_material : str
            Name or alias of the current material.
        requirements : dict, optional
            Filter keys: ``cheaper`` (bool), ``stronger`` (bool),
            ``corrosion_resistant`` (bool), ``machinability`` (str).
        """
        requirements = requirements or {}
        current_node = self._resolve_node(current_material)
        if not current_node:
            return []

        cur_props = current_node.properties
        all_materials = self.graph.find_nodes_by_type("material")

        alternatives: List[dict] = []
        for mat in all_materials:
            if mat.id == current_node.id:
                continue
            props = mat.properties
            tradeoffs: List[str] = []

            # cheaper filter
            if requirements.get("cheaper"):
                if props.get("price_per_kg", 999) >= cur_props.get("price_per_kg", 0):
                    continue
                saving = cur_props["price_per_kg"] - props["price_per_kg"]
                tradeoffs.append(f"每公斤便宜{saving:.1f}元")

            # stronger filter
            if requirements.get("stronger"):
                if props.get("tensile_strength_mpa", 0) <= cur_props.get("tensile_strength_mpa", 9999):
                    continue
                tradeoffs.append(f"抗拉强度更高({props['tensile_strength_mpa']}MPa)")

            # corrosion_resistant filter
            if requirements.get("corrosion_resistant"):
                rank = {"low": 0, "moderate": 1, "high": 2, "very_high": 3}
                if rank.get(props.get("corrosion_resistance", "low"), 0) < rank.get(
                    cur_props.get("corrosion_resistance", "low"), 0
                ):
                    continue
                tradeoffs.append(f"耐腐蚀性: {props.get('corrosion_resistance', '未知')}")

            # machinability filter
            req_mac = requirements.get("machinability")
            if req_mac:
                rank = {"poor": 0, "moderate": 1, "good": 2, "excellent": 3}
                if rank.get(props.get("machinability", "poor"), 0) < rank.get(req_mac, 0):
                    continue

            # Compute similarity score: shared suitable processes
            cur_procs = {n.id for n, _ in self.graph.get_neighbors(current_node.id, "suitable_for")}
            alt_procs = {n.id for n, _ in self.graph.get_neighbors(mat.id, "suitable_for")}
            overlap = len(cur_procs & alt_procs)
            total = len(cur_procs | alt_procs) if cur_procs or alt_procs else 1
            proc_similarity = overlap / total

            if not tradeoffs:
                # Provide a generic trade-off description
                price_diff = props.get("price_per_kg", 0) - cur_props.get("price_per_kg", 0)
                if price_diff < 0:
                    tradeoffs.append(f"每公斤便宜{abs(price_diff):.1f}元")
                elif price_diff > 0:
                    tradeoffs.append(f"每公斤贵{price_diff:.1f}元")

            alternatives.append({
                "material": mat.name,
                "material_id": mat.id,
                "price_per_kg": props.get("price_per_kg", 0),
                "tensile_strength_mpa": props.get("tensile_strength_mpa", 0),
                "machinability": props.get("machinability", "unknown"),
                "process_similarity": round(proc_similarity, 2),
                "tradeoffs": tradeoffs,
            })

        # Rank by process similarity descending
        alternatives.sort(key=lambda a: -a["process_similarity"])
        return alternatives

    # ------------------------------------------------------------------
    # explain_relationship
    # ------------------------------------------------------------------

    def explain_relationship(self, entity_a: str, entity_b: str) -> str:
        """Explain the relationship between two manufacturing entities.

        Uses path finding to connect entities and generates a Chinese
        natural-language explanation.
        """
        node_a = self._resolve_node(entity_a)
        node_b = self._resolve_node(entity_b)
        if not node_a or not node_b:
            missing = entity_a if not node_a else entity_b
            return f"未找到实体：{missing}"

        paths = self.graph.query_path(node_a.id, node_b.id, max_hops=3)
        if not paths:
            # Try reverse direction
            paths = self.graph.query_path(node_b.id, node_a.id, max_hops=3)

        if not paths:
            return f"未找到 {node_a.name} 和 {node_b.name} 之间的关联路径。"

        # Pick shortest path
        shortest = min(paths, key=len)

        # Build explanation
        segments: List[str] = []
        for i in range(len(shortest) - 1):
            src_id, tgt_id = shortest[i], shortest[i + 1]
            src_node = self.graph.get_node(src_id)
            tgt_node = self.graph.get_node(tgt_id)
            # Find the edge relation
            relation = ""
            for edge in self.graph._adjacency.get(src_id, []):
                if edge.target == tgt_id:
                    relation = edge.relation
                    break
            segments.append(self._relation_to_chinese(
                src_node.name if src_node else src_id,
                relation,
                tgt_node.name if tgt_node else tgt_id,
            ))

        return "，".join(segments) + "。"

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _resolve_node(self, text: str) -> Optional[KnowledgeNode]:
        """Resolve free text to a graph node via alias index or fuzzy match."""
        text = text.strip()
        # Direct alias lookup
        node = self.graph.find_node_by_alias(text)
        if node:
            return node
        # Try lowered
        node = self.graph.find_node_by_alias(text.lower())
        if node:
            return node
        # Try removing common suffixes
        for suffix in ("材料", "材质", "钢材", "合金", "工艺", "加工", "零件"):
            trimmed = text.rstrip(suffix)
            if trimmed != text:
                node = self.graph.find_node_by_alias(trimmed)
                if node:
                    return node
        # Substring match on aliases
        for nid, n in self.graph._nodes.items():
            if text.lower() in n.name.lower():
                return n
            for alias in n.aliases:
                if text.lower() in alias.lower():
                    return n
        return None

    @staticmethod
    def _relation_to_chinese(src: str, relation: str, tgt: str) -> str:
        """Convert a graph edge to a Chinese sentence fragment."""
        templates = {
            "suitable_for": f"{src}适合{tgt}",
            "commonly_made_from": f"{src}通常使用{tgt}制造",
            "produces": f"{src}可以达到{tgt}",
            "achieves": f"{src}可以实现{tgt}",
            "typically_requires": f"{src}通常需要{tgt}",
            "has_property": f"{src}具有{tgt}的特性",
        }
        return templates.get(relation, f"{src} → {relation} → {tgt}")
