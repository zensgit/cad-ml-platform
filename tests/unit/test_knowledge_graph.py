"""Tests for the manufacturing knowledge graph and query engine."""

from __future__ import annotations

import pytest

from src.ml.knowledge.graph import (
    KnowledgeEdge,
    KnowledgeNode,
    ManufacturingKnowledgeGraph,
)
from src.ml.knowledge.query_engine import GraphQueryEngine, QueryResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> ManufacturingKnowledgeGraph:
    g = ManufacturingKnowledgeGraph()
    g.build_default_graph()
    return g


@pytest.fixture()
def engine(graph: ManufacturingKnowledgeGraph) -> GraphQueryEngine:
    return GraphQueryEngine(graph)


# ---------------------------------------------------------------------------
# Graph structure tests
# ---------------------------------------------------------------------------


class TestGraphStructure:
    def test_build_default_graph_has_nodes(self, graph: ManufacturingKnowledgeGraph):
        """After build_default_graph, there should be at least 30 nodes."""
        stats = graph.get_stats()
        assert stats["node_count"] >= 30, f"Expected >=30 nodes, got {stats['node_count']}"

    def test_build_default_graph_has_edges(self, graph: ManufacturingKnowledgeGraph):
        """After build_default_graph, there should be at least 80 edges."""
        stats = graph.get_stats()
        assert stats["edge_count"] >= 80, f"Expected >=80 edges, got {stats['edge_count']}"

    def test_graph_stats(self, graph: ManufacturingKnowledgeGraph):
        stats = graph.get_stats()
        assert "node_count" in stats
        assert "edge_count" in stats
        assert "nodes_by_type" in stats
        assert "edges_by_relation" in stats
        # Should have at least 4 node types
        assert len(stats["nodes_by_type"]) >= 4
        # Check specific types exist
        assert stats["nodes_by_type"].get("material", 0) >= 10
        assert stats["nodes_by_type"].get("process", 0) >= 8
        assert stats["nodes_by_type"].get("part_type", 0) >= 8
        assert stats["nodes_by_type"].get("property", 0) >= 10

    def test_find_material_processes(self, graph: ManufacturingKnowledgeGraph):
        """SUS304 -> suitable_for -> should return processes."""
        neighbors = graph.get_neighbors("material:sus304", relation="suitable_for")
        assert len(neighbors) >= 3
        proc_names = {n.name for n, _ in neighbors}
        assert "CNC车削" in proc_names
        assert "CNC铣削" in proc_names

    def test_find_part_materials(self, graph: ManufacturingKnowledgeGraph):
        """法兰盘 -> commonly_made_from -> should return materials."""
        neighbors = graph.get_neighbors("part:flange", relation="commonly_made_from")
        assert len(neighbors) >= 2
        mat_ids = {n.id for n, _ in neighbors}
        assert "material:q235" in mat_ids

    def test_find_node_by_alias(self, graph: ManufacturingKnowledgeGraph):
        """'304不锈钢' should resolve to 'material:sus304'."""
        node = graph.find_node_by_alias("304不锈钢")
        assert node is not None
        assert node.id == "material:sus304"

    def test_find_node_by_alias_case_insensitive(self, graph: ManufacturingKnowledgeGraph):
        node = graph.find_node_by_alias("aisi 304")
        assert node is not None
        assert node.id == "material:sus304"

    def test_query_path_exists(self, graph: ManufacturingKnowledgeGraph):
        """There should be a path from Q235 to Ra3.2 within 3 hops."""
        paths = graph.query_path("material:q235", "property:ra3.2", max_hops=3)
        assert len(paths) >= 1
        # Each path should start with q235 and end with ra3.2
        for path in paths:
            assert path[0] == "material:q235"
            assert path[-1] == "property:ra3.2"
            assert len(path) <= 4  # max 3 hops = 4 nodes

    def test_to_dict_round_trip(self, graph: ManufacturingKnowledgeGraph):
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "stats" in d
        assert len(d["nodes"]) == d["stats"]["node_count"]
        assert len(d["edges"]) == d["stats"]["edge_count"]

    def test_get_reverse_neighbors(self, graph: ManufacturingKnowledgeGraph):
        """Processes pointing to Ra1.6 via 'produces'."""
        rev = graph.get_reverse_neighbors("property:ra1.6", relation="produces")
        assert len(rev) >= 2
        proc_names = {n.name for n, _ in rev}
        # CNC turning and milling should both produce Ra1.6
        assert "CNC车削" in proc_names or "CNC铣削" in proc_names


# ---------------------------------------------------------------------------
# Query engine tests
# ---------------------------------------------------------------------------


class TestQueryEngine:
    def test_optimal_process_recommendation(self, engine: GraphQueryEngine):
        """part=法兰盘, material=SUS304 -> ranked processes."""
        results = engine.find_optimal_process(part_type="法兰盘", material="SUS304")
        assert len(results) >= 1
        # Each result should have expected keys
        for r in results:
            assert "process" in r
            assert "score" in r
            assert "cost_per_hour" in r
        # CNC turning should appear (flange is rotational + SUS304 supports it)
        proc_names = {r["process"] for r in results}
        assert "CNC车削" in proc_names

    def test_optimal_process_with_surface_filter(self, engine: GraphQueryEngine):
        """Filter by Ra1.6 surface finish."""
        results = engine.find_optimal_process(
            part_type="法兰盘", material="SUS304", surface="Ra1.6",
        )
        # Results should only contain processes capable of Ra1.6
        assert len(results) >= 1

    def test_alternative_materials(self, engine: GraphQueryEngine):
        """current=SUS304, cheaper=True -> should return cheaper options."""
        alts = engine.find_alternative_materials("SUS304", requirements={"cheaper": True})
        assert len(alts) >= 1
        # All alternatives must be cheaper than SUS304 (22.0 CNY/kg)
        for alt in alts:
            assert alt["price_per_kg"] < 22.0

    def test_alternative_materials_no_filter(self, engine: GraphQueryEngine):
        alts = engine.find_alternative_materials("SUS304")
        # Should return all other materials
        assert len(alts) >= 5

    def test_explain_relationship(self, engine: GraphQueryEngine):
        """SUS304 and Ra1.6 -> explanation should mention a process."""
        explanation = engine.explain_relationship("SUS304", "Ra1.6")
        # Should contain Chinese text connecting the two
        assert "SUS304" in explanation or "不锈钢" in explanation
        assert "Ra1.6" in explanation or "粗糙度" in explanation

    def test_explain_relationship_missing_entity(self, engine: GraphQueryEngine):
        result = engine.explain_relationship("不存在的材料", "Ra1.6")
        assert "未找到" in result

    def test_query_engine_natural_language_material_process(self, engine: GraphQueryEngine):
        """'SUS304适合什么工艺' -> answer should mention CNC."""
        result = engine.query("SUS304适合什么加工工艺？")
        assert isinstance(result, QueryResult)
        assert "CNC" in result.answer or "车削" in result.answer or "铣削" in result.answer
        assert result.confidence > 0

    def test_query_engine_part_materials(self, engine: GraphQueryEngine):
        result = engine.query("哪些材料适合做法兰盘？")
        assert "Q235" in result.answer or "碳钢" in result.answer
        assert result.confidence > 0

    def test_query_engine_process_precision(self, engine: GraphQueryEngine):
        result = engine.query("CNC车削能达到什么精度？")
        assert "公差" in result.answer or "粗糙度" in result.answer

    def test_query_engine_multi_hop(self, engine: GraphQueryEngine):
        result = engine.query("法兰盘用SUS304做，推荐什么工艺？")
        assert "CNC" in result.answer or "车削" in result.answer
        assert len(result.reasoning) >= 1

    def test_query_engine_unknown_question(self, engine: GraphQueryEngine):
        result = engine.query("今天天气怎么样？")
        assert result.confidence <= 0.5
