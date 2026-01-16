"""Integration coverage for B-Rep face adjacency graph extraction."""

from __future__ import annotations

from typing import Any

import pytest

from src.core.geometry.engine import (
    BREP_GRAPH_EDGE_FEATURES,
    BREP_GRAPH_NODE_FEATURES,
    HAS_OCC,
    get_geometry_engine,
)

pytestmark = pytest.mark.skipif(not HAS_OCC, reason="pythonocc-core not installed")


def _make_box() -> Any:
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

    return BRepPrimAPI_MakeBox(10, 20, 30).Shape()


def test_extract_brep_graph_from_box() -> None:
    engine = get_geometry_engine()
    graph = engine.extract_brep_graph(_make_box())

    assert graph["valid_3d"] is True
    assert graph["node_schema"] == BREP_GRAPH_NODE_FEATURES
    assert graph["edge_schema"] == BREP_GRAPH_EDGE_FEATURES
    assert graph["node_count"] == len(graph["node_features"])
    assert graph["edge_count"] == len(graph["edge_features"])

    node_dim = len(BREP_GRAPH_NODE_FEATURES)
    assert node_dim > 0
    assert graph["node_count"] == 6
    for node in graph["node_features"]:
        assert len(node) == node_dim
        assert node[0] == pytest.approx(1.0)
        assert sum(node[:8]) == pytest.approx(1.0)

    assert len(graph["edge_index"]) == len(graph["edge_features"])
    for src, dst in graph["edge_index"]:
        assert 0 <= src < graph["node_count"]
        assert 0 <= dst < graph["node_count"]
