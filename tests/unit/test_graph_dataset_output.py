"""Unit coverage for graph dataset output formatting."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.core.geometry.engine import BREP_GRAPH_EDGE_FEATURES, BREP_GRAPH_NODE_FEATURES
from src.ml.train.dataset import ABCDataset


class _StubGeometryEngine:
    def load_step(self, content: bytes, file_name: str) -> Any:
        return object()

    def extract_brep_graph(self, shape: Any) -> dict[str, Any]:
        return {
            "valid_3d": True,
            "graph_schema_version": "v1",
            "node_schema": BREP_GRAPH_NODE_FEATURES,
            "edge_schema": BREP_GRAPH_EDGE_FEATURES,
            "node_features": [[1.0] * len(BREP_GRAPH_NODE_FEATURES)],
            "edge_index": [],
            "edge_features": [],
        }


def test_dataset_graph_output_dict(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    step_path = tmp_path / "sample.step"
    step_path.write_bytes(b"solid")

    monkeypatch.setattr(
        "src.core.geometry.engine.get_geometry_engine",
        lambda: _StubGeometryEngine(),
    )

    dataset = ABCDataset(
        str(tmp_path),
        output_format="graph",
        graph_backend="dict",
    )

    sample, label = dataset[0]

    assert isinstance(sample, dict)
    assert sample["graph_schema_version"] == "v1"
    assert sample["x"].shape == (1, len(BREP_GRAPH_NODE_FEATURES))
    assert sample["edge_index"].shape == (2, 0)
    assert sample["edge_attr"].shape == (0, len(BREP_GRAPH_EDGE_FEATURES))
    assert isinstance(label, int)
