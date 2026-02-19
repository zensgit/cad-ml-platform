from __future__ import annotations

import ezdxf
import pytest

pytest.importorskip("torch")
from src.ml.train.dataset_2d import DXFDataset, DXF_NODE_DIM


def test_dxf_graph_knn_fallback_bounds_edge_count(monkeypatch):
    """When no edges are found, kNN fallback should avoid fully-connected blowups."""
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Create 5 disconnected entities far apart so epsilon adjacency yields no edges.
    for i in range(5):
        x = float(i * 1000.0)
        msp.add_line((x, 0.0), (x + 10.0, 0.0))

    monkeypatch.setenv("DXF_EMPTY_EDGE_FALLBACK", "knn")
    monkeypatch.setenv("DXF_EMPTY_EDGE_K", "1")

    dataset = DXFDataset(root_dir=".", node_dim=DXF_NODE_DIM, return_edge_attr=False)
    x, edge_index = dataset._dxf_to_graph(msp, DXF_NODE_DIM)

    assert x.shape[0] == 5
    assert edge_index.shape[0] == 2
    # Fully connected would be N*(N-1)=20 directed edges. kNN should be much smaller.
    assert 0 < int(edge_index.shape[1]) < 20
