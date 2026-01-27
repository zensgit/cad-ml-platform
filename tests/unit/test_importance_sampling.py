from __future__ import annotations

import os

import pytest


def _build_dense_doc():
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Add multiple lines
    for i in range(6):
        msp.add_line((i, 0), (i + 1, 0))

    # Add circle and text to be prioritized
    msp.add_circle((5, 5), 2)
    msp.add_text("TITLE", dxfattribs={"height": 2, "insert": (9, 1)})

    return msp


def test_importance_sampler_respects_max_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from src.ml.importance_sampler import reset_importance_sampler
    from src.ml.train.dataset_2d import DXFDataset, DXF_NODE_FEATURES, DXF_NODE_DIM

    monkeypatch.setenv("DXF_MAX_NODES", "3")
    monkeypatch.setenv("DXF_SAMPLING_STRATEGY", "importance")
    monkeypatch.setenv("DXF_TEXT_PRIORITY_RATIO", "0.5")
    reset_importance_sampler()

    msp = _build_dense_doc()
    dataset = DXFDataset(root_dir=".", node_dim=DXF_NODE_DIM)
    x, edge_index = dataset._dxf_to_graph(msp, node_dim=DXF_NODE_DIM)

    assert x.shape[0] <= 3
    assert edge_index.shape[0] == 2

    idx_text = DXF_NODE_FEATURES.index("is_text")
    assert (x[:, idx_text] > 0.5).any()

    # Cleanup env to avoid affecting other tests
    monkeypatch.delenv("DXF_MAX_NODES", raising=False)
    monkeypatch.delenv("DXF_SAMPLING_STRATEGY", raising=False)
    monkeypatch.delenv("DXF_TEXT_PRIORITY_RATIO", raising=False)
