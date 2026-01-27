"""Unit coverage for DXF graph feature extraction."""

from __future__ import annotations

import pytest


def _build_sample_doc():
    pytest.importorskip("torch")
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    msp = doc.modelspace()

    msp.add_line((0, 0), (10, 0))
    msp.add_circle((5, 5), 2)
    msp.add_arc((0, 0), 3, 0, 90)
    msp.add_lwpolyline([(0, 0), (0, 1), (1, 1)], close=True)
    msp.add_text("NOTE", dxfattribs={"height": 2, "insert": (1, 1)})
    dim = msp.add_linear_dim(base=(0, 0), p1=(0, 0), p2=(10, 0), angle=0)
    dim.render()
    block = doc.blocks.new("B1")
    block.add_line((0, 0), (1, 0))
    msp.add_blockref("B1", (2, 2))

    return doc, msp


def test_dxf_graph_supports_extended_types() -> None:
    _doc, msp = _build_sample_doc()
    from src.ml.train.dataset_2d import DXFDataset, DXF_NODE_DIM, DXF_NODE_FEATURES

    dataset = DXFDataset(root_dir=".", node_dim=DXF_NODE_DIM)
    x, edge_index = dataset._dxf_to_graph(msp, node_dim=DXF_NODE_DIM)

    assert x.shape[1] == DXF_NODE_DIM
    assert edge_index.shape[0] == 2

    idx = {name: i for i, name in enumerate(DXF_NODE_FEATURES)}
    assert (x[:, idx["is_line"]] > 0.5).any()
    assert (x[:, idx["is_circle"]] > 0.5).any()
    assert (x[:, idx["is_arc"]] > 0.5).any()
    assert (x[:, idx["is_polyline"]] > 0.5).any()
    assert (x[:, idx["is_text"]] > 0.5).any()
    assert (x[:, idx["is_dimension"]] > 0.5).any()
    assert (x[:, idx["is_insert"]] > 0.5).any()
    assert (x[:, idx["text_density"]] >= 0.0).all()


def test_dxf_graph_legacy_layout_shape() -> None:
    _doc, msp = _build_sample_doc()
    from src.ml.train.dataset_2d import DXFDataset, DXF_NODE_FEATURES_LEGACY

    legacy_dim = len(DXF_NODE_FEATURES_LEGACY)
    dataset = DXFDataset(root_dir=".", node_dim=legacy_dim)
    x, _edge_index = dataset._dxf_to_graph(msp, node_dim=legacy_dim)

    assert x.shape[1] == legacy_dim
    idx_line = DXF_NODE_FEATURES_LEGACY.index("is_line")
    assert (x[:, idx_line] > 0.0).any()


def test_dxf_graph_edge_attr_shape() -> None:
    _doc, msp = _build_sample_doc()
    from src.ml.train.dataset_2d import DXFDataset, DXF_EDGE_DIM, DXF_NODE_DIM

    dataset = DXFDataset(root_dir=".", node_dim=DXF_NODE_DIM, return_edge_attr=True)
    x, edge_index, edge_attr = dataset._dxf_to_graph(
        msp, node_dim=DXF_NODE_DIM, return_edge_attr=True
    )

    assert x.shape[1] == DXF_NODE_DIM
    assert edge_index.shape[0] == 2
    assert edge_attr.shape[0] == edge_index.shape[1]
    assert edge_attr.shape[1] == DXF_EDGE_DIM
