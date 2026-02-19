import ezdxf
import pytest

pytest.importorskip("torch")
from src.ml.train.dataset_2d import DXFDataset


class _SampleResult:
    def __init__(self, sampled_entities):
        self.sampled_entities = sampled_entities


class _IdentitySampler:
    def sample(self, entities):
        return _SampleResult(entities)


def test_edge_augment_knn_adds_edges_for_isolated_nodes(monkeypatch):
    # Keep entity order stable and avoid sampling variance.
    import src.ml.importance_sampler as importance_sampler

    monkeypatch.setattr(
        importance_sampler, "get_importance_sampler", lambda: _IdentitySampler()
    )

    doc = ezdxf.new()
    msp = doc.modelspace()

    # Two connected lines (share an endpoint) + one far-away entity that would be isolated.
    msp.add_line((0, 0), (1, 0))
    msp.add_line((1, 0), (2, 0))
    msp.add_circle((100, 100), radius=1.0)

    dataset = DXFDataset(root_dir=".")

    monkeypatch.delenv("DXF_EDGE_AUGMENT_KNN_K", raising=False)
    x_base, edge_base = dataset._dxf_to_graph(msp)
    assert int(x_base.size(0)) == 3
    nodes_in_edges_base = set(edge_base.flatten().tolist()) if edge_base.numel() else set()
    assert 2 not in nodes_in_edges_base

    monkeypatch.setenv("DXF_EDGE_AUGMENT_KNN_K", "1")
    x_aug, edge_aug = dataset._dxf_to_graph(msp)
    assert int(x_aug.size(0)) == 3
    nodes_in_edges_aug = set(edge_aug.flatten().tolist()) if edge_aug.numel() else set()
    assert 2 in nodes_in_edges_aug
    assert int(edge_aug.size(1)) > int(edge_base.size(1))
