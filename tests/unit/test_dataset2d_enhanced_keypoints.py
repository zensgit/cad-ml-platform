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


def test_enhanced_keypoints_connect_circle_to_touching_line(monkeypatch):
    import src.ml.importance_sampler as importance_sampler

    monkeypatch.setattr(
        importance_sampler, "get_importance_sampler", lambda: _IdentitySampler()
    )

    # Ensure kNN augmentation does not mask the adjacency change.
    monkeypatch.setenv("DXF_EDGE_AUGMENT_KNN_K", "0")

    doc = ezdxf.new()
    msp = doc.modelspace()

    # Two connected lines to keep the epsilon graph non-empty in both modes.
    msp.add_line((1, 0), (2, 0))
    msp.add_line((2, 0), (3, 0))

    # Circle touches the first line at (1,0) on the circumference.
    msp.add_circle((0, 0), radius=1.0)

    dataset = DXFDataset(root_dir=".")

    monkeypatch.setenv("DXF_ENHANCED_KEYPOINTS", "false")
    x_base, edge_base = dataset._dxf_to_graph(msp)
    assert int(x_base.size(0)) == 3
    nodes_in_edges_base = set(edge_base.flatten().tolist()) if edge_base.numel() else set()
    assert 2 not in nodes_in_edges_base

    monkeypatch.setenv("DXF_ENHANCED_KEYPOINTS", "true")
    x_enh, edge_enh = dataset._dxf_to_graph(msp)
    assert int(x_enh.size(0)) == 3
    nodes_in_edges_enh = set(edge_enh.flatten().tolist()) if edge_enh.numel() else set()
    assert 2 in nodes_in_edges_enh
