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


def _edge_pairs(edge_index):
    if edge_index.numel() == 0:
        return set()
    return {tuple(pair) for pair in edge_index.t().tolist()}


def test_edge_augment_strategy_isolates_only_skips_when_no_isolates(monkeypatch):
    # Keep entity order stable and avoid sampling variance.
    import src.ml.importance_sampler as importance_sampler

    monkeypatch.setattr(
        importance_sampler, "get_importance_sampler", lambda: _IdentitySampler()
    )

    doc = ezdxf.new()
    msp = doc.modelspace()

    # Three connected lines in a chain: epsilon-adjacency already connects all nodes.
    msp.add_line((0, 0), (1, 0))
    msp.add_line((1, 0), (2, 0))
    msp.add_line((2, 0), (3, 0))

    dataset = DXFDataset(root_dir=".")

    monkeypatch.setenv("DXF_EDGE_AUGMENT_KNN_K", "2")
    monkeypatch.setenv("DXF_EDGE_AUGMENT_STRATEGY", "union_all")
    _x_union, edge_union = dataset._dxf_to_graph(msp)
    pairs_union = _edge_pairs(edge_union)
    assert (0, 2) in pairs_union
    assert (2, 0) in pairs_union

    monkeypatch.setenv("DXF_EDGE_AUGMENT_STRATEGY", "isolates_only")
    _x_iso, edge_iso = dataset._dxf_to_graph(msp)
    pairs_iso = _edge_pairs(edge_iso)
    assert (0, 2) not in pairs_iso
    assert (2, 0) not in pairs_iso
