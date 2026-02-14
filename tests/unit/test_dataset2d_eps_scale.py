import ezdxf

from src.ml.train.dataset_2d import DXFDataset


class _SampleResult:
    def __init__(self, sampled_entities):
        self.sampled_entities = sampled_entities


class _IdentitySampler:
    def sample(self, entities):
        return _SampleResult(entities)


def test_eps_scale_controls_endpoint_connection_distance(monkeypatch):
    import src.ml.importance_sampler as importance_sampler

    monkeypatch.setattr(
        importance_sampler, "get_importance_sampler", lambda: _IdentitySampler()
    )

    # Ensure kNN augmentation does not mask the adjacency change.
    monkeypatch.setenv("DXF_EDGE_AUGMENT_KNN_K", "0")

    doc = ezdxf.new()
    msp = doc.modelspace()

    # line0 and line1 share an endpoint to keep epsilon edges non-empty.
    msp.add_line((0, 0), (1000, 0))  # line0
    msp.add_line((0, 0), (0, 100))  # line1

    # line2 is near the endpoint of line0, but not within eps when scale=0.001.
    # max_dim ~ 1001.5 -> eps ~ 1.0, while the gap is 1.5.
    msp.add_line((1001.5, 0), (1001.5, 100))

    dataset = DXFDataset(root_dir=".")

    monkeypatch.setenv("DXF_EPS_SCALE", "0.001")
    x_base, edge_base = dataset._dxf_to_graph(msp)
    assert int(x_base.size(0)) == 3
    nodes_in_edges_base = (
        set(edge_base.flatten().tolist()) if edge_base.numel() else set()
    )
    assert 2 not in nodes_in_edges_base

    monkeypatch.setenv("DXF_EPS_SCALE", "0.002")
    x_big, edge_big = dataset._dxf_to_graph(msp)
    assert int(x_big.size(0)) == 3
    nodes_in_edges_big = set(edge_big.flatten().tolist()) if edge_big.numel() else set()
    assert 2 in nodes_in_edges_big

