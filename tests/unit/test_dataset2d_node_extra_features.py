import ezdxf

from src.ml.train.dataset_2d import DXFDataset


class _SampleResult:
    def __init__(self, sampled_entities):
        self.sampled_entities = sampled_entities


class _IdentitySampler:
    def sample(self, entities):
        return _SampleResult(entities)


def test_node_extra_features_appended_when_node_dim_exceeds_default(monkeypatch):
    import src.ml.importance_sampler as importance_sampler

    monkeypatch.setattr(
        importance_sampler, "get_importance_sampler", lambda: _IdentitySampler()
    )

    # Keep edge augmentation disabled to avoid unrelated graph changes.
    monkeypatch.setenv("DXF_EDGE_AUGMENT_KNN_K", "0")
    monkeypatch.setenv("DXF_ENHANCED_KEYPOINTS", "false")

    doc = ezdxf.new()
    msp = doc.modelspace()

    # Establish a stable drawing bbox: max_dim = 100.
    msp.add_line((0, 0), (100, 0))  # node 0
    msp.add_line((0, 0), (0, 100))  # node 1 (keeps bbox square)

    # Circle and arc share center/radius for predictable bbox norms.
    msp.add_circle((50, 50), radius=10.0)  # node 2
    msp.add_arc((50, 50), radius=10.0, start_angle=0.0, end_angle=90.0)  # node 3

    # Polyline with 4 vertices (not closed) -> vertex_norm = 4/64.
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 10), (0, 10)])  # node 4

    dataset = DXFDataset(root_dir=".")
    x, _edge_index = dataset._dxf_to_graph(msp, node_dim=23)
    assert tuple(x.shape) == (5, 23)

    # Extra feature indices (appended after the 19-base schema).
    idx_bbox_w = 19
    idx_bbox_h = 20
    idx_arc_sweep = 21
    idx_poly_vertices = 22

    # Circle: bbox_w/h = 20 -> 0.2 of max_dim=100; sweep=1.0.
    assert abs(float(x[2, idx_bbox_w]) - 0.2) < 1e-6
    assert abs(float(x[2, idx_bbox_h]) - 0.2) < 1e-6
    assert abs(float(x[2, idx_arc_sweep]) - 1.0) < 1e-6
    assert abs(float(x[2, idx_poly_vertices]) - 0.0) < 1e-6

    # Arc (quarter): bbox_w/h = 10 -> 0.1; sweep=0.25.
    assert abs(float(x[3, idx_bbox_w]) - 0.1) < 1e-6
    assert abs(float(x[3, idx_bbox_h]) - 0.1) < 1e-6
    assert abs(float(x[3, idx_arc_sweep]) - 0.25) < 1e-6

    # Polyline: bbox_w/h = 10 -> 0.1; vertex_norm = 4/64.
    assert abs(float(x[4, idx_bbox_w]) - 0.1) < 1e-6
    assert abs(float(x[4, idx_bbox_h]) - 0.1) < 1e-6
    assert abs(float(x[4, idx_poly_vertices]) - (4.0 / 64.0)) < 1e-6
