from __future__ import annotations

from argparse import Namespace


def _base_args() -> Namespace:
    return Namespace(
        training_profile="none",
        model="gcn",
        node_dim=19,
        hidden_dim=64,
        epochs=3,
        loss="focal",
        class_weighting="sqrt",
        sampler="balanced",
        distill=False,
        teacher="hybrid",
        distill_alpha=0.3,
        distill_temp=3.0,
        distill_mask_filename="auto",
        student_geometry_only=False,
        diagnose_no_text_no_filename=False,
        normalize_labels=False,
        clean_min_count=0,
        dxf_enhanced_keypoints="auto",
        dxf_edge_augment_knn_k=None,
        dxf_edge_augment_strategy="auto",
        dxf_eps_scale=0.001,
    )


def test_apply_training_profile_none_keeps_defaults() -> None:
    from scripts.run_graph2d_pipeline_local import _apply_training_profile

    args = _base_args()
    out = _apply_training_profile(args)
    assert out.training_profile == "none"
    assert out.model == "gcn"
    assert int(out.node_dim) == 19
    assert int(out.hidden_dim) == 64


def test_apply_training_profile_strict_node23_edgesage_v1() -> None:
    from scripts.run_graph2d_pipeline_local import _apply_training_profile

    args = _base_args()
    args.training_profile = "strict_node23_edgesage_v1"
    out = _apply_training_profile(args)

    assert out.training_profile == "strict_node23_edgesage_v1"
    assert out.model == "edge_sage"
    assert int(out.node_dim) == 23
    assert int(out.hidden_dim) == 128
    assert int(out.epochs) == 10
    assert bool(out.distill) is True
    assert out.teacher == "titleblock"
    assert bool(out.student_geometry_only) is True
    assert bool(out.diagnose_no_text_no_filename) is True
    assert bool(out.normalize_labels) is True
    assert int(out.clean_min_count) == 5
