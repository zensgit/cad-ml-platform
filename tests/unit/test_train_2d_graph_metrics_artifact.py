from __future__ import annotations

import argparse


def test_build_training_metrics_payload_fields() -> None:
    from scripts.train_2d_graph import _build_training_metrics_payload

    args = argparse.Namespace(
        epochs=10,
        early_stop_patience=2,
        seed=42,
        device="cpu",
        model="gcn",
        loss="focal",
        sampler="balanced",
        class_weighting="sqrt",
        config="config/graph2d_training.yaml",
        dxf_max_nodes=300,
        dxf_sampling_strategy="importance",
        dxf_sampling_seed=42,
        dxf_text_priority_ratio=0.3,
        dxf_frame_priority_ratio=0.25,
        dxf_long_line_ratio=0.2,
        dxf_edge_augment_knn_k=4,
        dxf_edge_augment_strategy="isolates_only",
        dxf_eps_scale=0.015,
        dxf_enhanced_keypoints="auto",
    )
    payload = _build_training_metrics_payload(
        args=args,
        best_val_acc=0.88,
        best_epoch=6,
        final_val_acc=0.85,
        final_loss=0.42,
        epochs_ran=8,
        train_size=120,
        val_size=30,
        class_stats={"num_classes": 10, "imbalance_ratio": 3.2},
        checkpoint_path="models/test_model.pth",
        epoch_history=[
            {"epoch": 1, "loss": 1.23, "val_acc": 0.5, "lr": 0.001, "is_best": True}
        ],
    )

    assert payload["status"] == "ok"
    assert payload["checkpoint_path"] == "models/test_model.pth"
    assert payload["epochs_requested"] == 10
    assert payload["epochs_ran"] == 8
    assert payload["stopped_early"] is True
    assert payload["best_val_acc"] == 0.88
    assert payload["best_epoch"] == 6
    assert payload["final_val_acc"] == 0.85
    assert payload["final_loss"] == 0.42
    assert payload["train_size"] == 120
    assert payload["val_size"] == 30
    assert payload["class_stats"]["num_classes"] == 10
    assert payload["sampling_overrides"]["dxf_max_nodes"] == 300
    assert len(payload["epoch_history"]) == 1
