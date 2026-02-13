from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def _base_args() -> Namespace:
    return Namespace(
        epochs=1,
        batch_size=4,
        hidden_dim=64,
        lr=1e-3,
        model="edge_sage",
        loss="cross_entropy",
        class_weighting="inverse",
        sampler="balanced",
        seed=42,
        device="cpu",
        dxf_max_nodes=200,
        dxf_sampling_strategy="importance",
        dxf_sampling_seed=42,
        dxf_text_priority_ratio=0.3,
        max_samples=0,
        distill=True,
        teacher="hybrid",
        distill_alpha=0.3,
        distill_temp=3.0,
        distill_mask_filename="auto",
    )


def test_pipeline_build_train_cmd_includes_distill_flags_and_masks_hybrid() -> None:
    from scripts.run_graph2d_pipeline_local import _build_train_cmd

    args = _base_args()
    cmd = _build_train_cmd(
        python="python",
        manifest_csv=Path("manifest.csv"),
        dxf_dir=Path("dxfs"),
        checkpoint_path=Path("model.pth"),
        args=args,
    )

    assert "--distill" in cmd
    assert "--teacher" in cmd and cmd[cmd.index("--teacher") + 1] == "hybrid"
    assert "--distill-alpha" in cmd
    assert "--distill-temp" in cmd
    assert "--distill-mask-filename" in cmd


def test_pipeline_build_train_cmd_auto_mask_skips_filename_teacher() -> None:
    from scripts.run_graph2d_pipeline_local import _build_train_cmd

    args = _base_args()
    args.teacher = "filename"
    cmd = _build_train_cmd(
        python="python",
        manifest_csv=Path("manifest.csv"),
        dxf_dir=Path("dxfs"),
        checkpoint_path=Path("model.pth"),
        args=args,
    )

    assert "--distill" in cmd
    assert "--teacher" in cmd and cmd[cmd.index("--teacher") + 1] == "filename"
    assert "--distill-mask-filename" not in cmd


def test_pipeline_build_train_cmd_respects_mask_false() -> None:
    from scripts.run_graph2d_pipeline_local import _build_train_cmd

    args = _base_args()
    args.teacher = "hybrid"
    args.distill_mask_filename = "false"
    cmd = _build_train_cmd(
        python="python",
        manifest_csv=Path("manifest.csv"),
        dxf_dir=Path("dxfs"),
        checkpoint_path=Path("model.pth"),
        args=args,
    )

    assert "--distill" in cmd
    assert "--distill-mask-filename" not in cmd

