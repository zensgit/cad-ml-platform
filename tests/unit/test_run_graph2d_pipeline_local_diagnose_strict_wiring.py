from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def _base_args() -> Namespace:
    return Namespace(
        min_label_confidence=0.8,
        diagnose_max_files=20,
        seed=42,
        diagnose_no_text_no_filename=False,
    )


def test_pipeline_build_diagnose_cmd_includes_strict_flags_when_enabled() -> None:
    from scripts.run_graph2d_pipeline_local import _build_diagnose_cmd

    args = _base_args()
    args.diagnose_no_text_no_filename = True

    cmd = _build_diagnose_cmd(
        python="python",
        dxf_dir=Path("dxfs"),
        checkpoint_path=Path("model.pth"),
        manifest_csv=Path("manifest.csv"),
        out_dir=Path("out"),
        args=args,
    )

    assert "scripts/diagnose_graph2d_on_dxf_dir.py" in cmd
    assert "--strip-text-entities" in cmd
    assert "--mask-filename" in cmd
