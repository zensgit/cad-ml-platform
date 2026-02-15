from __future__ import annotations

from pathlib import Path


def test_build_manifest_cmd_uses_parent_dir_label_mode() -> None:
    from scripts.run_graph2d_pipeline_local import _build_manifest_cmd

    cmd = _build_manifest_cmd(
        python="python",
        dxf_dir=Path("dxfs"),
        manifest_csv=Path("manifest.csv"),
        label_mode="parent_dir",
    )

    assert "scripts/build_dxf_label_manifest.py" in cmd
    assert "--label-mode" in cmd
    assert cmd[cmd.index("--label-mode") + 1] == "parent_dir"
