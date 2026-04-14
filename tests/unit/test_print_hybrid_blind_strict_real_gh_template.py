from __future__ import annotations

from typing import Any


def test_build_workflow_run_command_contains_expected_fields() -> None:
    from scripts.ci.print_hybrid_blind_strict_real_gh_template import (
        build_workflow_run_command,
    )

    command = build_workflow_run_command(
        workflow="evaluation-report.yml",
        ref="main",
        repo="zensgit/cad-ml-platform",
        dxf_dir="data/blind_dxf",
    )
    text = " ".join(command)
    assert "gh workflow run evaluation-report.yml" in text
    assert "--ref main" in text
    assert "--repo zensgit/cad-ml-platform" in text
    assert "hybrid_blind_enable=true" in text
    assert "hybrid_blind_dxf_dir=data/blind_dxf" in text
    assert "hybrid_blind_fail_on_gate_failed=true" in text
    assert "hybrid_blind_strict_require_real_data=true" in text


def test_main_prints_template_vars_and_watch_commands(capsys: Any) -> None:
    from scripts.ci import print_hybrid_blind_strict_real_gh_template as mod

    rc = mod.main(
        [
            "--workflow",
            "evaluation-report.yml",
            "--ref",
            "main",
            "--repo",
            "zensgit/cad-ml-platform",
            "--dxf-dir",
            "data/blind_dxf",
            "--print-vars",
            "--print-watch",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "gh workflow run evaluation-report.yml" in out
    assert "--ref main" in out
    assert "--repo zensgit/cad-ml-platform" in out
    assert "HYBRID_BLIND_ENABLE=true" in out
    assert "HYBRID_BLIND_DXF_DIR=data/blind_dxf" in out
    assert "HYBRID_BLIND_FAIL_ON_GATE_FAILED=true" in out
    assert "HYBRID_BLIND_STRICT_REQUIRE_REAL_DATA=true" in out
    assert "gh run watch <run_id> --exit-status" in out
    assert "gh run view <run_id> --json conclusion,url" in out
