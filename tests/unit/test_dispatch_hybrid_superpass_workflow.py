from __future__ import annotations

import json
import subprocess
from typing import Any


def test_build_workflow_run_command_contains_required_inputs() -> None:
    from scripts.ci.dispatch_hybrid_superpass_workflow import build_workflow_run_command

    command = build_workflow_run_command(
        workflow="evaluation-report.yml",
        ref="main",
        repo="zensgit/cad-ml-platform",
        hybrid_superpass_enable="true",
        hybrid_superpass_missing_mode="fail",
        hybrid_superpass_fail_on_failed="true",
        hybrid_blind_enable="true",
        hybrid_blind_dxf_dir="data/blind_dxf",
        hybrid_blind_fail_on_gate_failed="true",
        hybrid_blind_strict_require_real_data="false",
        hybrid_calibration_enable="true",
        hybrid_calibration_input_csv="reports/review.csv",
    )
    text = " ".join(command)
    assert "gh workflow run evaluation-report.yml" in text
    assert "--ref main" in text
    assert "--repo zensgit/cad-ml-platform" in text
    assert "hybrid_superpass_enable=true" in text
    assert "hybrid_superpass_missing_mode=fail" in text
    assert "hybrid_superpass_fail_on_failed=true" in text
    assert "hybrid_blind_enable=true" in text
    assert "hybrid_blind_dxf_dir=data/blind_dxf" in text
    assert "hybrid_blind_fail_on_gate_failed=true" in text
    assert "hybrid_blind_strict_require_real_data=false" in text
    assert "hybrid_calibration_enable=true" in text
    assert "hybrid_calibration_input_csv=reports/review.csv" in text


def test_main_print_only_outputs_dispatch_and_watch_commands(capsys: Any) -> None:
    from scripts.ci import dispatch_hybrid_superpass_workflow as mod

    rc = mod.main(
        [
            "--print-only",
            "--workflow",
            "evaluation-report.yml",
            "--ref",
            "main",
            "--repo",
            "zensgit/cad-ml-platform",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "hybrid_superpass_enable=true" in out
    assert "hybrid_superpass_missing_mode=fail" in out
    assert "hybrid_superpass_fail_on_failed=true" in out
    assert "gh run watch '<run_id>' --exit-status" in out
    assert "gh run view '<run_id>' --json conclusion,url" in out


def test_main_returns_nonzero_when_gh_not_ready(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_hybrid_superpass_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (False, "gh auth not ready"))
    rc = mod.main([])
    assert rc == 1


def test_main_success_when_expectation_matches(monkeypatch: Any, tmp_path: Any) -> None:
    from scripts.ci import dispatch_hybrid_superpass_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "list_dispatched_run_ids", lambda *_args, **_kwargs: [1, 2])
    monkeypatch.setattr(mod, "wait_for_new_dispatched_run_id", lambda **_kwargs: 3001)
    monkeypatch.setattr(mod, "watch_run", lambda _run_id, _repo: 0)
    monkeypatch.setattr(
        mod,
        "wait_for_run_conclusion",
        lambda **_kwargs: ("success", "https://example.com/r/3001"),
    )

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    output_json = tmp_path / "hybrid_superpass_dispatch.json"
    rc = mod.main([
        "--output-json",
        str(output_json),
        "--expected-conclusion",
        "success",
    ])
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["overall_exit_code"] == 0
    assert payload["matched_expectation"] is True
    assert payload["conclusion"] == "success"


def test_main_returns_nonzero_when_expectation_mismatch(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_hybrid_superpass_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "list_dispatched_run_ids", lambda *_args, **_kwargs: [1])
    monkeypatch.setattr(mod, "wait_for_new_dispatched_run_id", lambda **_kwargs: 4001)
    monkeypatch.setattr(mod, "watch_run", lambda _run_id, _repo: 1)
    monkeypatch.setattr(
        mod,
        "wait_for_run_conclusion",
        lambda **_kwargs: ("failure", "https://example.com/r/4001"),
    )

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(["--expected-conclusion", "success"])
    assert rc == 1


def test_find_missing_superpass_inputs_detects_expected_keys() -> None:
    from scripts.ci import dispatch_hybrid_superpass_workflow as mod

    missing = mod.find_missing_superpass_inputs(
        "name: Evaluation Report\non:\n  workflow_dispatch:\n    inputs:\n      min_combined:\n"
    )
    assert "hybrid_superpass_enable" in missing
    assert "hybrid_superpass_missing_mode" in missing
    assert "hybrid_superpass_fail_on_failed" in missing


def test_main_returns_nonzero_when_remote_workflow_missing_required_inputs(
    monkeypatch: Any, capsys: Any
) -> None:
    from scripts.ci import dispatch_hybrid_superpass_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(
        mod,
        "fetch_remote_workflow_text",
        lambda *_args, **_kwargs: "name: Evaluation Report\non:\n  workflow_dispatch:\n    inputs:\n      min_combined:\n",
    )
    monkeypatch.setattr(
        mod,
        "list_dispatched_run_ids",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not list runs")
        ),
    )

    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "missing required superpass inputs" in out
