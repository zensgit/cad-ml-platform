from __future__ import annotations

import json
import subprocess
from typing import Any


def test_build_workflow_run_command_contains_required_inputs() -> None:
    from scripts.ci.dispatch_graph2d_review_gate_strict_e2e import (
        build_workflow_run_command,
    )

    command = build_workflow_run_command(
        workflow="evaluation-report.yml",
        ref="main",
        review_pack_input_csv="tests/fixtures/ci/graph2d_review_pack_input.csv",
        review_pack_input_artifact_name="",
        review_pack_input_artifact_run_id="",
        review_pack_input_artifact_repository="",
        review_pack_input_artifact_path="",
        strict_value="true",
    )
    text = " ".join(command)
    assert "gh workflow run evaluation-report.yml" in text
    assert "--ref main" in text
    assert "review_pack_input_csv=tests/fixtures/ci/graph2d_review_pack_input.csv" in text
    assert "review_gate_strict=true" in text


def test_build_workflow_run_command_supports_artifact_input_fields() -> None:
    from scripts.ci.dispatch_graph2d_review_gate_strict_e2e import (
        build_workflow_run_command,
    )

    command = build_workflow_run_command(
        workflow="evaluation-report.yml",
        ref="main",
        review_pack_input_csv="",
        review_pack_input_artifact_name="batch-results-artifact",
        review_pack_input_artifact_run_id="123456789",
        review_pack_input_artifact_repository="zensgit/cad-ml-platform",
        review_pack_input_artifact_path="batch_results_sanitized.csv",
        strict_value="false",
    )
    text = " ".join(command)
    assert "review_pack_input_artifact_name=batch-results-artifact" in text
    assert "review_pack_input_artifact_run_id=123456789" in text
    assert "review_pack_input_artifact_repository=zensgit/cad-ml-platform" in text
    assert "review_pack_input_artifact_path=batch_results_sanitized.csv" in text
    assert "review_gate_strict=false" in text
    assert "review_pack_input_csv=" not in text


def test_main_print_only_outputs_dispatch_and_watch_commands(capsys: Any) -> None:
    from scripts.ci import dispatch_graph2d_review_gate_strict_e2e as mod

    rc = mod.main(["--print-only", "--workflow", "evaluation-report.yml", "--ref", "main"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "review_gate_strict=false" in out
    assert "review_gate_strict=true" in out
    assert "gh run list --workflow evaluation-report.yml" in out
    assert "gh run watch <run_id> --exit-status" in out


def test_main_returns_nonzero_when_gh_not_ready(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_graph2d_review_gate_strict_e2e as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (False, "gh auth not ready"))
    rc = mod.main([])
    assert rc == 1


def test_wait_for_run_conclusion_polls_until_non_empty(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_graph2d_review_gate_strict_e2e as mod

    calls = {"count": 0}

    def _fake_get_run_conclusion_and_url(_run_id: int) -> tuple[str, str]:
        calls["count"] += 1
        if calls["count"] < 3:
            return ("", "https://example.com/r")
        return ("success", "https://example.com/r")

    monkeypatch.setattr(mod, "get_run_conclusion_and_url", _fake_get_run_conclusion_and_url)
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
    conclusion, run_url = mod.wait_for_run_conclusion(
        run_id=1234,
        timeout_seconds=10,
        poll_interval_seconds=1,
    )
    assert conclusion == "success"
    assert run_url == "https://example.com/r"
    assert calls["count"] >= 3


def test_main_success_when_expectations_match(monkeypatch: Any, tmp_path: Any) -> None:
    from scripts.ci import dispatch_graph2d_review_gate_strict_e2e as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "list_dispatched_run_ids", lambda *_args, **_kwargs: [10, 11])

    run_ids = iter([1201, 1202])
    monkeypatch.setattr(
        mod,
        "wait_for_new_dispatched_run_id",
        lambda **_kwargs: next(run_ids),
    )

    watch_codes = iter([1, 1])
    monkeypatch.setattr(mod, "watch_run", lambda _run_id: next(watch_codes))

    conclusions = iter(
        [("success", "https://example.com/1201"), ("failure", "https://example.com/1202")]
    )
    monkeypatch.setattr(mod, "wait_for_run_conclusion", lambda **_kwargs: next(conclusions))

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    output_json = tmp_path / "strict_e2e_summary.json"
    rc = mod.main(["--output-json", str(output_json)])
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["overall_exit_code"] == 0
    assert payload["review_pack_input_artifact_name"] == ""
    assert len(payload["runs"]) == 2
    assert payload["runs"][0]["matched_expectation"] is True
    assert payload["runs"][1]["matched_expectation"] is True


def test_main_print_only_supports_artifact_input_flags(capsys: Any) -> None:
    from scripts.ci import dispatch_graph2d_review_gate_strict_e2e as mod

    rc = mod.main(
        [
            "--print-only",
            "--workflow",
            "evaluation-report.yml",
            "--ref",
            "main",
            "--review-pack-input-csv",
            "",
            "--review-pack-input-artifact-name",
            "batch-results-artifact",
            "--review-pack-input-artifact-run-id",
            "123456789",
            "--review-pack-input-artifact-repository",
            "zensgit/cad-ml-platform",
            "--review-pack-input-artifact-path",
            "batch_results_sanitized.csv",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "review_pack_input_artifact_name=batch-results-artifact" in out
    assert "review_pack_input_artifact_run_id=123456789" in out
    assert "review_pack_input_artifact_repository=zensgit/cad-ml-platform" in out
    assert "review_pack_input_artifact_path=batch_results_sanitized.csv" in out


def test_main_returns_nonzero_when_strict_true_unexpected_success(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_graph2d_review_gate_strict_e2e as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "list_dispatched_run_ids", lambda *_args, **_kwargs: [])

    run_ids = iter([2201, 2202])
    monkeypatch.setattr(
        mod,
        "wait_for_new_dispatched_run_id",
        lambda **_kwargs: next(run_ids),
    )
    monkeypatch.setattr(mod, "watch_run", lambda _run_id: 0)

    conclusions = iter(
        [("success", "https://example.com/2201"), ("success", "https://example.com/2202")]
    )
    monkeypatch.setattr(mod, "wait_for_run_conclusion", lambda **_kwargs: next(conclusions))

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main([])
    assert rc == 1
