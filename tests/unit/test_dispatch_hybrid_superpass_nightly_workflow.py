from __future__ import annotations

import json
import subprocess
from typing import Any


def test_build_workflow_run_command_contains_target_inputs() -> None:
    from scripts.ci.dispatch_hybrid_superpass_nightly_workflow import (
        build_workflow_run_command,
    )

    command = build_workflow_run_command(
        workflow="hybrid-superpass-nightly.yml",
        ref="main",
        repo="zensgit/cad-ml-platform",
        target_repo="zensgit/cad-ml-platform",
        target_ref="release/2026-03-13",
        target_workflow="hybrid-superpass-e2e.yml",
        dispatch_trace_id="nsp-test-trace",
    )
    text = " ".join(command)
    assert "gh workflow run hybrid-superpass-nightly.yml" in text
    assert "--ref main" in text
    assert "--repo zensgit/cad-ml-platform" in text
    assert "target_repo=zensgit/cad-ml-platform" in text
    assert "target_ref=release/2026-03-13" in text
    assert "target_workflow=hybrid-superpass-e2e.yml" in text
    assert "dispatch_trace_id=nsp-test-trace" in text


def test_main_print_only_writes_output_json(capsys: Any, tmp_path: Any) -> None:
    from scripts.ci import dispatch_hybrid_superpass_nightly_workflow as mod

    output_json = tmp_path / "nightly_dispatch_print_only.json"
    rc = mod.main(["--print-only", "--output-json", str(output_json)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dispatch_command=" in out
    assert "dispatch_trace_id=nsp-" in out
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["mode"] == "print_only"
    assert payload["overall_exit_code"] == 0
    assert payload["dispatch_trace_id"].startswith("nsp-")
    assert payload["run_id"] is None
    assert payload["expected_conclusion"] == "success"


def test_main_success_when_expectation_matches(monkeypatch: Any, tmp_path: Any) -> None:
    from scripts.ci import dispatch_hybrid_superpass_nightly_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "list_dispatched_run_ids", lambda *_args, **_kwargs: [1, 2])
    monkeypatch.setattr(mod, "wait_for_new_dispatched_run_id", lambda **_kwargs: 5001)
    monkeypatch.setattr(mod, "watch_run", lambda _run_id, _repo: 0)
    monkeypatch.setattr(
        mod,
        "wait_for_run_conclusion",
        lambda **_kwargs: ("success", "https://example.com/r/5001"),
    )

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    output_json = tmp_path / "nightly_dispatch_success.json"
    rc = mod.main(["--output-json", str(output_json), "--expected-conclusion", "success"])
    assert rc == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["run_id"] == 5001
    assert payload["conclusion"] == "success"
    assert payload["expected_conclusion"] == "success"
    assert payload["matched_expectation"] is True
    assert payload["overall_exit_code"] == 0


def test_main_returns_nonzero_when_expectation_mismatch(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    from scripts.ci import dispatch_hybrid_superpass_nightly_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "list_dispatched_run_ids", lambda *_args, **_kwargs: [1])
    monkeypatch.setattr(mod, "wait_for_new_dispatched_run_id", lambda **_kwargs: 5002)
    monkeypatch.setattr(mod, "watch_run", lambda _run_id, _repo: 1)
    monkeypatch.setattr(
        mod,
        "wait_for_run_conclusion",
        lambda **_kwargs: ("failure", "https://example.com/r/5002"),
    )

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    output_json = tmp_path / "nightly_dispatch_mismatch.json"
    rc = mod.main(["--output-json", str(output_json), "--expected-conclusion", "success"])
    assert rc == 1

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["run_id"] == 5002
    assert payload["conclusion"] == "failure"
    assert payload["expected_conclusion"] == "success"
    assert payload["matched_expectation"] is False
    assert payload["overall_exit_code"] == 1


def test_wait_for_new_dispatched_run_id_filters_by_trace_with_polling(
    monkeypatch: Any,
) -> None:
    from scripts.ci import dispatch_hybrid_superpass_nightly_workflow as mod

    calls = {"count": 0}

    def _fake_list_runs(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        calls["count"] += 1
        if calls["count"] == 1:
            return [
                {
                    "databaseId": 9001,
                    "displayTitle": "Hybrid Superpass Nightly (trace: nsp-other)",
                }
            ]
        return [
            {
                "databaseId": 9001,
                "displayTitle": "Hybrid Superpass Nightly (trace: nsp-other)",
            },
            {
                "databaseId": 9002,
                "displayTitle": "Hybrid Superpass Nightly (trace: nsp-target)",
            },
        ]

    monkeypatch.setattr(mod, "list_dispatched_runs", _fake_list_runs)
    monkeypatch.setattr(mod.time, "time", lambda: 0)
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)

    run_id = mod.wait_for_new_dispatched_run_id(
        workflow="hybrid-superpass-nightly.yml",
        ref="main",
        repo="zensgit/cad-ml-platform",
        known_run_ids=[9000],
        timeout_seconds=30,
        poll_interval_seconds=1,
        list_limit=20,
        dispatch_trace_id="nsp-target",
    )

    assert run_id == 9002
    assert calls["count"] >= 2
