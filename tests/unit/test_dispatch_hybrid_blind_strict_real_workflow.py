from __future__ import annotations

import json
import subprocess
from typing import Any


def test_build_workflow_run_command_contains_required_inputs() -> None:
    from scripts.ci.dispatch_hybrid_blind_strict_real_workflow import (
        build_workflow_run_command,
    )

    command = build_workflow_run_command(
        workflow="evaluation-report.yml",
        ref="main",
        repo="zensgit/cad-ml-platform",
        hybrid_blind_dxf_dir="data/blind_dxf",
        hybrid_blind_manifest_csv="reports/blind_manifest.csv",
        hybrid_blind_synth_manifest="tests/golden/golden_dxf_hybrid_cases.json",
        strict_fail_on_gate_failed="true",
        strict_require_real_data="true",
    )
    text = " ".join(command)
    assert "gh workflow run evaluation-report.yml" in text
    assert "--ref main" in text
    assert "--repo zensgit/cad-ml-platform" in text
    assert "hybrid_blind_enable=true" in text
    assert "hybrid_blind_dxf_dir=data/blind_dxf" in text
    assert "hybrid_blind_manifest_csv=reports/blind_manifest.csv" in text
    assert "hybrid_blind_synth_manifest=tests/golden/golden_dxf_hybrid_cases.json" in text
    assert "hybrid_blind_fail_on_gate_failed=true" in text
    assert "hybrid_blind_strict_require_real_data=true" in text


def test_main_print_only_outputs_dispatch_and_watch_commands(capsys: Any) -> None:
    from scripts.ci import dispatch_hybrid_blind_strict_real_workflow as mod

    rc = mod.main(
        [
            "--print-only",
            "--workflow",
            "evaluation-report.yml",
            "--ref",
            "main",
            "--repo",
            "zensgit/cad-ml-platform",
            "--hybrid-blind-dxf-dir",
            "data/blind_dxf",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "hybrid_blind_enable=true" in out
    assert "hybrid_blind_dxf_dir=data/blind_dxf" in out
    assert "hybrid_blind_fail_on_gate_failed=true" in out
    assert "hybrid_blind_strict_require_real_data=true" in out
    assert "--repo zensgit/cad-ml-platform" in out
    assert "gh run watch <run_id> --exit-status" in out
    assert "gh run view <run_id> --json conclusion,url" in out


def test_main_returns_nonzero_when_gh_not_ready(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_hybrid_blind_strict_real_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (False, "gh auth not ready"))
    rc = mod.main(["--hybrid-blind-dxf-dir", "data/blind_dxf"])
    assert rc == 1


def test_find_missing_strict_real_inputs_detects_expected_keys() -> None:
    from scripts.ci import dispatch_hybrid_blind_strict_real_workflow as mod

    missing = mod.find_missing_strict_real_inputs(
        (
            "name: Evaluation Report\n"
            "on:\n"
            "  workflow_dispatch:\n"
            "    inputs:\n"
            "      min_combined:\n"
        )
    )
    assert "hybrid_blind_enable" in missing
    assert "hybrid_blind_dxf_dir" in missing
    assert "hybrid_blind_fail_on_gate_failed" in missing
    assert "hybrid_blind_strict_require_real_data" in missing


def test_main_returns_nonzero_when_remote_workflow_missing_required_inputs(
    monkeypatch: Any, capsys: Any
) -> None:
    from scripts.ci import dispatch_hybrid_blind_strict_real_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(
        mod,
        "fetch_remote_workflow_text",
        lambda *_args, **_kwargs: (
            "name: Evaluation Report\n"
            "on:\n"
            "  workflow_dispatch:\n"
            "    inputs:\n"
            "      min_combined:\n"
        ),
    )
    monkeypatch.setattr(
        mod,
        "list_dispatched_run_ids",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not list runs")),
    )

    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--hybrid-blind-dxf-dir",
            "data/blind_dxf",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "missing required strict-real inputs" in out


def test_main_success_when_expectation_matches(monkeypatch: Any, tmp_path: Any) -> None:
    from scripts.ci import dispatch_hybrid_blind_strict_real_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "list_dispatched_run_ids", lambda *_args, **_kwargs: [10, 11])
    monkeypatch.setattr(mod, "wait_for_new_dispatched_run_id", lambda **_kwargs: 3201)
    monkeypatch.setattr(mod, "watch_run", lambda _run_id, _repo: 0)
    monkeypatch.setattr(
        mod,
        "wait_for_run_conclusion",
        lambda **_kwargs: ("success", "https://example.com/r/3201"),
    )

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    output_json = tmp_path / "hybrid_blind_strict_real_dispatch.json"
    rc = mod.main(
        [
            "--hybrid-blind-dxf-dir",
            "data/blind_dxf",
            "--output-json",
            str(output_json),
            "--expected-conclusion",
            "success",
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["overall_exit_code"] == 0
    assert payload["matched_expectation"] is True
    assert payload["conclusion"] == "success"


def test_main_returns_nonzero_when_expectation_mismatch(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_hybrid_blind_strict_real_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "list_dispatched_run_ids", lambda *_args, **_kwargs: [1])
    monkeypatch.setattr(mod, "wait_for_new_dispatched_run_id", lambda **_kwargs: 4201)
    monkeypatch.setattr(mod, "watch_run", lambda _run_id, _repo: 1)
    monkeypatch.setattr(
        mod,
        "wait_for_run_conclusion",
        lambda **_kwargs: ("failure", "https://example.com/r/4201"),
    )

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(
        [
            "--hybrid-blind-dxf-dir",
            "data/blind_dxf",
            "--expected-conclusion",
            "success",
        ]
    )
    assert rc == 1


def test_summarize_failed_jobs_extracts_first_failed_step() -> None:
    from scripts.ci import dispatch_hybrid_blind_strict_real_workflow as mod

    jobs_payload = [
        {
            "name": "prepare",
            "conclusion": "success",
            "steps": [{"name": "checkout", "conclusion": "success"}],
        },
        {
            "name": "strict-real-gate",
            "conclusion": "failure",
            "url": "https://example.com/job/strict-real",
            "steps": [
                {"name": "checkout", "conclusion": "success"},
                {"name": "gate", "conclusion": "failure"},
            ],
        },
    ]
    summary = mod.summarize_failed_jobs(jobs_payload, max_jobs=5)
    assert summary["total_jobs"] == 2
    assert summary["failed_job_count"] == 1
    failed_jobs = summary["failed_jobs"]
    assert isinstance(failed_jobs, list) and len(failed_jobs) == 1
    assert failed_jobs[0]["job_name"] == "strict-real-gate"
    assert failed_jobs[0]["failed_step_name"] == "gate"
    assert failed_jobs[0]["failed_step_conclusion"] == "failure"


def test_main_mismatch_writes_failure_diagnostics(
    monkeypatch: Any, tmp_path: Any
) -> None:
    from scripts.ci import dispatch_hybrid_blind_strict_real_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "list_dispatched_run_ids", lambda *_args, **_kwargs: [1])
    monkeypatch.setattr(mod, "wait_for_new_dispatched_run_id", lambda **_kwargs: 4301)
    monkeypatch.setattr(mod, "watch_run", lambda _run_id, _repo: 1)
    monkeypatch.setattr(
        mod,
        "wait_for_run_conclusion",
        lambda **_kwargs: ("failure", "https://example.com/r/4301"),
    )

    def _fake_run(*args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        command = args[0]
        if command[:3] == ["gh", "workflow", "run"]:
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="", stderr=""
            )
        if command[:4] == ["gh", "run", "view", "4301"] and "jobs" in command:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps(
                    {
                        "jobs": [
                            {
                                "name": "strict-real-gate",
                                "conclusion": "failure",
                                "steps": [
                                    {"name": "Checkout", "conclusion": "success"},
                                    {"name": "Check Hybrid Blind Gate", "conclusion": "failure"},
                                ],
                            }
                        ]
                    }
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    output_json = tmp_path / "hybrid_blind_strict_real_dispatch_mismatch.json"
    rc = mod.main(
        [
            "--hybrid-blind-dxf-dir",
            "data/blind_dxf",
            "--output-json",
            str(output_json),
            "--expected-conclusion",
            "success",
        ]
    )
    assert rc == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    diagnostics = payload.get("failure_diagnostics") or {}
    assert diagnostics.get("available") is True
    assert diagnostics.get("failed_job_count") == 1
    failed_jobs = diagnostics.get("failed_jobs") or []
    assert failed_jobs[0]["job_name"] == "strict-real-gate"
    assert failed_jobs[0]["failed_step_name"] == "Check Hybrid Blind Gate"


def test_main_returns_nonzero_with_actionable_hint_when_remote_inputs_missing(
    monkeypatch: Any, capsys: Any
) -> None:
    from scripts.ci import dispatch_hybrid_blind_strict_real_workflow as mod

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "list_dispatched_run_ids", lambda *_args, **_kwargs: [])

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr=(
                "could not create workflow dispatch event: HTTP 422: "
                'Unexpected inputs provided: ["hybrid_blind_enable"]'
            ),
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(["--hybrid-blind-dxf-dir", "data/blind_dxf"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "does not recognize hybrid blind strict-real inputs" in out
