from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_valid_hybrid_blind(path: Path) -> None:
    payload = {
        "schema_version": "1.0.0",
        "timestamp": "2026-03-13T00:00:00Z",
        "branch": "main",
        "commit": "abcdef1",
        "type": "hybrid_blind",
        "run_context": {
            "runner": "local",
            "machine": "test-machine",
            "os": "Darwin 25.3.0",
            "python": "3.11.0",
            "start_time": "2026-03-13T00:00:00Z",
            "ci_job_id": None,
            "ci_workflow": None,
        },
        "metrics": {
            "sample_size": 40,
            "weak_label_coverage": 0.9,
            "hybrid_accuracy": 0.72,
            "graph2d_accuracy": 0.31,
            "hybrid_gain_vs_graph2d": 0.41,
            "gate_status": "passed",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _run_validate(tmp_path: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    args = [sys.executable, "scripts/validate_eval_history.py", "--dir", str(tmp_path)]
    if extra_args:
        args.extend(extra_args)
    return subprocess.run(
        args,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_eval_history_skips_default_hybrid_blind_sidecar_reports(tmp_path: Path) -> None:
    _write_valid_hybrid_blind(tmp_path / "20260313_000000_main_hybrid_blind.json")
    (tmp_path / "hybrid_blind_drift_alert_report.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )

    proc = _run_validate(tmp_path)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Skipped 1 JSON files by exclude patterns" in proc.stdout


def test_validate_eval_history_still_fails_for_unknown_non_history_json(tmp_path: Path) -> None:
    _write_valid_hybrid_blind(tmp_path / "20260313_000000_main_hybrid_blind.json")
    (tmp_path / "custom_sidecar_report.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )

    proc = _run_validate(tmp_path)
    assert proc.returncode != 0
    assert "Missing required field: timestamp" in proc.stdout


def test_validate_eval_history_supports_custom_exclude_glob(tmp_path: Path) -> None:
    _write_valid_hybrid_blind(tmp_path / "20260313_000000_main_hybrid_blind.json")
    (tmp_path / "custom_sidecar_report.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )

    proc = _run_validate(
        tmp_path,
        ["--exclude-glob", "custom_sidecar_report.json"],
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Skipped 1 JSON files by exclude patterns" in proc.stdout
