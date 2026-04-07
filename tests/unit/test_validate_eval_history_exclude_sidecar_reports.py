from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_validator(tmp_path: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
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


def _write_valid_history_record(path: Path) -> None:
    payload = {
        "schema_version": "1.0.0",
        "timestamp": "2026-04-07T00:00:00Z",
        "branch": "main",
        "commit": "abcdef1",
        "type": "history_sequence",
        "run_context": {
            "runner": "local",
            "machine": "test-machine",
            "os": "Darwin 25.3.0",
            "python": "3.11.0",
            "start_time": "2026-04-07T00:00:00Z",
            "ci_job_id": None,
            "ci_workflow": None,
        },
        "history_metrics": {
            "coverage": 0.9,
            "accuracy_overall": 0.8,
            "macro_f1_overall": 0.79,
            "coarse_accuracy_on_ok": 0.85,
            "coarse_accuracy_overall": 0.82,
            "coarse_macro_f1_on_ok": 0.84,
            "coarse_macro_f1_overall": 0.81,
            "exact_top_mismatches": [],
            "coarse_top_mismatches": [],
        },
        "artifacts": {
            "summary_json": str(path.parent / "summary.json"),
            "results_csv": str(path.parent / "results.csv"),
            "prototypes_json": str(path.parent / "prototypes.json"),
            "tune_best_config_json": None,
            "recommended_env_file": str(path.parent / "recommended_history_sequence.env"),
        },
        "tuning": {
            "enabled": False,
            "source": "configured",
            "objective": "macro_f1_overall",
            "configured_token_weight": 1.0,
            "configured_bigram_weight": 1.0,
            "selected_token_weight": 1.0,
            "selected_bigram_weight": 1.0,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_validate_eval_history_skips_default_hybrid_blind_sidecar_reports(tmp_path: Path) -> None:
    _write_valid_history_record(tmp_path / "history_eval.json")
    for sidecar in (
        "hybrid_blind_drift_alert_report.json",
        "hybrid_blind_drift_threshold_suggestion.json",
    ):
        (tmp_path / sidecar).write_text(json.dumps({"surface_kind": sidecar}), encoding="utf-8")

    proc = _run_validator(tmp_path)

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Skipped 2 JSON files by exclude patterns" in proc.stdout


def test_validate_eval_history_supports_custom_exclude_glob(tmp_path: Path) -> None:
    _write_valid_history_record(tmp_path / "history_eval.json")
    (tmp_path / "custom_sidecar_report.json").write_text(
        json.dumps({"surface_kind": "custom_sidecar_report"}),
        encoding="utf-8",
    )

    proc = _run_validator(tmp_path, ["--exclude-glob", "custom_sidecar_report.json"])

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Skipped 1 JSON files by exclude patterns" in proc.stdout


def test_validate_eval_history_still_fails_for_unknown_non_history_json(tmp_path: Path) -> None:
    _write_valid_history_record(tmp_path / "history_eval.json")
    (tmp_path / "unexpected_surface.json").write_text(
        json.dumps({"surface_kind": "unexpected_surface"}),
        encoding="utf-8",
    )

    proc = _run_validator(tmp_path)

    assert proc.returncode == 1
    assert "unexpected_surface.json" in proc.stdout
    assert "Missing required field" in proc.stdout
