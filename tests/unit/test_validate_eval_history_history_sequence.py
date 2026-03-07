from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_validate_eval_history_accepts_history_sequence_record(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0.0",
        "timestamp": "2026-03-05T00:00:00Z",
        "branch": "main",
        "commit": "abcdef1",
        "type": "history_sequence",
        "run_context": {
            "runner": "local",
            "machine": "test-machine",
            "os": "Darwin 25.3.0",
            "python": "3.11.0",
            "start_time": "2026-03-05T00:00:00Z",
            "ci_job_id": None,
            "ci_workflow": None,
        },
        "history_metrics": {
            "coverage": 0.88,
            "accuracy_overall": 0.71,
            "macro_f1_overall": 0.69,
            "coarse_accuracy_on_ok": 0.82,
            "coarse_accuracy_overall": 0.77,
            "coarse_macro_f1_on_ok": 0.81,
            "coarse_macro_f1_overall": 0.75,
            "exact_top_mismatches": [
                {"expected": "人孔", "predicted": "捕集口", "count": 2}
            ],
            "coarse_top_mismatches": [],
        },
        "artifacts": {
            "summary_json": str(tmp_path / "summary.json"),
            "results_csv": str(tmp_path / "results.csv"),
            "prototypes_json": str(tmp_path / "prototypes.json"),
            "tune_best_config_json": None,
            "recommended_env_file": str(tmp_path / "recommended_history_sequence.env"),
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
    file_path = tmp_path / "history_eval.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "scripts/validate_eval_history.py", "--dir", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_validate_eval_history_rejects_invalid_history_mismatch_rows(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "1.0.0",
        "timestamp": "2026-03-05T00:00:00Z",
        "branch": "main",
        "commit": "abcdef1",
        "type": "history_sequence",
        "run_context": {
            "runner": "local",
            "machine": "test-machine",
            "os": "Darwin 25.3.0",
            "python": "3.11.0",
            "start_time": "2026-03-05T00:00:00Z",
            "ci_job_id": None,
            "ci_workflow": None,
        },
        "history_metrics": {
            "coverage": 0.88,
            "accuracy_overall": 0.71,
            "macro_f1_overall": 0.69,
            "exact_top_mismatches": [
                {"expected": "人孔", "predicted": "", "count": "bad"}
            ],
        },
        "artifacts": {
            "summary_json": str(tmp_path / "summary.json"),
            "results_csv": str(tmp_path / "results.csv"),
            "prototypes_json": str(tmp_path / "prototypes.json"),
            "tune_best_config_json": None,
            "recommended_env_file": str(tmp_path / "recommended_history_sequence.env"),
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
    file_path = tmp_path / "history_eval_invalid.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "scripts/validate_eval_history.py", "--dir", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "missing expected/predicted" in proc.stdout
    assert "count must be integer" in proc.stdout
