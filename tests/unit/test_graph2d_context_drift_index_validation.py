from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "validate_graph2d_context_drift_index.py"
)
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "graph2d_context_drift_index_schema.json"
)


def _valid_index_payload() -> dict:
    return {
        "schema_version": "1.0.0",
        "generated_at": "2026-02-17T00:00:00+00:00",
        "overview": {
            "status": "clear",
            "severity": "clear",
            "severity_reason": "no context drift signal",
            "alert_count": 0,
            "history_entries": 1,
            "recent_window": 1,
            "drift_key_count": 0,
            "top_drift_key": {"key": "", "count": 0},
            "artifact_coverage": {"present": 3, "total": 3},
        },
        "artifacts": {
            "alerts_report": {"path": "/tmp/a.json", "exists": True},
            "history_summary": {"path": "/tmp/b.json", "exists": True},
            "key_counts_summary": {"path": "/tmp/c.json", "exists": True},
            "history_raw": {"path": "/tmp/d.json", "exists": False},
        },
        "policy_sources": {},
        "summaries": {},
    }


def test_validate_index_script_accepts_valid_payload(tmp_path: Path) -> None:
    index_json = tmp_path / "index.json"
    index_json.write_text(json.dumps(_valid_index_payload()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--index-json",
            str(index_json),
            "--schema-json",
            str(SCHEMA_PATH),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "index_schema_valid=true" in result.stdout


def test_validate_index_script_rejects_invalid_severity(tmp_path: Path) -> None:
    payload = _valid_index_payload()
    payload["overview"]["severity"] = "unknown"
    index_json = tmp_path / "index.json"
    index_json.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--index-json",
            str(index_json),
            "--schema-json",
            str(SCHEMA_PATH),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "index_schema_valid=false" in result.stdout
