from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "index_graph2d_context_drift_artifacts.py"
)


def test_build_index_combines_core_summaries() -> None:
    from scripts.ci.index_graph2d_context_drift_artifacts import build_index

    payload = build_index(
        alerts_report={"summary": {"status": "alerted", "alert_count": 2, "policy_source": {}}},
        history_summary={
            "history_entries": 6,
            "recent_window": 5,
            "policy_source": {},
        },
        key_counts_summary={
            "key_counts": {"max_samples": 3, "seeds": 1},
            "policy_source": {},
        },
        history_payload=[{"run_id": "1"}],
        artifact_paths={
            "alerts_json": "/__missing__/a.json",
            "history_summary_json": "/__missing__/b.json",
            "key_counts_summary_json": "/__missing__/c.json",
            "history_json": "/__missing__/d.json",
        },
    )
    assert payload["schema_version"] == "1.0.0"
    assert payload["overview"]["status"] == "alerted"
    assert payload["overview"]["severity"] == "failed"
    assert payload["overview"]["alert_count"] == 2
    assert payload["overview"]["history_entries"] == 6
    assert payload["overview"]["drift_key_count"] == 2
    assert payload["overview"]["top_drift_key"]["key"] == "max_samples"


def test_index_script_handles_missing_inputs(tmp_path: Path) -> None:
    output_json = tmp_path / "index.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--alerts-json",
            str(tmp_path / "missing_alerts.json"),
            "--history-summary-json",
            str(tmp_path / "missing_history_summary.json"),
            "--key-counts-summary-json",
            str(tmp_path / "missing_key_counts_summary.json"),
            "--history-json",
            str(tmp_path / "missing_history.json"),
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["overview"]["status"] == "clear"
    assert payload["overview"]["severity"] == "failed"
    assert payload["overview"]["alert_count"] == 0
    assert payload["artifacts"]["alerts_report"]["exists"] is False


def test_index_script_with_realistic_payloads(tmp_path: Path) -> None:
    alerts_json = tmp_path / "alerts.json"
    history_summary_json = tmp_path / "history_summary.json"
    key_counts_summary_json = tmp_path / "key_counts_summary.json"
    history_json = tmp_path / "history.json"
    output_json = tmp_path / "index.json"

    alerts_json.write_text(
        json.dumps(
            {
                "summary": {
                    "status": "alerted",
                    "alert_count": 1,
                    "policy_source": {"config_loaded": True},
                }
            }
        ),
        encoding="utf-8",
    )
    history_summary_json.write_text(
        json.dumps(
            {
                "history_entries": 4,
                "recent_window": 4,
                "policy_source": {"config_loaded": True},
            }
        ),
        encoding="utf-8",
    )
    key_counts_summary_json.write_text(
        json.dumps(
            {
                "key_counts": {"max_samples": 2},
                "policy_source": {"config_loaded": True},
            }
        ),
        encoding="utf-8",
    )
    history_json.write_text(json.dumps([{"run_id": "1"}, {"run_id": "2"}]), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--alerts-json",
            str(alerts_json),
            "--history-summary-json",
            str(history_summary_json),
            "--key-counts-summary-json",
            str(key_counts_summary_json),
            "--history-json",
            str(history_json),
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["overview"]["status"] == "alerted"
    assert payload["overview"]["severity"] == "alerted"
    assert payload["overview"]["top_drift_key"]["key"] == "max_samples"
    assert payload["artifacts"]["history_raw"]["exists"] is True
