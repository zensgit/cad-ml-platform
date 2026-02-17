from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
UPDATE_SCRIPT = REPO_ROOT / "scripts" / "ci" / "update_graph2d_context_drift_history.py"
RENDER_HISTORY_SCRIPT = REPO_ROOT / "scripts" / "ci" / "render_graph2d_context_drift_history.py"
RENDER_KEY_COUNTS_SCRIPT = REPO_ROOT / "scripts" / "ci" / "render_graph2d_context_drift_key_counts.py"
CHECK_ALERTS_SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_graph2d_context_drift_alerts.py"


def _sample_report() -> dict:
    return {
        "channel": "strict",
        "status": "passed_with_warnings",
        "warnings": ["context mismatch"],
        "failures": [],
        "thresholds": {"context_mismatch_mode": "warn"},
        "baseline_metadata": {
            "context_match": False,
            "context_diff": {"max_samples": {"baseline": 120, "current": 80}},
        },
    }


def test_update_history_script_handles_invalid_config(tmp_path: Path) -> None:
    report_json = tmp_path / "report.json"
    report_json.write_text(json.dumps(_sample_report()), encoding="utf-8")

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("{broken", encoding="utf-8")

    output_json = tmp_path / "history.json"
    result = subprocess.run(
        [
            sys.executable,
            str(UPDATE_SCRIPT),
            "--config",
            str(bad_config),
            "--report-json",
            str(report_json),
            "--output-json",
            str(output_json),
            "--run-id",
            "2001",
            "--run-number",
            "2001",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "max_runs=20" in result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload
    policy_source = payload[0]["policy_source"]
    assert policy_source["config_loaded"] is False
    assert policy_source["resolved_policy"]["max_runs"] == 20


def test_render_history_script_handles_missing_config(tmp_path: Path) -> None:
    history_json = tmp_path / "history.json"
    history_json.write_text(
        json.dumps(
            [
                {
                    "run_number": "1001",
                    "status": "passed",
                    "warning_count": 0,
                    "failure_count": 0,
                    "drift_key_counts": {"max_samples": 1},
                }
            ]
        ),
        encoding="utf-8",
    )

    output_md = tmp_path / "history.md"
    missing_config = tmp_path / "missing.yaml"
    subprocess.run(
        [
            sys.executable,
            str(RENDER_HISTORY_SCRIPT),
            "--config",
            str(missing_config),
            "--history-json",
            str(history_json),
            "--title",
            "History",
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    content = output_md.read_text(encoding="utf-8")
    assert "loaded=False" in content
    assert "resolved_recent_runs=10" in content


def test_render_history_script_writes_summary_json(tmp_path: Path) -> None:
    history_json = tmp_path / "history.json"
    history_json.write_text(
        json.dumps(
            [
                {
                    "run_number": "1001",
                    "status": "passed",
                    "warning_count": 0,
                    "failure_count": 0,
                    "drift_key_counts": {"max_samples": 2},
                }
            ]
        ),
        encoding="utf-8",
    )
    output_md = tmp_path / "history.md"
    output_summary_json = tmp_path / "history_summary.json"
    subprocess.run(
        [
            sys.executable,
            str(RENDER_HISTORY_SCRIPT),
            "--history-json",
            str(history_json),
            "--title",
            "History",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_summary_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_summary_json.read_text(encoding="utf-8"))
    assert payload["history_entries"] == 1
    assert payload["recent_key_totals"]["max_samples"] == 2
    assert payload["policy_source"]["resolved_policy"]["recent_runs"] == 5


def test_render_key_counts_script_handles_invalid_config(tmp_path: Path) -> None:
    report_json = tmp_path / "report.json"
    report_json.write_text(json.dumps(_sample_report()), encoding="utf-8")

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("graph2d_context_drift_alerts: [broken", encoding="utf-8")

    output_md = tmp_path / "key_counts.md"
    subprocess.run(
        [
            sys.executable,
            str(RENDER_KEY_COUNTS_SCRIPT),
            "--config",
            str(bad_config),
            "--report-json",
            str(report_json),
            "--title",
            "Key Counts",
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    content = output_md.read_text(encoding="utf-8")
    assert "loaded=False" in content
    assert "resolved_recent_runs=5" in content
    assert "`max_samples` | 1" in content


def test_render_key_counts_script_writes_summary_json(tmp_path: Path) -> None:
    report_json = tmp_path / "report.json"
    report_json.write_text(json.dumps(_sample_report()), encoding="utf-8")

    output_md = tmp_path / "key_counts.md"
    output_summary_json = tmp_path / "key_counts_summary.json"
    subprocess.run(
        [
            sys.executable,
            str(RENDER_KEY_COUNTS_SCRIPT),
            "--report-json",
            str(report_json),
            "--title",
            "Key Counts",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_summary_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_summary_json.read_text(encoding="utf-8"))
    assert payload["report_count"] == 1
    assert payload["key_counts"]["max_samples"] == 1
    assert payload["policy_source"]["resolved_policy"]["recent_runs"] == 5


def test_check_alerts_script_writes_structured_summary(tmp_path: Path) -> None:
    history_json = tmp_path / "history.json"
    history_json.write_text(
        json.dumps(
            [
                {"run_number": "1", "drift_key_counts": {"max_samples": 1}},
                {"run_number": "2", "drift_key_counts": {"max_samples": 1}},
                {"run_number": "3", "drift_key_counts": {"max_samples": 1}},
            ]
        ),
        encoding="utf-8",
    )
    report_json = tmp_path / "alerts.json"
    subprocess.run(
        [
            sys.executable,
            str(CHECK_ALERTS_SCRIPT),
            "--history-json",
            str(history_json),
            "--output-json",
            str(report_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["status"] == "alerted"
    assert summary["alert_count"] == 1
    assert summary["rows"][0]["key"] == "max_samples"
