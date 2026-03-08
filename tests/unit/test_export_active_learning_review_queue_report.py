from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.active_learning import get_active_learner, reset_active_learner


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_active_learning_review_queue_report.py"
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def active_learning_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "active_learning"
    monkeypatch.setenv("ACTIVE_LEARNING_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ACTIVE_LEARNING_STORE", "file")
    reset_active_learner()
    yield data_dir
    reset_active_learner()


def test_export_review_queue_report_from_data_dir(active_learning_dir: Path) -> None:
    learner = get_active_learner()
    learner.flag_for_review(
        doc_id="doc-critical",
        predicted_type="法兰",
        confidence=0.83,
        alternatives=[],
        score_breakdown={
            "final_decision_source": "hybrid",
            "review_priority": "critical",
            "review_reasons": ["knowledge_conflict"],
        },
        uncertainty_reason="knowledge_conflict",
    )
    learner.flag_for_review(
        doc_id="doc-medium",
        predicted_type="人孔",
        confidence=0.31,
        alternatives=[],
        score_breakdown={
            "final_decision_source": "filename",
            "review_reasons": ["low_confidence"],
        },
        uncertainty_reason="low_confidence",
    )
    output_json = active_learning_dir / "review_queue_report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-dir",
            str(active_learning_dir),
            "--summary-json",
            str(output_json),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "critical_backlog"
    assert payload["total"] == 2
    assert payload["critical_priority_count"] == 1
    assert payload["high_priority_count"] == 1
    assert payload["automation_ready_count"] == 1
    assert payload["by_decision_source"]["hybrid"] == 1
    assert payload["by_decision_source"]["filename"] == 1
    assert payload["top_review_reasons"][0]["name"] == "knowledge_conflict"
    assert "prioritize_critical_review_queue" in payload["recommended_actions"]
    assert output_json.exists()


def test_export_review_queue_report_from_jsonl_input(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "review_queue.jsonl"
    input_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "sample-1",
                        "doc_id": "doc-1",
                        "status": "pending",
                        "sample_type": "knowledge_conflict",
                        "feedback_priority": "high",
                        "decision_source": "titleblock",
                        "uncertainty_reason": "knowledge_conflict",
                        "review_reasons": ["knowledge_conflict", "missing_fields"],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "sample-2",
                        "doc_id": "doc-2",
                        "status": "pending",
                        "sample_type": "low_confidence",
                        "feedback_priority": "medium",
                        "decision_source": "graph2d",
                        "uncertainty_reason": "low_confidence",
                        "review_reasons": ["low_confidence"],
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-path",
            str(input_jsonl),
            "--summary-json",
            str(output_json),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "managed_backlog"
    assert payload["high_priority_count"] == 1
    assert payload["critical_priority_count"] == 0
    assert payload["automation_ready_count"] == 0
    assert payload["by_sample_type"]["knowledge_conflict"] == 1
    assert payload["by_feedback_priority"]["medium"] == 1
    assert payload["top_decision_sources"][0]["name"] == "titleblock"
    assert payload["top_review_reasons"][0]["name"] == "knowledge_conflict"
    assert "drain_high_priority_review_queue" in payload["recommended_actions"]
    assert output_json.exists()
