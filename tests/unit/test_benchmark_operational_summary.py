from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import export_benchmark_operational_summary as module


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_operational_summary.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_operational_summary_uses_component_statuses() -> None:
    payload = module.build_operational_summary(
        title="Ops",
        benchmark_scorecard={
            "overall_status": "benchmark_ready_with_feedback_gap",
            "recommendations": ["close the loop"],
            "components": {
                "feedback_flywheel": {"status": "feedback_collected"},
                "assistant_explainability": {"status": "explainability_ready"},
                "review_queue": {"status": "under_control"},
                "ocr_review": {"status": "ocr_ready"},
            },
        },
        feedback_flywheel={"feedback_flywheel": {"feedback_total": 8, "correction_count": 5}},
        assistant_evidence={"total_records": 12},
        review_queue={"total": 0},
        ocr_review={"review_candidate_count": 0},
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "knowledge_outcome_drift_status": "regressed",
            "knowledge_outcome_drift_summary": "Operator outcomes regressed.",
            "recommended_actions": ["Investigate operator review regressions."],
        },
        artifact_paths={"benchmark_scorecard": "scorecard.json"},
    )

    assert payload["overall_status"] == "benchmark_ready_with_feedback_gap"
    assert payload["component_statuses"]["feedback_flywheel"] == "feedback_collected"
    assert payload["component_statuses"]["operator_adoption"] == "guided_manual"
    assert payload["blockers"] == [
        "feedback_flywheel:feedback_collected",
        "operator_adoption:guided_manual",
    ]
    assert payload["key_metrics"]["feedback_total"] == 8
    assert payload["operator_adoption_knowledge_outcome_drift_status"] == "regressed"


def test_build_operational_summary_derives_missing_statuses() -> None:
    payload = module.build_operational_summary(
        title="Ops",
        benchmark_scorecard={},
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={"operational_status": "managed_backlog", "total": 3},
        ocr_review={"review_candidate_count": 4, "automation_ready_count": 0},
        benchmark_operator_adoption={},
        artifact_paths={},
    )

    assert payload["component_statuses"]["feedback_flywheel"] == "missing"
    assert payload["component_statuses"]["assistant_explainability"] == "missing"
    assert payload["component_statuses"]["review_queue"] == "managed_backlog"
    assert payload["component_statuses"]["ocr_review"] == "review_heavy"
    assert payload["component_statuses"]["operator_adoption"] == "missing"
    assert "review_queue:managed_backlog" in payload["blockers"]


def test_export_benchmark_operational_summary_outputs_files(tmp_path: Path) -> None:
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        {
            "overall_status": "benchmark_ready_with_multisignal_evidence",
            "recommendations": ["freeze baseline"],
            "components": {
                "feedback_flywheel": {"status": "closed_loop_ready"},
                "assistant_explainability": {"status": "explainability_ready"},
                "review_queue": {"status": "under_control"},
                "ocr_review": {"status": "ocr_ready"},
            },
        },
    )
    feedback = _write_json(
        tmp_path / "feedback.json",
        {"feedback_flywheel": {"feedback_total": 10, "correction_count": 6}},
    )
    review_queue = _write_json(
        tmp_path / "review_queue.json",
        {"total": 2, "operational_status": "under_control"},
    )
    operator = _write_json(
        tmp_path / "operator.json",
        {
            "adoption_readiness": "operator_ready",
            "knowledge_outcome_drift_status": "stable",
            "knowledge_outcome_drift_summary": "Operator outcomes stable.",
            "recommended_actions": ["Keep operator automation healthy."],
        },
    )
    output_json = tmp_path / "ops.json"
    output_md = tmp_path / "ops.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--title",
            "Operational Summary",
            "--benchmark-scorecard",
            str(scorecard),
            "--feedback-flywheel",
            str(feedback),
            "--review-queue",
            str(review_queue),
            "--benchmark-operator-adoption",
            str(operator),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["component_statuses"]["feedback_flywheel"] == "closed_loop_ready"
    assert payload["component_statuses"]["operator_adoption"] == "operator_ready"
    assert payload["key_metrics"]["feedback_total"] == 10
    assert output_json.exists()
    assert output_md.exists()
    assert "Operational Summary" in output_md.read_text(encoding="utf-8")
