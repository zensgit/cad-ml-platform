import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_operator_adoption import (
    build_operator_adoption,
    render_markdown,
)


def test_build_operator_adoption_blocked() -> None:
    payload = build_operator_adoption(
        title="Operator Adoption",
        benchmark_release_decision={
            "release_status": "blocked",
            "automation_ready": False,
            "blocking_signals": ["review_queue:critical_backlog"],
        },
        benchmark_release_runbook={
            "release_status": "blocked",
            "next_action": "collect_artifacts",
            "blocking_signals": ["review_queue:critical_backlog"],
            "operator_steps": [
                {
                    "status": "required",
                    "action": "Regenerate missing benchmark artifacts.",
                }
            ],
        },
        review_queue={"operational_status": "critical_backlog", "critical_count": 3},
        feedback_flywheel={"status": "feedback_collected", "correction_count": 2},
        benchmark_knowledge_drift={},
        benchmark_knowledge_outcome_drift={},
        artifact_paths={"benchmark_release_runbook": "runbook.json"},
    )

    assert payload["adoption_readiness"] == "blocked"
    assert payload["operator_mode"] == "clear_blockers"
    assert payload["review_queue_critical_count"] == 3
    assert payload["correction_count"] == 2
    assert payload["artifacts"]["benchmark_release_runbook"]["present"] is True


def test_build_operator_adoption_freeze_ready() -> None:
    payload = build_operator_adoption(
        title="Operator Adoption",
        benchmark_release_decision={
            "release_status": "ready",
            "automation_ready": True,
        },
        benchmark_release_runbook={
            "release_status": "ready",
            "next_action": "freeze_release_baseline",
            "ready_to_freeze_baseline": True,
        },
        review_queue={"operational_status": "under_control", "critical_count": 0},
        feedback_flywheel={"status": "healthy", "feedback_total": 4},
        benchmark_knowledge_drift={},
        benchmark_knowledge_outcome_drift={},
        artifact_paths={},
    )

    assert payload["adoption_readiness"] == "operator_ready"
    assert payload["operator_mode"] == "freeze_ready"
    assert payload["recommended_actions"] == [
        "Continue benchmark monitoring and rerun evaluation on new data."
    ]


def test_build_operator_adoption_knowledge_drift_regressed() -> None:
    payload = build_operator_adoption(
        title="Operator Adoption",
        benchmark_release_decision={
            "release_status": "ready",
            "automation_ready": True,
            "knowledge_drift_status": "regressed",
            "knowledge_drift_summary": "Knowledge coverage regressed in GD&T.",
        },
        benchmark_release_runbook={
            "release_status": "ready",
            "next_action": "freeze_release_baseline",
            "ready_to_freeze_baseline": True,
        },
        review_queue={"operational_status": "under_control", "critical_count": 0},
        feedback_flywheel={"status": "healthy", "feedback_total": 4},
        benchmark_knowledge_drift={
            "knowledge_drift": {
                "status": "regressed",
                "summary": "Knowledge coverage regressed in GD&T.",
                "recommendations": ["Restore GD&T reference coverage before promotion."],
                "counts": {"regressions": 1},
            }
        },
        benchmark_knowledge_outcome_drift={},
        artifact_paths={},
    )

    assert payload["adoption_readiness"] == "guided_manual"
    assert payload["operator_mode"] == "stabilize_knowledge"
    assert payload["knowledge_drift_status"] == "regressed"
    assert payload["recommended_actions"][0] == (
        "Restore GD&T reference coverage before promotion."
    )


def test_build_operator_adoption_knowledge_outcome_drift_regressed() -> None:
    payload = build_operator_adoption(
        title="Operator Adoption",
        benchmark_release_decision={
            "release_status": "ready",
            "automation_ready": True,
            "knowledge_outcome_drift_status": "regressed",
            "knowledge_outcome_drift_summary": "Outcome alignment regressed in standards.",
        },
        benchmark_release_runbook={
            "release_status": "ready",
            "next_action": "freeze_release_baseline",
            "ready_to_freeze_baseline": True,
        },
        review_queue={"operational_status": "under_control", "critical_count": 0},
        feedback_flywheel={"status": "healthy", "feedback_total": 4},
        benchmark_knowledge_drift={},
        benchmark_knowledge_outcome_drift={
            "knowledge_outcome_drift": {
                "status": "regressed",
                "summary": "Outcome alignment regressed in standards.",
                "recommendations": [
                    "Reconcile standards outcome regressions before promotion."
                ],
                "current_status": "knowledge_outcome_correlation_partial",
                "previous_status": "knowledge_outcome_correlation_ready",
                "domain_regressions": ["standards"],
            }
        },
        artifact_paths={},
    )

    assert payload["adoption_readiness"] == "guided_manual"
    assert payload["operator_mode"] == "stabilize_knowledge"
    assert payload["knowledge_outcome_drift_status"] == "regressed"
    assert payload["knowledge_outcome_drift"]["domain_regressions"] == ["standards"]
    assert payload["recommended_actions"][0] == (
        "Reconcile standards outcome regressions before promotion."
    )


def test_render_markdown_and_cli_outputs(tmp_path: Path) -> None:
    release = tmp_path / "release.json"
    runbook = tmp_path / "runbook.json"
    review_queue = tmp_path / "queue.json"
    feedback = tmp_path / "feedback.json"
    drift = tmp_path / "drift.json"
    outcome_drift = tmp_path / "outcome_drift.json"
    output_json = tmp_path / "adoption.json"
    output_md = tmp_path / "adoption.md"

    release.write_text(
        json.dumps(
            {
                "release_status": "review_required",
                "automation_ready": False,
                "review_signals": ["Drain review queue"],
            }
        ),
        encoding="utf-8",
    )
    runbook.write_text(
        json.dumps(
            {
                "release_status": "review_required",
                "next_action": "review_signals",
                "review_signals": ["Drain review queue"],
                "operator_steps": [
                    {
                        "status": "required",
                        "action": "Drain review queue before promotion.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    review_queue.write_text(
        json.dumps({"operational_status": "managed_backlog", "high_count": 5}),
        encoding="utf-8",
    )
    feedback.write_text(
        json.dumps({"status": "feedback_collected", "feedback_total": 8}),
        encoding="utf-8",
    )
    drift.write_text(
        json.dumps(
            {
                "knowledge_drift": {
                    "status": "regressed",
                    "summary": "Knowledge coverage regressed in GD&T.",
                    "recommendations": [
                        "Restore GD&T reference coverage before promotion."
                    ],
                    "counts": {"regressions": 1},
                }
            }
        ),
        encoding="utf-8",
    )
    outcome_drift.write_text(
        json.dumps(
            {
                "knowledge_outcome_drift": {
                    "status": "regressed",
                    "summary": "Outcome alignment regressed in GD&T.",
                    "recommendations": [
                        "Reconcile outcome regressions before promotion."
                    ],
                    "current_status": "knowledge_outcome_correlation_partial",
                    "previous_status": "knowledge_outcome_correlation_ready",
                    "domain_regressions": ["gdt"],
                }
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/export_benchmark_operator_adoption.py",
            "--benchmark-release-decision",
            str(release),
            "--benchmark-release-runbook",
            str(runbook),
            "--review-queue",
            str(review_queue),
            "--feedback-flywheel",
            str(feedback),
            "--benchmark-knowledge-drift",
            str(drift),
            "--benchmark-knowledge-outcome-drift",
            str(outcome_drift),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["adoption_readiness"] == "guided_manual"
    assert payload["operator_mode"] == "stabilize_knowledge"
    assert payload["knowledge_drift_status"] == "regressed"
    assert payload["knowledge_outcome_drift_status"] == "regressed"
    assert payload["recommended_actions"][0] == (
        "Restore GD&T reference coverage before promotion."
    )
    assert output_md.exists()

    rendered = render_markdown(payload)
    assert "# Benchmark Operator Adoption" in rendered
    assert "`operator_mode`: `stabilize_knowledge`" in rendered
    assert "`knowledge_drift_status`: `regressed`" in rendered
    assert "`knowledge_outcome_drift_status`: `regressed`" in rendered
