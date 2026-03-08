import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_release_runbook import (
    build_release_runbook,
    render_markdown,
)


def test_build_release_runbook_requires_blocker_resolution() -> None:
    payload = build_release_runbook(
        title="Benchmark Release Runbook",
        benchmark_release_decision={
            "release_status": "blocked",
            "automation_ready": False,
            "primary_signal_source": "benchmark_companion_summary",
            "blocking_signals": ["review_queue:critical_backlog"],
            "artifacts": {
                "benchmark_scorecard": {"path": "scorecard.json"},
                "benchmark_operational_summary": {"path": "operational.json"},
            },
        },
        benchmark_companion_summary={
            "blockers": ["review_queue:critical_backlog"],
        },
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness={
            "knowledge_readiness": {
                "status": "knowledge_foundation_partial",
                "focus_areas_detail": [
                    {
                        "component": "gdt",
                        "status": "missing",
                        "priority": "high",
                        "action": "Expand GD&T coverage.",
                    }
                ],
            },
            "recommendations": ["Raise tolerance/GD&T readiness."],
        },
        benchmark_engineering_signals={
            "engineering_signals": {"status": "partial_engineering_semantics"},
            "recommendations": ["Close engineering gaps."],
        },
        benchmark_operator_adoption={
            "status": "attention_required",
            "summary": "Operator onboarding still needs a dry run.",
            "signals": ["operator_playbook:needs_walkthrough"],
            "actions": ["Schedule an operator handoff dry run."],
        },
        artifact_paths={
            "benchmark_release_decision": "release.json",
            "benchmark_engineering_signals": "engineering.json",
            "benchmark_operator_adoption": "operator_adoption.json",
        },
    )

    assert payload["release_status"] == "blocked"
    assert payload["engineering_status"] == "partial_engineering_semantics"
    assert payload["knowledge_status"] == "knowledge_foundation_partial"
    assert payload["knowledge_focus_areas"][0]["component"] == "gdt"
    assert payload["next_action"] == "collect_artifacts"
    assert "benchmark_artifact_bundle" in payload["missing_artifacts"]
    assert payload["artifacts"]["benchmark_engineering_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True
    assert payload["operator_adoption"]["actions"] == [
        "Schedule an operator handoff dry run."
    ]
    assert payload["operator_steps"][1]["key"] == "resolve_blockers"
    assert payload["operator_steps"][1]["status"] == "required"
    adoption_step = next(
        step
        for step in payload["operator_steps"]
        if step["key"] == "operator_adoption_guidance"
    )
    assert adoption_step["status"] == "guidance"
    assert "low-priority guidance" in adoption_step["action"]


def test_build_release_runbook_freezes_when_ready() -> None:
    payload = build_release_runbook(
        title="Benchmark Release Runbook",
        benchmark_release_decision={
            "release_status": "ready",
            "automation_ready": True,
            "primary_signal_source": "benchmark_release_decision",
            "artifacts": {
                "benchmark_scorecard": {"path": "scorecard.json"},
                "benchmark_operational_summary": {"path": "operational.json"},
                "benchmark_companion_summary": {"path": "companion.json"},
                "benchmark_artifact_bundle": {"path": "bundle.json"},
            },
        },
        benchmark_companion_summary={"overall_status": "healthy"},
        benchmark_artifact_bundle={"overall_status": "healthy"},
        benchmark_knowledge_readiness={
            "knowledge_readiness": {
                "status": "knowledge_foundation_ready",
                "focus_areas_detail": [],
            },
            "recommendations": [],
        },
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"},
        },
        benchmark_operator_adoption={},
        artifact_paths={
            "benchmark_release_decision": "release.json",
            "benchmark_companion_summary": "companion.json",
            "benchmark_artifact_bundle": "bundle.json",
            "benchmark_engineering_signals": "engineering.json",
        },
    )

    assert payload["ready_to_freeze_baseline"] is True
    assert payload["engineering_status"] == "engineering_semantics_ready"
    assert payload["knowledge_status"] == "knowledge_foundation_ready"
    assert payload["knowledge_focus_areas"] == []
    assert payload["next_action"] == "freeze_release_baseline"
    assert "benchmark_operator_adoption" not in payload["missing_artifacts"]
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is False
    assert payload["operator_steps"][-1]["status"] == "ready"


def test_render_markdown_and_cli_outputs(tmp_path: Path) -> None:
    release = tmp_path / "release.json"
    companion = tmp_path / "companion.json"
    bundle = tmp_path / "bundle.json"
    knowledge = tmp_path / "knowledge.json"
    engineering = tmp_path / "engineering.json"
    operator_adoption = tmp_path / "operator_adoption.json"
    output_json = tmp_path / "runbook.json"
    output_md = tmp_path / "runbook.md"

    release.write_text(
        json.dumps(
            {
                "release_status": "review_required",
                "automation_ready": False,
                "primary_signal_source": "benchmark_companion_summary",
                "review_signals": ["Drain review queue"],
                "artifacts": {
                    "benchmark_scorecard": {"path": "scorecard.json"},
                    "benchmark_operational_summary": {"path": "operational.json"},
                    "benchmark_companion_summary": {"path": "companion.json"},
                    "benchmark_artifact_bundle": {"path": "bundle.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    companion.write_text(
        json.dumps({"recommended_actions": ["Drain review queue"]}),
        encoding="utf-8",
    )
    bundle.write_text(json.dumps({"overall_status": "attention_required"}), encoding="utf-8")
    knowledge.write_text(
        json.dumps(
            {
                "knowledge_readiness": {
                    "status": "knowledge_foundation_partial",
                    "focus_areas_detail": [
                        {
                            "component": "tolerance",
                            "status": "partial",
                            "priority": "medium",
                            "action": "Backfill tolerance coverage.",
                        }
                    ],
                },
                "recommendations": ["Raise tolerance/GD&T readiness."],
            }
        ),
        encoding="utf-8",
    )
    engineering.write_text(
        json.dumps(
            {
                "engineering_signals": {"status": "partial_engineering_semantics"},
                "recommendations": ["Close engineering gaps."],
            }
        ),
        encoding="utf-8",
    )
    operator_adoption.write_text(
        json.dumps(
            {
                "status": "attention_required",
                "summary": "Operators still need rollout support.",
                "signals": ["operator_shift_handoff:pending"],
                "actions": ["Book an operator office-hours review."],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/export_benchmark_release_runbook.py",
            "--benchmark-release-decision",
            str(release),
            "--benchmark-companion-summary",
            str(companion),
            "--benchmark-artifact-bundle",
            str(bundle),
            "--benchmark-knowledge-readiness",
            str(knowledge),
            "--benchmark-engineering-signals",
            str(engineering),
            "--benchmark-operator-adoption",
            str(operator_adoption),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["release_status"] == "review_required"
    assert payload["engineering_status"] == "partial_engineering_semantics"
    assert payload["knowledge_status"] == "knowledge_foundation_partial"
    assert payload["knowledge_focus_areas"][0]["component"] == "tolerance"
    assert payload["next_action"] == "review_signals"
    assert payload["artifacts"]["benchmark_engineering_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True
    assert payload["operator_adoption"]["signals"] == [
        "operator_shift_handoff:pending"
    ]
    assert payload["operator_adoption"]["actions"] == [
        "Book an operator office-hours review."
    ]
    assert output_md.exists()

    rendered = render_markdown(payload)
    assert "# Benchmark Release Runbook" in rendered
    assert "`engineering_status`: `partial_engineering_semantics`" in rendered
    assert "`knowledge_status`: `knowledge_foundation_partial`" in rendered
    assert "`next_action`: `review_signals`" in rendered
    assert "Backfill tolerance coverage." in rendered
    assert "## Operator Adoption" in rendered
    assert "operator_shift_handoff:pending" in rendered
    assert "Book an operator office-hours review." in rendered
    assert "`benchmark_operator_adoption`: present=`True`" in rendered
