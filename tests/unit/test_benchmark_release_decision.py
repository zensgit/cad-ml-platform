import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_release_decision import (
    build_release_decision,
    render_markdown,
)


def test_build_release_decision_blocks_on_blockers() -> None:
    payload = build_release_decision(
        title="Release Decision",
        benchmark_scorecard={
            "components": {"hybrid": {"status": "healthy"}},
            "recommendations": ["keep monitoring"],
        },
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_companion_summary={
            "blockers": ["review_queue:critical_backlog"],
            "component_statuses": {
                "review_queue": "critical_backlog",
                "assistant_explainability": "partial_coverage",
            },
        },
        benchmark_knowledge_readiness={
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
        },
        benchmark_engineering_signals={
            "engineering_signals": {"status": "partial_engineering_semantics"},
            "recommendations": ["Close engineering gaps."],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "blocking_signals": ["operator:blocker"],
            "recommended_actions": ["Operator fallback only."],
        },
        artifact_paths={
            "benchmark_companion_summary": "companion.json",
            "benchmark_engineering_signals": "engineering.json",
            "benchmark_operator_adoption": "operator.json",
        },
    )

    assert payload["release_status"] == "blocked"
    assert payload["automation_ready"] is False
    assert payload["primary_signal_source"] == "benchmark_companion_summary"
    assert payload["blocking_signals"] == ["review_queue:critical_backlog"]
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_partial"
    assert payload["component_statuses"]["engineering_signals"] == "partial_engineering_semantics"
    assert payload["component_statuses"]["operator_adoption"] == "guided_manual"
    assert payload["knowledge_focus_areas"][0]["component"] == "tolerance"
    assert "Operator fallback only." not in payload["review_signals"]
    assert payload["artifacts"]["benchmark_engineering_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True


def test_build_release_decision_ready_without_blockers() -> None:
    payload = build_release_decision(
        title="Release Decision",
        benchmark_scorecard={
            "components": {
                "hybrid": {"status": "healthy"},
                "qdrant_backend": {"status": "healthy"},
            },
        },
        benchmark_operational_summary={
            "component_statuses": {
                "assistant_explainability": "healthy",
                "review_queue": "healthy",
                "ocr_review": "healthy",
            }
        },
        benchmark_artifact_bundle={"component_statuses": {"feedback_flywheel": "healthy"}},
        benchmark_companion_summary={},
        benchmark_knowledge_readiness={
            "knowledge_readiness": {
                "status": "knowledge_foundation_ready",
                "focus_areas_detail": [],
            },
            "recommendations": [],
        },
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"},
            "recommendations": ["Keep standards coverage stable."],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "operator_ready",
            "recommended_actions": ["Keep operator workflow stable."],
        },
        artifact_paths={"benchmark_scorecard": "scorecard.json"},
    )

    assert payload["release_status"] == "ready"
    assert payload["automation_ready"] is True
    assert payload["review_signals"] == []
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_ready"
    assert payload["component_statuses"]["engineering_signals"] == "engineering_semantics_ready"
    assert payload["component_statuses"]["operator_adoption"] == "operator_ready"
    assert payload["knowledge_focus_areas"] == []


def test_build_release_decision_uses_operator_adoption_blocker_as_fallback() -> None:
    payload = build_release_decision(
        title="Release Decision",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_companion_summary={},
        benchmark_knowledge_readiness={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"},
            "recommendations": [],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "blocked",
            "blocking_signals": ["operator_runbook:missing_dry_run"],
            "recommended_actions": ["Run operator dry run."],
        },
        artifact_paths={"benchmark_operator_adoption": "operator.json"},
    )

    assert payload["release_status"] == "blocked"
    assert payload["automation_ready"] is False
    assert payload["blocking_signals"] == ["operator_runbook:missing_dry_run"]
    assert payload["component_statuses"]["operator_adoption"] == "blocked"


def test_build_release_decision_uses_operator_adoption_recommendation_as_fallback() -> None:
    payload = build_release_decision(
        title="Release Decision",
        benchmark_scorecard={
            "components": {"hybrid": {"status": "healthy"}},
            "recommendations": [],
        },
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_companion_summary={},
        benchmark_knowledge_readiness={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"},
            "recommendations": [],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "blocking_signals": [],
            "recommended_actions": ["Walk operators through the guided manual path."],
        },
        artifact_paths={"benchmark_operator_adoption": "operator.json"},
    )

    assert payload["release_status"] == "review_required"
    assert payload["automation_ready"] is False
    assert payload["review_signals"] == ["Walk operators through the guided manual path."]


def test_render_markdown_and_cli_outputs(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    operational = tmp_path / "operational.json"
    bundle = tmp_path / "bundle.json"
    companion = tmp_path / "companion.json"
    knowledge = tmp_path / "knowledge.json"
    engineering = tmp_path / "engineering.json"
    operator = tmp_path / "operator.json"
    output_json = tmp_path / "release.json"
    output_md = tmp_path / "release.md"
    scorecard.write_text(
        json.dumps({"components": {"hybrid": {"status": "healthy"}}}),
        encoding="utf-8",
    )
    operational.write_text(
        json.dumps({"component_statuses": {"review_queue": "healthy"}}),
        encoding="utf-8",
    )
    bundle.write_text(
        json.dumps({"component_statuses": {"assistant_explainability": "healthy"}}),
        encoding="utf-8",
    )
    companion.write_text(
        json.dumps(
            {
                "component_statuses": {
                    "ocr_review": "healthy",
                    "qdrant_backend": "healthy",
                }
            }
        ),
        encoding="utf-8",
    )
    knowledge.write_text(
        json.dumps(
            {
                "knowledge_readiness": {
                    "status": "knowledge_foundation_ready",
                    "focus_areas_detail": [],
                },
                "recommendations": [],
            }
        ),
        encoding="utf-8",
    )
    engineering.write_text(
        json.dumps(
            {
                "engineering_signals": {
                    "status": "engineering_semantics_ready",
                },
                "recommendations": ["Keep standards coverage stable."],
            }
        ),
        encoding="utf-8",
    )
    operator.write_text(
        json.dumps(
            {
                "adoption_readiness": "operator_ready",
                "recommended_actions": ["Keep operator workflow stable."],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/export_benchmark_release_decision.py",
            "--benchmark-scorecard",
            str(scorecard),
            "--benchmark-operational-summary",
            str(operational),
            "--benchmark-artifact-bundle",
            str(bundle),
            "--benchmark-companion-summary",
            str(companion),
            "--benchmark-knowledge-readiness",
            str(knowledge),
            "--benchmark-engineering-signals",
            str(engineering),
            "--benchmark-operator-adoption",
            str(operator),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["release_status"] == "ready"
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_ready"
    assert payload["component_statuses"]["engineering_signals"] == "engineering_semantics_ready"
    assert payload["component_statuses"]["operator_adoption"] == "operator_ready"
    assert payload["artifacts"]["benchmark_knowledge_readiness"]["present"] is True
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True
    assert output_md.exists()

    rendered = render_markdown(payload)
    assert "# Benchmark Release Decision" in rendered
    assert "`release_status`: `ready`" in rendered
    assert "`knowledge_readiness`: `knowledge_foundation_ready`" in rendered
    assert "`operator_adoption`: `operator_ready`" in rendered
