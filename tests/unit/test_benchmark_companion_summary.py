import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_companion_summary import (
    build_companion_summary,
    render_markdown,
)


def test_build_companion_summary_prefers_bundle_and_flags_attention() -> None:
    payload = build_companion_summary(
        title="Benchmark Companion",
        benchmark_scorecard={
            "overall_status": "gap_detected",
            "components": {
                "hybrid": {"status": "healthy"},
                "review_queue": {"status": "healthy"},
            },
            "recommendations": ["improve scorecard coverage"],
        },
        benchmark_operational_summary={
            "overall_status": "attention_required",
            "component_statuses": {
                "assistant_explainability": "partial_coverage",
                "review_queue": "managed_backlog",
            },
            "blockers": ["assistant_explainability:partial_coverage"],
            "recommendations": ["raise assistant evidence coverage"],
        },
        benchmark_artifact_bundle={
            "overall_status": "attention_required",
            "component_statuses": {
                "assistant_explainability": "partial_coverage",
                "review_queue": "managed_backlog",
                "ocr_review": "managed_review",
            },
            "blockers": ["review_queue:managed_backlog"],
            "recommendations": ["reduce review queue backlog"],
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
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "partial_engineering_semantics"},
            "recommendations": ["Close engineering gaps."],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "recommended_actions": ["Review operator blockers."],
        },
        artifact_paths={
            "benchmark_scorecard": "scorecard.json",
            "benchmark_operational_summary": "operational.json",
            "benchmark_artifact_bundle": "bundle.json",
            "benchmark_engineering_signals": "engineering.json",
            "benchmark_operator_adoption": "operator.json",
        },
    )

    assert payload["overall_status"] == "attention_required"
    assert payload["review_surface"] == "attention_required"
    assert payload["primary_gap"] == "review_queue:managed_backlog"
    assert payload["component_statuses"]["assistant_explainability"] == "partial_coverage"
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_partial"
    assert payload["component_statuses"]["engineering_signals"] == "partial_engineering_semantics"
    assert payload["component_statuses"]["operator_adoption"] == "guided_manual"
    assert payload["knowledge_focus_areas"][0]["component"] == "tolerance"
    assert payload["recommended_actions"] == ["reduce review queue backlog"]
    assert payload["artifacts"]["benchmark_artifact_bundle"]["present"] is True
    assert payload["artifacts"]["benchmark_engineering_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True


def test_render_markdown_includes_sections() -> None:
    payload = {
        "title": "Benchmark Companion",
        "overall_status": "healthy",
        "review_surface": "ready",
        "primary_gap": "none",
        "knowledge_drift_summary": "status=stable; current=knowledge_foundation_ready",
        "component_statuses": {"hybrid": "healthy"},
        "recommended_actions": ["keep monitoring"],
        "blockers": [],
        "knowledge_drift_component_changes": [
            {
                "component": "standards",
                "previous_status": "ready",
                "current_status": "ready",
                "trend": "stable",
                "reference_item_delta": 0,
            }
        ],
        "knowledge_drift_recommendations": [
            "Knowledge readiness is stable against the previous benchmark baseline."
        ],
        "artifacts": {
            "benchmark_engineering_signals": {
                "present": True,
                "path": "engineering.json",
            },
            "benchmark_operator_adoption": {
                "present": True,
                "path": "operator.json",
            },
            "benchmark_artifact_bundle": {
                "present": True,
                "path": "bundle.json",
            },
            "benchmark_knowledge_readiness": {
                "present": True,
                "path": "knowledge.json",
            },
            "benchmark_knowledge_drift": {
                "present": True,
                "path": "knowledge_drift.json",
            },
        },
        "knowledge_focus_areas": [
            {
                "component": "gdt",
                "status": "missing",
                "priority": "high",
                "action": "Expand GD&T coverage.",
            }
        ],
    }

    rendered = render_markdown(payload)
    assert "# Benchmark Companion" in rendered
    assert "`review_surface`: `ready`" in rendered
    assert "## Recommended Actions" in rendered
    assert "bundle.json" in rendered
    assert "knowledge.json" in rendered
    assert "knowledge_drift.json" in rendered
    assert "engineering.json" in rendered
    assert "operator.json" in rendered
    assert "Expand GD&T coverage." in rendered
    assert "status=stable; current=knowledge_foundation_ready" in rendered


def test_cli_writes_outputs(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    operational = tmp_path / "operational.json"
    bundle = tmp_path / "bundle.json"
    knowledge = tmp_path / "knowledge.json"
    knowledge_drift = tmp_path / "knowledge_drift.json"
    engineering = tmp_path / "engineering.json"
    operator = tmp_path / "operator.json"
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    scorecard.write_text(
        json.dumps(
            {
                "overall_status": "healthy",
                "components": {"hybrid": {"status": "healthy"}},
                "recommendations": ["keep monitoring"],
            }
        ),
        encoding="utf-8",
    )
    operational.write_text(
        json.dumps(
            {
                "overall_status": "healthy",
                "component_statuses": {"review_queue": "healthy"},
            }
        ),
        encoding="utf-8",
    )
    bundle.write_text(
        json.dumps(
            {
                "overall_status": "healthy",
                "component_statuses": {
                    "assistant_explainability": "explainability_ready",
                    "review_queue": "healthy",
                    "ocr_review": "ocr_ready",
                },
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
    knowledge_drift.write_text(
        json.dumps(
            {
                "knowledge_drift": {
                    "status": "stable",
                    "current_status": "knowledge_foundation_ready",
                    "previous_status": "knowledge_foundation_ready",
                    "reference_item_delta": 0,
                    "regressions": [],
                    "improvements": [],
                    "new_focus_areas": [],
                    "component_changes": [],
                },
                "recommendations": [
                    "Knowledge readiness is stable against the previous benchmark baseline."
                ],
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
                "recommended_actions": ["Keep operator automation healthy."],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/export_benchmark_companion_summary.py",
            "--benchmark-scorecard",
            str(scorecard),
            "--benchmark-operational-summary",
            str(operational),
            "--benchmark-artifact-bundle",
            str(bundle),
            "--benchmark-knowledge-readiness",
            str(knowledge),
            "--benchmark-knowledge-drift",
            str(knowledge_drift),
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
    assert payload["overall_status"] == "healthy"
    assert payload["review_surface"] == "ready"
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_ready"
    assert payload["component_statuses"]["knowledge_drift"] == "stable"
    assert payload["component_statuses"]["engineering_signals"] == "engineering_semantics_ready"
    assert payload["component_statuses"]["operator_adoption"] == "operator_ready"
    assert payload["knowledge_focus_areas"] == []
    assert payload["knowledge_drift_summary"].startswith("status=stable")
    assert output_md.exists()


def test_build_companion_summary_exposes_knowledge_drift_passthrough() -> None:
    payload = build_companion_summary(
        title="Benchmark Companion",
        benchmark_scorecard={
            "overall_status": "healthy",
            "components": {"hybrid": {"status": "healthy"}},
        },
        benchmark_operational_summary={},
        benchmark_artifact_bundle={
            "overall_status": "healthy",
            "component_statuses": {
                "assistant_explainability": "explainability_ready",
                "review_queue": "healthy",
                "ocr_review": "ocr_ready",
            },
        },
        benchmark_knowledge_readiness={
            "knowledge_readiness": {
                "status": "knowledge_foundation_ready",
                "focus_areas_detail": [],
            }
        },
        benchmark_knowledge_drift={
            "knowledge_drift": {
                "status": "regressed",
                "current_status": "knowledge_foundation_partial",
                "previous_status": "knowledge_foundation_ready",
                "reference_item_delta": -20,
                "regressions": ["standards"],
                "improvements": [],
                "new_focus_areas": ["standards"],
                "component_changes": [
                    {
                        "component": "standards",
                        "previous_status": "ready",
                        "current_status": "partial",
                        "trend": "regressed",
                        "reference_item_delta": -20,
                    }
                ],
            },
            "recommendations": [
                "Resolve knowledge regressions before claiming the benchmark "
                "surpass baseline remains stable."
            ],
        },
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"},
            "recommendations": [],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "operator_ready",
            "recommended_actions": [],
        },
        artifact_paths={"benchmark_knowledge_drift": "knowledge_drift.json"},
    )

    assert payload["component_statuses"]["knowledge_drift"] == "regressed"
    assert payload["primary_gap"] == "knowledge_drift:regressed"
    assert payload["recommended_actions"] == [
        "Resolve knowledge regressions before claiming the benchmark "
        "surpass baseline remains stable."
    ]
    assert payload["artifacts"]["benchmark_knowledge_drift"]["present"] is True
    assert payload["knowledge_drift"]["status"] == "regressed"
    assert "regressions=standards" in payload["knowledge_drift_summary"]
