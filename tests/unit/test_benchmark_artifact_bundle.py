from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import export_benchmark_artifact_bundle as module

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "export_benchmark_artifact_bundle.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_build_bundle_prefers_operational_summary() -> None:
    payload = module.build_bundle(
        title="Benchmark Artifact Bundle",
        benchmark_scorecard={
            "overall_status": "benchmark_ready_with_feedback_gap",
            "components": {"hybrid": {"status": "ready"}},
        },
        benchmark_operational_summary={
            "overall_status": "attention_required",
            "component_statuses": {
                "feedback_flywheel": "feedback_collected",
                "assistant_explainability": "evidence_partial",
                "review_queue": "managed_backlog",
                "ocr_review": "review_heavy",
            },
            "blockers": ["feedback backlog"],
            "recommendations": ["Close the review queue."],
        },
        benchmark_companion_summary={},
        benchmark_release_decision={},
        benchmark_knowledge_readiness={
            "knowledge_readiness": {
                "status": "knowledge_foundation_partial",
                "focus_areas_detail": [
                    {
                        "component": "standards",
                        "status": "missing",
                        "priority": "high",
                        "missing_metrics": ["thread_count", "bearing_count", "oring_count"],
                    }
                ],
                "priority_domains": ["standards"],
                "domains": {
                    "standards": {
                        "status": "missing",
                        "focus_components": ["standards", "design_standards"],
                        "missing_metrics": ["thread_count", "bearing_count", "oring_count"],
                    }
                },
                "domain_focus_areas": [
                    {
                        "domain": "standards",
                        "status": "missing",
                        "priority": "high",
                        "action": "Expand standards coverage.",
                    }
                ],
            }
        },
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "partial_engineering_semantics"},
            "recommendations": ["Expand knowledge coverage."],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "knowledge_drift_status": "regressed",
            "knowledge_drift_summary": "Standards coverage regressed.",
            "knowledge_drift": {
                "recommendations": ["Restore standards baseline coverage."],
            },
            "recommended_actions": ["Resolve operator blockers."],
        },
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={
            "benchmark_operational_summary": "operational.json",
            "benchmark_engineering_signals": "engineering.json",
            "benchmark_operator_adoption": "operator.json",
        },
    )
    assert payload["overall_status"] == "attention_required"
    assert payload["available_artifact_count"] == 5
    assert payload["component_statuses"]["feedback_flywheel"] == "feedback_collected"
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_partial"
    assert payload["knowledge_focus_area_count"] == 1
    assert payload["knowledge_focus_areas"][0]["component"] == "standards"
    assert payload["knowledge_priority_domains"] == ["standards"]
    assert payload["knowledge_domains"]["standards"]["status"] == "missing"
    assert payload["knowledge_domain_focus_areas"][0]["domain"] == "standards"
    assert payload["component_statuses"]["engineering_signals"] == "partial_engineering_semantics"
    assert payload["component_statuses"]["operator_adoption"] == "guided_manual"
    assert payload["operator_adoption_knowledge_drift"]["status"] == "regressed"
    assert payload["blockers"] == ["feedback backlog"]
    assert payload["recommendations"] == ["Close the review queue."]


def test_build_bundle_falls_back_to_scorecard() -> None:
    payload = module.build_bundle(
        title="Benchmark Artifact Bundle",
        benchmark_scorecard={
            "overall_status": "benchmark_ready_with_multisignal_evidence",
            "components": {
                "hybrid": {"status": "ready"},
                "assistant_explainability": {"status": "evidence_ready"},
            },
            "recommendations": ["Freeze this run."],
        },
        benchmark_operational_summary={},
        benchmark_companion_summary={},
        benchmark_release_decision={},
        benchmark_knowledge_readiness={
            "knowledge_readiness": {
                "status": "knowledge_foundation_ready",
                "focus_areas_detail": [],
                "priority_domains": [],
                "domains": {
                    "tolerance": {
                        "status": "ready",
                        "focus_components": [],
                        "missing_metrics": [],
                    }
                },
                "domain_focus_areas": [],
            }
        },
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"},
            "recommendations": ["Keep standards coverage stable."],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "operator_ready",
            "knowledge_drift_status": "stable",
            "knowledge_drift_summary": "Knowledge baseline stable.",
            "recommended_actions": ["Keep operator automation healthy."],
        },
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={"benchmark_scorecard": "scorecard.json"},
    )
    assert payload["overall_status"] == "benchmark_ready_with_multisignal_evidence"
    assert payload["component_statuses"]["assistant_explainability"] == "evidence_ready"
    assert payload["recommendations"] == ["Freeze this run."]
    assert payload["knowledge_focus_area_count"] == 0
    assert payload["knowledge_priority_domains"] == []
    assert payload["knowledge_domains"]["tolerance"]["status"] == "ready"
    assert payload["knowledge_domain_focus_areas"] == []
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_ready"
    assert payload["component_statuses"]["engineering_signals"] == "engineering_semantics_ready"
    assert payload["component_statuses"]["operator_adoption"] == "operator_ready"
    assert payload["operator_adoption_knowledge_drift"]["status"] == "stable"


def test_export_benchmark_artifact_bundle_outputs_files(tmp_path: Path) -> None:
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        {
            "overall_status": "benchmark_ready_with_multisignal_evidence",
            "components": {"hybrid": {"status": "ready"}},
            "recommendations": ["Freeze this run."],
        },
    )
    operational = _write_json(
        tmp_path / "operational.json",
        {
            "overall_status": "attention_required",
            "component_statuses": {"review_queue": "managed_backlog"},
            "blockers": ["review queue backlog"],
            "recommendations": ["Drain critical samples."],
        },
    )
    companion = _write_json(
        tmp_path / "companion.json",
        {
            "overall_status": "attention_required",
            "review_surface": "attention_required",
            "primary_gap": "review_queue:managed_backlog",
            "component_statuses": {
                "assistant_explainability": "partial_coverage",
                "review_queue": "managed_backlog",
                "ocr_review": "managed_review",
                "qdrant_backend": "indexed_ready",
            },
            "blockers": ["review queue backlog"],
            "recommended_actions": ["Reduce review queue backlog"],
        },
    )
    knowledge = _write_json(
        tmp_path / "knowledge.json",
        {
            "knowledge_readiness": {
                "status": "knowledge_foundation_ready",
                "focus_areas_detail": [],
                "priority_domains": [],
                "domains": {
                    "gdt": {
                        "status": "ready",
                        "focus_components": [],
                        "missing_metrics": [],
                    }
                },
                "domain_focus_areas": [],
            }
        },
    )
    engineering = _write_json(
        tmp_path / "engineering.json",
        {
            "engineering_signals": {
                "status": "engineering_semantics_ready",
            },
            "recommendations": ["Keep standards coverage stable."],
        },
    )
    knowledge_drift = _write_json(
        tmp_path / "knowledge_drift.json",
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
        },
    )
    operator = _write_json(
        tmp_path / "operator.json",
        {
            "adoption_readiness": "guided_manual",
            "knowledge_drift_status": "regressed",
            "knowledge_drift_summary": "Knowledge baseline regressed.",
            "recommended_actions": ["Resolve operator blockers."],
            "blocking_signals": ["operator:blocker"],
        },
    )
    output_json = tmp_path / "bundle.json"
    output_md = tmp_path / "bundle.md"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--benchmark-scorecard",
            str(scorecard),
            "--benchmark-operational-summary",
            str(operational),
            "--benchmark-companion-summary",
            str(companion),
            "--benchmark-release-decision",
            str(companion),
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
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "attention_required"
    assert payload["artifacts"]["benchmark_scorecard"]["present"] is True
    assert payload["artifacts"]["benchmark_operational_summary"]["present"] is True
    assert payload["artifacts"]["benchmark_companion_summary"]["present"] is True
    assert payload["artifacts"]["benchmark_release_decision"]["present"] is True
    assert payload["artifacts"]["benchmark_knowledge_readiness"]["present"] is True
    assert payload["artifacts"]["benchmark_knowledge_drift"]["present"] is True
    assert payload["artifacts"]["benchmark_engineering_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True
    assert payload["component_statuses"]["assistant_explainability"] == "partial_coverage"
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_ready"
    assert payload["component_statuses"]["knowledge_drift"] == "stable"
    assert payload["knowledge_focus_area_count"] == 0
    assert payload["knowledge_drift_summary"].startswith("status=stable")
    assert payload["component_statuses"]["engineering_signals"] == "engineering_semantics_ready"
    assert payload["component_statuses"]["operator_adoption"] == "guided_manual"
    assert payload["operator_adoption_knowledge_drift"]["status"] == "regressed"
    assert payload["component_statuses"]["qdrant_backend"] == "indexed_ready"
    assert "Knowledge baseline regressed." in output_md.read_text(encoding="utf-8")
    assert payload["knowledge_domains"]["gdt"]["status"] == "ready"
    assert payload["knowledge_domain_focus_areas"] == []
    assert "review queue backlog" in output_md.read_text(encoding="utf-8")
    assert "## Knowledge Domains" in output_md.read_text(encoding="utf-8")


def test_build_bundle_prefers_companion_summary_when_present() -> None:
    payload = module.build_bundle(
        title="Benchmark Artifact Bundle",
        benchmark_scorecard={
            "overall_status": "benchmark_ready_with_multisignal_evidence",
            "components": {"hybrid": {"status": "ready"}},
            "recommendations": ["Keep monitoring."],
        },
        benchmark_operational_summary={
            "overall_status": "attention_required",
            "component_statuses": {"review_queue": "managed_backlog"},
            "blockers": ["review queue backlog"],
            "recommendations": ["Drain review queue."],
        },
        benchmark_companion_summary={
            "overall_status": "attention_required",
            "component_statuses": {
                "assistant_explainability": "partial_coverage",
                "review_queue": "managed_backlog",
                "ocr_review": "managed_review",
                "qdrant_backend": "indexed_ready",
                "knowledge_readiness": "knowledge_foundation_partial",
                "engineering_signals": "partial_engineering_semantics",
            },
            "blockers": ["review_queue:managed_backlog"],
            "recommended_actions": ["Reduce review queue backlog"],
        },
        benchmark_release_decision={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "recommended_actions": ["Review operator blockers."],
        },
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={"benchmark_companion_summary": "companion.json"},
    )

    assert payload["artifacts"]["benchmark_companion_summary"]["present"] is True
    assert payload["blockers"] == ["review_queue:managed_backlog"]
    assert payload["recommendations"] == ["Reduce review queue backlog"]
    assert payload["component_statuses"]["qdrant_backend"] == "indexed_ready"
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_partial"
    assert payload["component_statuses"]["engineering_signals"] == "partial_engineering_semantics"
    assert payload["component_statuses"]["operator_adoption"] == "guided_manual"


def test_build_bundle_prefers_release_decision_when_present() -> None:
    payload = module.build_bundle(
        title="Benchmark Artifact Bundle",
        benchmark_scorecard={
            "overall_status": "benchmark_ready_with_multisignal_evidence",
            "components": {"hybrid": {"status": "ready"}},
        },
        benchmark_operational_summary={
            "overall_status": "healthy",
            "component_statuses": {"review_queue": "healthy"},
        },
        benchmark_companion_summary={
            "overall_status": "healthy",
            "component_statuses": {
                "assistant_explainability": "healthy",
                "engineering_signals": "engineering_semantics_ready",
            },
        },
        benchmark_release_decision={
            "release_status": "review_required",
            "blocking_signals": [],
            "review_signals": ["review_queue:managed_backlog"],
        },
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "recommended_actions": ["Review operator blockers."],
        },
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={"benchmark_release_decision": "release.json"},
    )

    assert payload["overall_status"] == "review_required"
    assert payload["recommendations"] == ["review_queue:managed_backlog"]
    assert payload["artifacts"]["benchmark_release_decision"]["present"] is True
    assert payload["component_statuses"]["engineering_signals"] == "engineering_semantics_ready"
    assert payload["component_statuses"]["operator_adoption"] == "guided_manual"


def test_build_bundle_exposes_knowledge_drift_passthrough() -> None:
    payload = module.build_bundle(
        title="Benchmark Artifact Bundle",
        benchmark_scorecard={
            "overall_status": "benchmark_ready_with_multisignal_evidence",
            "components": {"hybrid": {"status": "ready"}},
        },
        benchmark_operational_summary={},
        benchmark_companion_summary={},
        benchmark_release_decision={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={
            "knowledge_drift": {
                "status": "regressed",
                "current_status": "knowledge_foundation_partial",
                "previous_status": "knowledge_foundation_ready",
                "reference_item_delta": -40,
                "regressions": ["gdt"],
                "improvements": [],
                "new_focus_areas": ["gdt"],
                "component_changes": [
                    {
                        "component": "gdt",
                        "previous_status": "ready",
                        "current_status": "missing",
                        "trend": "regressed",
                        "reference_item_delta": -40,
                    }
                ],
            },
            "recommendations": [
                "Resolve knowledge regressions before claiming the benchmark "
                "surpass baseline remains stable.",
                "Regressed components: gdt",
            ],
        },
        benchmark_engineering_signals={},
        benchmark_operator_adoption={},
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={"benchmark_knowledge_drift": "knowledge_drift.json"},
    )

    assert payload["component_statuses"]["knowledge_drift"] == "regressed"
    assert payload["artifacts"]["benchmark_knowledge_drift"]["present"] is True
    assert payload["knowledge_drift"]["status"] == "regressed"
    assert "regressions=gdt" in payload["knowledge_drift_summary"]
    assert payload["knowledge_drift_recommendations"] == [
        "Resolve knowledge regressions before claiming the benchmark "
        "surpass baseline remains stable.",
        "Regressed components: gdt",
    ]
    assert payload["recommendations"] == payload["knowledge_drift_recommendations"]
