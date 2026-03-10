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
            "components": {
                "hybrid": {"status": "ready"},
                "operator_adoption": {
                    "status": "guided_manual",
                    "operator_mode": "assisted_review",
                    "knowledge_outcome_drift_status": "regressed",
                    "knowledge_outcome_drift_summary": "Standards coverage regressed.",
                },
            },
        },
        benchmark_operational_summary={
            "overall_status": "attention_required",
            "component_statuses": {
                "feedback_flywheel": "feedback_collected",
                "assistant_explainability": "evidence_partial",
                "review_queue": "managed_backlog",
                "ocr_review": "review_heavy",
                "operator_adoption": "guided_manual",
            },
            "operator_adoption_knowledge_outcome_drift_status": "regressed",
            "operator_adoption_knowledge_outcome_drift_summary": (
                "Standards coverage regressed."
            ),
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
        benchmark_realdata_signals={
            "realdata_signals": {
                "status": "realdata_foundation_partial",
                "components": {
                    "hybrid_dxf": {"status": "ready", "sample_size": 110},
                    "history_h5": {"status": "ready", "sample_size": 1},
                    "step_dir": {"status": "partial", "sample_size": 3},
                },
            },
            "recommendations": ["Expand STEP/B-Rep directory validation."],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "knowledge_drift_status": "regressed",
            "knowledge_drift_summary": "Standards coverage regressed.",
            "knowledge_outcome_drift_status": "regressed",
            "knowledge_outcome_drift_summary": "Operator review outcomes regressed.",
            "release_surface_alignment_status": "aligned",
            "release_surface_alignment_summary": (
                "Release decision and runbook expose matching operator adoption surfaces."
            ),
            "release_surface_alignment": {
                "mismatches": [],
                "release_decision": {"scorecard_status": "guided_manual"},
                "release_runbook": {"scorecard_status": "guided_manual"},
            },
            "knowledge_outcome_drift": {
                "recommendations": ["Investigate regression-heavy operator cohorts."],
            },
            "knowledge_drift": {
                "recommendations": ["Restore standards baseline coverage."],
            },
            "recommended_actions": ["Resolve operator blockers."],
        },
        benchmark_knowledge_application={
            "knowledge_application": {
                "status": "knowledge_application_partial",
                "priority_domains": ["standards"],
                "domains": {
                    "standards": {
                        "status": "partial",
                        "readiness_status": "missing",
                        "evidence_status": "partial",
                        "signal_count": 2,
                    }
                },
                "focus_areas_detail": [
                    {
                        "domain": "standards",
                        "status": "partial",
                        "priority": "high",
                        "action": "Promote standards evidence.",
                    }
                ],
            },
            "recommendations": ["Promote standards application evidence."],
        },
        benchmark_knowledge_domain_matrix={
            "knowledge_domain_matrix": {
                "status": "knowledge_domain_matrix_partial",
                "priority_domains": ["standards"],
                "domains": {
                    "standards": {
                        "status": "blocked",
                        "readiness_status": "missing",
                        "application_status": "partial",
                        "realdata_status": "partial",
                    }
                },
            },
            "recommendations": ["Backfill standards foundation and raise real-data depth."],
        },
        benchmark_knowledge_domain_action_plan={
            "knowledge_domain_action_plan": {
                "status": "knowledge_domain_action_plan_blocked",
                "priority_domains": ["standards"],
                "actions": [
                    {
                        "id": "standards:foundation",
                        "domain": "standards",
                        "stage": "foundation",
                        "priority": "high",
                        "status": "blocked",
                    }
                ],
            },
            "recommendations": ["Backfill standards foundation metrics first."],
        },
        benchmark_knowledge_source_action_plan={
            "knowledge_source_action_plan": {
                "status": "knowledge_source_action_plan_blocked",
                "priority_domains": ["standards"],
                "total_action_count": 2,
                "high_priority_action_count": 1,
                "medium_priority_action_count": 1,
                "expansion_action_count": 1,
                "recommended_first_actions": [
                    {
                        "id": "standards:coverage",
                        "domain": "standards",
                        "stage": "source_group",
                        "priority": "high",
                    }
                ],
                "source_group_action_counts": {"standards": 1},
            },
            "recommendations": ["Prioritize standards source-group restoration."],
        },
        benchmark_knowledge_source_coverage={
            "knowledge_source_coverage": {
                "status": "knowledge_source_coverage_partial",
                "priority_domains": ["standards"],
                "domains": {
                    "standards": {
                        "status": "partial",
                        "focus_source_groups": ["standards"],
                    }
                },
                "expansion_candidates": [
                    {"name": "machining", "status": "ready"},
                    {"name": "welding", "status": "ready"},
                ],
            },
            "recommendations": ["Promote machining and standards source coverage."],
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
    assert payload["available_artifact_count"] == 11
    assert payload["component_statuses"]["feedback_flywheel"] == "feedback_collected"
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_partial"
    assert payload["knowledge_focus_area_count"] == 1
    assert payload["knowledge_focus_areas"][0]["component"] == "standards"
    assert payload["knowledge_priority_domains"] == ["standards"]
    assert payload["knowledge_domains"]["standards"]["status"] == "missing"
    assert payload["knowledge_domain_focus_areas"][0]["domain"] == "standards"
    assert payload["knowledge_drift_domain_regressions"] == []
    assert payload["knowledge_drift_new_priority_domains"] == []
    assert payload["component_statuses"]["engineering_signals"] == "partial_engineering_semantics"
    assert payload["component_statuses"]["realdata_signals"] == "realdata_foundation_partial"
    assert payload["component_statuses"]["operator_adoption"] == "guided_manual"
    assert payload["scorecard_operator_adoption"]["operator_mode"] == "assisted_review"
    assert payload["operational_operator_adoption"]["status"] == "guided_manual"
    assert payload["component_statuses"]["knowledge_application"] == "knowledge_application_partial"
    assert payload["component_statuses"]["knowledge_domain_matrix"] == (
        "knowledge_domain_matrix_partial"
    )
    assert payload["component_statuses"]["knowledge_domain_action_plan"] == (
        "knowledge_domain_action_plan_blocked"
    )
    assert payload["component_statuses"]["knowledge_source_action_plan"] == (
        "knowledge_source_action_plan_blocked"
    )
    assert payload["component_statuses"]["knowledge_source_coverage"] == (
        "knowledge_source_coverage_partial"
    )
    assert payload["operator_adoption_knowledge_drift"]["status"] == "regressed"
    assert payload["operator_adoption_knowledge_outcome_drift"]["status"] == "regressed"
    assert payload["operator_adoption_release_surface_alignment"]["status"] == "aligned"
    assert payload["operator_adoption_release_surface_alignment"]["mismatches"] == []
    assert payload["knowledge_application_status"] == "knowledge_application_partial"
    assert payload["knowledge_application_domains"]["standards"]["status"] == "partial"
    assert payload["knowledge_domain_matrix_status"] == "knowledge_domain_matrix_partial"
    assert payload["knowledge_domain_matrix_domains"]["standards"]["status"] == "blocked"
    assert payload["knowledge_domain_action_plan_status"] == (
        "knowledge_domain_action_plan_blocked"
    )
    assert payload["knowledge_domain_action_plan_actions"][0]["id"] == (
        "standards:foundation"
    )
    assert payload["knowledge_source_action_plan_status"] == (
        "knowledge_source_action_plan_blocked"
    )
    assert payload["knowledge_source_action_plan_recommended_first_actions"][0]["id"] == (
        "standards:coverage"
    )
    assert payload["knowledge_source_coverage_status"] == (
        "knowledge_source_coverage_partial"
    )
    assert payload["knowledge_source_coverage_domains"]["standards"]["status"] == (
        "partial"
    )
    assert payload["knowledge_source_coverage_expansion_candidates"][0]["name"] == (
        "machining"
    )
    assert payload["realdata_status"] == "realdata_foundation_partial"
    assert payload["artifacts"]["benchmark_realdata_signals"]["present"] is True
    assert payload["blockers"] == ["feedback backlog"]
    assert payload["recommendations"] == ["Close the review queue."]
    assert payload["realdata_recommendations"] == [
        "Expand STEP/B-Rep directory validation."
    ]
    markdown = module.render_markdown(payload)
    assert "## Operator Adoption Release Surface Alignment" in markdown
    assert "## Knowledge Source Coverage" in markdown


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
        benchmark_realdata_signals={
            "realdata_signals": {
                "status": "realdata_foundation_ready",
                "components": {
                    "hybrid_dxf": {"status": "ready", "sample_size": 110},
                    "history_h5": {"status": "ready", "sample_size": 1},
                    "step_dir": {"status": "ready", "sample_size": 3},
                },
            },
            "recommendations": [],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "operator_ready",
            "knowledge_drift_status": "stable",
            "knowledge_drift_summary": "Knowledge baseline stable.",
            "knowledge_outcome_drift_status": "stable",
            "knowledge_outcome_drift_summary": "Operator outcome drift stable.",
            "release_surface_alignment_status": "aligned",
            "release_surface_alignment_summary": "Release decision and runbook are aligned.",
            "release_surface_alignment": {"mismatches": []},
            "recommended_actions": ["Keep operator automation healthy."],
        },
        benchmark_knowledge_application={
            "knowledge_application": {
                "status": "knowledge_application_ready",
                "priority_domains": [],
                "domains": {
                    "tolerance": {
                        "status": "ready",
                        "readiness_status": "ready",
                        "evidence_status": "ready",
                        "signal_count": 5,
                    }
                },
                "focus_areas_detail": [],
            },
            "recommendations": [],
        },
        benchmark_knowledge_domain_matrix={
            "knowledge_domain_matrix": {
                "status": "knowledge_domain_matrix_ready",
                "priority_domains": [],
                "domains": {
                    "tolerance": {
                        "status": "ready",
                        "readiness_status": "ready",
                        "application_status": "ready",
                        "realdata_status": "ready",
                    }
                },
            },
            "recommendations": [],
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
    assert payload["component_statuses"]["realdata_signals"] == "realdata_foundation_ready"
    assert payload["component_statuses"]["operator_adoption"] == "operator_ready"
    assert payload["component_statuses"]["knowledge_application"] == "knowledge_application_ready"
    assert payload["component_statuses"]["knowledge_domain_matrix"] == (
        "knowledge_domain_matrix_ready"
    )
    assert payload["operator_adoption_knowledge_drift"]["status"] == "stable"
    assert payload["operator_adoption_knowledge_outcome_drift"]["status"] == "stable"
    assert payload["operator_adoption_release_surface_alignment"]["status"] == "aligned"
    assert payload["knowledge_drift_domain_improvements"] == []
    assert payload["realdata_status"] == "realdata_foundation_ready"
    assert payload["realdata_recommendations"] == []
    assert payload["knowledge_application_status"] == "knowledge_application_ready"
    assert payload["knowledge_domain_matrix_status"] == "knowledge_domain_matrix_ready"


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
    realdata = _write_json(
        tmp_path / "realdata.json",
        {
            "realdata_signals": {
                "status": "realdata_foundation_partial",
                "components": {
                    "hybrid_dxf": {"status": "ready", "sample_size": 110},
                    "history_h5": {"status": "ready", "sample_size": 1},
                    "step_dir": {"status": "partial", "sample_size": 3},
                },
            },
            "recommendations": ["Expand STEP/B-Rep directory validation."],
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
            "knowledge_outcome_drift_status": "regressed",
            "knowledge_outcome_drift_summary": "Operator review outcomes regressed.",
            "recommended_actions": ["Resolve operator blockers."],
            "blocking_signals": ["operator:blocker"],
            "release_surface_alignment_status": "mismatched",
            "release_surface_alignment_summary": (
                "Release runbook is missing operator adoption guidance."
            ),
            "release_surface_alignment": {
                "mismatches": ["release_runbook:missing"],
            },
        },
    )
    knowledge_source_coverage = _write_json(
        tmp_path / "knowledge_source_coverage.json",
        {
            "knowledge_source_coverage": {
                "status": "knowledge_source_coverage_ready",
                "priority_domains": [],
                "domains": {
                    "standards": {
                        "status": "ready",
                        "focus_source_groups": [],
                    }
                },
                "expansion_candidates": [{"name": "machining", "status": "ready"}],
            },
            "recommendations": ["Promote machining into benchmark views."],
        },
    )
    knowledge_source_action_plan = _write_json(
        tmp_path / "knowledge_source_action_plan.json",
        {
            "knowledge_source_action_plan": {
                "status": "knowledge_source_action_plan_ready",
                "priority_domains": [],
                "total_action_count": 1,
                "high_priority_action_count": 0,
                "medium_priority_action_count": 1,
                "expansion_action_count": 1,
                "recommended_first_actions": [
                    {
                        "id": "machining:coverage",
                        "domain": "machining",
                        "stage": "source_group",
                        "priority": "medium",
                    }
                ],
                "source_group_action_counts": {"machining": 1},
            },
            "recommendations": ["Promote machining source coverage next."],
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
            "--benchmark-realdata-signals",
            str(realdata),
            "--benchmark-operator-adoption",
            str(operator),
            "--benchmark-knowledge-source-action-plan",
            str(knowledge_source_action_plan),
            "--benchmark-knowledge-source-coverage",
            str(knowledge_source_coverage),
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
    assert payload["artifacts"]["benchmark_realdata_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True
    assert payload["component_statuses"]["assistant_explainability"] == "partial_coverage"
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_ready"
    assert payload["component_statuses"]["knowledge_drift"] == "stable"
    assert payload["knowledge_focus_area_count"] == 0
    assert payload["knowledge_drift_summary"].startswith("status=stable")
    assert payload["component_statuses"]["engineering_signals"] == "engineering_semantics_ready"
    assert payload["component_statuses"]["realdata_signals"] == "realdata_foundation_partial"
    assert payload["component_statuses"]["operator_adoption"] == "guided_manual"
    assert payload["component_statuses"]["knowledge_source_coverage"] == (
        "knowledge_source_coverage_ready"
    )
    assert payload["component_statuses"]["knowledge_source_action_plan"] == (
        "knowledge_source_action_plan_ready"
    )
    assert payload["operator_adoption_knowledge_drift"]["status"] == "regressed"
    assert payload["operator_adoption_knowledge_outcome_drift"]["status"] == "regressed"
    assert payload["operator_adoption_release_surface_alignment"]["status"] == (
        "mismatched"
    )
    assert payload["artifacts"]["benchmark_knowledge_source_coverage"]["present"] is True
    assert payload["artifacts"]["benchmark_knowledge_source_action_plan"]["present"] is True
    assert payload["knowledge_source_action_plan_status"] == (
        "knowledge_source_action_plan_ready"
    )
    assert payload["knowledge_source_coverage_status"] == "knowledge_source_coverage_ready"
    assert payload["realdata_status"] == "realdata_foundation_partial"
    assert payload["component_statuses"]["qdrant_backend"] == "indexed_ready"
    assert "Knowledge baseline regressed." in output_md.read_text(encoding="utf-8")
    assert payload["knowledge_domains"]["gdt"]["status"] == "ready"
    assert payload["knowledge_domain_focus_areas"] == []
    assert "domain_regressions" in output_md.read_text(encoding="utf-8")
    assert "review queue backlog" in output_md.read_text(encoding="utf-8")
    assert "## Knowledge Domains" in output_md.read_text(encoding="utf-8")
    assert "## Knowledge Domain Matrix" in output_md.read_text(encoding="utf-8")
    assert "## Knowledge Source Coverage" in output_md.read_text(encoding="utf-8")
    assert "## Operator Adoption Release Surface Alignment" in output_md.read_text(
        encoding="utf-8"
    )
    assert "## Operator Adoption Knowledge Outcome Drift" in output_md.read_text(
        encoding="utf-8"
    )
    assert "## Real-Data Signals" in output_md.read_text(encoding="utf-8")
    assert "Expand STEP/B-Rep directory validation." in output_md.read_text(
        encoding="utf-8"
    )


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
        benchmark_realdata_signals={},
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
        benchmark_realdata_signals={},
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
                "domain_regressions": ["gdt"],
                "domain_improvements": [],
                "new_focus_areas": ["gdt"],
                "resolved_priority_domains": [],
                "new_priority_domains": ["gdt"],
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
        benchmark_realdata_signals={},
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
    assert payload["knowledge_drift_domain_regressions"] == ["gdt"]
    assert payload["knowledge_drift_new_priority_domains"] == ["gdt"]
    assert payload["recommendations"] == payload["knowledge_drift_recommendations"]


def test_build_bundle_exposes_knowledge_outcome_drift_passthrough() -> None:
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
        benchmark_knowledge_drift={},
        benchmark_knowledge_outcome_drift={
            "knowledge_outcome_drift": {
                "status": "regressed",
                "current_status": "knowledge_outcome_correlation_partial",
                "previous_status": "knowledge_outcome_correlation_ready",
                "regressions": ["tolerance"],
                "improvements": [],
                "new_focus_areas": ["tolerance"],
                "domain_regressions": ["tolerance"],
                "domain_improvements": [],
                "new_priority_domains": ["tolerance"],
                "resolved_priority_domains": [],
            },
            "recommendations": [
                "Resolve knowledge outcome regressions before claiming benchmark outcome stability."
            ],
        },
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_operator_adoption={},
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={
            "benchmark_knowledge_outcome_drift": "knowledge_outcome_drift.json"
        },
    )

    assert payload["component_statuses"]["knowledge_outcome_drift"] == "regressed"
    assert payload["artifacts"]["benchmark_knowledge_outcome_drift"]["present"] is True
    assert payload["knowledge_outcome_drift"]["status"] == "regressed"
    assert "regressions=tolerance" in payload["knowledge_outcome_drift_summary"]
    assert payload["knowledge_outcome_drift_domain_regressions"] == ["tolerance"]
    assert payload["knowledge_outcome_drift_new_priority_domains"] == ["tolerance"]
    assert payload["recommendations"] == payload["knowledge_outcome_drift_recommendations"]
    markdown = module.render_markdown(payload)
    assert "## Operator Adoption Release Surface Alignment" in markdown
    assert "## Scorecard Operator Adoption" in markdown
    assert "## Operational Operator Adoption" in markdown


def test_build_bundle_exposes_competitive_surpass_index_passthrough() -> None:
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
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_realdata_scorecard={},
        benchmark_operator_adoption={},
        benchmark_competitive_surpass_index={
            "competitive_surpass_index": {
                "status": "competitive_surpass_attention_required",
                "score": 70,
                "primary_gaps": ["knowledge", "operator_adoption"],
            },
            "recommendations": [
                "Close tolerance/standards/GD&T knowledge gaps before claiming "
                "benchmark surpass readiness."
            ],
        },
        benchmark_knowledge_application={},
        benchmark_knowledge_realdata_correlation={},
        benchmark_knowledge_domain_matrix={},
        benchmark_knowledge_outcome_correlation={},
        benchmark_knowledge_outcome_drift={},
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={
            "benchmark_competitive_surpass_index": "competitive_surpass_index.json"
        },
    )

    assert payload["competitive_surpass_index_status"] == (
        "competitive_surpass_attention_required"
    )
    assert payload["competitive_surpass_primary_gaps"] == [
        "knowledge",
        "operator_adoption",
    ]
    assert payload["competitive_surpass_recommendations"] == [
        "Close tolerance/standards/GD&T knowledge gaps before claiming benchmark "
        "surpass readiness."
    ]
    assert payload["artifacts"]["benchmark_competitive_surpass_index"]["present"] is True
    markdown = module.render_markdown(payload)
    assert "## Competitive Surpass Index" in markdown
