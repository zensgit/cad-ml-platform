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
                "priority_domains": ["gdt"],
                "domains": {
                    "gdt": {
                        "status": "missing",
                        "focus_components": ["gdt"],
                        "missing_metrics": ["symbol_count"],
                    }
                },
                "domain_focus_areas": [
                    {
                        "domain": "gdt",
                        "status": "missing",
                        "priority": "high",
                        "action": "Expand GD&T coverage.",
                    }
                ],
            },
            "recommendations": ["Raise tolerance/GD&T readiness."],
        },
        benchmark_knowledge_drift={
            "knowledge_drift": {
                "status": "mixed",
                "regressions": ["gdt"],
                "improvements": ["standards"],
                "domain_regressions": ["gdt"],
                "domain_improvements": ["standards"],
                "new_focus_areas": ["gdt"],
                "resolved_focus_areas": [],
                "resolved_priority_domains": [],
                "new_priority_domains": ["gdt"],
            },
            "recommendations": [
                "Keep the previous baseline until knowledge regressions are cleared."
            ],
        },
        benchmark_engineering_signals={
            "engineering_signals": {"status": "partial_engineering_semantics"},
            "recommendations": ["Close engineering gaps."],
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
            "status": "attention_required",
            "summary": "Operator onboarding still needs a dry run.",
            "signals": ["operator_playbook:needs_walkthrough"],
            "actions": ["Schedule an operator handoff dry run."],
            "knowledge_drift_status": "regressed",
            "knowledge_drift_summary": "Tolerance coverage regressed.",
            "knowledge_drift": {
                "recommendations": ["Backfill tolerance knowledge coverage."],
            },
        },
        benchmark_knowledge_application={
            "knowledge_application": {
                "status": "knowledge_application_partial",
                "priority_domains": ["gdt"],
                "domains": {
                    "gdt": {
                        "status": "partial",
                        "readiness_status": "ready",
                        "evidence_status": "missing",
                        "signal_count": 0,
                    }
                },
                "focus_areas_detail": [
                    {
                        "domain": "gdt",
                        "status": "partial",
                        "priority": "high",
                        "action": "Promote GD&T application evidence.",
                    }
                ],
            },
            "recommendations": ["Promote GD&T application evidence."],
        },
        benchmark_knowledge_domain_matrix={
            "knowledge_domain_matrix": {
                "status": "knowledge_domain_matrix_partial",
                "priority_domains": ["gdt"],
                "domains": {
                    "gdt": {
                        "status": "blocked",
                        "readiness_status": "missing",
                        "application_status": "partial",
                        "realdata_status": "partial",
                    }
                },
            },
            "recommendations": ["Backfill GD&T foundation and real-data coverage."],
        },
        benchmark_knowledge_domain_action_plan={
            "knowledge_domain_action_plan": {
                "status": "knowledge_domain_action_plan_blocked",
                "priority_domains": ["gdt"],
                "actions": [
                    {
                        "id": "gdt:foundation",
                        "domain": "gdt",
                        "stage": "foundation",
                        "priority": "high",
                        "status": "blocked",
                    }
                ],
            },
            "recommendations": ["Backfill GD&T foundation metrics first."],
        },
        benchmark_knowledge_source_action_plan={
            "knowledge_source_action_plan": {
                "status": "knowledge_source_action_plan_blocked",
                "priority_domains": ["gdt"],
                "total_action_count": 2,
                "high_priority_action_count": 1,
                "medium_priority_action_count": 1,
                "expansion_action_count": 1,
                "recommended_first_actions": [
                    {
                        "id": "gdt:coverage",
                        "domain": "gdt",
                        "stage": "source_group",
                        "priority": "high",
                    }
                ],
                "source_group_action_counts": {"gdt": 1},
            },
            "recommendations": ["Backfill GD&T source actions first."],
        },
        benchmark_knowledge_source_coverage={
            "knowledge_source_coverage": {
                "status": "knowledge_source_coverage_partial",
                "priority_domains": ["gdt"],
                "domains": {
                    "gdt": {
                        "status": "partial",
                        "focus_source_groups": ["gdt"],
                    }
                },
                "expansion_candidates": [{"name": "machining", "status": "ready"}],
            },
            "recommendations": ["Backfill GD&T source coverage."],
        },
        benchmark_knowledge_domain_release_surface_alignment={
            "knowledge_domain_release_surface_alignment": {
                "status": "diverged",
                "summary": "gdt:blocked->partial",
                "mismatches": ["gdt:blocked->partial"],
                "domain_mismatches": ["gdt:blocked->partial"],
                "release_blocker_mismatches": [],
            },
            "recommendations": ["Reconcile release-surface mismatches before freeze."],
        },
        benchmark_knowledge_reference_inventory={
            "knowledge_reference_inventory": {
                "status": "knowledge_reference_inventory_partial",
                "summary": "GD&T reference tables are incomplete.",
                "priority_domains": ["gdt"],
                "total_reference_items": 18,
                "domains": {
                    "gdt": {
                        "status": "partial",
                        "total_reference_items": 18,
                        "populated_table_count": 1,
                        "total_table_count": 3,
                        "missing_tables": ["symbol_definitions", "feature_controls"],
                    }
                },
                "focus_tables_detail": [
                    {
                        "domain": "gdt",
                        "missing_tables": ["symbol_definitions", "feature_controls"],
                        "action": "Backfill GD&T reference tables.",
                    }
                ],
            },
            "recommendations": ["Backfill GD&T reference tables."],
        },
        benchmark_knowledge_domain_release_gate={
            "knowledge_domain_release_gate": {
                "status": "knowledge_domain_release_gate_blocked",
                "gate_open": False,
                "releasable_domains": ["standards"],
                "blocked_domains": ["gdt"],
                "priority_domains": ["gdt"],
            },
            "blocking_reasons": ["gdt:not_release_ready"],
            "warning_reasons": ["gdt:review_before_release"],
            "recommendations": ["Unblock GD&T before release freeze."],
        },
        benchmark_knowledge_domain_release_readiness_matrix={
            "knowledge_domain_release_readiness_matrix": {
                "status": "knowledge_domain_release_readiness_blocked",
                "summary": "ready=1; partial=0; blocked=1",
                "priority_domains": ["gdt"],
                "releasable_domains": ["standards"],
                "blocked_domains": ["gdt"],
                "domains": {
                    "gdt": {
                        "status": "blocked",
                        "validation_status": "blocked",
                        "inventory_status": "partial",
                        "release_gate_status": "blocked",
                        "alignment_warning": False,
                    }
                },
            },
            "recommendations": ["Unblock GD&T release readiness."],
        },
        artifact_paths={
            "benchmark_release_decision": "release.json",
            "benchmark_engineering_signals": "engineering.json",
            "benchmark_operator_adoption": "operator_adoption.json",
            "benchmark_knowledge_domain_release_gate": (
                "knowledge_domain_release_gate.json"
            ),
            "benchmark_knowledge_domain_release_readiness_matrix": (
                "knowledge_domain_release_readiness_matrix.json"
            ),
            "benchmark_knowledge_domain_release_surface_alignment": (
                "knowledge_domain_release_surface_alignment.json"
            ),
            "benchmark_knowledge_reference_inventory": (
                "knowledge_reference_inventory.json"
            ),
        },
    )

    assert payload["release_status"] == "blocked"
    assert payload["engineering_status"] == "partial_engineering_semantics"
    assert payload["knowledge_status"] == "knowledge_foundation_partial"
    assert payload["realdata_status"] == "realdata_foundation_partial"
    assert payload["knowledge_focus_areas"][0]["component"] == "gdt"
    assert payload["knowledge_drift_status"] == "mixed"
    assert payload["knowledge_drift_domain_regressions"] == ["gdt"]
    assert payload["knowledge_drift"]["counts"]["regressions"] == 1
    assert payload["knowledge_priority_domains"] == ["gdt"]
    assert payload["knowledge_domains"]["gdt"]["status"] == "missing"
    assert payload["knowledge_domain_focus_areas"][0]["domain"] == "gdt"
    assert payload["knowledge_application_status"] == "knowledge_application_partial"
    assert payload["knowledge_application_domains"]["gdt"]["status"] == "partial"
    assert payload["knowledge_domain_matrix_status"] == "knowledge_domain_matrix_partial"
    assert payload["knowledge_domain_matrix_domains"]["gdt"]["status"] == "blocked"
    assert payload["knowledge_domain_action_plan_status"] == (
        "knowledge_domain_action_plan_blocked"
    )
    assert payload["knowledge_domain_action_plan_actions"][0]["id"] == (
        "gdt:foundation"
    )
    assert payload["knowledge_domain_release_gate_status"] == (
        "knowledge_domain_release_gate_blocked"
    )
    assert payload["knowledge_domain_release_gate_gate_open"] is False
    assert payload["knowledge_domain_release_gate_blocked_domains"] == ["gdt"]
    assert payload["knowledge_domain_release_gate_releasable_domains"] == [
        "standards"
    ]
    assert payload["knowledge_domain_release_readiness_matrix_status"] == (
        "knowledge_domain_release_readiness_blocked"
    )
    assert payload["knowledge_domain_release_readiness_matrix_blocked_domains"] == [
        "gdt"
    ]
    assert payload["knowledge_domain_release_readiness_matrix_releasable_domains"] == [
        "standards"
    ]
    assert payload["knowledge_domain_release_surface_alignment_status"] == "diverged"
    assert payload["knowledge_domain_release_surface_alignment"]["mismatches"] == [
        "gdt:blocked->partial"
    ]
    assert payload["knowledge_reference_inventory_status"] == (
        "knowledge_reference_inventory_partial"
    )
    assert payload["knowledge_reference_inventory_priority_domains"] == ["gdt"]
    assert payload["knowledge_reference_inventory_total_reference_items"] == 18
    assert payload["knowledge_source_action_plan_status"] == (
        "knowledge_source_action_plan_blocked"
    )
    assert payload["knowledge_source_action_plan_recommended_first_actions"][0]["id"] == (
        "gdt:coverage"
    )
    assert payload["knowledge_source_coverage_status"] == (
        "knowledge_source_coverage_partial"
    )
    assert payload["knowledge_source_coverage_domains"]["gdt"]["status"] == "partial"
    assert payload["knowledge_source_coverage_expansion_candidates"][0]["name"] == (
        "machining"
    )
    assert payload["next_action"] == "collect_artifacts"
    assert "benchmark_artifact_bundle" in payload["missing_artifacts"]
    assert payload["artifacts"]["benchmark_engineering_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_realdata_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True
    assert (
        payload["artifacts"]["benchmark_knowledge_domain_release_gate"]["present"]
        is True
    )
    assert payload["artifacts"][
        "benchmark_knowledge_domain_release_readiness_matrix"
    ]["present"] is True
    assert (
        payload["artifacts"]["benchmark_knowledge_domain_release_surface_alignment"][
            "present"
        ]
        is True
    )
    assert payload["artifacts"]["benchmark_knowledge_reference_inventory"]["present"] is True
    rendered = render_markdown(payload)
    assert "## Knowledge Domain Release Gate" in rendered
    assert "## Knowledge Domain Release Readiness Matrix" in rendered
    assert "## Knowledge Domain Release Surface Alignment" in rendered
    assert "## Knowledge Reference Inventory" in rendered
    assert "gdt:not_release_ready" in payload["blocking_signals"]
    assert payload["operator_adoption"]["actions"] == [
        "Schedule an operator handoff dry run."
    ]
    assert payload["operator_adoption"]["knowledge_drift_status"] == "regressed"
    assert "Backfill GD&T source actions first." in payload["review_signals"]
    assert "Expand STEP/B-Rep directory validation." in payload["review_signals"]
    assert "Backfill GD&T source coverage." in payload["review_signals"]
    assert "Backfill GD&T reference tables." in payload["review_signals"]
    assert "Unblock GD&T release readiness." in payload["blocking_signals"]
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
                "benchmark_knowledge_domain_capability_matrix": {
                    "path": "knowledge_domain_capability_matrix.json"
                },
                "benchmark_knowledge_domain_capability_drift": {
                    "path": "knowledge_domain_capability_drift.json"
                },
                "benchmark_knowledge_domain_release_gate": {
                    "path": "knowledge_domain_release_gate.json"
                },
                "benchmark_knowledge_reference_inventory": {
                    "path": "knowledge_reference_inventory.json"
                },
            },
        },
        benchmark_companion_summary={"overall_status": "healthy"},
        benchmark_artifact_bundle={"overall_status": "healthy"},
        benchmark_knowledge_readiness={
            "knowledge_readiness": {
                "status": "knowledge_foundation_ready",
                "focus_areas_detail": [],
                "priority_domains": [],
                "domains": {
                    "standards": {
                        "status": "ready",
                        "focus_components": [],
                        "missing_metrics": [],
                    }
                },
                "domain_focus_areas": [],
            },
            "recommendations": [],
        },
        benchmark_knowledge_drift={
            "knowledge_drift": {
                "status": "improved",
                "regressions": [],
                "improvements": ["standards"],
                "domain_regressions": [],
                "domain_improvements": ["standards"],
                "new_focus_areas": [],
                "resolved_focus_areas": ["standards"],
                "resolved_priority_domains": ["standards"],
                "new_priority_domains": [],
            },
            "recommendations": [
                "Promote the improved knowledge baseline after CI and review surfaces "
                "are refreshed."
            ],
        },
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"},
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
        benchmark_operator_adoption={},
        benchmark_knowledge_application={
            "knowledge_application": {
                "status": "knowledge_application_ready",
                "priority_domains": [],
                "domains": {
                    "standards": {
                        "status": "ready",
                        "readiness_status": "ready",
                        "evidence_status": "ready",
                        "signal_count": 4,
                    }
                },
                "focus_areas_detail": [],
            },
            "recommendations": [],
        },
        benchmark_knowledge_realdata_correlation={
            "knowledge_realdata_correlation": {
                "status": "knowledge_realdata_ready",
                "priority_domains": [],
                "domains": {
                    "standards": {
                        "status": "ready",
                        "readiness_status": "ready",
                        "application_status": "ready",
                        "realdata_status": "ready",
                    }
                },
            },
            "recommendations": [],
        },
        benchmark_knowledge_domain_matrix={
            "knowledge_domain_matrix": {
                "status": "knowledge_domain_matrix_ready",
                "priority_domains": [],
                "domains": {
                    "standards": {
                        "status": "ready",
                        "readiness_status": "ready",
                        "application_status": "ready",
                        "realdata_status": "ready",
                    }
                },
            },
            "recommendations": [],
        },
        benchmark_knowledge_domain_capability_matrix={
            "knowledge_domain_capability_matrix": {
                "status": "knowledge_domain_capability_ready",
                "priority_domains": [],
                "focus_areas_detail": [],
                "domains": {
                    "standards": {
                        "domain": "standards",
                        "label": "Standards & Design Tables",
                        "status": "ready",
                        "foundation_status": "ready",
                        "application_status": "ready",
                        "matrix_status": "ready",
                        "provider_status": "ready",
                        "surface_status": "ready",
                        "reference_item_count": 12,
                    }
                },
            },
            "recommendations": [],
        },
        benchmark_knowledge_domain_capability_drift={
            "knowledge_domain_capability_drift": {
                "status": "stable",
                "domain_regressions": [],
                "domain_improvements": [],
            },
            "recommendations": [],
        },
        benchmark_knowledge_domain_validation_matrix={
            "knowledge_domain_validation_matrix": {
                "status": "knowledge_domain_validation_matrix_ready",
                "summary": "Validation coverage is complete.",
                "priority_domains": [],
                "total_test_count": 12,
                "domains": {
                    "standards": {
                        "status": "ready",
                        "provider_status": "ready",
                        "api_status": "ready",
                        "integration_status": "ready",
                        "assistant_status": "ready",
                    }
                },
            },
            "recommendations": [],
        },
        benchmark_knowledge_domain_release_surface_alignment={
            "knowledge_domain_release_surface_alignment": {
                "status": "aligned",
                "summary": "all release surfaces aligned",
                "mismatches": [],
                "domain_mismatches": [],
                "release_blocker_mismatches": [],
            },
            "recommendations": [],
        },
        benchmark_knowledge_reference_inventory={
            "knowledge_reference_inventory": {
                "status": "knowledge_reference_inventory_ready",
                "summary": "Standards reference inventory is complete.",
                "priority_domains": [],
                "total_reference_items": 42,
                "domains": {
                    "standards": {
                        "status": "ready",
                        "total_reference_items": 42,
                        "populated_table_count": 4,
                        "total_table_count": 4,
                        "missing_tables": [],
                    }
                },
                "focus_tables_detail": [],
            },
            "recommendations": [],
        },
        benchmark_knowledge_domain_action_plan={
            "knowledge_domain_action_plan": {
                "status": "knowledge_domain_action_plan_ready",
                "priority_domains": [],
                "actions": [
                    {
                        "id": "standards:realdata",
                        "domain": "standards",
                        "stage": "realdata",
                        "priority": "medium",
                        "status": "ready",
                    }
                ],
            },
            "recommendations": [],
        },
        benchmark_knowledge_domain_control_plane={
            "knowledge_domain_control_plane": {
                "status": "knowledge_domain_control_plane_ready",
                "domains": {"standards": {"status": "ready"}},
                "focus_areas": [],
                "release_blockers": [],
            },
            "recommendations": [],
        },
        benchmark_knowledge_domain_control_plane_drift={
            "knowledge_domain_control_plane_drift": {
                "status": "improved",
                "domain_regressions": [],
                "domain_improvements": ["standards"],
                "new_release_blockers": [],
                "resolved_release_blockers": ["standards"],
            },
            "recommendations": [],
        },
        benchmark_knowledge_domain_release_gate={
            "knowledge_domain_release_gate": {
                "status": "knowledge_domain_release_gate_ready",
                "gate_open": True,
                "releasable_domains": ["standards"],
                "blocked_domains": [],
                "priority_domains": [],
            },
            "blocking_reasons": [],
            "warning_reasons": [],
            "recommendations": [],
        },
        benchmark_knowledge_domain_release_readiness_matrix={
            "knowledge_domain_release_readiness_matrix": {
                "status": "knowledge_domain_release_readiness_ready",
                "summary": "ready=1; partial=0; blocked=0",
                "priority_domains": [],
                "releasable_domains": ["standards"],
                "blocked_domains": [],
                "domains": {
                    "standards": {
                        "status": "ready",
                        "validation_status": "ready",
                        "inventory_status": "ready",
                        "release_gate_status": "ready",
                        "alignment_warning": False,
                    }
                },
            },
            "recommendations": [],
        },
        benchmark_knowledge_source_action_plan={
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
            "recommendations": [],
        },
        benchmark_knowledge_source_coverage={
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
        benchmark_knowledge_source_drift={
            "knowledge_source_drift": {
                "status": "improved",
                "current_status": "knowledge_source_coverage_ready",
                "previous_status": "knowledge_source_coverage_partial",
                "source_group_regressions": [],
                "source_group_improvements": ["standards"],
                "resolved_priority_domains": ["standards"],
                "new_priority_domains": [],
                "counts": {"regressions": 0, "improvements": 1},
            },
            "summary": "Knowledge source coverage improved.",
            "recommendations": [
                "Promote the improved knowledge source coverage after CI surfaces refresh."
            ],
        },
        benchmark_knowledge_outcome_correlation={
            "knowledge_outcome_correlation": {
                "status": "knowledge_outcome_correlation_ready",
                "priority_domains": [],
                "domains": {
                    "standards": {
                        "status": "ready",
                        "matrix_status": "ready",
                        "best_surface": "hybrid_dxf",
                        "best_surface_score": 0.91,
                    }
                },
            },
            "recommendations": [],
        },
        benchmark_knowledge_outcome_drift={
            "knowledge_outcome_drift": {
                "status": "improved",
                "current_status": "knowledge_outcome_correlation_ready",
                "previous_status": "knowledge_outcome_correlation_partial",
                "domain_regressions": [],
                "domain_improvements": ["standards"],
                "resolved_priority_domains": ["standards"],
                "new_priority_domains": [],
            },
            "recommendations": [
                "Promote the improved knowledge outcome correlation after CI surfaces refresh."
            ],
        },
        benchmark_competitive_surpass_index={
            "competitive_surpass_index": {
                "status": "competitive_surpass_ready",
                "score": 91,
                "primary_gaps": [],
            },
            "recommendations": [],
        },
        benchmark_competitive_surpass_trend={
            "competitive_surpass_trend": {
                "status": "improved",
                "score_delta": 8,
                "pillar_improvements": ["release_alignment"],
                "pillar_regressions": [],
                "resolved_primary_gaps": ["release_alignment"],
                "new_primary_gaps": [],
            },
            "summary": "status=improved; score_delta=8",
            "recommendations": [
                "Promote the improved competitive surpass posture after CI surfaces refresh."
            ],
        },
        benchmark_competitive_surpass_action_plan={
            "competitive_surpass_action_plan": {
                "status": "competitive_surpass_action_plan_ready",
                "total_action_count": 0,
                "high_priority_action_count": 0,
                "medium_priority_action_count": 0,
                "priority_pillars": [],
                "recommended_first_actions": [],
            },
            "recommendations": [],
        },
        artifact_paths={
            "benchmark_release_decision": "release.json",
            "benchmark_companion_summary": "companion.json",
            "benchmark_artifact_bundle": "bundle.json",
            "benchmark_engineering_signals": "engineering.json",
            "benchmark_realdata_signals": "realdata.json",
            "benchmark_knowledge_domain_capability_matrix": (
                "knowledge_domain_capability_matrix.json"
            ),
            "benchmark_knowledge_domain_capability_drift": (
                "knowledge_domain_capability_drift.json"
            ),
            "benchmark_knowledge_domain_validation_matrix": (
                "knowledge_domain_validation_matrix.json"
            ),
            "benchmark_knowledge_domain_release_surface_alignment": (
                "knowledge_domain_release_surface_alignment.json"
            ),
            "benchmark_knowledge_reference_inventory": (
                "knowledge_reference_inventory.json"
            ),
            "benchmark_knowledge_domain_action_plan": "knowledge_domain_action_plan.json",
            "benchmark_knowledge_domain_control_plane": (
                "knowledge_domain_control_plane.json"
            ),
            "benchmark_knowledge_domain_control_plane_drift": (
                "knowledge_domain_control_plane_drift.json"
            ),
            "benchmark_knowledge_domain_release_gate": (
                "knowledge_domain_release_gate.json"
            ),
            "benchmark_knowledge_domain_release_readiness_matrix": (
                "knowledge_domain_release_readiness_matrix.json"
            ),
            "benchmark_knowledge_source_action_plan": (
                "knowledge_source_action_plan.json"
            ),
            "benchmark_knowledge_source_coverage": "knowledge_source_coverage.json",
            "benchmark_knowledge_outcome_correlation": (
                "knowledge_outcome_correlation.json"
            ),
            "benchmark_knowledge_outcome_drift": "knowledge_outcome_drift.json",
            "benchmark_knowledge_source_drift": "knowledge_source_drift.json",
            "benchmark_competitive_surpass_index": "competitive_surpass.json",
            "benchmark_competitive_surpass_trend": "competitive_surpass_trend.json",
            "benchmark_competitive_surpass_action_plan": (
                "competitive_surpass_action_plan.json"
            ),
        },
    )

    assert payload["ready_to_freeze_baseline"] is True
    assert payload["engineering_status"] == "engineering_semantics_ready"
    assert payload["knowledge_status"] == "knowledge_foundation_ready"
    assert payload["realdata_status"] == "realdata_foundation_ready"
    assert payload["knowledge_focus_areas"] == []
    assert payload["knowledge_drift_status"] == "improved"
    assert payload["knowledge_drift_domain_improvements"] == ["standards"]
    assert payload["knowledge_priority_domains"] == []
    assert payload["knowledge_domains"]["standards"]["status"] == "ready"
    assert payload["knowledge_domain_focus_areas"] == []
    assert payload["knowledge_application_status"] == "knowledge_application_ready"
    assert payload["knowledge_domain_matrix_status"] == "knowledge_domain_matrix_ready"
    assert payload["knowledge_domain_action_plan_status"] == (
        "knowledge_domain_action_plan_ready"
    )
    assert payload["knowledge_domain_control_plane_status"] == (
        "knowledge_domain_control_plane_ready"
    )
    assert payload["knowledge_domain_control_plane_drift_status"] == "improved"
    assert payload["knowledge_domain_validation_matrix_status"] == (
        "knowledge_domain_validation_matrix_ready"
    )
    assert payload["knowledge_domain_release_gate_status"] == (
        "knowledge_domain_release_gate_ready"
    )
    assert payload["knowledge_domain_release_gate_gate_open"] is True
    assert payload["knowledge_domain_release_gate_releasable_domains"] == [
        "standards"
    ]
    assert payload["knowledge_domain_release_readiness_matrix_status"] == (
        "knowledge_domain_release_readiness_ready"
    )
    assert payload["knowledge_domain_release_readiness_matrix_releasable_domains"] == [
        "standards"
    ]
    assert payload["knowledge_domain_release_surface_alignment_status"] == "aligned"
    assert payload["knowledge_reference_inventory_status"] == (
        "knowledge_reference_inventory_ready"
    )
    assert payload["knowledge_reference_inventory_total_reference_items"] == 42
    assert payload["knowledge_source_coverage_status"] == (
        "knowledge_source_coverage_ready"
    )
    assert payload["knowledge_domain_action_plan_actions"][0]["id"] == (
        "standards:realdata"
    )
    assert payload["knowledge_outcome_drift_status"] == "improved"
    assert payload["competitive_surpass_index_status"] == "competitive_surpass_ready"
    assert payload["competitive_surpass_index"]["score"] == 91
    assert payload["competitive_surpass_primary_gaps"] == []
    assert payload["competitive_surpass_recommendations"] == []
    assert payload["competitive_surpass_trend_status"] == "improved"
    assert payload["competitive_surpass_trend_score_delta"] == 8
    assert payload["competitive_surpass_trend_pillar_improvements"] == [
        "release_alignment"
    ]
    assert payload["competitive_surpass_trend_resolved_primary_gaps"] == [
        "release_alignment"
    ]
    assert payload["competitive_surpass_trend_recommendations"] == [
        "Promote the improved competitive surpass posture after CI surfaces refresh."
    ]
    assert payload["competitive_surpass_action_plan_status"] == (
        "competitive_surpass_action_plan_ready"
    )
    assert payload["competitive_surpass_action_plan_total_action_count"] == 0
    assert payload["competitive_surpass_action_plan_priority_pillars"] == []
    assert payload["competitive_surpass_action_plan_recommendations"] == []
    assert payload["next_action"] == "freeze_release_baseline"
    assert "benchmark_operator_adoption" not in payload["missing_artifacts"]
    assert "benchmark_knowledge_drift" not in payload["missing_artifacts"]
    assert "benchmark_knowledge_domain_action_plan" not in payload["missing_artifacts"]
    assert "benchmark_knowledge_domain_control_plane" not in payload["missing_artifacts"]
    assert (
        "benchmark_knowledge_domain_control_plane_drift"
        not in payload["missing_artifacts"]
    )
    assert "benchmark_knowledge_domain_release_gate" not in payload["missing_artifacts"]
    assert (
        "benchmark_knowledge_domain_release_readiness_matrix"
        not in payload["missing_artifacts"]
    )
    assert (
        "benchmark_knowledge_domain_release_surface_alignment"
        not in payload["missing_artifacts"]
    )
    assert "benchmark_knowledge_reference_inventory" not in payload["missing_artifacts"]
    assert "benchmark_knowledge_source_action_plan" not in payload["missing_artifacts"]
    assert "benchmark_knowledge_source_coverage" not in payload["missing_artifacts"]
    assert "benchmark_knowledge_outcome_drift" not in payload["missing_artifacts"]
    assert "benchmark_realdata_signals" not in payload["missing_artifacts"]
    assert "benchmark_competitive_surpass_index" not in payload["missing_artifacts"]
    assert "benchmark_competitive_surpass_trend" not in payload["missing_artifacts"]
    assert "benchmark_competitive_surpass_action_plan" not in payload["missing_artifacts"]
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is False
    assert payload["artifacts"]["benchmark_realdata_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_knowledge_outcome_drift"]["present"] is True
    assert (
        payload["artifacts"]["benchmark_knowledge_domain_validation_matrix"]["present"]
        is True
    )
    assert (
        payload["artifacts"]["benchmark_knowledge_domain_control_plane"]["present"]
        is True
    )
    assert (
        payload["artifacts"]["benchmark_knowledge_domain_control_plane_drift"][
            "present"
        ]
        is True
    )
    assert payload["artifacts"]["benchmark_knowledge_domain_release_gate"]["present"] is True
    assert payload["artifacts"][
        "benchmark_knowledge_domain_release_readiness_matrix"
    ]["present"] is True
    assert (
        payload["artifacts"]["benchmark_knowledge_domain_release_surface_alignment"][
            "present"
        ]
        is True
    )
    assert payload["artifacts"]["benchmark_knowledge_reference_inventory"]["present"] is True
    assert payload["artifacts"]["benchmark_competitive_surpass_index"]["present"] is True
    assert payload["artifacts"]["benchmark_competitive_surpass_trend"]["present"] is True
    assert payload["artifacts"]["benchmark_competitive_surpass_action_plan"][
        "present"
    ] is True
    assert payload["operator_steps"][-1]["status"] == "ready"
    assert payload["operator_adoption"]["knowledge_outcome_drift_status"] == "unknown"
    assert "## Knowledge Domain Release Readiness Matrix" in render_markdown(payload)
    assert "## Knowledge Reference Inventory" in render_markdown(payload)
    assert "## Competitive Surpass Action Plan" in render_markdown(payload)


def test_render_markdown_and_cli_outputs(tmp_path: Path) -> None:
    release = tmp_path / "release.json"
    companion = tmp_path / "companion.json"
    bundle = tmp_path / "bundle.json"
    knowledge = tmp_path / "knowledge.json"
    drift = tmp_path / "drift.json"
    engineering = tmp_path / "engineering.json"
    realdata = tmp_path / "realdata.json"
    operator_adoption = tmp_path / "operator_adoption.json"
    knowledge_application = tmp_path / "knowledge_application.json"
    knowledge_realdata_correlation = tmp_path / "knowledge_realdata_correlation.json"
    knowledge_domain_matrix = tmp_path / "knowledge_domain_matrix.json"
    knowledge_domain_capability_matrix = tmp_path / "knowledge_domain_capability_matrix.json"
    knowledge_domain_capability_drift = (
        tmp_path / "knowledge_domain_capability_drift.json"
    )
    knowledge_domain_validation_matrix = (
        tmp_path / "knowledge_domain_validation_matrix.json"
    )
    knowledge_domain_control_plane = tmp_path / "knowledge_domain_control_plane.json"
    knowledge_domain_control_plane_drift = (
        tmp_path / "knowledge_domain_control_plane_drift.json"
    )
    knowledge_domain_release_gate = tmp_path / "knowledge_domain_release_gate.json"
    knowledge_domain_release_readiness_matrix = (
        tmp_path / "knowledge_domain_release_readiness_matrix.json"
    )
    knowledge_domain_release_surface_alignment = (
        tmp_path / "knowledge_domain_release_surface_alignment.json"
    )
    knowledge_reference_inventory = tmp_path / "knowledge_reference_inventory.json"
    knowledge_domain_action_plan = tmp_path / "knowledge_domain_action_plan.json"
    knowledge_source_action_plan = tmp_path / "knowledge_source_action_plan.json"
    knowledge_source_coverage = tmp_path / "knowledge_source_coverage.json"
    knowledge_source_drift = tmp_path / "knowledge_source_drift.json"
    knowledge_outcome_correlation = tmp_path / "knowledge_outcome_correlation.json"
    knowledge_outcome_drift = tmp_path / "knowledge_outcome_drift.json"
    competitive_surpass = tmp_path / "competitive_surpass.json"
    competitive_surpass_trend = tmp_path / "competitive_surpass_trend.json"
    competitive_surpass_action_plan = tmp_path / "competitive_surpass_action_plan.json"
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
                    "benchmark_knowledge_domain_capability_drift": {
                        "path": "knowledge_domain_capability_drift.json"
                    },
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
                    "priority_domains": ["tolerance"],
                    "domains": {
                        "tolerance": {
                            "status": "partial",
                            "focus_components": ["tolerance"],
                            "missing_metrics": ["common_fit_count"],
                        }
                    },
                    "domain_focus_areas": [
                        {
                            "domain": "tolerance",
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
    drift.write_text(
        json.dumps(
            {
                "knowledge_drift": {
                    "status": "regressed",
                    "regressions": ["tolerance"],
                    "improvements": [],
                    "new_focus_areas": ["tolerance"],
                    "resolved_focus_areas": [],
                },
                "recommendations": [
                    "Resolve knowledge regressions before claiming the benchmark "
                    "surpass baseline remains stable."
                ],
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
    realdata.write_text(
        json.dumps(
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
                "knowledge_drift_status": "regressed",
                "knowledge_drift_summary": "Tolerance coverage regressed.",
                "knowledge_outcome_drift_status": "regressed",
                "knowledge_outcome_drift_summary": "Tolerance outcome coverage regressed.",
                "knowledge_drift": {
                    "recommendations": ["Backfill tolerance knowledge coverage."],
                },
                "knowledge_outcome_drift": {
                    "recommendations": ["Backfill tolerance outcome coverage."]
                },
            }
        ),
        encoding="utf-8",
    )
    knowledge_application.write_text(
        json.dumps(
            {
                "knowledge_application": {
                    "status": "knowledge_application_partial",
                    "priority_domains": ["tolerance"],
                    "domains": {
                        "tolerance": {
                            "status": "partial",
                            "readiness_status": "partial",
                            "evidence_status": "partial",
                            "signal_count": 1,
                        }
                    },
                    "focus_areas_detail": [
                        {
                            "domain": "tolerance",
                            "status": "partial",
                            "priority": "medium",
                            "action": "Raise tolerance application coverage.",
                        }
                    ],
                },
                "recommendations": ["Raise tolerance application coverage."],
            }
        ),
        encoding="utf-8",
    )
    knowledge_realdata_correlation.write_text(
        json.dumps(
            {
                "knowledge_realdata_correlation": {
                    "status": "knowledge_realdata_partial",
                    "priority_domains": ["tolerance"],
                    "domains": {
                        "tolerance": {
                            "status": "partial",
                            "readiness_status": "partial",
                            "application_status": "partial",
                            "realdata_status": "partial",
                        }
                    },
                },
                "recommendations": ["Raise tolerance real-data depth."],
            }
        ),
        encoding="utf-8",
    )
    knowledge_domain_matrix.write_text(
        json.dumps(
            {
                "knowledge_domain_matrix": {
                    "status": "knowledge_domain_matrix_partial",
                    "priority_domains": ["tolerance"],
                    "domains": {
                        "tolerance": {
                            "status": "partial",
                            "readiness_status": "partial",
                            "application_status": "partial",
                            "realdata_status": "partial",
                        }
                    },
                },
                "recommendations": [
                    "Backfill tolerance foundation and raise real-data depth."
                ],
            }
        ),
        encoding="utf-8",
    )
    knowledge_domain_capability_matrix.write_text(
        json.dumps(
            {
                "knowledge_domain_capability_matrix": {
                    "status": "knowledge_domain_capability_partial",
                    "priority_domains": ["gdt"],
                    "focus_areas_detail": [
                        {
                            "domain": "gdt",
                            "label": "GD&T & Datums",
                            "status": "blocked",
                            "priority": "high",
                            "provider_status": "missing",
                            "surface_status": "partial",
                            "primary_gaps": [
                                "provider_missing:gdt",
                                "public_surface_missing",
                            ],
                            "action": "Promote GD&T into provider-backed surfaces.",
                        }
                    ],
                    "domains": {
                        "gdt": {
                            "domain": "gdt",
                            "label": "GD&T & Datums",
                            "status": "blocked",
                            "foundation_status": "ready",
                            "application_status": "ready",
                            "matrix_status": "ready",
                            "provider_status": "missing",
                            "surface_status": "partial",
                            "reference_item_count": 2,
                            "primary_gaps": [
                                "provider_missing:gdt",
                                "public_surface_missing",
                            ],
                        }
                    },
                },
                "recommendations": [
                    "Backfill GD&T provider coverage: gdt",
                    "Add a public benchmark/API surface for GD&T & Datums",
                ],
            }
        ),
        encoding="utf-8",
    )
    knowledge_domain_capability_drift.write_text(
        json.dumps(
            {
                "knowledge_domain_capability_drift": {
                    "status": "regressed",
                    "domain_regressions": ["gdt"],
                    "domain_improvements": [],
                },
                "recommendations": [
                    "Restore regressed knowledge domain capabilities before claiming "
                    "benchmark capability stability."
                ],
            }
        ),
        encoding="utf-8",
    )
    knowledge_domain_validation_matrix.write_text(
        json.dumps(
            {
                "knowledge_domain_validation_matrix": {
                    "status": "knowledge_domain_validation_matrix_partial",
                    "summary": "Tolerance validation coverage is still partial.",
                    "priority_domains": ["tolerance"],
                    "total_test_count": 7,
                    "domains": {
                        "tolerance": {
                            "status": "partial",
                            "provider_status": "ready",
                            "api_status": "partial",
                            "integration_status": "partial",
                            "assistant_status": "ready",
                        }
                    },
                },
                "recommendations": [
                    "Raise tolerance validation coverage before promoting the next baseline."
                ],
            }
        ),
        encoding="utf-8",
    )
    knowledge_domain_control_plane.write_text(
        json.dumps(
            {
                "knowledge_domain_control_plane": {
                    "status": "knowledge_domain_control_plane_ready",
                    "domains": {"tolerance": {"status": "ready"}},
                    "focus_areas": [],
                    "release_blockers": [],
                },
                "recommendations": [],
            }
        ),
        encoding="utf-8",
    )
    knowledge_domain_control_plane_drift.write_text(
        json.dumps(
            {
                "knowledge_domain_control_plane_drift": {
                    "status": "improved",
                    "domain_regressions": [],
                    "domain_improvements": ["tolerance"],
                    "new_release_blockers": [],
                    "resolved_release_blockers": ["tolerance"],
                },
                "recommendations": [],
            }
        ),
        encoding="utf-8",
    )
    knowledge_domain_release_gate.write_text(
        json.dumps(
            {
                "knowledge_domain_release_gate": {
                    "status": "knowledge_domain_release_gate_partial",
                    "gate_open": True,
                    "releasable_domains": ["standards"],
                    "blocked_domains": [],
                    "priority_domains": ["tolerance"],
                },
                "blocking_reasons": [],
                "warning_reasons": ["tolerance:review_before_release"],
                "recommendations": [
                    "Review tolerance release readiness before promotion."
                ],
            }
        ),
        encoding="utf-8",
    )
    knowledge_domain_release_readiness_matrix.write_text(
        json.dumps(
            {
                "knowledge_domain_release_readiness_matrix": {
                    "status": "knowledge_domain_release_readiness_partial",
                    "summary": "ready=0; partial=1; blocked=0",
                    "priority_domains": ["tolerance"],
                    "releasable_domains": ["standards"],
                    "blocked_domains": [],
                    "domains": {
                        "tolerance": {
                            "status": "partial",
                            "validation_status": "partial",
                            "inventory_status": "partial",
                            "release_gate_status": "partial",
                            "alignment_warning": False,
                        }
                    },
                },
                "recommendations": [
                    "Review tolerance release readiness before promotion."
                ],
            }
        ),
        encoding="utf-8",
    )
    knowledge_domain_release_surface_alignment.write_text(
        json.dumps(
            {
                "knowledge_domain_release_surface_alignment": {
                    "status": "aligned",
                    "summary": "all release surfaces aligned",
                    "mismatches": [],
                    "domain_mismatches": [],
                    "release_blocker_mismatches": [],
                },
                "recommendations": [],
            }
        ),
        encoding="utf-8",
    )
    knowledge_domain_action_plan.write_text(
        json.dumps(
            {
                "knowledge_domain_action_plan": {
                    "status": "knowledge_domain_action_plan_partial",
                    "priority_domains": ["tolerance"],
                    "actions": [
                        {
                            "id": "tolerance:realdata",
                            "domain": "tolerance",
                            "stage": "realdata",
                            "priority": "high",
                            "status": "partial",
                        }
                    ],
                },
                "recommendations": [
                    "Raise tolerance real-data depth before promoting the next baseline."
                ],
            }
        ),
        encoding="utf-8",
    )
    knowledge_reference_inventory.write_text(
        json.dumps(
            {
                "knowledge_reference_inventory": {
                    "status": "knowledge_reference_inventory_partial",
                    "summary": "Tolerance reference tables are incomplete.",
                    "priority_domains": ["tolerance"],
                    "total_reference_items": 19,
                    "domains": {
                        "tolerance": {
                            "status": "partial",
                            "total_reference_items": 19,
                            "populated_table_count": 2,
                            "total_table_count": 4,
                            "missing_tables": ["hole_limits", "shaft_limits"],
                        }
                    },
                    "focus_tables_detail": [
                        {
                            "domain": "tolerance",
                            "missing_tables": ["hole_limits", "shaft_limits"],
                            "action": "Backfill tolerance reference tables.",
                        }
                    ],
                },
                "recommendations": ["Backfill tolerance reference tables."],
            }
        ),
        encoding="utf-8",
    )
    knowledge_source_coverage.write_text(
        json.dumps(
            {
                "knowledge_source_coverage": {
                    "status": "knowledge_source_coverage_partial",
                    "priority_domains": ["tolerance"],
                    "domains": {
                        "tolerance": {
                            "status": "partial",
                            "focus_source_groups": ["tolerance"],
                        }
                    },
                    "expansion_candidates": [{"name": "machining", "status": "ready"}],
                },
                "recommendations": ["Backfill tolerance source coverage."],
            }
        ),
        encoding="utf-8",
    )
    knowledge_source_action_plan.write_text(
        json.dumps(
            {
                "knowledge_source_action_plan": {
                    "status": "knowledge_source_action_plan_partial",
                    "priority_domains": ["tolerance"],
                    "total_action_count": 1,
                    "high_priority_action_count": 1,
                    "medium_priority_action_count": 0,
                    "expansion_action_count": 0,
                    "recommended_first_actions": [
                        {
                            "id": "tolerance:coverage",
                            "domain": "tolerance",
                            "stage": "source_group",
                            "priority": "high",
                        }
                    ],
                    "source_group_action_counts": {"tolerance": 1},
                },
                "recommendations": ["Backfill tolerance source actions first."],
            }
        ),
        encoding="utf-8",
    )
    knowledge_source_drift.write_text(
        json.dumps(
            {
                "knowledge_source_drift": {
                    "status": "regressed",
                    "current_status": "knowledge_source_coverage_partial",
                    "previous_status": "knowledge_source_coverage_ready",
                    "source_group_regressions": ["tolerance"],
                    "source_group_improvements": [],
                    "new_priority_domains": ["tolerance"],
                    "resolved_priority_domains": [],
                },
                "summary": "Tolerance source coverage regressed.",
                "recommendations": [
                    "Restore regressed knowledge source groups before claiming "
                    "benchmark source stability."
                ],
            }
        ),
        encoding="utf-8",
    )
    knowledge_outcome_correlation.write_text(
        json.dumps(
            {
                "knowledge_outcome_correlation": {
                    "status": "knowledge_outcome_correlation_partial",
                    "priority_domains": ["tolerance"],
                    "domains": {
                        "tolerance": {
                            "status": "partial",
                            "matrix_status": "partial",
                            "best_surface": "hybrid_dxf",
                            "best_surface_score": 0.88,
                        }
                    },
                },
                "recommendations": [
                    "Use companion, bundle, release decision, and runbook surfaces "
                    "to track which knowledge domains are still weak on real-data outcomes."
                ],
            }
        ),
        encoding="utf-8",
    )
    knowledge_outcome_drift.write_text(
        json.dumps(
            {
                "knowledge_outcome_drift": {
                    "status": "regressed",
                    "current_status": "knowledge_outcome_correlation_partial",
                    "previous_status": "knowledge_outcome_correlation_ready",
                    "domain_regressions": ["tolerance"],
                    "domain_improvements": [],
                    "new_priority_domains": ["tolerance"],
                    "resolved_priority_domains": [],
                },
                "recommendations": [
                    "Resolve knowledge outcome regressions before claiming "
                    "benchmark outcome stability."
                ],
            }
        ),
        encoding="utf-8",
    )
    competitive_surpass.write_text(
        json.dumps(
            {
                "competitive_surpass_index": {
                    "status": "competitive_surpass_partial",
                    "score": 73,
                    "primary_gaps": ["history_realdata", "step_dir_depth"],
                },
                "recommendations": [
                    "Expand history and STEP real-data depth before claiming "
                    "benchmark surpass readiness."
                ],
            }
        ),
        encoding="utf-8",
    )
    competitive_surpass_trend.write_text(
        json.dumps(
            {
                "competitive_surpass_trend": {
                    "status": "mixed",
                    "score_delta": 3,
                    "pillar_improvements": ["knowledge"],
                    "pillar_regressions": ["realdata"],
                    "resolved_primary_gaps": [],
                    "new_primary_gaps": ["step_dir_depth"],
                },
                "summary": "status=mixed; score_delta=3",
                "recommendations": [
                    "Keep the current competitive surpass rollout under review until "
                    "regressions are cleared."
                ],
            }
        ),
        encoding="utf-8",
    )
    competitive_surpass_action_plan.write_text(
        json.dumps(
            {
                "competitive_surpass_action_plan": {
                    "status": "competitive_surpass_action_plan_partial",
                    "total_action_count": 2,
                    "high_priority_action_count": 1,
                    "medium_priority_action_count": 1,
                    "priority_pillars": ["realdata", "knowledge"],
                    "recommended_first_actions": [
                        {
                            "pillar": "realdata",
                            "action": "Expand STEP/B-Rep depth before promotion.",
                        }
                    ],
                },
                "recommendations": [
                    "realdata: Expand STEP/B-Rep depth before promotion."
                ],
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
            "--benchmark-knowledge-drift",
            str(drift),
            "--benchmark-engineering-signals",
            str(engineering),
            "--benchmark-realdata-signals",
            str(realdata),
            "--benchmark-operator-adoption",
            str(operator_adoption),
            "--benchmark-knowledge-application",
            str(knowledge_application),
            "--benchmark-knowledge-realdata-correlation",
            str(knowledge_realdata_correlation),
            "--benchmark-knowledge-domain-matrix",
            str(knowledge_domain_matrix),
            "--benchmark-knowledge-domain-capability-matrix",
            str(knowledge_domain_capability_matrix),
            "--benchmark-knowledge-domain-capability-drift",
            str(knowledge_domain_capability_drift),
            "--benchmark-knowledge-domain-validation-matrix",
            str(knowledge_domain_validation_matrix),
            "--benchmark-knowledge-domain-control-plane",
            str(knowledge_domain_control_plane),
            "--benchmark-knowledge-domain-control-plane-drift",
            str(knowledge_domain_control_plane_drift),
            "--benchmark-knowledge-domain-release-gate",
            str(knowledge_domain_release_gate),
            "--benchmark-knowledge-domain-release-readiness-matrix",
            str(knowledge_domain_release_readiness_matrix),
            "--benchmark-knowledge-domain-release-surface-alignment",
            str(knowledge_domain_release_surface_alignment),
            "--benchmark-knowledge-reference-inventory",
            str(knowledge_reference_inventory),
            "--benchmark-knowledge-domain-action-plan",
            str(knowledge_domain_action_plan),
            "--benchmark-knowledge-source-action-plan",
            str(knowledge_source_action_plan),
            "--benchmark-knowledge-source-coverage",
            str(knowledge_source_coverage),
            "--benchmark-knowledge-source-drift",
            str(knowledge_source_drift),
            "--benchmark-knowledge-outcome-correlation",
            str(knowledge_outcome_correlation),
            "--benchmark-knowledge-outcome-drift",
            str(knowledge_outcome_drift),
            "--benchmark-competitive-surpass-index",
            str(competitive_surpass),
            "--benchmark-competitive-surpass-trend",
            str(competitive_surpass_trend),
            "--benchmark-competitive-surpass-action-plan",
            str(competitive_surpass_action_plan),
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
    assert payload["realdata_status"] == "realdata_foundation_partial"
    assert payload["knowledge_focus_areas"][0]["component"] == "tolerance"
    assert payload["knowledge_drift_status"] == "regressed"
    assert payload["knowledge_domains"]["tolerance"]["status"] == "partial"
    assert payload["knowledge_domain_focus_areas"][0]["domain"] == "tolerance"
    assert payload["knowledge_application_status"] == "knowledge_application_partial"
    assert payload["knowledge_application_domains"]["tolerance"]["status"] == "partial"
    assert payload["knowledge_domain_action_plan_status"] == (
        "knowledge_domain_action_plan_partial"
    )
    assert payload["knowledge_domain_control_plane_status"] == (
        "knowledge_domain_control_plane_ready"
    )
    assert payload["knowledge_domain_control_plane_drift_status"] == "improved"
    assert payload["knowledge_domain_validation_matrix_status"] == (
        "knowledge_domain_validation_matrix_partial"
    )
    assert payload["knowledge_domain_release_gate_status"] == (
        "knowledge_domain_release_gate_partial"
    )
    assert payload["knowledge_domain_release_gate_gate_open"] is True
    assert payload["knowledge_domain_release_gate_priority_domains"] == ["tolerance"]
    assert payload["knowledge_domain_release_readiness_matrix_status"] == (
        "knowledge_domain_release_readiness_partial"
    )
    assert payload["knowledge_domain_release_readiness_matrix_priority_domains"] == [
        "tolerance"
    ]
    assert payload["knowledge_domain_release_surface_alignment_status"] == "aligned"
    assert payload["knowledge_reference_inventory_status"] == (
        "knowledge_reference_inventory_partial"
    )
    assert payload["knowledge_reference_inventory_total_reference_items"] == 19
    assert payload["knowledge_source_action_plan_status"] == (
        "knowledge_source_action_plan_partial"
    )
    assert payload["knowledge_source_coverage_status"] == (
        "knowledge_source_coverage_partial"
    )
    assert payload["knowledge_source_drift_status"] == "regressed"
    assert payload["knowledge_source_drift_source_group_regressions"] == ["tolerance"]
    assert payload["knowledge_source_coverage_domains"]["tolerance"]["status"] == (
        "partial"
    )
    assert payload["knowledge_domain_action_plan_actions"][0]["id"] == (
        "tolerance:realdata"
    )
    assert payload["knowledge_outcome_correlation_status"] == (
        "knowledge_outcome_correlation_partial"
    )
    assert payload["knowledge_outcome_drift_status"] == "regressed"
    assert payload["competitive_surpass_index_status"] == "competitive_surpass_partial"
    assert payload["competitive_surpass_index"]["score"] == 73
    assert payload["competitive_surpass_primary_gaps"] == [
        "history_realdata",
        "step_dir_depth",
    ]
    assert payload["competitive_surpass_recommendations"] == [
        "Expand history and STEP real-data depth before claiming benchmark surpass readiness."
    ]
    assert payload["competitive_surpass_trend_status"] == "mixed"
    assert payload["competitive_surpass_trend_score_delta"] == 3
    assert payload["competitive_surpass_trend_pillar_improvements"] == ["knowledge"]
    assert payload["competitive_surpass_trend_pillar_regressions"] == ["realdata"]
    assert payload["competitive_surpass_trend_new_primary_gaps"] == ["step_dir_depth"]
    assert payload["competitive_surpass_trend_recommendations"] == [
        "Keep the current competitive surpass rollout under review until regressions are cleared."
    ]
    assert payload["competitive_surpass_action_plan_status"] == (
        "competitive_surpass_action_plan_partial"
    )
    assert payload["competitive_surpass_action_plan_total_action_count"] == 2
    assert payload["competitive_surpass_action_plan_priority_pillars"] == [
        "realdata",
        "knowledge",
    ]
    assert payload["competitive_surpass_action_plan_recommendations"] == [
        "realdata: Expand STEP/B-Rep depth before promotion."
    ]
    assert payload["next_action"] == "review_signals"
    assert "benchmark_knowledge_domain_action_plan" not in payload["missing_artifacts"]
    assert "benchmark_knowledge_domain_control_plane" not in payload["missing_artifacts"]
    assert (
        "benchmark_knowledge_domain_control_plane_drift"
        not in payload["missing_artifacts"]
    )
    assert (
        "benchmark_knowledge_domain_validation_matrix"
        not in payload["missing_artifacts"]
    )
    assert "benchmark_knowledge_domain_release_gate" not in payload["missing_artifacts"]
    assert (
        "benchmark_knowledge_domain_release_readiness_matrix"
        not in payload["missing_artifacts"]
    )
    assert (
        "benchmark_knowledge_domain_release_surface_alignment"
        not in payload["missing_artifacts"]
    )
    assert "benchmark_knowledge_reference_inventory" not in payload["missing_artifacts"]
    assert "benchmark_knowledge_source_action_plan" not in payload["missing_artifacts"]
    assert "benchmark_knowledge_source_coverage" not in payload["missing_artifacts"]
    assert payload["artifacts"]["benchmark_knowledge_drift"]["present"] is True
    assert payload["artifacts"]["benchmark_knowledge_outcome_drift"]["present"] is True
    assert payload["artifacts"]["benchmark_engineering_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_realdata_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True
    assert (
        payload["artifacts"]["benchmark_knowledge_domain_validation_matrix"]["present"]
        is True
    )
    assert (
        payload["artifacts"]["benchmark_knowledge_domain_control_plane"]["present"]
        is True
    )
    assert (
        payload["artifacts"]["benchmark_knowledge_domain_control_plane_drift"][
            "present"
        ]
        is True
    )
    assert payload["artifacts"]["benchmark_knowledge_domain_release_gate"]["present"] is True
    assert payload["artifacts"][
        "benchmark_knowledge_domain_release_readiness_matrix"
    ]["present"] is True
    assert (
        payload["artifacts"]["benchmark_knowledge_domain_release_surface_alignment"][
            "present"
        ]
        is True
    )
    assert payload["artifacts"]["benchmark_knowledge_reference_inventory"]["present"] is True
    assert payload["artifacts"]["benchmark_competitive_surpass_index"]["present"] is True
    assert payload["artifacts"]["benchmark_competitive_surpass_trend"]["present"] is True
    assert payload["artifacts"]["benchmark_competitive_surpass_action_plan"][
        "present"
    ] is True
    assert payload["operator_adoption"]["signals"] == [
        "operator_shift_handoff:pending"
    ]
    assert payload["operator_adoption"]["actions"] == [
        "Book an operator office-hours review."
    ]
    assert payload["operator_adoption"]["knowledge_drift_status"] == "regressed"
    assert payload["operator_adoption"]["knowledge_outcome_drift_status"] == "regressed"
    assert output_md.exists()

    rendered = render_markdown(payload)
    assert "# Benchmark Release Runbook" in rendered
    assert "`engineering_status`: `partial_engineering_semantics`" in rendered
    assert "## Knowledge Reference Inventory" in rendered
    assert "`knowledge_status`: `knowledge_foundation_partial`" in rendered
    assert "`realdata_status`: `realdata_foundation_partial`" in rendered
    assert "`next_action`: `review_signals`" in rendered
    assert "Backfill tolerance coverage." in rendered
    assert "## Knowledge Drift" in rendered
    assert "`status`: `regressed`" in rendered
    assert "## Knowledge Domains" in rendered
    assert "## Knowledge Domain Focus Areas" in rendered
    assert "## Knowledge Application" in rendered
    assert "## Knowledge Domain Matrix" in rendered
    assert "## Knowledge Domain Validation Matrix" in rendered
    assert "## Knowledge Domain Release Gate" in rendered
    assert "## Knowledge Domain Release Readiness Matrix" in rendered
    assert "## Knowledge Domain Action Plan" in rendered
    assert "## Knowledge Source Drift" in rendered
    assert "## Knowledge Outcome Drift" in rendered
    assert "## Competitive Surpass Index" in rendered
    assert "`competitive_surpass_index_status`: `competitive_surpass_partial`" in rendered
    assert "history_realdata, step_dir_depth" in rendered
    assert (
        "Expand history and STEP real-data depth before claiming benchmark surpass readiness."
        in rendered
    )
    assert "## Competitive Surpass Trend" in rendered
    assert "`competitive_surpass_trend_status`: `mixed`" in rendered
    assert (
        "Keep the current competitive surpass rollout under review until regressions are cleared."
        in rendered
    )
    assert "## Competitive Surpass Action Plan" in rendered
    assert (
        "`status`: `competitive_surpass_action_plan_partial`" in rendered
    )
    assert "realdata, knowledge" in rendered
    assert "Expand STEP/B-Rep depth before promotion." in rendered
    assert "## Real-Data Signals" in rendered
    assert "## Operator Adoption" in rendered
    assert "operator_shift_handoff:pending" in rendered
    assert "Book an operator office-hours review." in rendered
    assert "Tolerance source coverage regressed." in rendered
    assert "Tolerance coverage regressed." in rendered
    assert "Tolerance outcome coverage regressed." in rendered
    assert "Backfill tolerance outcome coverage." in rendered
    assert "Expand STEP/B-Rep directory validation." in rendered
    assert "`benchmark_operator_adoption`: present=`True`" in rendered


def test_build_release_runbook_exposes_scorecard_and_operational_operator_adoption() -> None:
    payload = build_release_runbook(
        title="Benchmark Release Runbook",
        benchmark_release_decision={"release_status": "review_required"},
        benchmark_scorecard={
            "components": {
                "operator_adoption": {
                    "status": "guided_manual",
                    "operator_mode": "shadow_review",
                    "knowledge_outcome_drift_status": "regressed",
                    "knowledge_outcome_drift_summary": "Operator outcome drift needs review.",
                }
            }
        },
        benchmark_operational_summary={
            "component_statuses": {"operator_adoption": "attention_required"},
            "operator_adoption_knowledge_outcome_drift_status": "partial",
            "operator_adoption_knowledge_outcome_drift_summary": "Operational rollout is partial.",
        },
        benchmark_companion_summary={},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_operator_adoption={},
        artifact_paths={},
    )

    assert payload["scorecard_operator_adoption"]["status"] == "guided_manual"
    assert payload["scorecard_operator_adoption"]["operator_mode"] == "shadow_review"
    assert (
        payload["scorecard_operator_adoption"]["knowledge_outcome_drift_status"]
        == "regressed"
    )
    assert payload["operational_operator_adoption"]["status"] == "attention_required"
    assert (
        payload["operational_operator_adoption"]["knowledge_outcome_drift_status"]
        == "partial"
    )
    assert payload["operator_adoption_release_surface_alignment"]["status"] == "unknown"

    rendered = render_markdown(payload)
    assert "## Scorecard Operator Adoption" in rendered
    assert "`operator_mode`: `shadow_review`" in rendered
    assert "Operator outcome drift needs review." in rendered
    assert "## Operational Operator Adoption" in rendered
    assert "Operational rollout is partial." in rendered
    assert "## Operator Adoption Release Surface Alignment" in rendered


def test_build_release_runbook_exposes_knowledge_source_drift_passthrough() -> None:
    payload = build_release_runbook(
        title="Benchmark Release Runbook",
        benchmark_release_decision={"release_status": "review_required"},
        benchmark_scorecard={},
        benchmark_operational_summary={},
        benchmark_companion_summary={},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_knowledge_outcome_drift={},
        benchmark_knowledge_source_drift={
            "knowledge_source_drift": {
                "status": "regressed",
                "current_status": "knowledge_source_coverage_partial",
                "previous_status": "knowledge_source_coverage_ready",
                "source_group_regressions": ["gdt"],
                "source_group_improvements": [],
                "new_priority_domains": ["gdt"],
                "resolved_priority_domains": [],
                "counts": {"regressions": 1, "improvements": 0},
            },
            "summary": "GD&T source coverage regressed.",
            "recommendations": [
                "Restore regressed knowledge source groups before claiming "
                "benchmark source stability."
            ],
        },
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_operator_adoption={},
        artifact_paths={"benchmark_knowledge_source_drift": "knowledge_source_drift.json"},
    )

    assert payload["knowledge_source_drift_status"] == "regressed"
    assert payload["knowledge_source_drift_source_group_regressions"] == ["gdt"]
    assert payload["knowledge_source_drift_recommendations"] == [
        "Restore regressed knowledge source groups before claiming benchmark source stability."
    ]
    assert payload["artifacts"]["benchmark_knowledge_source_drift"]["present"] is True
    assert (
        "Restore regressed knowledge source groups before claiming benchmark source stability."
        in payload["review_signals"]
    )

    rendered = render_markdown(payload)
    assert "## Knowledge Source Drift" in rendered
    assert "GD&T source coverage regressed." in rendered


def test_build_release_runbook_exposes_operator_adoption_release_alignment() -> None:
    payload = build_release_runbook(
        title="Benchmark Release Runbook",
        benchmark_release_decision={"release_status": "review_required"},
        benchmark_scorecard={},
        benchmark_operational_summary={},
        benchmark_companion_summary={},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "release_surface_alignment_status": "aligned",
            "release_surface_alignment_summary": (
                "Release decision and runbook are aligned."
            ),
            "release_surface_alignment": {"mismatches": []},
        },
        artifact_paths={},
    )

    assert payload["operator_adoption_release_surface_alignment"]["status"] == "aligned"
    assert payload["operator_adoption_release_surface_alignment"]["mismatches"] == []

    rendered = render_markdown(payload)
    assert "Release decision and runbook are aligned." in rendered
