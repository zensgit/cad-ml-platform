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
        },
        benchmark_knowledge_drift={
            "knowledge_drift": {
                "status": "regressed",
                "regressions": ["gdt"],
                "improvements": [],
                "domain_regressions": ["gdt"],
                "domain_improvements": [],
                "new_focus_areas": ["gdt"],
                "resolved_focus_areas": [],
                "resolved_priority_domains": [],
                "new_priority_domains": ["gdt"],
            },
            "recommendations": [
                "Resolve knowledge regressions before claiming the benchmark surpass "
                "baseline remains stable."
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
            "adoption_readiness": "guided_manual",
            "blocking_signals": ["operator:blocker"],
            "knowledge_drift_status": "regressed",
            "knowledge_drift_summary": "Tolerance coverage regressed.",
            "knowledge_drift": {
                "recommendations": ["Backfill tolerance knowledge coverage."],
            },
            "recommended_actions": ["Operator fallback only."],
        },
        benchmark_knowledge_application={
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
        },
        benchmark_knowledge_domain_matrix={
            "knowledge_domain_matrix": {
                "status": "knowledge_domain_matrix_partial",
                "priority_domains": ["tolerance"],
                "domains": {
                    "tolerance": {
                        "status": "blocked",
                        "readiness_status": "partial",
                        "application_status": "partial",
                        "realdata_status": "partial",
                    }
                },
            },
            "recommendations": ["Backfill tolerance foundation and real-data coverage."],
        },
        benchmark_knowledge_domain_action_plan={
            "knowledge_domain_action_plan": {
                "status": "knowledge_domain_action_plan_blocked",
                "priority_domains": ["tolerance"],
                "actions": [
                    {
                        "id": "tolerance:foundation",
                        "domain": "tolerance",
                        "stage": "foundation",
                        "priority": "high",
                        "status": "blocked",
                    }
                ],
            },
            "recommendations": ["Backfill tolerance foundation metrics first."],
        },
        benchmark_knowledge_source_action_plan={
            "knowledge_source_action_plan": {
                "status": "knowledge_source_action_plan_blocked",
                "priority_domains": ["tolerance"],
                "total_action_count": 2,
                "high_priority_action_count": 1,
                "medium_priority_action_count": 1,
                "expansion_action_count": 1,
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
        },
        benchmark_knowledge_source_coverage={
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
                "summary": "Tolerance reference tables are incomplete.",
                "priority_domains": ["tolerance"],
                "total_reference_items": 24,
                "domains": {
                    "tolerance": {
                        "status": "partial",
                        "total_reference_items": 24,
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
        },
        benchmark_knowledge_domain_release_gate={
            "knowledge_domain_release_gate": {
                "status": "knowledge_domain_release_gate_blocked",
                "gate_open": False,
                "releasable_domains": ["standards"],
                "blocked_domains": ["tolerance"],
                "priority_domains": ["tolerance"],
            },
            "blocking_reasons": ["tolerance:not_release_ready"],
            "warning_reasons": ["tolerance:review_before_release"],
            "recommendations": ["Unblock tolerance before freeze."],
        },
        artifact_paths={
            "benchmark_companion_summary": "companion.json",
            "benchmark_engineering_signals": "engineering.json",
            "benchmark_operator_adoption": "operator.json",
            "benchmark_knowledge_domain_release_gate": (
                "knowledge_domain_release_gate.json"
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
    assert payload["automation_ready"] is False
    assert payload["primary_signal_source"] == "benchmark_companion_summary"
    assert payload["blocking_signals"] == [
        "review_queue:critical_backlog",
        "tolerance:not_release_ready",
    ]
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_partial"
    assert payload["component_statuses"]["knowledge_drift"] == "regressed"
    assert payload["component_statuses"]["engineering_signals"] == "partial_engineering_semantics"
    assert payload["component_statuses"]["realdata_signals"] == "realdata_foundation_partial"
    assert payload["component_statuses"]["operator_adoption"] == "guided_manual"
    assert payload["component_statuses"]["knowledge_application"] == "knowledge_application_partial"
    assert payload["component_statuses"]["knowledge_domain_matrix"] == (
        "knowledge_domain_matrix_partial"
    )
    assert payload["component_statuses"]["knowledge_domain_action_plan"] == (
        "knowledge_domain_action_plan_blocked"
    )
    assert payload["component_statuses"]["knowledge_domain_release_gate"] == (
        "knowledge_domain_release_gate_blocked"
    )
    assert payload["component_statuses"]["knowledge_source_action_plan"] == (
        "knowledge_source_action_plan_blocked"
    )
    assert payload["component_statuses"]["knowledge_source_coverage"] == (
        "knowledge_source_coverage_partial"
    )
    assert payload["component_statuses"]["knowledge_domain_release_surface_alignment"] == (
        "diverged"
    )
    assert payload["component_statuses"]["knowledge_reference_inventory"] == (
        "knowledge_reference_inventory_partial"
    )
    assert payload["operator_adoption_knowledge_drift"]["status"] == "regressed"
    assert payload["operator_adoption_knowledge_outcome_drift"]["status"] == "unknown"
    assert payload["knowledge_domain_release_gate_status"] == (
        "knowledge_domain_release_gate_blocked"
    )
    assert payload["knowledge_domain_release_gate_gate_open"] is False
    assert payload["knowledge_domain_release_gate_releasable_domains"] == [
        "standards"
    ]
    assert payload["knowledge_domain_release_gate_blocked_domains"] == ["tolerance"]
    assert payload["knowledge_domain_release_surface_alignment_status"] == "diverged"
    assert payload["knowledge_domain_release_surface_alignment"]["mismatches"] == [
        "gdt:blocked->partial"
    ]
    assert payload["knowledge_reference_inventory_status"] == (
        "knowledge_reference_inventory_partial"
    )
    assert payload["knowledge_reference_inventory_priority_domains"] == [
        "tolerance"
    ]
    assert payload["knowledge_reference_inventory_total_reference_items"] == 24
    assert payload["knowledge_focus_areas"][0]["component"] == "tolerance"
    assert payload["knowledge_drift_status"] == "regressed"
    assert payload["knowledge_drift_domain_regressions"] == ["gdt"]
    assert payload["realdata_status"] == "realdata_foundation_partial"
    assert payload["realdata_signals"]["components"]["step_dir"]["status"] == "partial"
    assert payload["realdata_recommendations"] == [
        "Expand STEP/B-Rep directory validation."
    ]
    assert payload["knowledge_drift"]["counts"]["regressions"] == 1
    assert "Backfill tolerance knowledge coverage." in payload["review_signals"]
    assert "Expand STEP/B-Rep directory validation." in payload["review_signals"]
    assert payload["knowledge_priority_domains"] == ["tolerance"]
    assert payload["knowledge_domains"]["tolerance"]["status"] == "partial"
    assert payload["knowledge_domain_focus_areas"][0]["domain"] == "tolerance"
    assert payload["knowledge_application_status"] == "knowledge_application_partial"
    assert payload["knowledge_application_domains"]["tolerance"]["status"] == "partial"
    assert payload["knowledge_domain_matrix_status"] == "knowledge_domain_matrix_partial"
    assert payload["knowledge_domain_matrix_domains"]["tolerance"]["status"] == "blocked"
    assert payload["knowledge_domain_action_plan_status"] == (
        "knowledge_domain_action_plan_blocked"
    )
    assert payload["knowledge_domain_action_plan_actions"][0]["id"] == (
        "tolerance:foundation"
    )
    assert payload["knowledge_source_action_plan_status"] == (
        "knowledge_source_action_plan_blocked"
    )
    assert payload["knowledge_source_action_plan_recommended_first_actions"][0]["id"] == (
        "tolerance:coverage"
    )
    assert payload["knowledge_source_coverage_status"] == (
        "knowledge_source_coverage_partial"
    )
    assert payload["knowledge_source_coverage_domains"]["tolerance"]["status"] == (
        "partial"
    )
    assert payload["knowledge_source_coverage_expansion_candidates"][0]["name"] == (
        "machining"
    )
    assert "Operator fallback only." not in payload["review_signals"]
    assert "Backfill tolerance source actions first." in payload["review_signals"]
    assert "Backfill tolerance source coverage." in payload["review_signals"]
    assert "Backfill tolerance reference tables." in payload["review_signals"]
    assert payload["artifacts"]["benchmark_engineering_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_realdata_signals"]["present"] is False
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True
    assert (
        payload["artifacts"]["benchmark_knowledge_domain_release_surface_alignment"][
            "present"
        ]
        is True
    )
    assert payload["artifacts"]["benchmark_knowledge_domain_release_gate"]["present"] is True
    assert payload["artifacts"]["benchmark_knowledge_reference_inventory"]["present"] is True
    rendered = render_markdown(payload)
    assert "## Knowledge Domain Release Gate" in rendered
    assert "## Knowledge Domain Release Surface Alignment" in rendered
    assert "## Knowledge Reference Inventory" in rendered


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
                "priority_domains": [],
                "domains": {
                    "gdt": {
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
            "knowledge_outcome_drift_status": "stable",
            "recommended_actions": ["Keep operator workflow stable."],
        },
        benchmark_knowledge_application={
            "knowledge_application": {
                "status": "knowledge_application_ready",
                "priority_domains": [],
                "domains": {
                    "gdt": {
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
        benchmark_knowledge_domain_matrix={
            "knowledge_domain_matrix": {
                "status": "knowledge_domain_matrix_ready",
                "priority_domains": [],
                "domains": {
                    "gdt": {
                        "status": "ready",
                        "readiness_status": "ready",
                        "application_status": "ready",
                        "realdata_status": "ready",
                    }
                },
            },
            "recommendations": [],
        },
        benchmark_competitive_surpass_index={
            "competitive_surpass_index": {
                "status": "competitive_surpass_ready",
                "score": 92,
                "primary_gaps": [],
            },
            "recommendations": [],
        },
        benchmark_competitive_surpass_trend={
            "competitive_surpass_trend": {
                "status": "improved",
                "score_delta": 12,
                "pillar_improvements": ["realdata"],
                "pillar_regressions": [],
                "resolved_primary_gaps": ["realdata"],
                "new_primary_gaps": [],
            },
            "summary": "status=improved; score_delta=12",
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
        artifact_paths={"benchmark_scorecard": "scorecard.json"},
    )

    assert payload["release_status"] == "ready"
    assert payload["automation_ready"] is True
    assert payload["review_signals"] == []
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_ready"
    assert payload["component_statuses"]["knowledge_drift"] == "improved"
    assert payload["component_statuses"]["engineering_signals"] == "engineering_semantics_ready"
    assert payload["component_statuses"]["realdata_signals"] == "realdata_foundation_ready"
    assert payload["component_statuses"]["operator_adoption"] == "operator_ready"
    assert payload["component_statuses"]["knowledge_application"] == "knowledge_application_ready"
    assert payload["component_statuses"]["knowledge_domain_matrix"] == (
        "knowledge_domain_matrix_ready"
    )
    assert payload["component_statuses"]["competitive_surpass_index"] == (
        "competitive_surpass_ready"
    )
    assert payload["operator_adoption_knowledge_drift"]["status"] == "stable"
    assert payload["operator_adoption_knowledge_outcome_drift"]["status"] == "stable"
    assert payload["knowledge_focus_areas"] == []
    assert payload["knowledge_drift_status"] == "improved"
    assert payload["knowledge_drift_domain_improvements"] == ["standards"]
    assert payload["realdata_status"] == "realdata_foundation_ready"
    assert payload["realdata_recommendations"] == []
    assert payload["knowledge_drift"]["counts"]["improvements"] == 1
    assert payload["knowledge_priority_domains"] == []
    assert payload["knowledge_domains"]["gdt"]["status"] == "ready"
    assert payload["knowledge_domain_focus_areas"] == []
    assert payload["knowledge_application_status"] == "knowledge_application_ready"
    assert payload["knowledge_domain_matrix_status"] == "knowledge_domain_matrix_ready"
    assert payload["competitive_surpass_index_status"] == "competitive_surpass_ready"
    assert payload["competitive_surpass_index"]["score"] == 92
    assert payload["competitive_surpass_primary_gaps"] == []
    assert payload["competitive_surpass_recommendations"] == []
    assert payload["component_statuses"]["competitive_surpass_trend"] == "improved"
    assert payload["competitive_surpass_trend_status"] == "improved"
    assert payload["competitive_surpass_trend_score_delta"] == 12
    assert payload["competitive_surpass_trend_pillar_improvements"] == ["realdata"]
    assert payload["competitive_surpass_trend_resolved_primary_gaps"] == ["realdata"]
    assert payload["competitive_surpass_trend_recommendations"] == [
        "Promote the improved competitive surpass posture after CI surfaces refresh."
    ]
    assert payload["component_statuses"]["competitive_surpass_action_plan"] == (
        "competitive_surpass_action_plan_ready"
    )
    assert payload["competitive_surpass_action_plan_status"] == (
        "competitive_surpass_action_plan_ready"
    )
    assert payload["competitive_surpass_action_plan_total_action_count"] == 0
    assert payload["competitive_surpass_action_plan_priority_pillars"] == []
    assert payload["competitive_surpass_action_plan_recommendations"] == []
    assert payload["review_signals"] == []
    assert "## Competitive Surpass Action Plan" in render_markdown(payload)


def test_build_release_decision_uses_operator_adoption_blocker_as_fallback() -> None:
    payload = build_release_decision(
        title="Release Decision",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_companion_summary={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"},
            "recommendations": [],
        },
        benchmark_realdata_signals={},
        benchmark_operator_adoption={
            "adoption_readiness": "blocked",
            "blocking_signals": ["operator_runbook:missing_dry_run"],
            "knowledge_drift_status": "stable",
            "knowledge_outcome_drift_status": "stable",
            "recommended_actions": ["Run operator dry run."],
        },
        benchmark_knowledge_application={},
        artifact_paths={"benchmark_operator_adoption": "operator.json"},
    )

    assert payload["release_status"] == "blocked"
    assert payload["automation_ready"] is False
    assert payload["blocking_signals"] == ["operator_runbook:missing_dry_run"]
    assert payload["component_statuses"]["operator_adoption"] == "blocked"
    assert payload["component_statuses"]["knowledge_application"] == "unknown"


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
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"},
            "recommendations": [],
        },
        benchmark_realdata_signals={},
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "blocking_signals": [],
            "knowledge_drift_status": "stable",
            "knowledge_outcome_drift_status": "stable",
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
    drift = tmp_path / "drift.json"
    engineering = tmp_path / "engineering.json"
    realdata = tmp_path / "realdata.json"
    operator = tmp_path / "operator.json"
    competitive_surpass = tmp_path / "competitive_surpass.json"
    competitive_surpass_trend = tmp_path / "competitive_surpass_trend.json"
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
            }
        ),
        encoding="utf-8",
    )
    drift.write_text(
        json.dumps(
            {
                "knowledge_drift": {
                    "status": "improved",
                    "regressions": [],
                    "improvements": ["standards"],
                    "new_focus_areas": [],
                    "resolved_focus_areas": ["standards"],
                },
                "recommendations": [
                    "Promote the improved knowledge baseline after CI and review "
                    "surfaces are refreshed."
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
    realdata.write_text(
        json.dumps(
            {
                "realdata_signals": {
                    "status": "realdata_foundation_ready",
                    "components": {
                        "hybrid_dxf": {"status": "ready", "sample_size": 110},
                        "history_h5": {"status": "ready", "sample_size": 1},
                        "step_dir": {"status": "ready", "sample_size": 3},
                    },
                },
                "recommendations": [],
            }
        ),
        encoding="utf-8",
    )
    operator.write_text(
        json.dumps(
            {
                "adoption_readiness": "operator_ready",
                "knowledge_drift_status": "stable",
                "knowledge_drift_summary": "No knowledge regressions detected.",
                "knowledge_outcome_drift_status": "regressed",
                "knowledge_outcome_drift_summary": "Tolerance outcome coverage regressed.",
                "knowledge_outcome_drift": {
                    "recommendations": ["Backfill tolerance outcome coverage."]
                },
                "recommended_actions": ["Keep operator workflow stable."],
            }
        ),
        encoding="utf-8",
    )
    competitive_surpass.write_text(
        json.dumps(
            {
                "competitive_surpass_index": {
                    "status": "competitive_surpass_partial",
                    "score": 78,
                    "primary_gaps": ["step_dir_depth", "knowledge_realdata"],
                },
                "recommendations": [
                    "Expand STEP/B-Rep depth before claiming benchmark surpass readiness."
                ],
            }
        ),
        encoding="utf-8",
    )
    competitive_surpass_trend.write_text(
        json.dumps(
            {
                "competitive_surpass_trend": {
                    "status": "regressed",
                    "score_delta": -7,
                    "pillar_improvements": [],
                    "pillar_regressions": ["realdata"],
                    "resolved_primary_gaps": [],
                    "new_primary_gaps": ["step_dir_depth"],
                },
                "summary": "status=regressed; score_delta=-7",
                "recommendations": [
                    "Resolve competitive surpass regressions before claiming benchmark progress."
                ],
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
            "--benchmark-knowledge-drift",
            str(drift),
            "--benchmark-engineering-signals",
            str(engineering),
            "--benchmark-realdata-signals",
            str(realdata),
            "--benchmark-operator-adoption",
            str(operator),
            "--benchmark-competitive-surpass-index",
            str(competitive_surpass),
            "--benchmark-competitive-surpass-trend",
            str(competitive_surpass_trend),
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
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_ready"
    assert payload["component_statuses"]["knowledge_drift"] == "improved"
    assert payload["component_statuses"]["engineering_signals"] == "engineering_semantics_ready"
    assert payload["component_statuses"]["realdata_signals"] == "realdata_foundation_ready"
    assert payload["component_statuses"]["operator_adoption"] == "operator_ready"
    assert payload["component_statuses"]["competitive_surpass_index"] == (
        "competitive_surpass_partial"
    )
    assert payload["operator_adoption_knowledge_drift"]["status"] == "stable"
    assert payload["operator_adoption_knowledge_outcome_drift"]["status"] == "regressed"
    assert payload["artifacts"]["benchmark_knowledge_readiness"]["present"] is True
    assert payload["artifacts"]["benchmark_knowledge_drift"]["present"] is True
    assert payload["artifacts"]["benchmark_realdata_signals"]["present"] is True
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True
    assert payload["knowledge_domains"]["standards"]["status"] == "ready"
    assert payload["knowledge_domain_focus_areas"] == []
    assert payload["competitive_surpass_index_status"] == "competitive_surpass_partial"
    assert payload["competitive_surpass_index"]["score"] == 78
    assert payload["competitive_surpass_primary_gaps"] == [
        "step_dir_depth",
        "knowledge_realdata",
    ]
    assert payload["competitive_surpass_recommendations"] == [
        "Expand STEP/B-Rep depth before claiming benchmark surpass readiness."
    ]
    assert payload["artifacts"]["benchmark_competitive_surpass_index"]["present"] is True
    assert payload["component_statuses"]["competitive_surpass_trend"] == "regressed"
    assert payload["competitive_surpass_trend_status"] == "regressed"
    assert payload["competitive_surpass_trend_score_delta"] == -7
    assert payload["competitive_surpass_trend_pillar_regressions"] == ["realdata"]
    assert payload["competitive_surpass_trend_new_primary_gaps"] == ["step_dir_depth"]
    assert payload["competitive_surpass_trend_recommendations"] == [
        "Resolve competitive surpass regressions before claiming benchmark progress."
    ]
    assert payload["artifacts"]["benchmark_competitive_surpass_trend"]["present"] is True
    assert output_md.exists()

    rendered = render_markdown(payload)
    assert "# Benchmark Release Decision" in rendered
    assert "`release_status`: `review_required`" in rendered
    assert "`knowledge_readiness`: `knowledge_foundation_ready`" in rendered
    assert "## Knowledge Drift" in rendered
    assert "`status`: `improved`" in rendered
    assert "`operator_adoption`: `operator_ready`" in rendered
    assert "`operator_adoption_knowledge_drift`: `stable`" in rendered
    assert "`operator_adoption_knowledge_outcome_drift`: `regressed`" in rendered
    assert "## Competitive Surpass Index" in rendered
    assert "`competitive_surpass_index_status`: `competitive_surpass_partial`" in rendered
    assert "step_dir_depth, knowledge_realdata" in rendered
    assert "Expand STEP/B-Rep depth before claiming benchmark surpass readiness." in rendered
    assert "## Competitive Surpass Trend" in rendered
    assert "## Competitive Surpass Trend" in rendered
    assert "- `status`: `regressed`" in rendered
    assert "Resolve competitive surpass regressions before claiming benchmark progress." in rendered
    assert "## Knowledge Domains" in rendered
    assert "## Knowledge Domain Matrix" in rendered
    assert "## Real-Data Signals" in rendered
    assert "realdata.json" in rendered


def test_build_release_decision_exposes_knowledge_outcome_drift_passthrough() -> None:
    payload = build_release_decision(
        title="Release Decision",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_companion_summary={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_knowledge_outcome_drift={
            "knowledge_outcome_drift": {
                "status": "regressed",
                "current_status": "knowledge_outcome_correlation_partial",
                "previous_status": "knowledge_outcome_correlation_ready",
                "domain_regressions": ["gdt"],
                "domain_improvements": [],
                "new_priority_domains": ["gdt"],
                "resolved_priority_domains": [],
            },
            "recommendations": [
                "Resolve knowledge outcome regressions before claiming benchmark outcome stability."
            ],
        },
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_operator_adoption={"adoption_readiness": "operator_ready"},
        artifact_paths={
            "benchmark_knowledge_outcome_drift": "knowledge_outcome_drift.json"
        },
    )

    assert payload["component_statuses"]["knowledge_outcome_drift"] == "regressed"
    assert payload["artifacts"]["benchmark_knowledge_outcome_drift"]["present"] is True
    assert payload["knowledge_outcome_drift_status"] == "regressed"
    assert payload["knowledge_outcome_drift_domain_regressions"] == ["gdt"]
    assert payload["knowledge_outcome_drift_recommendations"] == [
        "Resolve knowledge outcome regressions before claiming benchmark outcome stability."
    ]
    assert (
        "Resolve knowledge outcome regressions before claiming benchmark outcome stability."
        in payload["review_signals"]
    )


def test_build_release_decision_exposes_operator_adoption_knowledge_outcome_drift() -> None:
    payload = build_release_decision(
        title="Release Decision",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_companion_summary={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "knowledge_outcome_drift_status": "regressed",
            "knowledge_outcome_drift_summary": "Tolerance outcome coverage regressed.",
            "knowledge_outcome_drift": {
                "recommendations": ["Backfill tolerance outcome coverage."]
            },
            "recommended_actions": ["Walk operators through the guided manual path."],
        },
        artifact_paths={"benchmark_operator_adoption": "operator.json"},
    )

    assert payload["operator_adoption_knowledge_outcome_drift"]["status"] == "regressed"
    assert (
        payload["operator_adoption_knowledge_outcome_drift"]["summary"]
        == "Tolerance outcome coverage regressed."
    )
    assert "Backfill tolerance outcome coverage." in payload["review_signals"]


def test_build_release_decision_exposes_knowledge_source_drift_passthrough() -> None:
    payload = build_release_decision(
        title="Release Decision",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_companion_summary={},
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
            },
            "recommendations": [
                "Restore regressed knowledge source groups before claiming "
                "benchmark source stability."
            ],
        },
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_operator_adoption={"adoption_readiness": "operator_ready"},
        artifact_paths={
            "benchmark_knowledge_source_drift": "knowledge_source_drift.json"
        },
    )

    assert payload["component_statuses"]["knowledge_source_drift"] == "regressed"
    assert payload["artifacts"]["benchmark_knowledge_source_drift"]["present"] is True
    assert payload["knowledge_source_drift_status"] == "regressed"
    assert payload["knowledge_source_drift_source_group_regressions"] == ["gdt"]
    assert payload["knowledge_source_drift_recommendations"] == [
        "Restore regressed knowledge source groups before claiming benchmark source stability."
    ]
    assert (
        "Restore regressed knowledge source groups before claiming benchmark source stability."
        in payload["review_signals"]
    )


def test_build_release_decision_exposes_scorecard_and_operational_operator_adoption() -> None:
    payload = build_release_decision(
        title="Release Decision",
        benchmark_scorecard={
            "components": {
                "hybrid": {"status": "healthy"},
                "operator_adoption": {
                    "status": "guided_manual",
                    "operator_mode": "shadow_review",
                    "knowledge_outcome_drift_status": "regressed",
                    "knowledge_outcome_drift_summary": "Operator outcome drift needs review.",
                },
            }
        },
        benchmark_operational_summary={
            "component_statuses": {"operator_adoption": "attention_required"},
            "operator_adoption_knowledge_outcome_drift_status": "partial",
            "operator_adoption_knowledge_outcome_drift_summary": "Operational rollout is partial.",
        },
        benchmark_artifact_bundle={},
        benchmark_companion_summary={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_operator_adoption={"adoption_readiness": "operator_ready"},
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


def test_build_release_decision_exposes_operator_adoption_release_alignment() -> None:
    payload = build_release_decision(
        title="Release Decision",
        benchmark_scorecard={},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_companion_summary={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "release_surface_alignment_status": "mismatched",
            "release_surface_alignment_summary": (
                "Release runbook is missing operator adoption guidance."
            ),
            "release_surface_alignment": {
                "mismatches": ["release_runbook:missing"],
                "release_decision": {"scorecard_status": "guided_manual"},
                "release_runbook": {},
            },
        },
        artifact_paths={},
    )

    assert payload["operator_adoption_release_surface_alignment"]["status"] == (
        "mismatched"
    )
    assert payload["operator_adoption_release_surface_alignment"]["mismatches"] == [
        "release_runbook:missing"
    ]

    rendered = render_markdown(payload)
    assert "Release runbook is missing operator adoption guidance." in rendered
    assert "release_runbook:missing" in rendered
