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
                "operator_adoption": {
                    "status": "guided_manual",
                    "operator_mode": "assisted_review",
                    "knowledge_outcome_drift_status": "regressed",
                    "knowledge_outcome_drift_summary": "Tolerance coverage regressed.",
                },
            },
            "recommendations": ["improve scorecard coverage"],
        },
        benchmark_operational_summary={
            "overall_status": "attention_required",
            "component_statuses": {
                "assistant_explainability": "partial_coverage",
                "review_queue": "managed_backlog",
                "operator_adoption": "guided_manual",
            },
            "operator_adoption_knowledge_outcome_drift_status": "regressed",
            "operator_adoption_knowledge_outcome_drift_summary": (
                "Tolerance coverage regressed."
            ),
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
                "priority_domains": ["tolerance", "standards"],
                "domains": {
                    "tolerance": {
                        "status": "partial",
                        "focus_components": ["tolerance"],
                        "missing_metrics": ["common_fit_count"],
                    },
                    "standards": {
                        "status": "missing",
                        "focus_components": ["standards", "design_standards"],
                        "missing_metrics": ["thread_count"],
                    },
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
        benchmark_knowledge_drift={},
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
                    "step_dir": {"status": "ready", "sample_size": 3},
                },
            },
            "recommendations": ["Expand STEP/B-Rep directory validation."],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "guided_manual",
            "knowledge_drift_status": "regressed",
            "knowledge_drift_summary": "Tolerance coverage regressed.",
            "knowledge_outcome_drift_status": "regressed",
            "knowledge_outcome_drift_summary": "Operator review outcomes regressed.",
            "release_surface_alignment_status": "mismatched",
            "release_surface_alignment_summary": (
                "Release runbook is missing operator adoption guidance."
            ),
            "release_surface_alignment": {
                "mismatches": ["release_runbook:missing"],
            },
            "knowledge_outcome_drift": {
                "recommendations": ["Investigate operator review regressions."],
            },
            "knowledge_drift": {
                "recommendations": ["Backfill tolerance knowledge coverage."],
            },
            "recommended_actions": ["Review operator blockers."],
        },
        artifact_paths={
            "benchmark_scorecard": "scorecard.json",
            "benchmark_operational_summary": "operational.json",
            "benchmark_artifact_bundle": "bundle.json",
            "benchmark_engineering_signals": "engineering.json",
            "benchmark_operator_adoption": "operator.json",
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
                        "action": "Promote GD&T evidence.",
                    }
                ],
            },
            "recommendations": ["Promote gdt application evidence into benchmark surfaces."],
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
                        "realdata_status": "blocked",
                    }
                },
            },
            "recommendations": ["Backfill standards foundation and real-data coverage."],
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
                "expansion_candidates": [{"name": "machining", "status": "ready"}],
            },
            "recommendations": ["Promote machining and standards source coverage."],
        },
    )

    assert payload["overall_status"] == "attention_required"
    assert payload["review_surface"] == "attention_required"
    assert payload["primary_gap"] == "review_queue:managed_backlog"
    assert payload["component_statuses"]["assistant_explainability"] == "partial_coverage"
    assert payload["component_statuses"]["knowledge_readiness"] == "knowledge_foundation_partial"
    assert payload["component_statuses"]["engineering_signals"] == "partial_engineering_semantics"
    assert payload["component_statuses"]["realdata_signals"] == "realdata_foundation_partial"
    assert payload["component_statuses"]["operator_adoption"] == "guided_manual"
    assert payload["scorecard_operator_adoption"]["operator_mode"] == "assisted_review"
    assert payload["operational_operator_adoption"]["status"] == "guided_manual"
    assert payload["component_statuses"]["knowledge_application"] == "knowledge_application_partial"
    assert payload["operator_adoption_knowledge_drift"]["status"] == "regressed"
    assert payload["operator_adoption_knowledge_outcome_drift"]["status"] == "regressed"
    assert payload["operator_adoption_release_surface_alignment"]["status"] == (
        "mismatched"
    )
    assert (
        payload["operator_adoption_knowledge_drift"]["summary"]
        == "Tolerance coverage regressed."
    )
    assert payload["knowledge_focus_areas"][0]["component"] == "tolerance"
    assert payload["knowledge_priority_domains"] == ["tolerance", "standards"]
    assert payload["knowledge_domains"]["tolerance"]["status"] == "partial"
    assert payload["knowledge_domain_focus_areas"][0]["domain"] == "tolerance"
    assert payload["knowledge_application_status"] == "knowledge_application_partial"
    assert payload["knowledge_application_domains"]["gdt"]["status"] == "partial"
    assert payload["knowledge_application_priority_domains"] == ["gdt"]
    assert payload["knowledge_domain_matrix_status"] == "knowledge_domain_matrix_partial"
    assert payload["knowledge_domain_matrix_domains"]["standards"]["status"] == "blocked"
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
    assert payload["knowledge_drift_domain_regressions"] == []
    assert payload["recommended_actions"] == ["reduce review queue backlog"]
    assert payload["artifacts"]["benchmark_artifact_bundle"]["present"] is True
    assert payload["artifacts"]["benchmark_engineering_signals"]["present"] is True
    assert payload["realdata_status"] == "realdata_foundation_partial"
    assert payload["artifacts"]["benchmark_operator_adoption"]["present"] is True
    markdown = render_markdown(payload)
    assert "## Operator Adoption Release Surface Alignment" in markdown
    assert "## Scorecard Operator Adoption" in markdown
    assert "## Operational Operator Adoption" in markdown
    assert "## Knowledge Source Coverage" in markdown


def test_render_markdown_includes_sections() -> None:
    payload = {
        "title": "Benchmark Companion",
        "overall_status": "healthy",
        "review_surface": "ready",
        "primary_gap": "none",
        "knowledge_drift_summary": "status=stable; current=knowledge_foundation_ready",
        "knowledge_application_status": "knowledge_application_partial",
        "knowledge_domain_matrix_status": "knowledge_domain_matrix_partial",
        "realdata_status": "realdata_foundation_ready",
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
        "knowledge_drift_domain_regressions": ["gdt"],
        "knowledge_drift_new_priority_domains": ["gdt"],
        "artifacts": {
            "benchmark_engineering_signals": {
                "present": True,
                "path": "engineering.json",
            },
            "benchmark_realdata_signals": {
                "present": True,
                "path": "realdata.json",
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
        "realdata_signals": {
            "components": {
                "hybrid_dxf": {"status": "ready", "sample_size": 110},
                "history_h5": {"status": "ready", "sample_size": 1},
                "step_dir": {"status": "ready", "sample_size": 3},
            }
        },
        "realdata_recommendations": ["Expand STEP/B-Rep directory validation."],
        "knowledge_focus_areas": [
            {
                "component": "gdt",
                "status": "missing",
                "priority": "high",
                "action": "Expand GD&T coverage.",
            }
        ],
        "operator_adoption_knowledge_drift": {
            "status": "regressed",
            "summary": "Tolerance coverage regressed.",
            "recommendations": ["Backfill tolerance knowledge coverage."],
        },
        "operator_adoption_knowledge_outcome_drift": {
            "status": "regressed",
            "summary": "Operator review outcomes regressed.",
            "recommendations": ["Investigate operator review regressions."],
        },
        "operator_adoption_release_surface_alignment": {
            "status": "mismatched",
            "summary": "Release runbook is missing operator adoption guidance.",
            "mismatches": ["release_runbook:missing"],
        },
        "knowledge_application_domains": {
            "gdt": {
                "status": "partial",
                "readiness_status": "ready",
                "evidence_status": "missing",
                "signal_count": 0,
            }
        },
        "knowledge_application_recommendations": [
            "Promote gdt application evidence into benchmark surfaces."
        ],
        "knowledge_domain_matrix_domains": {
            "gdt": {
                "status": "partial",
                "readiness_status": "missing",
                "application_status": "partial",
                "realdata_status": "partial",
            }
        },
        "knowledge_domain_matrix_recommendations": [
            "Backfill gdt foundation and expand real-data coverage."
        ],
        "knowledge_domains": {
            "gdt": {
                "status": "missing",
                "focus_components": ["gdt"],
                "missing_metrics": ["symbol_count"],
            }
        },
        "knowledge_domain_focus_areas": [
            {
                "domain": "gdt",
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
    assert "realdata.json" in rendered
    assert "operator.json" in rendered
    assert "Tolerance coverage regressed." in rendered
    assert "Operator review outcomes regressed." in rendered
    assert "## Operator Adoption Release Surface Alignment" in rendered
    assert "Expand GD&T coverage." in rendered
    assert "status=stable; current=knowledge_foundation_ready" in rendered
    assert "domain_regressions" in rendered
    assert "## Real-Data Signals" in rendered
    assert "## Knowledge Domains" in rendered
    assert "## Knowledge Application" in rendered
    assert "## Knowledge Domain Focus Areas" in rendered
    assert "## Knowledge Domain Matrix" in rendered
    assert "## Operator Adoption Knowledge Outcome Drift" in rendered


def test_cli_writes_outputs(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    operational = tmp_path / "operational.json"
    bundle = tmp_path / "bundle.json"
    knowledge = tmp_path / "knowledge.json"
    knowledge_drift = tmp_path / "knowledge_drift.json"
    engineering = tmp_path / "engineering.json"
    realdata = tmp_path / "realdata.json"
    operator = tmp_path / "operator.json"
    knowledge_application = tmp_path / "knowledge_application.json"
    knowledge_realdata_correlation = tmp_path / "knowledge_realdata_correlation.json"
    knowledge_domain_matrix = tmp_path / "knowledge_domain_matrix.json"
    knowledge_outcome_correlation = tmp_path / "knowledge_outcome_correlation.json"
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
    realdata.write_text(
        json.dumps(
            {
                "realdata_signals": {
                    "status": "realdata_foundation_partial",
                    "components": {
                        "hybrid_dxf": {"status": "ready", "sample_size": 10},
                        "history_h5": {"status": "ready", "sample_size": 1},
                        "step_dir": {"status": "ready", "sample_size": 3},
                    },
                },
                "recommendations": ["Expand STEP/B-Rep directory validation."],
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
                "knowledge_outcome_drift_status": "stable",
                "knowledge_outcome_drift_summary": "Operator outcome drift stable.",
                "recommended_actions": ["Keep operator automation healthy."],
            }
        ),
        encoding="utf-8",
    )
    knowledge_application.write_text(
        json.dumps(
            {
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
            }
        ),
        encoding="utf-8",
    )
    knowledge_realdata_correlation.write_text(
        json.dumps(
            {
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
            }
        ),
        encoding="utf-8",
    )
    knowledge_domain_matrix.write_text(
        json.dumps(
            {
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
            }
        ),
        encoding="utf-8",
    )
    knowledge_outcome_correlation.write_text(
        json.dumps(
            {
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
            "--benchmark-realdata-signals",
            str(realdata),
            "--benchmark-operator-adoption",
            str(operator),
            "--benchmark-knowledge-application",
            str(knowledge_application),
            "--benchmark-knowledge-realdata-correlation",
            str(knowledge_realdata_correlation),
            "--benchmark-knowledge-domain-matrix",
            str(knowledge_domain_matrix),
            "--benchmark-knowledge-outcome-correlation",
            str(knowledge_outcome_correlation),
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
    assert payload["component_statuses"]["realdata_signals"] == "realdata_foundation_partial"
    assert payload["component_statuses"]["operator_adoption"] == "operator_ready"
    assert payload["component_statuses"]["knowledge_application"] == "knowledge_application_ready"
    assert payload["component_statuses"]["knowledge_domain_matrix"] == (
        "knowledge_domain_matrix_ready"
    )
    assert payload["component_statuses"]["knowledge_outcome_correlation"] == (
        "knowledge_outcome_correlation_ready"
    )
    assert payload["operator_adoption_knowledge_drift"]["status"] == "stable"
    assert payload["operator_adoption_knowledge_outcome_drift"]["status"] == "stable"
    assert payload["operator_adoption_release_surface_alignment"]["status"] == "unknown"
    assert payload["knowledge_focus_areas"] == []
    assert payload["knowledge_drift_summary"].startswith("status=stable")
    assert payload["knowledge_priority_domains"] == []
    assert payload["knowledge_domains"]["standards"]["status"] == "ready"
    assert payload["knowledge_domain_focus_areas"] == []
    assert payload["knowledge_application_status"] == "knowledge_application_ready"
    assert payload["knowledge_domain_matrix_status"] == "knowledge_domain_matrix_ready"
    assert payload["knowledge_outcome_correlation_status"] == (
        "knowledge_outcome_correlation_ready"
    )
    assert payload["knowledge_drift_domain_improvements"] == []
    assert payload["artifacts"]["benchmark_realdata_signals"]["present"] is True
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
                "domain_regressions": ["standards"],
                "domain_improvements": [],
                "new_focus_areas": ["standards"],
                "resolved_priority_domains": [],
                "new_priority_domains": ["standards"],
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
        benchmark_realdata_signals={
            "realdata_signals": {
                "status": "realdata_foundation_ready",
                "components": {
                    "hybrid_dxf": {"status": "ready", "sample_size": 110},
                },
            },
            "recommendations": [],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "operator_ready",
            "recommended_actions": [],
        },
        artifact_paths={
            "benchmark_knowledge_drift": "knowledge_drift.json",
            "benchmark_realdata_signals": "realdata.json",
        },
    )

    assert payload["component_statuses"]["knowledge_drift"] == "regressed"
    assert payload["primary_gap"] == "knowledge_drift:regressed"
    assert payload["recommended_actions"] == [
        "Resolve knowledge regressions before claiming the benchmark "
        "surpass baseline remains stable."
    ]
    assert payload["artifacts"]["benchmark_knowledge_drift"]["present"] is True
    assert payload["artifacts"]["benchmark_realdata_signals"]["present"] is True
    assert payload["knowledge_drift"]["status"] == "regressed"
    assert "regressions=standards" in payload["knowledge_drift_summary"]
    assert payload["knowledge_drift_domain_regressions"] == ["standards"]
    assert payload["knowledge_drift_new_priority_domains"] == ["standards"]


def test_build_companion_summary_exposes_knowledge_outcome_drift_passthrough() -> None:
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
            "knowledge_readiness": {"status": "knowledge_foundation_ready"}
        },
        benchmark_knowledge_drift={},
        benchmark_knowledge_outcome_drift={
            "knowledge_outcome_drift": {
                "status": "mixed",
                "current_status": "knowledge_outcome_correlation_partial",
                "previous_status": "knowledge_outcome_correlation_ready",
                "regressions": ["tolerance"],
                "improvements": ["standards"],
                "new_focus_areas": ["tolerance"],
                "domain_regressions": ["tolerance"],
                "domain_improvements": ["standards"],
                "new_priority_domains": ["tolerance"],
                "resolved_priority_domains": ["standards"],
            },
            "recommendations": [
                "Keep the previous knowledge outcome baseline until regressions are cleared."
            ],
        },
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"},
            "recommendations": [],
        },
        benchmark_realdata_signals={
            "realdata_signals": {"status": "realdata_foundation_ready"},
            "recommendations": [],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "operator_ready",
            "recommended_actions": [],
        },
        artifact_paths={
            "benchmark_knowledge_outcome_drift": "knowledge_outcome_drift.json"
        },
    )

    assert payload["component_statuses"]["knowledge_outcome_drift"] == "mixed"
    assert payload["primary_gap"] == "knowledge_outcome_drift:mixed"
    assert payload["artifacts"]["benchmark_knowledge_outcome_drift"]["present"] is True
    assert payload["knowledge_outcome_drift"]["status"] == "mixed"
    assert "regressions=tolerance" in payload["knowledge_outcome_drift_summary"]
    assert payload["knowledge_outcome_drift_domain_regressions"] == ["tolerance"]
    assert payload["knowledge_outcome_drift_recommendations"] == [
        "Keep the previous knowledge outcome baseline until regressions are cleared."
    ]


def test_build_companion_summary_exposes_competitive_surpass_index_passthrough() -> None:
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
            "knowledge_readiness": {"status": "knowledge_foundation_ready"}
        },
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"},
            "recommendations": [],
        },
        benchmark_realdata_signals={
            "realdata_signals": {"status": "realdata_foundation_ready"},
            "recommendations": [],
        },
        benchmark_realdata_scorecard={
            "realdata_scorecard": {"status": "realdata_scorecard_ready"},
            "recommendations": [],
        },
        benchmark_operator_adoption={
            "adoption_readiness": "operator_ready",
            "release_surface_alignment_status": "aligned",
            "release_surface_alignment_summary": "Release surfaces are aligned.",
            "release_surface_alignment": {"mismatches": []},
            "recommended_actions": [],
        },
        benchmark_competitive_surpass_index={
            "competitive_surpass_index": {
                "status": "competitive_surpass_ready",
                "score": 100,
                "primary_gaps": [],
            },
            "recommendations": [
                "Competitive-surpass benchmark pillars are aligned across "
                "engineering, knowledge, real-data, operator adoption, and "
                "release surfaces."
            ],
        },
        artifact_paths={
            "benchmark_competitive_surpass_index": "competitive_surpass_index.json"
        },
    )

    assert payload["competitive_surpass_index_status"] == "competitive_surpass_ready"
    assert payload["competitive_surpass_primary_gaps"] == []
    assert payload["competitive_surpass_recommendations"] == [
        "Competitive-surpass benchmark pillars are aligned across engineering, "
        "knowledge, real-data, operator adoption, and release surfaces."
    ]
    assert payload["artifacts"]["benchmark_competitive_surpass_index"]["present"] is True
    markdown = render_markdown(payload)
    assert "## Competitive Surpass Index" in markdown
