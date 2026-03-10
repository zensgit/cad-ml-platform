from __future__ import annotations

import json
from pathlib import Path

from scripts.export_benchmark_competitive_surpass_index import (
    build_competitive_surpass_summary,
    main as competitive_surpass_main,
)
from src.core.benchmark import (
    build_competitive_surpass_index,
    competitive_surpass_index_recommendations,
    render_competitive_surpass_markdown,
)


def _ready_inputs() -> dict:
    return {
        "benchmark_engineering_signals": {
            "engineering_signals": {
                "status": "engineering_semantics_ready",
                "coverage_ratio": 0.92,
                "rows_with_standards_candidates": 18,
                "rows_with_violations": 2,
            },
            "recommendations": [],
        },
        "benchmark_knowledge_readiness": {
            "knowledge_readiness": {
                "status": "knowledge_foundation_ready",
                "focus_areas_detail": [],
                "priority_domains": [],
                "domains": {},
            }
        },
        "benchmark_knowledge_application": {
            "knowledge_application": {
                "status": "knowledge_application_ready",
                "focus_areas_detail": [],
                "priority_domains": [],
                "domains": {},
            },
            "recommendations": [],
        },
        "benchmark_knowledge_realdata_correlation": {
            "knowledge_realdata_correlation": {
                "status": "knowledge_realdata_ready",
                "focus_areas": [],
                "priority_domains": [],
                "domains": {},
            },
            "recommendations": [],
        },
        "benchmark_knowledge_domain_matrix": {
            "knowledge_domain_matrix": {
                "status": "knowledge_domain_matrix_ready",
                "priority_domains": [],
                "domains": {},
            },
            "recommendations": [],
        },
        "benchmark_knowledge_domain_action_plan": {
            "knowledge_domain_action_plan": {
                "status": "knowledge_domain_action_plan_ready",
                "priority_domains": [],
                "actions": [],
            },
            "recommendations": [],
        },
        "benchmark_knowledge_source_coverage": {
            "knowledge_source_coverage": {
                "status": "knowledge_source_coverage_ready",
                "priority_domains": [],
                "domains": {},
                "expansion_candidates": [
                    {"name": "machining", "status": "ready"},
                    {"name": "welding", "status": "ready"},
                ],
            },
            "recommendations": [],
        },
        "benchmark_knowledge_source_action_plan": {
            "knowledge_source_action_plan": {
                "status": "knowledge_source_action_plan_ready",
                "priority_domains": [],
                "actions": [],
            },
            "recommendations": [],
        },
        "benchmark_knowledge_source_drift": {
            "knowledge_source_drift": {
                "status": "stable",
                "source_group_regressions": [],
                "source_group_improvements": ["machining"],
                "resolved_priority_domains": [],
                "new_priority_domains": [],
            },
            "recommendations": [],
        },
        "benchmark_knowledge_outcome_correlation": {
            "knowledge_outcome_correlation": {
                "status": "knowledge_outcome_correlation_ready",
                "priority_domains": [],
                "domains": {},
            },
            "recommendations": [],
        },
        "benchmark_knowledge_outcome_drift": {
            "knowledge_outcome_drift": {
                "status": "stable",
                "current_status": "knowledge_outcome_correlation_ready",
                "previous_status": "knowledge_outcome_correlation_ready",
                "new_priority_domains": [],
            },
            "recommendations": [],
        },
        "benchmark_realdata_signals": {
            "realdata_signals": {
                "status": "realdata_foundation_ready",
                "ready_component_count": 4,
            },
            "recommendations": [],
        },
        "benchmark_realdata_scorecard": {
            "realdata_scorecard": {
                "status": "realdata_scorecard_ready",
                "best_surface": "history_h5",
            },
            "recommendations": [],
        },
        "benchmark_operator_adoption": {
            "adoption_readiness": "operator_ready",
            "operator_mode": "freeze_ready",
            "knowledge_outcome_drift_status": "stable",
            "recommended_actions": [],
            "release_surface_alignment_status": "aligned",
            "release_surface_alignment_summary": "Release surfaces are aligned.",
            "release_surface_alignment": {"mismatches": []},
        },
        "benchmark_knowledge_domain_release_surface_alignment": {
            "knowledge_domain_release_surface_alignment": {
                "status": "aligned",
                "summary": "Knowledge-domain release surfaces are aligned.",
                "mismatches": [],
            }
        },
    }


def test_build_competitive_surpass_index_ready() -> None:
    payload = build_competitive_surpass_index(**_ready_inputs())

    assert payload["status"] == "competitive_surpass_ready"
    assert payload["score"] == 100
    assert payload["ready_pillars"] == [
        "engineering",
        "knowledge",
        "realdata",
        "operator_adoption",
        "release_alignment",
    ]
    assert (
        payload["pillars"]["knowledge"]["details"]["component_statuses"]["source_coverage"]
        == "knowledge_source_coverage_ready"
    )
    assert (
        payload["pillars"]["knowledge"]["details"]["component_statuses"][
            "source_action_plan"
        ]
        == "knowledge_source_action_plan_ready"
    )
    assert (
        payload["pillars"]["knowledge"]["details"]["component_statuses"]["source_drift"]
        == "stable"
    )
    assert payload["primary_gaps"] == []
    assert competitive_surpass_index_recommendations(payload) == [
        "Competitive-surpass benchmark pillars are aligned across engineering, "
        "knowledge, real-data, operator adoption, and release surfaces."
    ]
    rendered = render_competitive_surpass_markdown(
        {"competitive_surpass_index": payload, "recommendations": []},
        "Benchmark Competitive Surpass Index",
    )
    assert "## Pillars" in rendered
    assert "release_alignment" in rendered


def test_export_benchmark_competitive_surpass_index_outputs_files(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _ready_inputs()
    paths = {}
    argv = [
        "export_benchmark_competitive_surpass_index.py",
        "--output-json",
        str(tmp_path / "competitive_surpass.json"),
        "--output-md",
        str(tmp_path / "competitive_surpass.md"),
    ]
    for flag, key in (
        ("--benchmark-engineering-signals", "benchmark_engineering_signals"),
        ("--benchmark-knowledge-readiness", "benchmark_knowledge_readiness"),
        ("--benchmark-knowledge-application", "benchmark_knowledge_application"),
        (
            "--benchmark-knowledge-realdata-correlation",
            "benchmark_knowledge_realdata_correlation",
        ),
        ("--benchmark-knowledge-domain-matrix", "benchmark_knowledge_domain_matrix"),
        (
            "--benchmark-knowledge-domain-action-plan",
            "benchmark_knowledge_domain_action_plan",
        ),
        (
            "--benchmark-knowledge-source-coverage",
            "benchmark_knowledge_source_coverage",
        ),
        (
            "--benchmark-knowledge-source-action-plan",
            "benchmark_knowledge_source_action_plan",
        ),
        ("--benchmark-knowledge-source-drift", "benchmark_knowledge_source_drift"),
        (
            "--benchmark-knowledge-outcome-correlation",
            "benchmark_knowledge_outcome_correlation",
        ),
        ("--benchmark-knowledge-outcome-drift", "benchmark_knowledge_outcome_drift"),
        ("--benchmark-realdata-signals", "benchmark_realdata_signals"),
        ("--benchmark-realdata-scorecard", "benchmark_realdata_scorecard"),
        ("--benchmark-operator-adoption", "benchmark_operator_adoption"),
        (
            "--benchmark-knowledge-domain-release-surface-alignment",
            "benchmark_knowledge_domain_release_surface_alignment",
        ),
    ):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(inputs[key]), encoding="utf-8")
        paths[key] = path
        argv.extend([flag, str(path)])

    monkeypatch.setattr("sys.argv", argv)
    competitive_surpass_main()

    output_json = tmp_path / "competitive_surpass.json"
    output_md = tmp_path / "competitive_surpass.md"
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["competitive_surpass_index"]["status"] == (
        "competitive_surpass_ready"
    )
    assert payload["recommendations"] == [
        "Competitive-surpass benchmark pillars are aligned across engineering, "
        "knowledge, real-data, operator adoption, and release surfaces."
    ]
    rendered = output_md.read_text(encoding="utf-8")
    assert "# Benchmark Competitive Surpass Index" in rendered
    assert "## Recommendations" in rendered


def test_build_competitive_surpass_summary_blocks_on_realdata_and_alignment() -> None:
    inputs = _ready_inputs()
    inputs["benchmark_realdata_signals"] = {
        "realdata_signals": {
            "status": "realdata_foundation_missing",
            "ready_component_count": 0,
        },
        "recommendations": ["Collect production DXF and STEP data."],
    }
    inputs["benchmark_realdata_scorecard"] = {
        "realdata_scorecard": {"status": "realdata_scorecard_missing"},
        "recommendations": ["Expand real-data scorecard coverage."],
    }
    inputs["benchmark_operator_adoption"] = {
        "adoption_readiness": "guided_manual",
        "operator_mode": "assisted_review",
        "knowledge_outcome_drift_status": "regressed",
        "recommended_actions": ["Stabilize operator review outcomes."],
        "release_surface_alignment_status": "unavailable",
        "release_surface_alignment_summary": "Release surfaces are not aligned.",
        "release_surface_alignment": {"mismatches": ["release_runbook:missing"]},
    }
    inputs["benchmark_knowledge_domain_release_surface_alignment"] = {
        "knowledge_domain_release_surface_alignment": {
            "status": "unavailable",
            "summary": "Knowledge-domain release surfaces are unavailable.",
            "mismatches": ["release_decision:missing_knowledge_domain_status"],
        }
    }
    inputs["benchmark_knowledge_source_coverage"] = {
        "knowledge_source_coverage": {
            "status": "knowledge_source_coverage_partial",
            "priority_domains": ["standards"],
            "domains": {"standards": {"status": "partial"}},
            "expansion_candidates": [{"name": "machining", "status": "ready"}],
        },
        "recommendations": ["Expose machining and standards source coverage."],
    }
    inputs["benchmark_knowledge_source_action_plan"] = {
        "knowledge_source_action_plan": {
            "status": "knowledge_source_action_plan_partial",
            "priority_domains": ["standards"],
            "actions": [{"id": "machining:expansion"}],
        },
        "recommendations": ["Promote machining source coverage into benchmark surfaces."],
    }
    inputs["benchmark_knowledge_source_drift"] = {
        "knowledge_source_drift": {
            "status": "regressed",
            "source_group_regressions": ["standards"],
            "source_group_improvements": [],
            "resolved_priority_domains": [],
            "new_priority_domains": ["standards"],
        },
        "recommendations": ["Recover standards source coverage before release gating."],
    }
    payload = build_competitive_surpass_summary(
        title="Benchmark Competitive Surpass Index",
        artifact_paths={},
        **inputs,
    )

    component = payload["competitive_surpass_index"]
    assert component["status"] == "competitive_surpass_blocked"
    assert component["blocked_pillars"] == ["realdata", "release_alignment"]
    assert (
        component["pillars"]["knowledge"]["details"]["component_statuses"]["source_coverage"]
        == "knowledge_source_coverage_partial"
    )
    assert (
        component["pillars"]["knowledge"]["details"]["component_statuses"][
            "source_action_plan"
        ]
        == "knowledge_source_action_plan_partial"
    )
    assert (
        component["pillars"]["knowledge"]["details"]["component_statuses"][
            "source_drift"
        ]
        == "regressed"
    )
    assert payload["recommendations"][0] == (
        "Expand DXF/STEP/history real-data validation so benchmark claims are "
        "grounded in production evidence."
    )
