from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_competitive_surpass_action_plan import build_summary
from src.core.benchmark import (
    build_competitive_surpass_action_plan,
    competitive_surpass_action_plan_recommendations,
    render_competitive_surpass_action_plan_markdown,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_competitive_surpass_action_plan.py"
)


def _ready_inputs() -> dict:
    return {
        "benchmark_competitive_surpass_index": {
            "competitive_surpass_index": {
                "status": "competitive_surpass_ready",
                "primary_gaps": [],
            }
        },
        "benchmark_competitive_surpass_trend": {
            "competitive_surpass_trend": {
                "status": "regressed",
                "score_delta": -4,
                "pillar_regressions": ["realdata"],
                "new_primary_gaps": ["step_dir_depth"],
            },
            "recommendations": [
                "realdata: Stabilize STEP directory coverage before promotion."
            ],
        },
        "benchmark_engineering_signals": {
            "engineering_signals": {
                "status": "engineering_semantics_ready",
                "coverage_ratio": 0.94,
                "rows_with_standards_candidates": 12,
            },
            "recommendations": [],
        },
        "benchmark_knowledge_domain_action_plan": {
            "knowledge_domain_action_plan": {
                "status": "knowledge_domain_action_plan_ready",
                "priority_domains": [],
                "total_action_count": 0,
                "actions": [],
            },
            "recommendations": [],
        },
        "benchmark_realdata_signals": {
            "realdata_signals": {
                "status": "realdata_foundation_partial",
                "ready_component_count": 2,
                "partial_component_count": 1,
                "environment_blocked_count": 0,
            },
            "recommendations": [
                "realdata: Expand STEP smoke coverage and history manifests."
            ],
        },
        "benchmark_realdata_scorecard": {
            "realdata_scorecard": {
                "status": "realdata_scorecard_partial",
                "ready_component_count": 2,
                "partial_component_count": 1,
                "environment_blocked_count": 0,
            },
            "recommendations": [
                "realdata: Close the remaining real-data scorecard gaps."
            ],
        },
        "benchmark_operator_adoption": {
            "adoption_readiness": "guided_manual",
            "operator_mode": "guided_review",
            "release_surface_alignment_status": "diverged",
            "release_surface_alignment_summary": "Release surfaces still diverge.",
            "recommended_actions": [
                "release_alignment: Refresh operator-facing release surfaces."
            ],
        },
    }


def test_build_competitive_surpass_action_plan_blocks_and_prioritizes() -> None:
    payload = build_competitive_surpass_action_plan(**_ready_inputs())

    assert payload["status"] == "competitive_surpass_action_plan_blocked"
    assert payload["high_priority_action_count"] >= 1
    assert payload["total_action_count"] >= 2
    assert "realdata" in payload["priority_pillars"]
    assert payload["recommended_first_actions"][0]["pillar"] in {
        "competitive_surpass",
        "realdata",
        "release_alignment",
    }
    recommendations = competitive_surpass_action_plan_recommendations(payload)
    assert recommendations
    assert any("realdata" in item for item in recommendations)


def test_export_benchmark_competitive_surpass_action_plan_cli(tmp_path: Path) -> None:
    inputs = _ready_inputs()
    argv = [
        sys.executable,
        str(SCRIPT),
        "--output-json",
        str(tmp_path / "competitive_surpass_action_plan.json"),
        "--output-md",
        str(tmp_path / "competitive_surpass_action_plan.md"),
    ]
    for flag, key in (
        ("--benchmark-competitive-surpass-index", "benchmark_competitive_surpass_index"),
        ("--benchmark-competitive-surpass-trend", "benchmark_competitive_surpass_trend"),
        ("--benchmark-engineering-signals", "benchmark_engineering_signals"),
        (
            "--benchmark-knowledge-domain-action-plan",
            "benchmark_knowledge_domain_action_plan",
        ),
        ("--benchmark-realdata-signals", "benchmark_realdata_signals"),
        ("--benchmark-realdata-scorecard", "benchmark_realdata_scorecard"),
        ("--benchmark-operator-adoption", "benchmark_operator_adoption"),
    ):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(inputs[key]), encoding="utf-8")
        argv.extend([flag, str(path)])

    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["competitive_surpass_action_plan"]["status"] == (
        "competitive_surpass_action_plan_blocked"
    )
    assert (tmp_path / "competitive_surpass_action_plan.json").exists()
    assert (tmp_path / "competitive_surpass_action_plan.md").exists()


def test_render_competitive_surpass_action_plan_markdown() -> None:
    rendered = render_competitive_surpass_action_plan_markdown(
        build_summary(
            title="Benchmark Competitive Surpass Action Plan",
            artifact_paths={},
            **_ready_inputs(),
        ),
        "Benchmark Competitive Surpass Action Plan",
    )
    assert "# Benchmark Competitive Surpass Action Plan" in rendered
    assert "## Actions" in rendered
    assert "## Recommendations" in rendered
