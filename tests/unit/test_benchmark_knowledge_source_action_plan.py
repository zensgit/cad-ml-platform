from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_knowledge_source_action_plan import build_summary
from src.core.benchmark import (
    build_knowledge_source_action_plan,
    knowledge_source_action_plan_recommendations,
    render_knowledge_source_action_plan_markdown,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_source_action_plan.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_knowledge_source_action_plan_blocked() -> None:
    payload = build_knowledge_source_action_plan(
        {
            "knowledge_source_coverage": {
                "source_groups": {
                    "standards": {
                        "label": "Standard Parts",
                        "domain": "standards",
                        "status": "missing",
                        "missing_source_tables": ["metric_threads"],
                        "source_item_count": 0,
                        "reference_standard_count": 0,
                    }
                },
                "domains": {
                    "standards": {
                        "label": "Standards & Design Tables",
                        "status": "missing",
                        "focus_source_groups": ["standards"],
                    }
                },
                "expansion_candidates": [
                    {
                        "name": "machining",
                        "label": "Machining Knowledge",
                        "domain": "manufacturing",
                        "status": "ready",
                        "source_table_count": 7,
                        "source_item_count": 120,
                    }
                ],
            }
        }
    )
    assert payload["status"] == "knowledge_source_action_plan_blocked"
    assert payload["total_action_count"] == 3
    assert payload["high_priority_action_count"] == 2
    assert payload["recommended_first_actions"][0]["id"] == "standards:domain"
    assert any(
        item.startswith("source_group:")
        for item in knowledge_source_action_plan_recommendations(payload)
    )


def test_export_benchmark_knowledge_source_action_plan_cli(tmp_path: Path) -> None:
    coverage = _write_json(
        tmp_path / "coverage.json",
        {
            "knowledge_source_coverage": {
                "source_groups": {
                    "tolerance": {
                        "label": "Tolerance & Fits",
                        "domain": "tolerance",
                        "status": "partial",
                        "missing_source_tables": ["fit_selection_rules"],
                        "source_item_count": 42,
                        "reference_standard_count": 3,
                    }
                },
                "domains": {
                    "tolerance": {
                        "label": "Tolerance & Fits",
                        "status": "partial",
                        "focus_source_groups": ["tolerance"],
                    }
                },
                "expansion_candidates": [],
            }
        },
    )
    output_json = tmp_path / "action_plan.json"
    output_md = tmp_path / "action_plan.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-knowledge-source-coverage",
            str(coverage),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["knowledge_source_action_plan"]["status"] == (
        "knowledge_source_action_plan_partial"
    )
    assert output_json.exists()
    assert output_md.exists()
    rendered = output_md.read_text(encoding="utf-8")
    assert "## Actions" in rendered


def test_render_knowledge_source_action_plan_markdown() -> None:
    rendered = render_knowledge_source_action_plan_markdown(
        build_summary(
            title="Benchmark Knowledge Source Action Plan",
            knowledge_source_coverage_summary={
                "knowledge_source_coverage": {
                    "source_groups": {},
                    "domains": {},
                    "expansion_candidates": [
                        {
                            "name": "welding",
                            "label": "Welding Knowledge",
                            "domain": "manufacturing",
                            "status": "ready",
                            "source_table_count": 6,
                            "source_item_count": 54,
                        }
                    ],
                }
            },
            artifact_paths={"benchmark_knowledge_source_coverage": "coverage.json"},
        ),
        "Benchmark Knowledge Source Action Plan",
    )
    assert "# Benchmark Knowledge Source Action Plan" in rendered
    assert "## Recommendations" in rendered
