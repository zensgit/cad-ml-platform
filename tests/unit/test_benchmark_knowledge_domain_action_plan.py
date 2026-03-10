from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_knowledge_domain_action_plan import (
    build_summary,
)
from src.core.benchmark import (
    build_knowledge_domain_action_plan,
    knowledge_domain_action_plan_recommendations,
    render_knowledge_domain_action_plan_markdown,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_domain_action_plan.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_knowledge_domain_action_plan_blocked() -> None:
    payload = build_knowledge_domain_action_plan(
        {
            "knowledge_domain_matrix": {
                "domains": {
                    "standards": {
                        "label": "Standards & Design Tables",
                        "status": "blocked",
                        "readiness_status": "missing",
                        "application_status": "partial",
                        "realdata_status": "blocked",
                        "missing_metrics": ["thread_count"],
                        "focus_components": ["standards"],
                        "application_signal_count": 1,
                        "blocked_realdata_components": ["step_smoke"],
                    }
                }
            }
        }
    )
    assert payload["status"] == "knowledge_domain_action_plan_blocked"
    assert payload["total_action_count"] == 3
    assert payload["high_priority_action_count"] == 2
    assert payload["recommended_first_actions"][0]["id"] == "standards:foundation"
    assert any(
        item.startswith("foundation: Backfill Standards & Design Tables")
        for item in knowledge_domain_action_plan_recommendations(payload)
    )


def test_export_benchmark_knowledge_domain_action_plan_cli(tmp_path: Path) -> None:
    matrix = _write_json(
        tmp_path / "matrix.json",
        {
            "knowledge_domain_matrix": {
                "domains": {
                    "tolerance": {
                        "label": "Tolerance & Fits",
                        "status": "partial",
                        "readiness_status": "partial",
                        "application_status": "ready",
                        "realdata_status": "partial",
                        "missing_metrics": ["common_fit_count"],
                        "focus_components": ["tolerance"],
                        "application_signal_count": 3,
                        "partial_realdata_components": ["history_h5"],
                    }
                }
            }
        },
    )
    output_json = tmp_path / "action_plan.json"
    output_md = tmp_path / "action_plan.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-knowledge-domain-matrix",
            str(matrix),
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
    assert payload["knowledge_domain_action_plan"]["status"] == (
        "knowledge_domain_action_plan_partial"
    )
    assert output_json.exists()
    assert output_md.exists()
    rendered = output_md.read_text(encoding="utf-8")
    assert "## Actions" in rendered


def test_render_knowledge_domain_action_plan_markdown() -> None:
    rendered = render_knowledge_domain_action_plan_markdown(
        build_summary(
            title="Benchmark Knowledge Domain Action Plan",
            knowledge_domain_matrix_summary={
                "knowledge_domain_matrix": {
                    "domains": {
                        "gdt": {
                            "label": "GD&T & Datums",
                            "status": "partial",
                            "readiness_status": "ready",
                            "application_status": "partial",
                            "realdata_status": "ready",
                            "missing_metrics": [],
                            "focus_components": ["gdt"],
                            "application_signal_count": 0,
                        }
                    }
                }
            },
            artifact_paths={"benchmark_knowledge_domain_matrix": "matrix.json"},
        ),
        "Benchmark Knowledge Domain Action Plan",
    )
    assert "# Benchmark Knowledge Domain Action Plan" in rendered
    assert "## Recommendations" in rendered
