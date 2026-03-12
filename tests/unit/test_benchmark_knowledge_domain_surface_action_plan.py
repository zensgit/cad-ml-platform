from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_knowledge_domain_surface_action_plan import (
    build_summary,
)
from src.core.benchmark import (
    build_knowledge_domain_surface_action_plan,
    knowledge_domain_surface_action_plan_recommendations,
    render_knowledge_domain_surface_action_plan_markdown,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_domain_surface_action_plan.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_knowledge_domain_surface_action_plan_blocked() -> None:
    payload = build_knowledge_domain_surface_action_plan(
        {
            "knowledge_domain_surface_matrix": {
                "domains": {
                    "gdt": {
                        "label": "GD&T & Datums",
                        "status": "blocked",
                        "subcapabilities": {
                            "gdt_public_api": {
                                "label": "GD&T public API",
                                "status": "blocked",
                                "present_route_count": 0,
                                "expected_route_count": 1,
                                "missing_routes": ["/api/v1/gdt"],
                                "reference_item_count": 0,
                                "action": "Promote GD&T from reference-only knowledge.",
                            }
                        },
                    }
                }
            }
        }
    )
    assert payload["status"] == "knowledge_domain_surface_action_plan_blocked"
    assert payload["total_action_count"] == 1
    assert payload["high_priority_action_count"] == 1
    assert payload["recommended_first_actions"][0]["id"] == "gdt:gdt_public_api"
    assert any(
        "GD&T public API" in item
        for item in knowledge_domain_surface_action_plan_recommendations(payload)
    )


def test_export_benchmark_knowledge_domain_surface_action_plan_cli(
    tmp_path: Path,
) -> None:
    matrix = _write_json(
        tmp_path / "surface-matrix.json",
        {
            "knowledge_domain_surface_matrix": {
                "domains": {
                    "standards": {
                        "label": "Standards & Design Standards",
                        "status": "partial",
                        "subcapabilities": {
                            "bearing_lookup": {
                                "label": "Bearing lookup",
                                "status": "partial",
                                "present_route_count": 1,
                                "expected_route_count": 2,
                                "missing_routes": ["/api/v1/standards/bearing/by-bore"],
                                "reference_item_count": 42,
                                "action": "Expose bearing lookup through stable public routes.",
                            }
                        },
                    }
                }
            }
        },
    )
    output_json = tmp_path / "surface-action-plan.json"
    output_md = tmp_path / "surface-action-plan.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-knowledge-domain-surface-matrix",
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
    assert payload["knowledge_domain_surface_action_plan"]["status"] == (
        "knowledge_domain_surface_action_plan_partial"
    )
    assert output_json.exists()
    assert output_md.exists()
    rendered = output_md.read_text(encoding="utf-8")
    assert "## Actions" in rendered


def test_render_knowledge_domain_surface_action_plan_markdown() -> None:
    rendered = render_knowledge_domain_surface_action_plan_markdown(
        build_summary(
            title="Benchmark Knowledge Domain Surface Action Plan",
            knowledge_domain_surface_matrix_summary={
                "knowledge_domain_surface_matrix": {
                    "domains": {
                        "tolerance": {
                            "label": "Tolerance & Fits",
                            "status": "partial",
                            "subcapabilities": {
                                "fit_deviations": {
                                    "label": "Fit deviations",
                                    "status": "partial",
                                    "present_route_count": 1,
                                    "expected_route_count": 2,
                                    "missing_routes": ["/api/v1/tolerance/fit"],
                                    "reference_item_count": 8,
                                    "action": "Backfill fit deviation routes.",
                                }
                            },
                        }
                    }
                }
            },
            artifact_paths={"benchmark_knowledge_domain_surface_matrix": "surface.json"},
        ),
        "Benchmark Knowledge Domain Surface Action Plan",
    )
    assert "# Benchmark Knowledge Domain Surface Action Plan" in rendered
    assert "## Recommendations" in rendered
