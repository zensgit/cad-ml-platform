from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_knowledge_domain_release_readiness_action_plan import (
    build_summary,
)
from src.core.benchmark import (
    build_knowledge_domain_release_readiness_action_plan,
    knowledge_domain_release_readiness_action_plan_recommendations,
    render_knowledge_domain_release_readiness_action_plan_markdown,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_domain_release_readiness_action_plan.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_release_readiness_action_plan_blocked() -> None:
    payload = build_knowledge_domain_release_readiness_action_plan(
        benchmark_knowledge_domain_release_readiness_matrix={
            "knowledge_domain_release_readiness_matrix": {
                "domains": {
                    "standards": {
                        "label": "Standards & Design Tables",
                        "status": "blocked",
                        "release_gate_status": "blocked",
                        "validation_status": "partial",
                        "inventory_status": "partial",
                        "alignment_warning": True,
                        "action": "Backfill standards release-readiness coverage.",
                        "blocking_reasons": ["missing_tables"],
                        "warning_reasons": ["alignment_warning"],
                    }
                }
            }
        },
        benchmark_knowledge_domain_release_readiness_drift={
            "knowledge_domain_release_readiness_drift": {
                "domain_changes": [
                    {
                        "domain": "standards",
                        "label": "Standards & Design Tables",
                        "trend": "regressed",
                        "previous_status": "ready",
                        "current_status": "blocked",
                    }
                ]
            }
        },
        benchmark_knowledge_domain_release_gate={
            "knowledge_domain_release_gate": {
                "gate_open": False,
                "blocked_domains": ["standards"],
                "priority_domains": ["standards"],
            }
        },
    )
    assert payload["status"] == (
        "knowledge_domain_release_readiness_action_plan_blocked"
    )
    assert payload["total_action_count"] == 3
    assert payload["high_priority_action_count"] == 3
    assert payload["recommended_first_actions"][0]["stage"] in {
        "readiness",
        "drift",
        "release_gate",
    }
    assert any(
        item.startswith("readiness: Backfill standards release-readiness coverage.")
        for item in knowledge_domain_release_readiness_action_plan_recommendations(
            payload
        )
    )


def test_export_release_readiness_action_plan_cli(tmp_path: Path) -> None:
    matrix = _write_json(
        tmp_path / "matrix.json",
        {
            "knowledge_domain_release_readiness_matrix": {
                "domains": {
                    "tolerance": {
                        "label": "Tolerance & Fits",
                        "status": "partial",
                        "release_gate_status": "partial",
                        "validation_status": "ready",
                        "inventory_status": "partial",
                        "action": "Raise tolerance release-readiness coverage.",
                    }
                }
            }
        },
    )
    drift = _write_json(
        tmp_path / "drift.json",
        {
            "knowledge_domain_release_readiness_drift": {
                "domain_changes": [
                    {
                        "domain": "tolerance",
                        "label": "Tolerance & Fits",
                        "trend": "mixed",
                        "previous_status": "ready",
                        "current_status": "partial",
                    }
                ]
            }
        },
    )
    gate = _write_json(
        tmp_path / "gate.json",
        {
            "knowledge_domain_release_gate": {
                "gate_open": True,
                "partial_domains": ["tolerance"],
                "priority_domains": ["tolerance"],
            }
        },
    )
    output_json = tmp_path / "action_plan.json"
    output_md = tmp_path / "action_plan.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-knowledge-domain-release-readiness-matrix",
            str(matrix),
            "--benchmark-knowledge-domain-release-readiness-drift",
            str(drift),
            "--benchmark-knowledge-domain-release-gate",
            str(gate),
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
    assert payload["knowledge_domain_release_readiness_action_plan"]["status"] == (
        "knowledge_domain_release_readiness_action_plan_blocked"
    )
    assert output_json.exists()
    assert output_md.exists()
    rendered = output_md.read_text(encoding="utf-8")
    assert "## Actions" in rendered


def test_render_release_readiness_action_plan_markdown() -> None:
    rendered = render_knowledge_domain_release_readiness_action_plan_markdown(
        build_summary(
            title="Benchmark Knowledge Domain Release Readiness Action Plan",
            benchmark_knowledge_domain_release_readiness_matrix={
                "knowledge_domain_release_readiness_matrix": {
                    "domains": {
                        "gdt": {
                            "label": "GD&T & Datums",
                            "status": "partial",
                            "release_gate_status": "partial",
                            "validation_status": "ready",
                            "inventory_status": "ready",
                            "action": "Raise GD&T release-readiness coverage.",
                        }
                    }
                }
            },
            benchmark_knowledge_domain_release_readiness_drift={},
            benchmark_knowledge_domain_release_gate={
                "knowledge_domain_release_gate": {
                    "gate_open": True,
                    "partial_domains": ["gdt"],
                }
            },
            artifact_paths={
                "benchmark_knowledge_domain_release_readiness_matrix": "matrix.json",
                "benchmark_knowledge_domain_release_gate": "gate.json",
            },
        ),
        "Benchmark Knowledge Domain Release Readiness Action Plan",
    )
    assert "# Benchmark Knowledge Domain Release Readiness Action Plan" in rendered
    assert "## Recommendations" in rendered
