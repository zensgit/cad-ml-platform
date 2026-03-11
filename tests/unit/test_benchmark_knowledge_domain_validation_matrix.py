from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_artifact_bundle import build_bundle
from scripts.export_benchmark_companion_summary import build_companion_summary
from scripts.export_benchmark_knowledge_domain_validation_matrix import (
    build_summary,
)
from scripts.export_benchmark_release_decision import build_release_decision
from scripts.export_benchmark_release_runbook import build_release_runbook
from src.core.benchmark import (
    build_knowledge_domain_validation_matrix,
    knowledge_domain_validation_matrix_recommendations,
    render_knowledge_domain_validation_matrix_markdown,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_domain_validation_matrix.py"
)


def _engineering() -> dict:
    return {
        "engineering_signals": {"status": "engineering_semantics_ready"},
        "recommendations": [],
    }


def _realdata() -> dict:
    return {
        "realdata_signals": {
            "status": "realdata_foundation_ready",
            "components": {
                "hybrid_dxf": {"status": "ready", "sample_size": 10},
                "history_h5": {"status": "ready", "sample_size": 1},
                "step_dir": {"status": "ready", "sample_size": 3},
            },
        },
        "recommendations": [],
    }


def _operator_adoption() -> dict:
    return {
        "adoption_readiness": "guided_manual",
        "recommended_actions": ["Review operator blockers."],
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_knowledge_domain_validation_matrix_detects_gdt_validation_gap() -> None:
    component = build_knowledge_domain_validation_matrix()

    assert component["status"] == "knowledge_domain_validation_blocked"
    assert component["ready_domain_count"] == 2
    assert component["blocked_domain_count"] == 1
    assert component["priority_domains"] == ["gdt"]

    assert component["domains"]["tolerance"]["status"] == "ready"
    assert component["domains"]["standards"]["status"] == "ready"
    assert component["domains"]["gdt"]["status"] == "blocked"
    assert component["domains"]["gdt"]["provider_status"] == "missing"
    assert component["domains"]["gdt"]["missing_layers"] == [
        "provider",
        "api",
        "integration_tests",
    ]


def test_recommendations_and_markdown_highlight_gdt_gaps() -> None:
    payload = build_summary(
        title="Benchmark Knowledge Domain Validation Matrix",
        artifact_paths={"benchmark_knowledge_domain_validation_matrix": "matrix.json"},
    )
    component = payload["knowledge_domain_validation_matrix"]
    recommendations = knowledge_domain_validation_matrix_recommendations(component)
    rendered = render_knowledge_domain_validation_matrix_markdown(
        payload,
        "Benchmark Knowledge Domain Validation Matrix",
    )

    assert recommendations == [
        "gdt: close validation gaps in provider, api, integration_tests"
    ]
    assert "## Domains" in rendered
    assert "### GD&T & Datums" in rendered
    assert "- `status`: `blocked`" in rendered
    assert "provider, api, integration_tests" in rendered


def test_cli_exports_validation_matrix_json_and_markdown(tmp_path: Path) -> None:
    output_json = tmp_path / "validation_matrix.json"
    output_md = tmp_path / "validation_matrix.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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

    assert payload["knowledge_domain_validation_matrix"]["status"] == (
        "knowledge_domain_validation_blocked"
    )
    assert output_json.exists()
    assert output_md.exists()
    rendered = output_md.read_text(encoding="utf-8")
    assert "## Recommendations" in rendered


def test_bundle_companion_and_release_surfaces_include_validation_matrix() -> None:
    validation = build_summary(
        title="Benchmark Knowledge Domain Validation Matrix",
        artifact_paths={"benchmark_knowledge_domain_validation_matrix": "matrix.json"},
    )
    bundle = build_bundle(
        title="Bundle",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={"component_statuses": {"review_queue": "healthy"}},
        benchmark_companion_summary={},
        benchmark_release_decision={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals=_engineering(),
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        benchmark_knowledge_domain_validation_matrix=validation,
        artifact_paths={"benchmark_knowledge_domain_validation_matrix": "matrix.json"},
    )
    companion = build_companion_summary(
        title="Companion",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={"component_statuses": {"review_queue": "healthy"}},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals=_engineering(),
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        benchmark_knowledge_domain_validation_matrix=validation,
        artifact_paths={"benchmark_knowledge_domain_validation_matrix": "matrix.json"},
    )
    decision = build_release_decision(
        title="Decision",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_companion_summary={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals=_engineering(),
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        benchmark_knowledge_domain_validation_matrix=validation,
        artifact_paths={"benchmark_knowledge_domain_validation_matrix": "matrix.json"},
    )
    runbook = build_release_runbook(
        title="Runbook",
        benchmark_release_decision={},
        benchmark_companion_summary={},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals=_engineering(),
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        benchmark_knowledge_domain_validation_matrix=validation,
        artifact_paths={"benchmark_knowledge_domain_validation_matrix": "matrix.json"},
    )

    assert bundle["knowledge_domain_validation_matrix_status"] == (
        "knowledge_domain_validation_blocked"
    )
    assert companion["knowledge_domain_validation_matrix_domains"]["gdt"]["status"] == (
        "blocked"
    )
    assert decision["knowledge_domain_validation_matrix_domains"]["gdt"]["status"] == (
        "blocked"
    )
    assert runbook["knowledge_domain_validation_matrix_domains"]["gdt"]["status"] == (
        "blocked"
    )
    assert decision["review_signals"] == [
        "gdt: close validation gaps in provider, api, integration_tests"
    ]
    assert runbook["review_signals"] == [
        "gdt: close validation gaps in provider, api, integration_tests"
    ]
