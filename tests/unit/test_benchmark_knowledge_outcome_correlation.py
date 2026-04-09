from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_artifact_bundle import build_bundle
from scripts.export_benchmark_companion_summary import build_companion_summary
from scripts.export_benchmark_knowledge_outcome_correlation import build_payload
from scripts.export_benchmark_release_decision import build_release_decision
from scripts.export_benchmark_release_runbook import build_release_runbook
from src.core.benchmark import build_knowledge_outcome_correlation_status


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_outcome_correlation.py"
)


def _knowledge_domain_matrix() -> dict:
    return {
        "knowledge_domain_matrix": {
            "status": "knowledge_domain_matrix_partial",
            "domains": {
                "tolerance": {
                    "label": "Tolerance & Fits",
                    "status": "partial",
                    "focus_components": ["tolerance"],
                    "missing_metrics": ["common_fit_count"],
                    "action": "Backfill tolerance coverage.",
                },
                "standards": {
                    "label": "Standards & Design Tables",
                    "status": "ready",
                    "focus_components": [],
                    "missing_metrics": [],
                    "action": "Keep standards healthy.",
                },
                "gdt": {
                    "label": "GD&T & Datums",
                    "status": "blocked",
                    "focus_components": ["gdt"],
                    "missing_metrics": ["datums_count"],
                    "action": "Raise GD&T readiness.",
                },
            },
        },
        "recommendations": [
            "Backfill tolerance coverage.",
            "Raise GD&T readiness.",
        ],
    }


def _realdata_scorecard() -> dict:
    return {
        "realdata_scorecard": {
            "status": "realdata_scorecard_partial",
            "components": {
                "hybrid_dxf": {
                    "status": "ready",
                    "coarse_accuracy": 0.91,
                    "exact_accuracy": 0.88,
                },
                "history_h5": {
                    "status": "partial",
                    "coarse_accuracy": 0.62,
                    "exact_accuracy": 0.54,
                },
                "step_smoke": {
                    "status": "environment_blocked",
                },
                "step_dir": {
                    "status": "ready",
                    "coverage_ratio": 1.0,
                    "hint_coverage_ratio": 0.67,
                },
            },
        },
        "recommendations": [
            "Expand STEP smoke reliability.",
        ],
    }


def test_build_knowledge_outcome_correlation_status() -> None:
    payload = build_knowledge_outcome_correlation_status(
        _knowledge_domain_matrix(),
        _realdata_scorecard(),
    )

    assert payload["status"] == "knowledge_outcome_correlation_partial"
    assert payload["ready_domain_count"] == 1
    assert payload["partial_domain_count"] == 1
    assert payload["blocked_domain_count"] == 1
    assert payload["domains"]["tolerance"]["best_surface"] == "step_dir"
    assert payload["domains"]["tolerance"]["surface_statuses"]["history_h5"] == (
        "partial"
    )
    assert payload["domains"]["gdt"]["status"] == "blocked"
    assert payload["priority_domains"] == ["gdt"]


def test_export_benchmark_knowledge_outcome_correlation_cli(
    tmp_path: Path,
) -> None:
    matrix_json = tmp_path / "knowledge_domain_matrix.json"
    realdata_json = tmp_path / "realdata_scorecard.json"
    output_json = tmp_path / "knowledge_outcome_correlation.json"
    output_md = tmp_path / "knowledge_outcome_correlation.md"
    matrix_json.write_text(
        json.dumps(_knowledge_domain_matrix(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    realdata_json.write_text(
        json.dumps(_realdata_scorecard(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-knowledge-domain-matrix",
            str(matrix_json),
            "--benchmark-realdata-scorecard",
            str(realdata_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        },
    )

    assert result.returncode == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["knowledge_outcome_correlation"]["status"] == (
        "knowledge_outcome_correlation_partial"
    )
    assert (
        payload["knowledge_outcome_correlation"]["domains"]["standards"][
            "best_surface"
        ]
        == "hybrid_dxf"
    )
    rendered = output_md.read_text(encoding="utf-8")
    assert "# Benchmark Knowledge Outcome Correlation" in rendered
    assert "## Domains" in rendered
    assert "### Tolerance & Fits" in rendered


def test_knowledge_outcome_correlation_surfaces_propagate() -> None:
    summary = build_payload(
        title="Benchmark Knowledge Outcome Correlation",
        benchmark_knowledge_domain_matrix=_knowledge_domain_matrix(),
        benchmark_realdata_scorecard=_realdata_scorecard(),
        artifact_paths={
            "benchmark_knowledge_domain_matrix": "knowledge_domain_matrix.json",
            "benchmark_realdata_scorecard": "realdata_scorecard.json",
        },
    )
    companion = build_companion_summary(
        title="Benchmark Companion",
        benchmark_scorecard={},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_realdata_scorecard=_realdata_scorecard(),
        benchmark_operator_adoption={},
        benchmark_knowledge_application={},
        benchmark_knowledge_realdata_correlation={},
        benchmark_knowledge_domain_matrix=_knowledge_domain_matrix(),
        benchmark_knowledge_outcome_correlation=summary,
        artifact_paths={
            "benchmark_knowledge_outcome_correlation": (
                "knowledge_outcome_correlation.json"
            )
        },
    )
    bundle = build_bundle(
        title="Benchmark Artifact Bundle",
        benchmark_scorecard={},
        benchmark_operational_summary={},
        benchmark_companion_summary=companion,
        benchmark_release_decision={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_realdata_scorecard=_realdata_scorecard(),
        benchmark_operator_adoption={},
        benchmark_knowledge_application={},
        benchmark_knowledge_realdata_correlation={},
        benchmark_knowledge_domain_matrix=_knowledge_domain_matrix(),
        benchmark_knowledge_outcome_correlation=summary,
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={
            "benchmark_knowledge_outcome_correlation": (
                "knowledge_outcome_correlation.json"
            )
        },
    )
    decision = build_release_decision(
        title="Benchmark Release Decision",
        benchmark_scorecard={},
        benchmark_operational_summary={},
        benchmark_artifact_bundle=bundle,
        benchmark_companion_summary=companion,
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_realdata_scorecard=_realdata_scorecard(),
        benchmark_operator_adoption={},
        benchmark_knowledge_application={},
        benchmark_knowledge_realdata_correlation={},
        benchmark_knowledge_domain_matrix=_knowledge_domain_matrix(),
        benchmark_knowledge_outcome_correlation=summary,
        artifact_paths={
            "benchmark_knowledge_outcome_correlation": (
                "knowledge_outcome_correlation.json"
            )
        },
    )
    runbook = build_release_runbook(
        title="Benchmark Release Runbook",
        benchmark_release_decision=decision,
        benchmark_companion_summary=companion,
        benchmark_artifact_bundle=bundle,
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_realdata_scorecard=_realdata_scorecard(),
        benchmark_operator_adoption={},
        benchmark_knowledge_application={},
        benchmark_knowledge_realdata_correlation={},
        benchmark_knowledge_domain_matrix=_knowledge_domain_matrix(),
        benchmark_knowledge_outcome_correlation=summary,
        artifact_paths={
            "benchmark_knowledge_outcome_correlation": (
                "knowledge_outcome_correlation.json"
            )
        },
    )

    assert companion["component_statuses"]["knowledge_outcome_correlation"] == (
        "knowledge_outcome_correlation_partial"
    )
    assert bundle["component_statuses"]["knowledge_outcome_correlation"] == (
        "knowledge_outcome_correlation_partial"
    )
    assert decision["knowledge_outcome_correlation_status"] == (
        "knowledge_outcome_correlation_partial"
    )
    assert runbook["knowledge_outcome_correlation_status"] == (
        "knowledge_outcome_correlation_partial"
    )
    assert "benchmark_knowledge_outcome_correlation" in bundle["artifacts"]
    assert "benchmark_knowledge_outcome_correlation" in decision["artifacts"]
    assert "benchmark_knowledge_outcome_correlation" in runbook["artifacts"]
    assert any(
        "Use companion, bundle, release decision, and runbook surfaces" in item
        for item in decision["review_signals"]
    )
