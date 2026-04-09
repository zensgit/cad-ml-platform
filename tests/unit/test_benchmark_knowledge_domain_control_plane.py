from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_artifact_bundle import build_bundle
from scripts.export_benchmark_companion_summary import build_companion_summary
from scripts.export_benchmark_knowledge_domain_control_plane import build_payload
from scripts.export_benchmark_release_decision import build_release_decision
from scripts.export_benchmark_release_runbook import build_release_runbook
from src.core.benchmark import build_knowledge_domain_control_plane


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_domain_control_plane.py"
)


def _capability_matrix() -> dict:
    return {
        "knowledge_domain_capability_matrix": {
            "status": "knowledge_domain_capability_matrix_partial",
            "domains": {
                "tolerance": {
                    "label": "Tolerance & Fits",
                    "status": "ready",
                    "foundation_status": "ready",
                    "application_status": "ready",
                    "matrix_status": "ready",
                    "provider_status": "ready",
                    "surface_status": "ready",
                    "primary_gaps": [],
                    "action": "Keep tolerance healthy.",
                },
                "standards": {
                    "label": "Standards & Design Tables",
                    "status": "partial",
                    "foundation_status": "partial",
                    "application_status": "partial",
                    "matrix_status": "partial",
                    "provider_status": "ready",
                    "surface_status": "partial",
                    "primary_gaps": ["public_surface_missing"],
                    "action": "Promote standards into release surfaces.",
                },
                "gdt": {
                    "label": "GD&T & Datums",
                    "status": "blocked",
                    "foundation_status": "ready",
                    "application_status": "partial",
                    "matrix_status": "blocked",
                    "provider_status": "missing",
                    "surface_status": "partial",
                    "primary_gaps": ["provider_missing:gdt"],
                    "action": "Restore GD&T provider and release coverage.",
                },
            },
        },
        "recommendations": ["Promote standards and GD&T coverage."],
    }


def _capability_drift() -> dict:
    return {
        "knowledge_domain_capability_drift": {
            "status": "mixed",
            "domain_regressions": ["gdt"],
            "domain_improvements": ["tolerance"],
            "new_priority_domains": ["gdt"],
            "resolved_priority_domains": ["tolerance"],
        },
        "recommendations": ["Resolve GD&T regressions."],
    }


def _realdata_correlation() -> dict:
    return {
        "knowledge_realdata_correlation": {
            "status": "knowledge_realdata_partial",
            "domains": {
                "tolerance": {
                    "status": "ready",
                    "realdata_components": {"hybrid_dxf": "ready", "step_dir": "ready"},
                    "missing_realdata_components": [],
                    "action": "Keep tolerance real-data healthy.",
                },
                "standards": {
                    "status": "partial",
                    "realdata_components": {"hybrid_dxf": "ready", "history_h5": "partial"},
                    "missing_realdata_components": [],
                    "action": "Raise standards history coverage.",
                },
                "gdt": {
                    "status": "blocked",
                    "realdata_components": {"hybrid_dxf": "partial", "step_smoke": "missing"},
                    "missing_realdata_components": ["step_smoke"],
                    "action": "Raise GD&T STEP validation coverage.",
                },
            },
        },
        "recommendations": ["Improve standards and GD&T real-data depth."],
    }


def _outcome_correlation() -> dict:
    return {
        "knowledge_outcome_correlation": {
            "status": "knowledge_outcome_correlation_partial",
            "domains": {
                "tolerance": {
                    "status": "ready",
                    "best_surface": "hybrid_dxf",
                    "best_surface_score": 0.92,
                    "weak_surfaces": [],
                    "missing_surfaces": [],
                    "action": "Keep tolerance outcomes healthy.",
                },
                "standards": {
                    "status": "partial",
                    "best_surface": "hybrid_dxf",
                    "best_surface_score": 0.68,
                    "weak_surfaces": ["history_h5"],
                    "missing_surfaces": [],
                    "action": "Raise standards history outcomes.",
                },
                "gdt": {
                    "status": "blocked",
                    "best_surface": "hybrid_dxf",
                    "best_surface_score": 0.41,
                    "weak_surfaces": ["hybrid_dxf"],
                    "missing_surfaces": ["step_smoke"],
                    "action": "Lift GD&T hybrid and STEP outcomes.",
                },
            },
        },
        "recommendations": ["Improve standards/GD&T outcome strength."],
    }


def _domain_action_plan() -> dict:
    return {
        "knowledge_domain_action_plan": {
            "status": "knowledge_domain_action_plan_blocked",
            "priority_domains": ["gdt", "standards"],
            "actions": [
                {
                    "id": "gdt:foundation",
                    "domain": "gdt",
                    "stage": "foundation",
                    "priority": "high",
                    "status": "blocked",
                    "action": "Restore GD&T provider and release coverage.",
                },
                {
                    "id": "standards:realdata",
                    "domain": "standards",
                    "stage": "realdata",
                    "priority": "medium",
                    "status": "required",
                    "action": "Raise standards history coverage.",
                },
            ],
        },
        "recommendations": ["Fix GD&T first, then standards."],
    }


def test_build_knowledge_domain_control_plane() -> None:
    payload = build_knowledge_domain_control_plane(
        benchmark_knowledge_domain_capability_matrix=_capability_matrix(),
        benchmark_knowledge_domain_capability_drift=_capability_drift(),
        benchmark_knowledge_realdata_correlation=_realdata_correlation(),
        benchmark_knowledge_outcome_correlation=_outcome_correlation(),
        benchmark_knowledge_domain_action_plan=_domain_action_plan(),
    )

    assert payload["status"] == "knowledge_domain_control_plane_blocked"
    assert payload["ready_domain_count"] == 1
    assert payload["partial_domain_count"] == 1
    assert payload["blocked_domain_count"] == 1
    assert payload["release_blockers"] == ["gdt"]
    assert payload["domains"]["tolerance"]["status"] == "ready"
    assert payload["domains"]["standards"]["status"] == "partial"
    assert payload["domains"]["gdt"]["status"] == "blocked"
    assert payload["domains"]["gdt"]["drift_status"] == "regressed"


def test_export_benchmark_knowledge_domain_control_plane_cli(tmp_path: Path) -> None:
    capability = tmp_path / "capability.json"
    drift = tmp_path / "drift.json"
    realdata = tmp_path / "realdata.json"
    outcome = tmp_path / "outcome.json"
    action_plan = tmp_path / "action_plan.json"
    output_json = tmp_path / "control_plane.json"
    output_md = tmp_path / "control_plane.md"
    for path, payload in (
        (capability, _capability_matrix()),
        (drift, _capability_drift()),
        (realdata, _realdata_correlation()),
        (outcome, _outcome_correlation()),
        (action_plan, _domain_action_plan()),
    ):
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark-knowledge-domain-capability-matrix",
            str(capability),
            "--benchmark-knowledge-domain-capability-drift",
            str(drift),
            "--benchmark-knowledge-realdata-correlation",
            str(realdata),
            "--benchmark-knowledge-outcome-correlation",
            str(outcome),
            "--benchmark-knowledge-domain-action-plan",
            str(action_plan),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )

    assert result.returncode == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["knowledge_domain_control_plane"]["status"] == (
        "knowledge_domain_control_plane_blocked"
    )
    assert payload["knowledge_domain_control_plane"]["domains"]["standards"]["status"] == (
        "partial"
    )
    rendered = output_md.read_text(encoding="utf-8")
    assert "# Benchmark Knowledge Domain Control Plane" in rendered
    assert "## Domains" in rendered
    assert "### Tolerance & Fits" in rendered


def test_knowledge_domain_control_plane_surfaces_propagate() -> None:
    summary = build_payload(
        title="Benchmark Knowledge Domain Control Plane",
        benchmark_knowledge_domain_capability_matrix=_capability_matrix(),
        benchmark_knowledge_domain_capability_drift=_capability_drift(),
        benchmark_knowledge_realdata_correlation=_realdata_correlation(),
        benchmark_knowledge_outcome_correlation=_outcome_correlation(),
        benchmark_knowledge_domain_action_plan=_domain_action_plan(),
        artifact_paths={
            "benchmark_knowledge_domain_capability_matrix": "capability.json",
            "benchmark_knowledge_domain_capability_drift": "drift.json",
            "benchmark_knowledge_realdata_correlation": "realdata.json",
            "benchmark_knowledge_outcome_correlation": "outcome.json",
            "benchmark_knowledge_domain_action_plan": "action_plan.json",
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
        benchmark_operator_adoption={},
        benchmark_knowledge_application={},
        benchmark_knowledge_realdata_correlation=_realdata_correlation(),
        benchmark_knowledge_domain_capability_matrix=_capability_matrix(),
        benchmark_knowledge_domain_capability_drift=_capability_drift(),
        benchmark_knowledge_domain_action_plan=_domain_action_plan(),
        benchmark_knowledge_outcome_correlation=_outcome_correlation(),
        benchmark_knowledge_domain_control_plane=summary,
        artifact_paths={"benchmark_knowledge_domain_control_plane": "control_plane.json"},
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
        benchmark_operator_adoption={},
        benchmark_knowledge_application={},
        benchmark_knowledge_realdata_correlation=_realdata_correlation(),
        benchmark_knowledge_domain_capability_matrix=_capability_matrix(),
        benchmark_knowledge_domain_capability_drift=_capability_drift(),
        benchmark_knowledge_domain_action_plan=_domain_action_plan(),
        benchmark_knowledge_outcome_correlation=_outcome_correlation(),
        benchmark_knowledge_domain_control_plane=summary,
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={"benchmark_knowledge_domain_control_plane": "control_plane.json"},
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
        benchmark_operator_adoption={},
        benchmark_knowledge_application={},
        benchmark_knowledge_realdata_correlation=_realdata_correlation(),
        benchmark_knowledge_domain_capability_matrix=_capability_matrix(),
        benchmark_knowledge_domain_capability_drift=_capability_drift(),
        benchmark_knowledge_domain_action_plan=_domain_action_plan(),
        benchmark_knowledge_outcome_correlation=_outcome_correlation(),
        benchmark_knowledge_domain_control_plane=summary,
        artifact_paths={"benchmark_knowledge_domain_control_plane": "control_plane.json"},
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
        benchmark_operator_adoption={},
        benchmark_knowledge_application={},
        benchmark_knowledge_realdata_correlation=_realdata_correlation(),
        benchmark_knowledge_domain_capability_matrix=_capability_matrix(),
        benchmark_knowledge_domain_capability_drift=_capability_drift(),
        benchmark_knowledge_domain_action_plan=_domain_action_plan(),
        benchmark_knowledge_outcome_correlation=_outcome_correlation(),
        benchmark_knowledge_domain_control_plane=summary,
        artifact_paths={"benchmark_knowledge_domain_control_plane": "control_plane.json"},
    )

    assert companion["component_statuses"]["knowledge_domain_control_plane"] == (
        "knowledge_domain_control_plane_blocked"
    )
    assert bundle["component_statuses"]["knowledge_domain_control_plane"] == (
        "knowledge_domain_control_plane_blocked"
    )
    assert decision["knowledge_domain_control_plane_status"] == (
        "knowledge_domain_control_plane_blocked"
    )
    assert runbook["knowledge_domain_control_plane_status"] == (
        "knowledge_domain_control_plane_blocked"
    )
    assert "benchmark_knowledge_domain_control_plane" in bundle["artifacts"]
    assert "benchmark_knowledge_domain_control_plane" in decision["artifacts"]
    assert "benchmark_knowledge_domain_control_plane" in runbook["artifacts"]
