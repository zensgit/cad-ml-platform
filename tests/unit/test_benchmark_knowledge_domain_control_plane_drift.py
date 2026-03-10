from __future__ import annotations

from scripts.export_benchmark_artifact_bundle import build_bundle
from scripts.export_benchmark_companion_summary import build_companion_summary
from scripts.export_benchmark_knowledge_domain_control_plane_drift import build_summary
from scripts.export_benchmark_release_decision import build_release_decision
from scripts.export_benchmark_release_runbook import build_release_runbook
from src.core.benchmark.knowledge_domain_control_plane_drift import (
    build_knowledge_domain_control_plane_drift_status,
    knowledge_domain_control_plane_drift_recommendations,
    render_knowledge_domain_control_plane_drift_markdown,
)


def _current_control_plane() -> dict:
    return {
        "knowledge_domain_control_plane": {
            "status": "knowledge_domain_control_plane_blocked",
            "ready_domain_count": 1,
            "partial_domain_count": 1,
            "blocked_domain_count": 1,
            "missing_domain_count": 0,
            "total_action_count": 3,
            "high_priority_action_count": 2,
            "release_blockers": ["gdt"],
            "domains": {
                "tolerance": {
                    "domain": "tolerance",
                    "label": "Tolerance & Fits",
                    "status": "ready",
                    "priority": "low",
                    "release_blocker": False,
                    "primary_gaps": [],
                    "next_action": "Keep tolerance healthy.",
                },
                "standards": {
                    "domain": "standards",
                    "label": "Standards & Design Tables",
                    "status": "partial",
                    "priority": "medium",
                    "release_blocker": False,
                    "primary_gaps": ["public_surface_missing"],
                    "next_action": "Promote standards release coverage.",
                },
                "gdt": {
                    "domain": "gdt",
                    "label": "GD&T & Datums",
                    "status": "blocked",
                    "priority": "high",
                    "release_blocker": True,
                    "primary_gaps": ["provider_missing:gdt"],
                    "next_action": "Restore GD&T provider coverage.",
                },
            },
        },
        "recommendations": ["Resolve GD&T control-plane blockers."],
    }


def _previous_control_plane() -> dict:
    return {
        "knowledge_domain_control_plane": {
            "status": "knowledge_domain_control_plane_partial",
            "ready_domain_count": 1,
            "partial_domain_count": 2,
            "blocked_domain_count": 0,
            "missing_domain_count": 0,
            "total_action_count": 2,
            "high_priority_action_count": 1,
            "release_blockers": [],
            "domains": {
                "tolerance": {
                    "domain": "tolerance",
                    "label": "Tolerance & Fits",
                    "status": "ready",
                    "priority": "low",
                    "release_blocker": False,
                    "primary_gaps": [],
                    "next_action": "Keep tolerance healthy.",
                },
                "standards": {
                    "domain": "standards",
                    "label": "Standards & Design Tables",
                    "status": "partial",
                    "priority": "medium",
                    "release_blocker": False,
                    "primary_gaps": ["public_surface_missing"],
                    "next_action": "Promote standards release coverage.",
                },
                "gdt": {
                    "domain": "gdt",
                    "label": "GD&T & Datums",
                    "status": "partial",
                    "priority": "medium",
                    "release_blocker": False,
                    "primary_gaps": ["provider_partial:gdt"],
                    "next_action": "Keep GD&T provider on track.",
                },
            },
        },
        "recommendations": ["Keep GD&T on track."],
    }


def _engineering() -> dict:
    return {"engineering_signals": {"status": "engineering_semantics_ready"}}


def _realdata() -> dict:
    return {"realdata_signals": {"status": "realdata_foundation_ready"}}


def _operator_adoption() -> dict:
    return {"adoption_readiness": "guided_manual"}


def test_build_knowledge_domain_control_plane_drift_detects_gdt_regression() -> None:
    component = build_knowledge_domain_control_plane_drift_status(
        _current_control_plane(),
        _previous_control_plane(),
    )

    assert component["status"] == "regressed"
    assert component["domain_regressions"] == ["gdt"]
    assert component["new_release_blockers"] == ["gdt"]
    assert component["blocked_domain_delta"] == 1
    assert component["total_action_delta"] == 1


def test_drift_recommendations_and_markdown_highlight_release_blocker() -> None:
    payload = build_summary(
        title="Control Plane Drift",
        current_summary=_current_control_plane(),
        previous_summary=_previous_control_plane(),
        artifact_paths={},
    )
    recommendations = knowledge_domain_control_plane_drift_recommendations(
        payload["knowledge_domain_control_plane_drift"]
    )
    rendered = render_knowledge_domain_control_plane_drift_markdown(
        payload,
        "Control Plane Drift",
    )

    assert any("Regressed domains" in item for item in recommendations)
    assert "new_release_blockers" in rendered
    assert "GD&T & Datums" in rendered


def test_control_plane_drift_surfaces_propagate() -> None:
    drift = build_summary(
        title="Control Plane Drift",
        current_summary=_current_control_plane(),
        previous_summary=_previous_control_plane(),
        artifact_paths={},
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
        benchmark_knowledge_domain_control_plane_drift=drift,
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={
            "benchmark_knowledge_domain_control_plane_drift": "control_plane_drift.json"
        },
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
        benchmark_knowledge_domain_control_plane_drift=drift,
        artifact_paths={
            "benchmark_knowledge_domain_control_plane_drift": "control_plane_drift.json"
        },
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
        benchmark_knowledge_domain_control_plane_drift=drift,
        artifact_paths={
            "benchmark_knowledge_domain_control_plane_drift": "control_plane_drift.json"
        },
    )
    runbook = build_release_runbook(
        title="Runbook",
        benchmark_release_decision={},
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={},
        benchmark_companion_summary={},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals=_engineering(),
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        benchmark_knowledge_domain_control_plane_drift=drift,
        artifact_paths={
            "benchmark_knowledge_domain_control_plane_drift": "control_plane_drift.json"
        },
    )

    assert bundle["knowledge_domain_control_plane_drift_status"] == "regressed"
    assert bundle["component_statuses"]["knowledge_domain_control_plane_drift"] == (
        "regressed"
    )
    assert (
        bundle["artifacts"]["benchmark_knowledge_domain_control_plane_drift"]["present"]
        is True
    )

    assert companion["knowledge_domain_control_plane_drift_status"] == "regressed"
    assert companion["component_statuses"]["knowledge_domain_control_plane_drift"] == (
        "regressed"
    )

    assert decision["knowledge_domain_control_plane_drift_status"] == "regressed"
    assert decision["knowledge_domain_control_plane_drift_domain_regressions"] == [
        "gdt"
    ]
    assert decision["knowledge_domain_control_plane_drift_new_release_blockers"] == [
        "gdt"
    ]

    assert runbook["knowledge_domain_control_plane_drift_status"] == "regressed"
    assert runbook["knowledge_domain_control_plane_drift_domain_regressions"] == [
        "gdt"
    ]
