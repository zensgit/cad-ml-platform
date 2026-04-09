from __future__ import annotations

from scripts.export_benchmark_artifact_bundle import build_bundle
from scripts.export_benchmark_companion_summary import build_companion_summary
from scripts.export_benchmark_knowledge_domain_capability_drift import build_summary
from scripts.export_benchmark_release_decision import build_release_decision
from scripts.export_benchmark_release_runbook import build_release_runbook
from src.core.benchmark.knowledge_domain_capability_drift import (
    build_knowledge_domain_capability_drift_status,
    knowledge_domain_capability_drift_recommendations,
    render_knowledge_domain_capability_drift_markdown,
)


def _current_matrix() -> dict:
    return {
        "knowledge_domain_capability_matrix": {
            "status": "knowledge_domain_capability_partial",
            "ready_domain_count": 1,
            "partial_domain_count": 1,
            "blocked_domain_count": 1,
            "priority_domains": ["gdt"],
            "provider_gap_domains": ["gdt"],
            "surface_gap_domains": ["gdt", "standards"],
            "focus_areas_detail": [
                {"domain": "standards", "priority": "medium"},
                {"domain": "gdt", "priority": "high"},
            ],
            "domains": {
                "tolerance": {
                    "domain": "tolerance",
                    "label": "Tolerance & Fits",
                    "status": "ready",
                    "provider_status": "ready",
                    "surface_status": "ready",
                    "priority": "low",
                    "primary_gaps": [],
                },
                "standards": {
                    "domain": "standards",
                    "label": "Standards & Design Tables",
                    "status": "partial",
                    "provider_status": "ready",
                    "surface_status": "partial",
                    "priority": "medium",
                    "primary_gaps": ["public_surface_missing"],
                },
                "gdt": {
                    "domain": "gdt",
                    "label": "GD&T & Datums",
                    "status": "blocked",
                    "provider_status": "missing",
                    "surface_status": "partial",
                    "priority": "high",
                    "primary_gaps": ["provider_missing:gdt", "public_surface_missing"],
                },
            },
        },
        "recommendations": ["Backfill GD&T provider coverage."],
    }


def _previous_matrix() -> dict:
    return {
        "knowledge_domain_capability_matrix": {
            "status": "knowledge_domain_capability_partial",
            "ready_domain_count": 2,
            "partial_domain_count": 1,
            "blocked_domain_count": 0,
            "priority_domains": ["standards"],
            "provider_gap_domains": [],
            "surface_gap_domains": ["standards"],
            "focus_areas_detail": [
                {"domain": "standards", "priority": "high"},
            ],
            "domains": {
                "tolerance": {
                    "domain": "tolerance",
                    "label": "Tolerance & Fits",
                    "status": "ready",
                    "provider_status": "ready",
                    "surface_status": "ready",
                    "priority": "low",
                    "primary_gaps": [],
                },
                "standards": {
                    "domain": "standards",
                    "label": "Standards & Design Tables",
                    "status": "partial",
                    "provider_status": "ready",
                    "surface_status": "partial",
                    "priority": "high",
                    "primary_gaps": ["public_surface_missing"],
                },
                "gdt": {
                    "domain": "gdt",
                    "label": "GD&T & Datums",
                    "status": "ready",
                    "provider_status": "ready",
                    "surface_status": "partial",
                    "priority": "low",
                    "primary_gaps": [],
                },
            },
        },
        "recommendations": ["Keep GD&T provider-backed."],
    }


def _engineering() -> dict:
    return {"engineering_signals": {"status": "engineering_semantics_ready"}}


def _realdata() -> dict:
    return {"realdata_signals": {"status": "realdata_foundation_ready"}}


def _operator_adoption() -> dict:
    return {"adoption_readiness": "guided_manual"}


def test_build_knowledge_domain_capability_drift_detects_gdt_regression() -> None:
    component = build_knowledge_domain_capability_drift_status(
        _current_matrix(),
        _previous_matrix(),
    )
    assert component["status"] == "regressed"
    assert component["domain_regressions"] == ["gdt"]
    assert component["new_provider_gap_domains"] == ["gdt"]
    assert component["new_priority_domains"] == ["gdt"]
    assert component["ready_domain_delta"] == -1


def test_recommendations_and_markdown_highlight_regression() -> None:
    payload = build_summary(
        title="Capability Drift",
        current_summary=_current_matrix(),
        previous_summary=_previous_matrix(),
        artifact_paths={},
    )
    recommendations = knowledge_domain_capability_drift_recommendations(
        payload["knowledge_domain_capability_drift"]
    )
    rendered = render_knowledge_domain_capability_drift_markdown(
        payload,
        "Capability Drift",
    )
    assert any("Regressed domains" in item for item in recommendations)
    assert "`gdt`" in rendered
    assert "provider=`ready` -> `missing`" in rendered


def test_bundle_companion_release_surfaces_include_capability_drift() -> None:
    drift = build_summary(
        title="Capability Drift",
        current_summary=_current_matrix(),
        previous_summary=_previous_matrix(),
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
        benchmark_knowledge_domain_capability_drift=drift,
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={"benchmark_knowledge_domain_capability_drift": "capability_drift.json"},
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
        benchmark_knowledge_domain_capability_drift=drift,
        artifact_paths={"benchmark_knowledge_domain_capability_drift": "capability_drift.json"},
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
        benchmark_knowledge_domain_capability_drift=drift,
        artifact_paths={"benchmark_knowledge_domain_capability_drift": "capability_drift.json"},
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
        benchmark_knowledge_domain_capability_drift=drift,
        artifact_paths={"benchmark_knowledge_domain_capability_drift": "capability_drift.json"},
    )

    assert bundle["knowledge_domain_capability_drift_status"] == "regressed"
    assert bundle["component_statuses"]["knowledge_domain_capability_drift"] == "regressed"
    assert bundle["artifacts"]["benchmark_knowledge_domain_capability_drift"]["present"] is True

    assert companion["knowledge_domain_capability_drift_status"] == "regressed"
    assert companion["component_statuses"]["knowledge_domain_capability_drift"] == "regressed"

    assert decision["knowledge_domain_capability_drift_status"] == "regressed"
    assert decision["component_statuses"]["knowledge_domain_capability_drift"] == "regressed"
    assert decision["knowledge_domain_capability_drift_domain_regressions"] == ["gdt"]

    assert runbook["knowledge_domain_capability_drift_status"] == "regressed"
    assert runbook["knowledge_domain_capability_drift_domain_regressions"] == ["gdt"]
