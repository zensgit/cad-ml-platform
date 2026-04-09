from __future__ import annotations

from scripts.export_benchmark_artifact_bundle import build_bundle
from scripts.export_benchmark_companion_summary import build_companion_summary
from scripts.export_benchmark_knowledge_domain_capability_matrix import build_summary
from scripts.export_benchmark_release_decision import build_release_decision
from scripts.export_benchmark_release_runbook import build_release_runbook
from src.core.benchmark.knowledge_domain_capability_matrix import (
    build_knowledge_domain_capability_matrix,
    knowledge_domain_capability_matrix_recommendations,
    render_knowledge_domain_capability_matrix_markdown,
)


def _knowledge_readiness() -> dict:
    return {
        "knowledge_readiness": {
            "status": "knowledge_foundation_partial",
            "domains": {
                "tolerance": {"status": "ready"},
                "standards": {"status": "partial"},
                "gdt": {"status": "ready"},
            },
            "priority_domains": ["standards"],
            "focus_areas_detail": [
                {
                    "component": "standards",
                    "status": "partial",
                    "priority": "medium",
                    "action": "Backfill standards readiness.",
                }
            ],
        },
        "recommendations": ["Backfill standards readiness."],
    }


def _knowledge_application() -> dict:
    return {
        "knowledge_application": {
            "status": "knowledge_application_partial",
            "domains": {
                "tolerance": {
                    "status": "ready",
                    "readiness_status": "ready",
                    "evidence_status": "ready",
                    "signal_count": 3,
                },
                "standards": {
                    "status": "partial",
                    "readiness_status": "partial",
                    "evidence_status": "partial",
                    "signal_count": 1,
                },
                "gdt": {
                    "status": "ready",
                    "readiness_status": "ready",
                    "evidence_status": "ready",
                    "signal_count": 1,
                },
            },
            "priority_domains": ["standards"],
            "focus_areas_detail": [
                {
                    "domain": "standards",
                    "status": "partial",
                    "priority": "medium",
                    "action": "Raise standards application evidence.",
                }
            ],
        },
        "recommendations": ["Raise standards application evidence."],
    }


def _knowledge_domain_matrix() -> dict:
    return {
        "knowledge_domain_matrix": {
            "status": "knowledge_domain_matrix_partial",
            "domains": {
                "tolerance": {
                    "status": "ready",
                    "readiness_status": "ready",
                    "application_status": "ready",
                    "realdata_status": "ready",
                    "action": "Keep tolerance aligned.",
                },
                "standards": {
                    "status": "partial",
                    "readiness_status": "partial",
                    "application_status": "partial",
                    "realdata_status": "partial",
                    "action": "Promote standards into benchmark surfaces.",
                },
                "gdt": {
                    "status": "ready",
                    "readiness_status": "ready",
                    "application_status": "ready",
                    "realdata_status": "ready",
                    "action": "Promote GD&T into provider-backed surfaces.",
                },
            },
            "priority_domains": ["standards", "gdt"],
            "focus_areas_detail": [
                {
                    "domain": "standards",
                    "status": "partial",
                    "priority": "medium",
                    "action": "Promote standards into benchmark surfaces.",
                }
            ],
        },
        "recommendations": ["Promote standards into benchmark surfaces."],
    }


def _capability_matrix() -> dict:
    return build_summary(
        title="Knowledge Domain Capability Matrix",
        knowledge_readiness_summary=_knowledge_readiness(),
        knowledge_application_summary=_knowledge_application(),
        knowledge_domain_matrix_summary=_knowledge_domain_matrix(),
        artifact_paths={},
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


def test_build_knowledge_domain_capability_matrix_detects_provider_and_surface_gaps() -> None:
    component = build_knowledge_domain_capability_matrix(
        knowledge_readiness_summary=_knowledge_readiness(),
        knowledge_application_summary=_knowledge_application(),
        knowledge_domain_matrix_summary=_knowledge_domain_matrix(),
    )

    assert component["status"] == "knowledge_domain_capability_partial"
    assert component["domains"]["tolerance"]["status"] == "ready"
    assert component["domains"]["standards"]["status"] == "partial"
    assert component["domains"]["gdt"]["status"] == "blocked"
    assert component["domains"]["gdt"]["provider_status"] == "missing"
    assert component["domains"]["gdt"]["surface_status"] == "partial"
    assert component["domains"]["gdt"]["missing_providers"] == ["gdt"]
    assert component["provider_gap_domains"] == ["gdt"]
    assert "gdt" in component["surface_gap_domains"]
    assert component["priority_domains"] == ["gdt"]


def test_capability_matrix_recommendations_and_markdown_highlight_gdt_gap() -> None:
    payload = _capability_matrix()
    component = payload["knowledge_domain_capability_matrix"]
    recommendations = knowledge_domain_capability_matrix_recommendations(component)
    rendered = render_knowledge_domain_capability_matrix_markdown(
        payload,
        "Knowledge Domain Capability Matrix",
    )

    assert any("GD&T" in item and "provider coverage" in item for item in recommendations)
    assert any("public benchmark/API surface" in item for item in recommendations)
    assert "## Domains" in rendered
    assert "`gdt` status=`blocked`" in rendered
    assert "provider_missing:gdt" in rendered


def test_build_bundle_and_companion_include_capability_matrix_passthrough() -> None:
    capability = _capability_matrix()
    bundle = build_bundle(
        title="Bundle",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={"component_statuses": {"review_queue": "healthy"}},
        benchmark_companion_summary={},
        benchmark_release_decision={},
        benchmark_knowledge_readiness=_knowledge_readiness(),
        benchmark_knowledge_drift={},
        benchmark_engineering_signals=_engineering(),
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        benchmark_knowledge_application=_knowledge_application(),
        benchmark_knowledge_domain_matrix=_knowledge_domain_matrix(),
        benchmark_knowledge_domain_capability_matrix=capability,
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={"benchmark_knowledge_domain_capability_matrix": "capability.json"},
    )
    companion = build_companion_summary(
        title="Companion",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={"component_statuses": {"review_queue": "healthy"}},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness=_knowledge_readiness(),
        benchmark_knowledge_drift={},
        benchmark_engineering_signals=_engineering(),
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        benchmark_knowledge_application=_knowledge_application(),
        benchmark_knowledge_domain_matrix=_knowledge_domain_matrix(),
        benchmark_knowledge_domain_capability_matrix=capability,
        artifact_paths={"benchmark_knowledge_domain_capability_matrix": "capability.json"},
    )

    assert bundle["knowledge_domain_capability_matrix_status"] == (
        "knowledge_domain_capability_partial"
    )
    assert bundle["component_statuses"]["knowledge_domain_capability_matrix"] == (
        "knowledge_domain_capability_partial"
    )
    assert bundle["artifacts"]["benchmark_knowledge_domain_capability_matrix"]["present"] is True
    assert bundle["knowledge_domain_capability_matrix_domains"]["gdt"]["status"] == "blocked"

    assert companion["knowledge_domain_capability_matrix_status"] == (
        "knowledge_domain_capability_partial"
    )
    assert companion["component_statuses"]["knowledge_domain_capability_matrix"] == (
        "knowledge_domain_capability_partial"
    )
    assert companion["knowledge_domain_capability_matrix_domains"]["gdt"]["provider_status"] == (
        "missing"
    )
    assert any(
        "GD&T" in item
        for item in companion["knowledge_domain_capability_matrix_recommendations"]
    )


def test_build_release_surfaces_include_capability_matrix_review_signals() -> None:
    capability = _capability_matrix()
    decision = build_release_decision(
        title="Decision",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_companion_summary={},
        benchmark_knowledge_readiness=_knowledge_readiness(),
        benchmark_knowledge_drift={},
        benchmark_engineering_signals=_engineering(),
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        benchmark_knowledge_application=_knowledge_application(),
        benchmark_knowledge_domain_matrix=_knowledge_domain_matrix(),
        benchmark_knowledge_domain_capability_matrix=capability,
        artifact_paths={"benchmark_knowledge_domain_capability_matrix": "capability.json"},
    )
    runbook = build_release_runbook(
        title="Runbook",
        benchmark_release_decision={},
        benchmark_companion_summary={},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness=_knowledge_readiness(),
        benchmark_knowledge_drift={},
        benchmark_engineering_signals=_engineering(),
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        benchmark_knowledge_application=_knowledge_application(),
        benchmark_knowledge_domain_matrix=_knowledge_domain_matrix(),
        benchmark_knowledge_domain_capability_matrix=capability,
        artifact_paths={"benchmark_knowledge_domain_capability_matrix": "capability.json"},
    )

    assert decision["knowledge_domain_capability_matrix_status"] == (
        "knowledge_domain_capability_partial"
    )
    assert decision["knowledge_domain_capability_matrix_domains"]["gdt"]["status"] == "blocked"
    assert any("GD&T" in item for item in decision["review_signals"])

    assert runbook["knowledge_domain_capability_matrix_status"] == (
        "knowledge_domain_capability_partial"
    )
    assert runbook["knowledge_domain_capability_matrix_domains"]["gdt"]["status"] == "blocked"
    assert any("GD&T" in item for item in runbook["review_signals"])
