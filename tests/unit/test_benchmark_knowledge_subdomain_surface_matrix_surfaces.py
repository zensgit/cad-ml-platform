from __future__ import annotations

from scripts.export_benchmark_artifact_bundle import build_bundle
from scripts.export_benchmark_companion_summary import build_companion_summary
from scripts.export_benchmark_knowledge_subdomain_surface_matrix import build_summary
from scripts.export_benchmark_release_decision import build_release_decision
from scripts.export_benchmark_release_runbook import build_release_runbook


def _knowledge_readiness() -> dict:
    return {
        "knowledge_readiness": {
            "status": "knowledge_foundation_partial",
            "domains": {"standards": {"status": "partial"}},
            "priority_domains": ["standards"],
        },
        "recommendations": ["Backfill standards readiness."],
    }


def _knowledge_application() -> dict:
    return {
        "knowledge_application": {
            "status": "knowledge_application_partial",
            "domains": {"standards": {"status": "partial", "signal_count": 2}},
            "priority_domains": ["standards"],
        },
        "recommendations": ["Raise standards application evidence."],
    }


def _knowledge_domain_matrix() -> dict:
    return {
        "knowledge_domain_matrix": {
            "status": "knowledge_domain_matrix_partial",
            "domains": {"standards": {"status": "partial"}},
            "priority_domains": ["standards"],
        },
        "recommendations": ["Promote standards benchmark surfaces."],
    }


def _capability_matrix() -> dict:
    return {
        "knowledge_domain_capability_matrix": {
            "status": "knowledge_domain_capability_partial",
            "domains": {"standards": {"status": "partial"}},
            "priority_domains": ["standards"],
        },
        "recommendations": ["Add standards provider depth."],
    }


def _engineering() -> dict:
    return {"engineering_signals": {"status": "engineering_semantics_ready"}}


def _realdata() -> dict:
    return {
        "realdata_signals": {
            "status": "realdata_foundation_ready",
            "components": {"hybrid_dxf": {"status": "ready", "sample_size": 10}},
        }
    }


def _operator_adoption() -> dict:
    return {
        "adoption_readiness": "guided_manual",
        "recommended_actions": ["Review operator blockers."],
    }


def _subdomain_surface_matrix() -> dict:
    return build_summary(
        title="Benchmark Knowledge Subdomain Surface Matrix",
        artifact_paths={},
    )


def test_build_bundle_and_companion_include_subdomain_surface_matrix() -> None:
    component = _subdomain_surface_matrix()
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
        benchmark_knowledge_domain_capability_matrix=_capability_matrix(),
        benchmark_knowledge_subdomain_surface_matrix=component,
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={
            "benchmark_knowledge_subdomain_surface_matrix": (
                "knowledge-subdomain-surface-matrix.json"
            ),
        },
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
        benchmark_knowledge_domain_capability_matrix=_capability_matrix(),
        benchmark_knowledge_subdomain_surface_matrix=component,
        artifact_paths={
            "benchmark_knowledge_subdomain_surface_matrix": (
                "knowledge-subdomain-surface-matrix.json"
            ),
        },
    )

    assert bundle["knowledge_subdomain_surface_matrix_status"].startswith(
        "knowledge_subdomain_surface_matrix_"
    )
    assert (
        bundle["component_statuses"]["knowledge_subdomain_surface_matrix"]
        == bundle["knowledge_subdomain_surface_matrix_status"]
    )
    assert bundle["artifacts"]["benchmark_knowledge_subdomain_surface_matrix"]["present"] is True
    assert companion["knowledge_subdomain_surface_matrix_subdomains"]


def test_build_release_surfaces_include_subdomain_surface_matrix() -> None:
    component = _subdomain_surface_matrix()
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
        benchmark_knowledge_domain_capability_matrix=_capability_matrix(),
        benchmark_knowledge_subdomain_surface_matrix=component,
        artifact_paths={
            "benchmark_knowledge_subdomain_surface_matrix": (
                "knowledge-subdomain-surface-matrix.json"
            ),
        },
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
        benchmark_knowledge_domain_capability_matrix=_capability_matrix(),
        benchmark_knowledge_subdomain_surface_matrix=component,
        artifact_paths={
            "benchmark_knowledge_subdomain_surface_matrix": (
                "knowledge-subdomain-surface-matrix.json"
            ),
        },
    )

    assert decision["knowledge_subdomain_surface_matrix_status"].startswith(
        "knowledge_subdomain_surface_matrix_"
    )
    assert runbook["knowledge_subdomain_surface_matrix_status"].startswith(
        "knowledge_subdomain_surface_matrix_"
    )
