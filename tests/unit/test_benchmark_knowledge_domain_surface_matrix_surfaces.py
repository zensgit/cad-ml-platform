from __future__ import annotations

from scripts.export_benchmark_artifact_bundle import build_bundle
from scripts.export_benchmark_companion_summary import build_companion_summary
from scripts.export_benchmark_knowledge_domain_surface_matrix import build_summary
from scripts.export_benchmark_release_decision import build_release_decision
from scripts.export_benchmark_release_runbook import build_release_runbook


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
        },
        "recommendations": ["Backfill standards readiness."],
    }


def _knowledge_application() -> dict:
    return {
        "knowledge_application": {
            "status": "knowledge_application_partial",
            "domains": {
                "tolerance": {"status": "ready", "signal_count": 3},
                "standards": {"status": "partial", "signal_count": 2},
                "gdt": {"status": "blocked", "signal_count": 0},
            },
            "priority_domains": ["gdt"],
        },
        "recommendations": ["Raise GD&T application evidence."],
    }


def _knowledge_domain_matrix() -> dict:
    return {
        "knowledge_domain_matrix": {
            "status": "knowledge_domain_matrix_partial",
            "domains": {
                "tolerance": {"status": "ready"},
                "standards": {"status": "partial"},
                "gdt": {"status": "blocked"},
            },
            "priority_domains": ["gdt"],
        },
        "recommendations": ["Promote GD&T into benchmark surfaces."],
    }


def _capability_matrix() -> dict:
    return {
        "knowledge_domain_capability_matrix": {
            "status": "knowledge_domain_capability_partial",
            "domains": {
                "tolerance": {"status": "ready"},
                "standards": {"status": "ready"},
                "gdt": {"status": "blocked"},
            },
            "priority_domains": ["gdt"],
        },
        "recommendations": ["Add GD&T provider and public benchmark surface."],
    }


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


def _surface_matrix() -> dict:
    return build_summary(title="Knowledge Domain Surface Matrix")


def test_build_bundle_and_companion_include_surface_matrix_passthrough() -> None:
    surface_matrix = _surface_matrix()
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
        benchmark_knowledge_domain_surface_matrix=surface_matrix,
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={
            "benchmark_knowledge_domain_surface_matrix": "surface-matrix.json",
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
        benchmark_knowledge_domain_surface_matrix=surface_matrix,
        artifact_paths={
            "benchmark_knowledge_domain_surface_matrix": "surface-matrix.json",
        },
    )

    assert bundle["knowledge_domain_surface_matrix_status"] in {
        "knowledge_domain_surface_matrix_partial",
        "knowledge_domain_surface_matrix_blocked",
    }
    assert bundle["component_statuses"]["knowledge_domain_surface_matrix"] in {
        "knowledge_domain_surface_matrix_partial",
        "knowledge_domain_surface_matrix_blocked",
    }
    assert bundle["artifacts"]["benchmark_knowledge_domain_surface_matrix"]["present"] is True
    assert companion["knowledge_domain_surface_matrix_domains"]["gdt"]["status"] == "partial"
    assert companion["knowledge_domain_surface_matrix_domains"]["gdt"][
        "subcapabilities"
    ]["gdt_public_api"]["present_route_count"] == 0
    assert any(
        "GD&T" in item or "gdt" in item.lower()
        for item in companion["knowledge_domain_surface_matrix_recommendations"]
    )


def test_build_release_surfaces_include_surface_matrix_review_signals() -> None:
    surface_matrix = _surface_matrix()
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
        benchmark_knowledge_domain_surface_matrix=surface_matrix,
        artifact_paths={
            "benchmark_knowledge_domain_surface_matrix": "surface-matrix.json",
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
        benchmark_knowledge_domain_surface_matrix=surface_matrix,
        artifact_paths={
            "benchmark_knowledge_domain_surface_matrix": "surface-matrix.json",
        },
    )

    assert decision["knowledge_domain_surface_matrix_domains"]["gdt"]["status"] == "partial"
    assert any(
        "GD&T" in item or "gdt" in item.lower() for item in decision["review_signals"]
    )
    assert runbook["knowledge_domain_surface_matrix_domains"]["gdt"]["status"] == "partial"
    assert any(
        "GD&T" in item or "gdt" in item.lower() for item in runbook["review_signals"]
    )
