from __future__ import annotations

from scripts.export_benchmark_artifact_bundle import build_bundle
from scripts.export_benchmark_companion_summary import build_companion_summary
from scripts.export_benchmark_release_decision import build_release_decision
from scripts.export_benchmark_release_runbook import build_release_runbook


def _knowledge_readiness() -> dict:
    return {
        "knowledge_readiness": {
            "status": "knowledge_foundation_partial",
            "domains": {
                "tolerance": {
                    "status": "partial",
                    "focus_components": ["tolerance"],
                    "missing_metrics": ["common_fit_count"],
                },
                "standards": {
                    "status": "ready",
                    "focus_components": [],
                    "missing_metrics": [],
                },
                "gdt": {
                    "status": "ready",
                    "focus_components": [],
                    "missing_metrics": [],
                },
            },
            "domain_focus_areas": [
                {
                    "domain": "tolerance",
                    "status": "partial",
                    "priority": "medium",
                    "action": "Backfill tolerance coverage.",
                }
            ],
            "priority_domains": ["tolerance"],
            "focus_areas_detail": [
                {
                    "component": "tolerance",
                    "status": "partial",
                    "priority": "medium",
                    "action": "Backfill tolerance coverage.",
                }
            ],
        },
        "recommendations": ["Backfill tolerance coverage."],
    }


def _knowledge_application() -> dict:
    return {
        "knowledge_application": {
            "status": "knowledge_application_partial",
            "domains": {
                "tolerance": {
                    "status": "partial",
                    "readiness_status": "partial",
                    "evidence_status": "partial",
                    "signal_count": 1,
                }
            },
            "priority_domains": ["tolerance"],
            "focus_areas_detail": [
                {
                    "domain": "tolerance",
                    "status": "partial",
                    "priority": "medium",
                    "action": "Raise tolerance application coverage.",
                }
            ],
        },
        "recommendations": ["Raise tolerance application coverage."],
    }


def _knowledge_realdata_correlation() -> dict:
    return {
        "knowledge_realdata_correlation": {
            "status": "knowledge_realdata_partial",
            "domains": {
                "tolerance": {
                    "status": "partial",
                    "readiness_status": "partial",
                    "application_status": "partial",
                    "realdata_status": "partial",
                }
            },
            "priority_domains": ["tolerance"],
            "focus_areas_detail": [
                {
                    "domain": "tolerance",
                    "status": "partial",
                    "priority": "medium",
                    "action": "Raise tolerance real-data depth.",
                }
            ],
        },
        "recommendations": ["Raise tolerance real-data depth."],
    }


def _realdata() -> dict:
    return {
        "realdata_signals": {
            "status": "realdata_foundation_partial",
            "components": {
                "hybrid_dxf": {"status": "ready", "sample_size": 10},
                "history_h5": {"status": "partial", "sample_size": 1},
                "step_smoke": {"status": "environment_blocked", "sample_size": 0},
                "step_dir": {"status": "partial", "sample_size": 3},
            },
            "component_statuses": {
                "hybrid_dxf": "ready",
                "history_h5": "partial",
                "step_smoke": "environment_blocked",
                "step_dir": "partial",
            },
        },
        "recommendations": ["Expand STEP/B-Rep directory validation."],
    }


def _operator_adoption() -> dict:
    return {
        "adoption_readiness": "guided_manual",
        "recommended_actions": ["Operator fallback only."],
    }


def test_build_companion_summary_includes_knowledge_realdata_correlation() -> None:
    payload = build_companion_summary(
        title="Companion",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={"component_statuses": {"review_queue": "healthy"}},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness=_knowledge_readiness(),
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"}
        },
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        artifact_paths={},
        benchmark_knowledge_application=_knowledge_application(),
        benchmark_knowledge_realdata_correlation=_knowledge_realdata_correlation(),
    )

    assert payload["knowledge_realdata_correlation_status"] == "knowledge_realdata_partial"
    assert payload["knowledge_realdata_correlation_domains"]["tolerance"]["realdata_status"] == (
        "partial"
    )


def test_build_bundle_includes_knowledge_realdata_correlation() -> None:
    payload = build_bundle(
        title="Bundle",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={"component_statuses": {}},
        benchmark_companion_summary={},
        benchmark_release_decision={},
        benchmark_knowledge_readiness=_knowledge_readiness(),
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"}
        },
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        benchmark_knowledge_application=_knowledge_application(),
        benchmark_knowledge_realdata_correlation=_knowledge_realdata_correlation(),
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={},
    )

    assert payload["knowledge_realdata_correlation_status"] == "knowledge_realdata_partial"
    assert payload["component_statuses"]["knowledge_realdata_correlation"] == (
        "knowledge_realdata_partial"
    )


def test_build_release_decision_includes_knowledge_realdata_correlation() -> None:
    payload = build_release_decision(
        title="Release",
        benchmark_scorecard={"components": {"hybrid": {"status": "healthy"}}},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_companion_summary={},
        benchmark_knowledge_readiness=_knowledge_readiness(),
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"}
        },
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        artifact_paths={},
        benchmark_knowledge_application=_knowledge_application(),
        benchmark_knowledge_realdata_correlation=_knowledge_realdata_correlation(),
    )

    assert payload["knowledge_realdata_correlation_status"] == "knowledge_realdata_partial"
    assert "Raise tolerance real-data depth." in payload["review_signals"]


def test_build_release_runbook_includes_knowledge_realdata_correlation() -> None:
    payload = build_release_runbook(
        title="Runbook",
        benchmark_release_decision={},
        benchmark_companion_summary={},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness=_knowledge_readiness(),
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={
            "engineering_signals": {"status": "engineering_semantics_ready"}
        },
        benchmark_realdata_signals=_realdata(),
        benchmark_operator_adoption=_operator_adoption(),
        benchmark_knowledge_application=_knowledge_application(),
        benchmark_knowledge_realdata_correlation=_knowledge_realdata_correlation(),
        artifact_paths={},
    )

    assert payload["knowledge_realdata_correlation_status"] == "knowledge_realdata_partial"
    assert payload["knowledge_realdata_correlation_domains"]["tolerance"]["status"] == (
        "partial"
    )
