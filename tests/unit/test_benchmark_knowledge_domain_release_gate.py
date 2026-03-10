from __future__ import annotations

from scripts.export_benchmark_artifact_bundle import build_bundle, render_markdown
from scripts.export_benchmark_companion_summary import build_companion_summary
from scripts.export_benchmark_knowledge_domain_release_gate import build_summary
from src.core.benchmark import (
    build_knowledge_domain_release_gate,
    knowledge_domain_release_gate_recommendations,
)


def _ready_inputs() -> dict:
    return {
        "benchmark_knowledge_domain_capability_matrix": {
            "knowledge_domain_capability_matrix": {
                "status": "knowledge_domain_capability_ready",
                "priority_domains": [],
                "domains": {
                    "tolerance": {"status": "ready"},
                    "gdt": {"status": "ready"},
                },
            }
        },
        "benchmark_knowledge_domain_capability_drift": {
            "knowledge_domain_capability_drift": {"status": "stable"}
        },
        "benchmark_knowledge_domain_action_plan": {
            "knowledge_domain_action_plan": {
                "status": "knowledge_domain_action_plan_ready",
                "priority_domains": [],
                "high_priority_action_count": 0,
                "medium_priority_action_count": 0,
                "recommended_first_actions": [],
            }
        },
        "benchmark_knowledge_domain_control_plane": {
            "knowledge_domain_control_plane": {
                "status": "knowledge_domain_control_plane_ready",
                "release_blockers": [],
                "domains": {
                    "tolerance": {"status": "ready"},
                    "gdt": {"status": "ready"},
                },
            }
        },
        "benchmark_knowledge_domain_control_plane_drift": {
            "knowledge_domain_control_plane_drift": {"status": "stable"}
        },
        "benchmark_knowledge_domain_release_surface_alignment": {
            "knowledge_domain_release_surface_alignment": {
                "status": "aligned",
                "summary": "release surfaces aligned",
                "mismatches": [],
            }
        },
    }


def _blocked_inputs() -> dict:
    return {
        "benchmark_knowledge_domain_capability_matrix": {
            "knowledge_domain_capability_matrix": {
                "status": "knowledge_domain_capability_partial",
                "priority_domains": ["standards"],
                "domains": {
                    "standards": {"status": "blocked"},
                    "tolerance": {"status": "ready"},
                },
            }
        },
        "benchmark_knowledge_domain_capability_drift": {
            "knowledge_domain_capability_drift": {
                "status": "regressed",
                "new_priority_domains": ["standards"],
            }
        },
        "benchmark_knowledge_domain_action_plan": {
            "knowledge_domain_action_plan": {
                "status": "knowledge_domain_action_plan_blocked",
                "priority_domains": ["standards"],
                "high_priority_action_count": 1,
                "medium_priority_action_count": 0,
                "recommended_first_actions": [
                    {
                        "id": "standards:foundation",
                        "domain": "standards",
                        "stage": "foundation",
                        "action": "Backfill standards coverage.",
                    }
                ],
            },
            "recommendations": ["Backfill standards coverage first."],
        },
        "benchmark_knowledge_domain_control_plane": {
            "knowledge_domain_control_plane": {
                "status": "knowledge_domain_control_plane_blocked",
                "release_blockers": ["standards"],
                "domains": {
                    "standards": {"status": "blocked"},
                    "tolerance": {"status": "ready"},
                },
            }
        },
        "benchmark_knowledge_domain_control_plane_drift": {
            "knowledge_domain_control_plane_drift": {
                "status": "regressed",
                "new_release_blockers": ["standards"],
            }
        },
        "benchmark_knowledge_domain_release_surface_alignment": {
            "knowledge_domain_release_surface_alignment": {
                "status": "diverged",
                "summary": "runbook missing standards blocker",
                "mismatches": ["standards:blocked->partial"],
            }
        },
    }


def test_build_release_gate_ready() -> None:
    payload = build_knowledge_domain_release_gate(**_ready_inputs())

    assert payload["status"] == "knowledge_domain_release_gate_ready"
    assert payload["gate_open"] is True
    assert payload["blocked_domains"] == []
    assert payload["releasable_domains"] == ["gdt", "tolerance"]
    assert knowledge_domain_release_gate_recommendations(payload) == [
        "Knowledge-domain release gate is open; keep capability, control-plane, "
        "and release-surface alignment baselines synchronized."
    ]


def test_build_release_gate_blocked() -> None:
    payload = build_knowledge_domain_release_gate(**_blocked_inputs())

    assert payload["status"] == "knowledge_domain_release_gate_blocked"
    assert payload["gate_open"] is False
    assert "standards" in payload["blocked_domains"]
    assert "control_plane:knowledge_domain_control_plane_blocked" in payload[
        "blocking_reasons"
    ]
    assert "release_surface_alignment:diverged" in payload["blocking_reasons"]
    assert payload["recommended_first_action_id"] == "standards:foundation"


def test_export_release_gate_summary() -> None:
    payload = build_summary(
        title="Benchmark Knowledge Domain Release Gate",
        artifact_paths={"benchmark_knowledge_domain_control_plane": "control_plane.json"},
        **_blocked_inputs(),
    )

    assert payload["knowledge_domain_release_gate"]["status"] == (
        "knowledge_domain_release_gate_blocked"
    )
    assert payload["artifacts"]["benchmark_knowledge_domain_control_plane"] == (
        "control_plane.json"
    )


def test_bundle_surfaces_include_release_gate() -> None:
    payload = build_bundle(
        title="Benchmark Artifact Bundle",
        benchmark_scorecard={"components": {}},
        benchmark_operational_summary={},
        benchmark_companion_summary={},
        benchmark_release_decision={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_operator_adoption={},
        benchmark_knowledge_domain_release_gate={
            "knowledge_domain_release_gate": build_knowledge_domain_release_gate(
                **_blocked_inputs()
            )
        },
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={"benchmark_knowledge_domain_release_gate": "release_gate.json"},
    )

    assert payload["component_statuses"]["knowledge_domain_release_gate"] == (
        "knowledge_domain_release_gate_blocked"
    )
    assert payload["knowledge_domain_release_gate_status"] == (
        "knowledge_domain_release_gate_blocked"
    )
    assert payload["artifacts"]["benchmark_knowledge_domain_release_gate"]["present"] is True
    assert "## Knowledge Domain Release Gate" in render_markdown(payload)


def test_companion_surfaces_include_release_gate() -> None:
    payload = build_companion_summary(
        title="Benchmark Companion Summary",
        benchmark_scorecard={"components": {}},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_operator_adoption={},
        benchmark_knowledge_domain_release_gate={
            "knowledge_domain_release_gate": build_knowledge_domain_release_gate(
                **_blocked_inputs()
            )
        },
        artifact_paths={"benchmark_knowledge_domain_release_gate": "release_gate.json"},
    )

    assert payload["component_statuses"]["knowledge_domain_release_gate"] == (
        "knowledge_domain_release_gate_blocked"
    )
    assert payload["knowledge_domain_release_gate_status"] == (
        "knowledge_domain_release_gate_blocked"
    )
    assert payload["knowledge_domain_release_gate_blocked_domains"] == ["standards"]
