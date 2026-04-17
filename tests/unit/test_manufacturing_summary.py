from __future__ import annotations

from src.core.process.manufacturing_summary import build_manufacturing_decision_summary


def test_build_manufacturing_decision_summary_returns_none_when_inputs_empty():
    assert build_manufacturing_decision_summary() is None


def test_build_manufacturing_decision_summary_builds_full_l4_summary():
    result = build_manufacturing_decision_summary(
        quality_payload={
            "mode": "L4_DFM",
            "manufacturability": "high",
            "issues": ["sharp_corner"],
        },
        process_payload={
            "primary_recommendation": {
                "process": "turning",
                "method": "cnc_lathe",
            },
            "alternatives": [],
        },
        cost_payload={
            "total_unit_cost": 12.5,
            "currency": "USD",
        },
    )

    assert result == {
        "feasibility": "high",
        "risks": ["sharp_corner"],
        "process": {
            "process": "turning",
            "method": "cnc_lathe",
        },
        "cost_estimate": {
            "total_unit_cost": 12.5,
            "currency": "USD",
        },
        "cost_range": {
            "low": 11.25,
            "high": 13.75,
        },
        "currency": "USD",
    }


def test_build_manufacturing_decision_summary_falls_back_to_legacy_process_shape():
    result = build_manufacturing_decision_summary(
        quality_payload={
            "manufacturability": "medium",
            "issues": ["thin_wall"],
        },
        process_payload={
            "process": "cnc_milling",
            "method": "standard",
            "rule_version": "rules-v1",
        },
    )

    assert result == {
        "feasibility": "medium",
        "risks": ["thin_wall"],
        "process": {
            "process": "cnc_milling",
            "method": "standard",
        },
        "cost_estimate": {},
        "cost_range": None,
        "currency": None,
    }


def test_build_manufacturing_decision_summary_uses_legacy_process_when_primary_empty():
    result = build_manufacturing_decision_summary(
        process_payload={
            "primary_recommendation": {},
            "process": "laser_cutting",
            "method": "rule_based",
        }
    )

    assert result == {
        "feasibility": None,
        "risks": [],
        "process": {
            "process": "laser_cutting",
            "method": "rule_based",
        },
        "cost_estimate": {},
        "cost_range": None,
        "currency": None,
    }


def test_build_manufacturing_decision_summary_ignores_non_mapping_payloads():
    result = build_manufacturing_decision_summary(
        quality_payload={"issues": []},
        process_payload=None,
        cost_payload=None,
    )

    assert result == {
        "feasibility": None,
        "risks": [],
        "process": None,
        "cost_estimate": {},
        "cost_range": None,
        "currency": None,
    }
