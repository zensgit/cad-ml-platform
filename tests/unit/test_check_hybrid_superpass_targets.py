from __future__ import annotations

from scripts.ci.check_hybrid_superpass_targets import evaluate_superpass_targets


def _thresholds() -> dict:
    return {
        "min_hybrid_accuracy": 0.60,
        "min_hybrid_gain_vs_graph2d": 0.00,
        "max_calibration_ece": 0.08,
        "missing_mode": "skip",
    }


def test_superpass_gate_passes_when_all_targets_met() -> None:
    report = evaluate_superpass_targets(
        hybrid_blind_gate_report={
            "metrics": {
                "hybrid_accuracy": 0.72,
                "hybrid_gain_vs_graph2d": 0.11,
            }
        },
        hybrid_calibration_json={"metrics_after": {"ece": 0.041}},
        thresholds=_thresholds(),
        missing_mode="skip",
    )

    assert report["status"] == "passed"
    assert report["failures"] == []
    assert all(item["passed"] for item in report["checks"])


def test_superpass_gate_fails_when_target_below_threshold() -> None:
    report = evaluate_superpass_targets(
        hybrid_blind_gate_report={
            "metrics": {
                "hybrid_accuracy": 0.52,
                "hybrid_gain_vs_graph2d": -0.03,
            }
        },
        hybrid_calibration_json={"metrics_after": {"ece": 0.12}},
        thresholds=_thresholds(),
        missing_mode="skip",
    )

    assert report["status"] == "failed"
    assert len(report["failures"]) >= 3
    assert any("hybrid_accuracy" in item for item in report["failures"])
    assert any("hybrid_gain_vs_graph2d" in item for item in report["failures"])
    assert any("calibration_ece" in item for item in report["failures"])


def test_superpass_gate_missing_inputs_skip_mode_warns_but_passes() -> None:
    report = evaluate_superpass_targets(
        hybrid_blind_gate_report=None,
        hybrid_calibration_json=None,
        thresholds=_thresholds(),
        missing_mode="skip",
    )

    assert report["status"] == "passed"
    assert report["failures"] == []
    assert len(report["warnings"]) >= 2
    assert any(item["skipped"] for item in report["checks"])


def test_superpass_gate_missing_inputs_fail_mode_fails() -> None:
    report = evaluate_superpass_targets(
        hybrid_blind_gate_report=None,
        hybrid_calibration_json=None,
        thresholds=_thresholds(),
        missing_mode="fail",
    )

    assert report["status"] == "failed"
    assert len(report["failures"]) >= 2
    assert any(not item["passed"] for item in report["checks"])
