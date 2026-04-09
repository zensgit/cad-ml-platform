from __future__ import annotations

from scripts.ci.check_hybrid_superpass_targets import evaluate_superpass_targets


def _thresholds() -> dict:
    return {
        "min_hybrid_accuracy": 0.60,
        "min_hybrid_gain_vs_graph2d": 0.00,
        "max_calibration_ece": 0.08,
        "missing_mode": "skip",
        "require_real_blind_dataset": True,
        "allowed_blind_dataset_sources": ["configured_dxf_dir"],
    }


def test_superpass_gate_passes_when_all_targets_met() -> None:
    report = evaluate_superpass_targets(
        hybrid_blind_gate_report={
            "input_summary": {"dataset_source": "configured_dxf_dir"},
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
            "input_summary": {"dataset_source": "configured_dxf_dir"},
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


def test_superpass_gate_skips_synthetic_blind_metrics_and_keeps_calibration_active() -> None:
    report = evaluate_superpass_targets(
        hybrid_blind_gate_report={
            "input_summary": {"dataset_source": "synthetic_manifest"},
            "metrics": {
                "hybrid_accuracy": 0.0,
                "hybrid_gain_vs_graph2d": -0.2,
            },
        },
        hybrid_calibration_json={"metrics_after": {"ece": 0.041}},
        thresholds=_thresholds(),
        missing_mode="fail",
    )

    assert report["status"] == "passed"
    assert report["failures"] == []
    assert report["inputs"]["hybrid_blind_dataset_source"] == "synthetic_manifest"
    assert report["inputs"]["hybrid_blind_dataset_qualified"] is False
    assert any("advisory only" in item for item in report["warnings"])
    blind_checks = [item for item in report["checks"] if item["source"] == "hybrid_blind_gate"]
    assert blind_checks
    assert all(item["skipped"] for item in blind_checks)
    calibration_checks = [item for item in report["checks"] if item["source"] == "hybrid_calibration"]
    assert calibration_checks and calibration_checks[0]["passed"] is True
