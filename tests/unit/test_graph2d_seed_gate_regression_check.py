from __future__ import annotations


def test_regression_check_passes_within_thresholds() -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import evaluate_regression

    summary = {
        "strict_accuracy_mean": 0.36,
        "strict_accuracy_min": 0.29,
        "strict_top_pred_ratio_max": 0.72,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
    }
    baseline = {
        "strict_accuracy_mean": 0.37,
        "strict_accuracy_min": 0.30,
        "strict_top_pred_ratio_max": 0.70,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
    }
    report = evaluate_regression(
        summary=summary,
        baseline_channel=baseline,
        channel="standard",
        max_accuracy_mean_drop=0.02,
        max_accuracy_min_drop=0.02,
        max_top_pred_ratio_increase=0.03,
        max_low_conf_ratio_increase=0.01,
        max_distinct_labels_drop=0,
    )
    assert report["status"] == "passed"
    assert report["failures"] == []


def test_regression_check_fails_when_metrics_regress() -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import evaluate_regression

    summary = {
        "strict_accuracy_mean": 0.22,
        "strict_accuracy_min": 0.10,
        "strict_top_pred_ratio_max": 0.90,
        "strict_low_conf_ratio_max": 0.20,
        "manifest_distinct_labels_min": 3,
    }
    baseline = {
        "strict_accuracy_mean": 0.36,
        "strict_accuracy_min": 0.29,
        "strict_top_pred_ratio_max": 0.70,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
    }
    report = evaluate_regression(
        summary=summary,
        baseline_channel=baseline,
        channel="standard",
        max_accuracy_mean_drop=0.02,
        max_accuracy_min_drop=0.02,
        max_top_pred_ratio_increase=0.03,
        max_low_conf_ratio_increase=0.01,
        max_distinct_labels_drop=0,
    )
    assert report["status"] == "failed"
    failures = "\n".join(report["failures"])
    assert "strict_accuracy_mean" in failures
    assert "strict_accuracy_min" in failures
    assert "strict_top_pred_ratio_max" in failures
    assert "strict_low_conf_ratio_max" in failures
    assert "manifest_distinct_labels_min" in failures


def test_resolve_thresholds_uses_channel_config_and_cli_override() -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import _resolve_thresholds

    resolved = _resolve_thresholds(
        channel="strict",
        config_payload={
            "max_accuracy_mean_drop": 0.10,
            "channels": {
                "strict": {
                    "max_accuracy_mean_drop": 0.04,
                    "max_top_pred_ratio_increase": 0.06,
                }
            },
        },
        cli_overrides={
            "max_accuracy_mean_drop": None,
            "max_accuracy_min_drop": None,
            "max_top_pred_ratio_increase": 0.02,
            "max_low_conf_ratio_increase": None,
            "max_distinct_labels_drop": None,
        },
    )
    assert resolved["max_accuracy_mean_drop"] == 0.04
    assert resolved["max_top_pred_ratio_increase"] == 0.02
    assert resolved["max_low_conf_ratio_increase"] == 0.05
