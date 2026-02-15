from __future__ import annotations


def test_regression_summary_includes_key_rows() -> None:
    from scripts.ci.summarize_graph2d_seed_gate_regression import build_summary

    report = {
        "channel": "standard",
        "status": "passed",
        "failures": [],
        "thresholds": {
            "max_accuracy_mean_drop": 0.08,
            "max_accuracy_min_drop": 0.08,
            "max_top_pred_ratio_increase": 0.10,
            "max_low_conf_ratio_increase": 0.05,
            "max_distinct_labels_drop": 0,
        },
        "baseline": {
            "strict_accuracy_mean": 0.3625,
            "strict_accuracy_min": 0.291667,
            "strict_top_pred_ratio_max": 0.708333,
            "strict_low_conf_ratio_max": 0.05,
            "manifest_distinct_labels_min": 5,
        },
        "current": {
            "strict_accuracy_mean": 0.3625,
            "strict_accuracy_min": 0.291667,
            "strict_top_pred_ratio_max": 0.708333,
            "strict_low_conf_ratio_max": 0.05,
            "manifest_distinct_labels_min": 5,
        },
    }
    text = build_summary(report, "Graph2D Seed Gate Regression Check")
    assert "Graph2D Seed Gate Regression Check" in text
    assert "Regression status" in text
    assert "`passed`" in text
    assert "strict_top_pred_ratio_max (cur/base)" in text
    assert "strict_low_conf_ratio_max (cur/base)" in text


def test_regression_summary_prints_failures() -> None:
    from scripts.ci.summarize_graph2d_seed_gate_regression import build_summary

    report = {
        "channel": "strict",
        "status": "failed",
        "failures": ["strict_accuracy_mean: current=0.100000 < allowed=0.200000"],
    }
    text = build_summary(report, "Graph2D Seed Gate Regression Check")
    assert "Regression failures:" in text
    assert "strict_accuracy_mean: current=0.100000 < allowed=0.200000" in text

