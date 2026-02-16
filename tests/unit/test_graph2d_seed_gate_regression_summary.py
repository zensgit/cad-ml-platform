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
            "max_baseline_age_days": 365,
            "require_snapshot_ref_exists": True,
            "require_snapshot_metrics_match": True,
            "require_integrity_hash_match": True,
            "require_snapshot_date_match": True,
            "require_snapshot_ref_date_match": True,
            "require_context_match": True,
            "context_keys": [
                "training_profile",
                "manifest_label_mode",
                "max_samples",
            ],
        },
        "threshold_source": {
            "config": "config/graph2d_seed_gate_regression.yaml",
            "config_loaded": True,
            "current_source": "/tmp/graph2d-seed-gate/seed_sweep_summary.json",
            "cli_overrides": {},
        },
        "baseline_metadata": {
            "date": "2026-02-15",
            "age_days": 0,
            "snapshot_ref": "reports/experiments/20260215/graph2d_seed_gate_baseline_snapshot_20260215.json",
            "snapshot_path": "/tmp/repo/reports/experiments/20260215/graph2d_seed_gate_baseline_snapshot_20260215.json",
            "snapshot_exists": True,
            "snapshot_channel_present": True,
            "snapshot_metrics_match": True,
            "snapshot_metrics_diff": {},
            "baseline_channel_hash_match": True,
            "snapshot_channel_hash_match": True,
            "snapshot_vs_baseline_hash_match": True,
            "baseline_core_hash_match": True,
            "snapshot_payload_date": "2026-02-16",
            "snapshot_date_match": True,
            "snapshot_ref_date_stamp": "20260216",
            "expected_date_stamp": "20260216",
            "snapshot_ref_date_match": True,
            "context_match": True,
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
    assert "Threshold source" in text
    assert "config/graph2d_seed_gate_regression.yaml" in text
    assert "/tmp/graph2d-seed-gate/seed_sweep_summary.json" in text
    assert "Baseline metadata" in text
    assert "snapshot_exists=True" in text
    assert "snapshot_match=True" in text
    assert "snapshot_metrics_match=True" in text
    assert "integrity_match=True" in text
    assert "snapshot_vs_baseline_hash_match=True" in text
    assert "snapshot_date_match=True" in text
    assert "snapshot_ref_date_match=True" in text
    assert "context_match=True" in text
    assert "context_keys=3" in text


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
