from __future__ import annotations


def test_build_baseline_contains_expected_metrics() -> None:
    from scripts.ci.update_graph2d_seed_gate_baseline import build_baseline

    standard = {
        "config": "config/graph2d_seed_gate.yaml",
        "training_profile": "none",
        "manifest_label_mode": "parent_dir",
        "seeds": [7, 21],
        "num_runs": 2,
        "max_samples": 120,
        "min_label_confidence": 0.0,
        "force_normalize_labels": "auto",
        "force_clean_min_count": -1,
        "strict_accuracy_mean": 0.36,
        "strict_accuracy_min": 0.29,
        "strict_accuracy_max": 0.43,
        "strict_top_pred_ratio_mean": 0.60,
        "strict_top_pred_ratio_max": 0.70,
        "strict_low_conf_threshold": 0.2,
        "strict_low_conf_ratio_mean": 0.05,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
        "manifest_distinct_labels_max": 5,
        "gate": {"passed": True},
    }
    strict = {
        "config": "config/graph2d_seed_gate_strict.yaml",
        "training_profile": "strict_node23_edgesage_v1",
        "manifest_label_mode": "parent_dir",
        "seeds": [7, 21],
        "num_runs": 2,
        "max_samples": 120,
        "min_label_confidence": 0.0,
        "force_normalize_labels": "false",
        "force_clean_min_count": 0,
        "strict_accuracy_mean": 0.95,
        "strict_accuracy_min": 0.94,
        "strict_accuracy_max": 0.95,
        "strict_top_pred_ratio_mean": 0.25,
        "strict_top_pred_ratio_max": 0.27,
        "strict_low_conf_threshold": 0.2,
        "strict_low_conf_ratio_mean": 0.05,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
        "manifest_distinct_labels_max": 5,
        "gate": {"passed": True},
    }
    payload = build_baseline(
        standard_summary=standard,
        strict_summary=strict,
        standard_summary_path="/tmp/std.json",
        strict_summary_path="/tmp/strict.json",
        snapshot_ref="reports/experiments/20260215/snap.json",
    )
    assert payload["standard"]["strict_accuracy_mean"] == 0.36
    assert payload["standard"]["gate_passed"] is True
    assert payload["standard"]["context"]["training_profile"] == "none"
    assert payload["standard"]["context"]["seeds"] == [7, 21]
    assert payload["strict"]["strict_top_pred_ratio_max"] == 0.27
    assert payload["strict"]["context"]["training_profile"] == "strict_node23_edgesage_v1"
    assert payload["strict"]["context"]["force_normalize_labels"] == "false"
    assert payload["source"]["snapshot_ref"].endswith("snap.json")
    integrity = payload.get("integrity") or {}
    assert integrity.get("algorithm") == "sha256-canonical-json"
    assert isinstance(integrity.get("standard_channel_sha256"), str)
    assert len(integrity.get("standard_channel_sha256") or "") == 64
    assert isinstance(integrity.get("strict_channel_sha256"), str)
    assert len(integrity.get("strict_channel_sha256") or "") == 64
    assert isinstance(integrity.get("payload_core_sha256"), str)
    assert len(integrity.get("payload_core_sha256") or "") == 64
