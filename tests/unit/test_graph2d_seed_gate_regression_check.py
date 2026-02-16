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


def test_resolve_current_summary_uses_baseline_when_enabled() -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import _resolve_current_summary

    baseline_channel = {
        "strict_accuracy_mean": 0.36,
        "strict_accuracy_min": 0.29,
        "strict_top_pred_ratio_max": 0.70,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
    }
    summary_payload = {
        "strict_accuracy_mean": 0.1,
        "strict_accuracy_min": 0.1,
        "strict_top_pred_ratio_max": 0.9,
        "strict_low_conf_ratio_max": 0.9,
        "manifest_distinct_labels_min": 1,
    }
    out = _resolve_current_summary(
        use_baseline_as_current=True,
        baseline_channel=baseline_channel,
        summary_payload=summary_payload,
    )
    assert out["strict_accuracy_mean"] == 0.36
    assert out["manifest_distinct_labels_min"] == 5


def test_resolve_baseline_policy_prefers_config_then_cli() -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import _resolve_baseline_policy

    resolved = _resolve_baseline_policy(
        config_payload={
            "max_baseline_age_days": 120,
            "require_snapshot_ref_exists": False,
            "require_snapshot_metrics_match": False,
            "require_integrity_hash_match": False,
            "require_snapshot_date_match": False,
            "require_snapshot_ref_date_match": False,
            "require_context_match": False,
            "context_keys": ["training_profile", "max_samples"],
        },
        cli_overrides={
            "max_baseline_age_days": 30,
            "require_snapshot_ref_exists": "true",
            "require_snapshot_metrics_match": "true",
            "require_integrity_hash_match": "true",
            "require_snapshot_date_match": "true",
            "require_snapshot_ref_date_match": "true",
            "require_context_match": "true",
            "context_keys": "training_profile,manifest_label_mode",
        },
    )
    assert resolved["max_baseline_age_days"] == 30
    assert resolved["require_snapshot_ref_exists"] is True
    assert resolved["require_snapshot_metrics_match"] is True
    assert resolved["require_integrity_hash_match"] is True
    assert resolved["require_snapshot_date_match"] is True
    assert resolved["require_snapshot_ref_date_match"] is True
    assert resolved["require_context_match"] is True
    assert resolved["context_keys"] == ["training_profile", "manifest_label_mode"]


def test_resolve_current_context_uses_baseline_when_enabled() -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import _resolve_current_context

    baseline_channel = {
        "context": {
            "training_profile": "strict_node23_edgesage_v1",
            "manifest_label_mode": "parent_dir",
        }
    }
    summary_payload = {"training_profile": "none", "manifest_label_mode": "filename"}
    out = _resolve_current_context(
        use_baseline_as_current=True,
        baseline_channel=baseline_channel,
        summary_payload=summary_payload,
    )
    assert out["training_profile"] == "strict_node23_edgesage_v1"
    assert out["manifest_label_mode"] == "parent_dir"


def test_regression_check_passes_when_context_matches() -> None:
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
        "context": {
            "training_profile": "none",
            "manifest_label_mode": "parent_dir",
            "max_samples": 120,
            "min_label_confidence": 0.0,
            "strict_low_conf_threshold": 0.2,
        },
    }
    current_context = {
        "training_profile": "none",
        "manifest_label_mode": "parent_dir",
        "max_samples": 120,
        "min_label_confidence": 0.0,
        "strict_low_conf_threshold": 0.2,
    }
    report = evaluate_regression(
        summary=summary,
        current_context=current_context,
        baseline_channel=baseline,
        channel="standard",
        max_accuracy_mean_drop=0.02,
        max_accuracy_min_drop=0.02,
        max_top_pred_ratio_increase=0.03,
        max_low_conf_ratio_increase=0.01,
        max_distinct_labels_drop=0,
        require_context_match=True,
        context_keys=[
            "training_profile",
            "manifest_label_mode",
            "max_samples",
            "min_label_confidence",
            "strict_low_conf_threshold",
        ],
    )
    assert report["status"] == "passed"
    assert report["baseline_metadata"]["context_match"] is True
    assert report["baseline_metadata"]["context_diff"] == {}


def test_regression_check_fails_when_context_mismatch() -> None:
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
        "context": {
            "training_profile": "strict_node23_edgesage_v1",
            "manifest_label_mode": "parent_dir",
            "max_samples": 120,
            "min_label_confidence": 0.0,
            "strict_low_conf_threshold": 0.2,
        },
    }
    current_context = {
        "training_profile": "none",
        "manifest_label_mode": "filename",
        "max_samples": 120,
        "min_label_confidence": 0.0,
        "strict_low_conf_threshold": 0.25,
    }
    report = evaluate_regression(
        summary=summary,
        current_context=current_context,
        baseline_channel=baseline,
        channel="standard",
        max_accuracy_mean_drop=0.02,
        max_accuracy_min_drop=0.02,
        max_top_pred_ratio_increase=0.03,
        max_low_conf_ratio_increase=0.01,
        max_distinct_labels_drop=0,
        require_context_match=True,
        context_keys=[
            "training_profile",
            "manifest_label_mode",
            "max_samples",
            "min_label_confidence",
            "strict_low_conf_threshold",
        ],
    )
    assert report["status"] == "failed"
    failures = "\n".join(report["failures"])
    assert "context: mismatch on keys" in failures
    context_diff = report["baseline_metadata"]["context_diff"]
    assert "training_profile" in context_diff
    assert "manifest_label_mode" in context_diff
    assert "strict_low_conf_threshold" in context_diff


def test_regression_check_fails_when_baseline_is_stale() -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import evaluate_regression

    summary = {
        "strict_accuracy_mean": 0.36,
        "strict_accuracy_min": 0.29,
        "strict_top_pred_ratio_max": 0.70,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
    }
    baseline = {
        "date": "2000-01-01",
        "source": {
            "snapshot_ref": "reports/experiments/20260215/graph2d_seed_gate_baseline_snapshot_20260215.json"
        },
        "standard": {
            "strict_accuracy_mean": 0.36,
            "strict_accuracy_min": 0.29,
            "strict_top_pred_ratio_max": 0.70,
            "strict_low_conf_ratio_max": 0.05,
            "manifest_distinct_labels_min": 5,
        },
    }
    report = evaluate_regression(
        summary=summary,
        baseline_channel=baseline["standard"],
        channel="standard",
        max_accuracy_mean_drop=0.02,
        max_accuracy_min_drop=0.02,
        max_top_pred_ratio_increase=0.03,
        max_low_conf_ratio_increase=0.01,
        max_distinct_labels_drop=0,
        baseline_payload=baseline,
        baseline_json_path="config/graph2d_seed_gate_baseline.json",
        max_baseline_age_days=30,
        require_snapshot_ref_exists=False,
        require_snapshot_metrics_match=False,
        require_integrity_hash_match=False,
        require_snapshot_date_match=False,
        require_snapshot_ref_date_match=False,
    )
    assert report["status"] == "failed"
    failures = "\n".join(report["failures"])
    assert "baseline_date: age_days=" in failures


def test_regression_check_fails_when_snapshot_ref_missing_required() -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import evaluate_regression

    summary = {
        "strict_accuracy_mean": 0.36,
        "strict_accuracy_min": 0.29,
        "strict_top_pred_ratio_max": 0.70,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
    }
    baseline = {
        "date": "2026-02-15",
        "source": {},
        "standard": {
            "strict_accuracy_mean": 0.36,
            "strict_accuracy_min": 0.29,
            "strict_top_pred_ratio_max": 0.70,
            "strict_low_conf_ratio_max": 0.05,
            "manifest_distinct_labels_min": 5,
        },
    }
    report = evaluate_regression(
        summary=summary,
        baseline_channel=baseline["standard"],
        channel="standard",
        max_accuracy_mean_drop=0.02,
        max_accuracy_min_drop=0.02,
        max_top_pred_ratio_increase=0.03,
        max_low_conf_ratio_increase=0.01,
        max_distinct_labels_drop=0,
        baseline_payload=baseline,
        baseline_json_path="config/graph2d_seed_gate_baseline.json",
        max_baseline_age_days=365,
        require_snapshot_ref_exists=True,
        require_snapshot_metrics_match=False,
        require_integrity_hash_match=False,
        require_snapshot_date_match=False,
        require_snapshot_ref_date_match=False,
    )
    assert report["status"] == "failed"
    failures = "\n".join(report["failures"])
    assert "snapshot_ref: missing in baseline source" in failures


def test_regression_check_fails_when_snapshot_metrics_mismatch(tmp_path) -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import evaluate_regression

    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        "{\n"
        '  "standard": {\n'
        '    "strict_accuracy_mean": 0.99,\n'
        '    "strict_accuracy_min": 0.99,\n'
        '    "strict_top_pred_ratio_max": 0.1,\n'
        '    "strict_low_conf_ratio_max": 0.01,\n'
        '    "manifest_distinct_labels_min": 9\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    summary = {
        "strict_accuracy_mean": 0.36,
        "strict_accuracy_min": 0.29,
        "strict_top_pred_ratio_max": 0.70,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
    }
    baseline = {
        "date": "2026-02-15",
        "source": {"snapshot_ref": str(snapshot)},
        "standard": {
            "strict_accuracy_mean": 0.36,
            "strict_accuracy_min": 0.29,
            "strict_top_pred_ratio_max": 0.70,
            "strict_low_conf_ratio_max": 0.05,
            "manifest_distinct_labels_min": 5,
        },
    }
    report = evaluate_regression(
        summary=summary,
        baseline_channel=baseline["standard"],
        channel="standard",
        max_accuracy_mean_drop=0.02,
        max_accuracy_min_drop=0.02,
        max_top_pred_ratio_increase=0.03,
        max_low_conf_ratio_increase=0.01,
        max_distinct_labels_drop=0,
        baseline_payload=baseline,
        baseline_json_path=str(tmp_path / "baseline.json"),
        max_baseline_age_days=365,
        require_snapshot_ref_exists=True,
        require_snapshot_metrics_match=True,
        require_integrity_hash_match=False,
        require_snapshot_date_match=False,
        require_snapshot_ref_date_match=False,
    )
    assert report["status"] == "failed"
    failures = "\n".join(report["failures"])
    assert "snapshot_metrics: channel 'standard' differs from baseline" in failures
    assert report["baseline_metadata"]["snapshot_metrics_match"] is False
    assert report["baseline_metadata"]["snapshot_metrics_diff"]


def test_regression_check_fails_when_integrity_hash_mismatch(tmp_path) -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import evaluate_regression

    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        "{\n"
        '  "integrity": {\n'
        '    "standard_channel_sha256": "deadbeef"\n'
        "  },\n"
        '  "standard": {\n'
        '    "strict_accuracy_mean": 0.36,\n'
        '    "strict_accuracy_min": 0.29,\n'
        '    "strict_top_pred_ratio_max": 0.70,\n'
        '    "strict_low_conf_ratio_max": 0.05,\n'
        '    "manifest_distinct_labels_min": 5\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    summary = {
        "strict_accuracy_mean": 0.36,
        "strict_accuracy_min": 0.29,
        "strict_top_pred_ratio_max": 0.70,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
    }
    baseline = {
        "date": "2026-02-15",
        "source": {"snapshot_ref": str(snapshot)},
        "integrity": {
            "standard_channel_sha256": "badbadbad",
            "payload_core_sha256": "badcore",
        },
        "standard": {
            "strict_accuracy_mean": 0.36,
            "strict_accuracy_min": 0.29,
            "strict_top_pred_ratio_max": 0.70,
            "strict_low_conf_ratio_max": 0.05,
            "manifest_distinct_labels_min": 5,
        },
        "strict": {},
    }
    report = evaluate_regression(
        summary=summary,
        baseline_channel=baseline["standard"],
        channel="standard",
        max_accuracy_mean_drop=0.02,
        max_accuracy_min_drop=0.02,
        max_top_pred_ratio_increase=0.03,
        max_low_conf_ratio_increase=0.01,
        max_distinct_labels_drop=0,
        baseline_payload=baseline,
        baseline_json_path=str(tmp_path / "baseline.json"),
        max_baseline_age_days=365,
        require_snapshot_ref_exists=True,
        require_snapshot_metrics_match=False,
        require_integrity_hash_match=True,
        require_snapshot_date_match=False,
        require_snapshot_ref_date_match=False,
    )
    assert report["status"] == "failed"
    failures = "\n".join(report["failures"])
    assert "integrity: baseline standard_channel_sha256 mismatch" in failures


def test_regression_check_fails_when_snapshot_date_mismatch(tmp_path) -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import evaluate_regression

    snapshot = tmp_path / "graph2d_seed_gate_baseline_snapshot_20260216.json"
    snapshot.write_text(
        "{\n"
        '  "date": "2026-02-15",\n'
        '  "standard": {\n'
        '    "strict_accuracy_mean": 0.36,\n'
        '    "strict_accuracy_min": 0.29,\n'
        '    "strict_top_pred_ratio_max": 0.70,\n'
        '    "strict_low_conf_ratio_max": 0.05,\n'
        '    "manifest_distinct_labels_min": 5\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    summary = {
        "strict_accuracy_mean": 0.36,
        "strict_accuracy_min": 0.29,
        "strict_top_pred_ratio_max": 0.70,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
    }
    baseline = {
        "date": "2026-02-16",
        "source": {"snapshot_ref": str(snapshot)},
        "standard": {
            "strict_accuracy_mean": 0.36,
            "strict_accuracy_min": 0.29,
            "strict_top_pred_ratio_max": 0.70,
            "strict_low_conf_ratio_max": 0.05,
            "manifest_distinct_labels_min": 5,
        },
    }
    report = evaluate_regression(
        summary=summary,
        baseline_channel=baseline["standard"],
        channel="standard",
        max_accuracy_mean_drop=0.02,
        max_accuracy_min_drop=0.02,
        max_top_pred_ratio_increase=0.03,
        max_low_conf_ratio_increase=0.01,
        max_distinct_labels_drop=0,
        baseline_payload=baseline,
        baseline_json_path=str(tmp_path / "baseline.json"),
        max_baseline_age_days=365,
        require_snapshot_ref_exists=True,
        require_snapshot_metrics_match=False,
        require_integrity_hash_match=False,
        require_snapshot_date_match=True,
        require_snapshot_ref_date_match=False,
    )
    assert report["status"] == "failed"
    failures = "\n".join(report["failures"])
    assert "snapshot_date: snapshot date '2026-02-15' != baseline date '2026-02-16'" in failures


def test_regression_check_fails_when_snapshot_ref_date_mismatch(tmp_path) -> None:
    from scripts.ci.check_graph2d_seed_gate_regression import evaluate_regression

    snapshot = tmp_path / "graph2d_seed_gate_baseline_snapshot_20260215.json"
    snapshot.write_text("{\"date\":\"2026-02-16\", \"standard\": {}}", encoding="utf-8")
    summary = {
        "strict_accuracy_mean": 0.36,
        "strict_accuracy_min": 0.29,
        "strict_top_pred_ratio_max": 0.70,
        "strict_low_conf_ratio_max": 0.05,
        "manifest_distinct_labels_min": 5,
    }
    baseline = {
        "date": "2026-02-16",
        "source": {"snapshot_ref": str(snapshot)},
        "standard": {
            "strict_accuracy_mean": 0.36,
            "strict_accuracy_min": 0.29,
            "strict_top_pred_ratio_max": 0.70,
            "strict_low_conf_ratio_max": 0.05,
            "manifest_distinct_labels_min": 5,
        },
    }
    report = evaluate_regression(
        summary=summary,
        baseline_channel=baseline["standard"],
        channel="standard",
        max_accuracy_mean_drop=0.02,
        max_accuracy_min_drop=0.02,
        max_top_pred_ratio_increase=0.03,
        max_low_conf_ratio_increase=0.01,
        max_distinct_labels_drop=0,
        baseline_payload=baseline,
        baseline_json_path=str(tmp_path / "baseline.json"),
        max_baseline_age_days=365,
        require_snapshot_ref_exists=True,
        require_snapshot_metrics_match=False,
        require_integrity_hash_match=False,
        require_snapshot_date_match=False,
        require_snapshot_ref_date_match=True,
    )
    assert report["status"] == "failed"
    failures = "\n".join(report["failures"])
    assert "snapshot_ref_date: snapshot_ref stamp '20260215' != baseline stamp '20260216'" in failures
