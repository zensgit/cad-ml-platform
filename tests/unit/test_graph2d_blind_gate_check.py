from __future__ import annotations


def test_graph2d_blind_gate_passes_with_healthy_metrics() -> None:
    from scripts.ci.check_graph2d_blind_gate import evaluate_blind_gate

    summary = {
        "sampled_files": 120,
        "status_counts": {"ok": 120},
        "accuracy": 0.31,
        "eval_options": {"strip_text_entities": True, "mask_filename": True},
        "pred_labels": {"distinct_canon_count": 9},
        "top_pred_labels_canon": [["传动件", 58], ["人孔", 14]],
        "confidence": {
            "low_conf_threshold": 0.2,
            "low_conf_rate": 0.72,
        },
    }
    thresholds = {
        "min_accuracy": 0.25,
        "max_top_pred_ratio": 0.65,
        "min_distinct_pred_labels": 5,
        "max_low_conf_rate": 0.8,
        "low_conf_threshold": 0.2,
        "require_strip_text_entities": True,
        "require_mask_filename": True,
    }
    report = evaluate_blind_gate(summary, thresholds)
    assert report["status"] == "passed"
    assert report["failures"] == []


def test_graph2d_blind_gate_fails_on_label_collapse_and_low_accuracy() -> None:
    from scripts.ci.check_graph2d_blind_gate import evaluate_blind_gate

    summary = {
        "sampled_files": 30,
        "status_counts": {"ok": 30},
        "accuracy": 0.06,
        "eval_options": {"strip_text_entities": True, "mask_filename": True},
        "pred_labels": {"distinct_canon_count": 1},
        "top_pred_labels_canon": [["传动件", 28]],
        "confidence": {
            "low_conf_threshold": 0.2,
            "low_conf_rate": 0.95,
        },
    }
    thresholds = {
        "min_accuracy": 0.20,
        "max_top_pred_ratio": 0.60,
        "min_distinct_pred_labels": 5,
        "max_low_conf_rate": 0.80,
        "low_conf_threshold": 0.2,
        "require_strip_text_entities": True,
        "require_mask_filename": True,
    }
    report = evaluate_blind_gate(summary, thresholds)
    assert report["status"] == "failed"
    joined = "\n".join(report["failures"])
    assert "accuracy" in joined
    assert "distinct predicted labels" in joined
    assert "top_pred_ratio" in joined
    assert "low_conf_rate" in joined
