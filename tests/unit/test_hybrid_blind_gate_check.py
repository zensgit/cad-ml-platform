from __future__ import annotations


def test_hybrid_blind_gate_passes_with_gain_and_coverage() -> None:
    from scripts.ci.check_hybrid_blind_gate import evaluate_hybrid_blind_gate

    summary = {
        "geometry_only": True,
        "mask_filename": True,
        "strip_text": True,
        "sample_size": 120,
        "weak_labels": {
            "covered_count": 108,
            "covered_rate": 0.9,
            "accuracy": {
                "hybrid_label": {"evaluated": 108, "correct": 49, "missing_pred": 10, "accuracy": 0.4537},
                "graph2d_label": {"evaluated": 108, "correct": 42, "missing_pred": 11, "accuracy": 0.3889},
            },
        },
    }
    thresholds = {
        "min_sample_size": 20,
        "min_weak_label_coverage": 0.7,
        "min_hybrid_accuracy": 0.30,
        "min_gain_vs_graph2d": 0.0,
        "max_hybrid_missing_pred_rate": 0.30,
        "require_geometry_only": True,
        "require_mask_filename": True,
        "require_strip_text": True,
    }
    report = evaluate_hybrid_blind_gate(summary, thresholds)
    assert report["status"] == "passed"
    assert report["failures"] == []
    assert report["metrics"]["hybrid_gain_vs_graph2d"] > 0


def test_hybrid_blind_gate_fails_on_drop_and_bad_flags() -> None:
    from scripts.ci.check_hybrid_blind_gate import evaluate_hybrid_blind_gate

    summary = {
        "geometry_only": False,
        "mask_filename": False,
        "strip_text": False,
        "sample_size": 10,
        "weak_labels": {
            "covered_count": 2,
            "covered_rate": 0.2,
            "accuracy": {
                "hybrid_label": {"evaluated": 2, "correct": 0, "missing_pred": 2, "accuracy": 0.0},
                "graph2d_label": {"evaluated": 2, "correct": 1, "missing_pred": 0, "accuracy": 0.5},
            },
        },
    }
    thresholds = {
        "min_sample_size": 20,
        "min_weak_label_coverage": 0.7,
        "min_hybrid_accuracy": 0.30,
        "min_gain_vs_graph2d": 0.0,
        "max_hybrid_missing_pred_rate": 0.30,
        "require_geometry_only": True,
        "require_mask_filename": True,
        "require_strip_text": True,
    }
    report = evaluate_hybrid_blind_gate(summary, thresholds)
    assert report["status"] == "failed"
    joined = "\n".join(report["failures"])
    assert "geometry_only" in joined
    assert "mask_filename" in joined
    assert "strip_text" in joined
    assert "sample_size" in joined
    assert "weak_label_coverage" in joined
    assert "hybrid_accuracy" in joined
    assert "hybrid_gain_vs_graph2d" in joined
    assert "hybrid_missing_pred_rate" in joined
