from __future__ import annotations

import json
from pathlib import Path


def test_hybrid_confidence_gate_passes_with_improvement() -> None:
    from scripts.ci import check_hybrid_confidence_calibration_gate as mod

    current = {
        "n_samples": 100,
        "metrics_before": {"ece": 0.14, "brier_score": 0.19, "mce": 0.24},
        "metrics_after": {"ece": 0.11, "brier_score": 0.17, "mce": 0.20},
    }
    baseline = {
        "metrics_after": {"ece": 0.12, "brier_score": 0.18, "mce": 0.21},
    }
    thresholds = {
        "min_samples": 30,
        "max_ece_increase": 0.02,
        "max_brier_increase": 0.02,
        "max_mce_increase": 0.03,
        "max_ece_vs_before_increase": 0.0,
        "max_brier_vs_before_increase": 0.0,
        "max_mce_vs_before_increase": 0.0,
    }
    report = mod.evaluate_gate(
        current=current,
        baseline=baseline,
        thresholds=thresholds,
        missing_mode="skip",
    )
    assert report["status"] == "passed"
    assert report["failures"] == []


def test_hybrid_confidence_gate_fails_on_regression() -> None:
    from scripts.ci import check_hybrid_confidence_calibration_gate as mod

    current = {
        "n_samples": 100,
        "metrics_before": {"ece": 0.14, "brier_score": 0.19, "mce": 0.24},
        "metrics_after": {"ece": 0.20, "brier_score": 0.27, "mce": 0.33},
    }
    baseline = {
        "metrics_after": {"ece": 0.12, "brier_score": 0.18, "mce": 0.21},
    }
    thresholds = {
        "min_samples": 30,
        "max_ece_increase": 0.02,
        "max_brier_increase": 0.02,
        "max_mce_increase": 0.03,
        "max_ece_vs_before_increase": 0.0,
        "max_brier_vs_before_increase": 0.0,
        "max_mce_vs_before_increase": 0.0,
    }
    report = mod.evaluate_gate(
        current=current,
        baseline=baseline,
        thresholds=thresholds,
        missing_mode="skip",
    )
    assert report["status"] == "failed"
    assert len(report["failures"]) > 0


def test_hybrid_confidence_gate_missing_current_skips(tmp_path: Path) -> None:
    from scripts.ci import check_hybrid_confidence_calibration_gate as mod

    output_json = tmp_path / "gate_report.json"
    rc = mod.main(
        [
            "--current-json",
            str(tmp_path / "missing_current.json"),
            "--baseline-json",
            str(tmp_path / "missing_baseline.json"),
            "--output-json",
            str(output_json),
            "--missing-mode",
            "skip",
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"


def test_hybrid_confidence_gate_summary_markdown(tmp_path: Path) -> None:
    from scripts.ci import summarize_hybrid_confidence_calibration_gate as mod

    report = {
        "status": "passed",
        "reason": "ok",
        "current": {
            "n_samples": 50,
            "metrics_before": {"ece": 0.2, "brier_score": 0.3, "mce": 0.4},
            "metrics_after": {"ece": 0.15, "brier_score": 0.25, "mce": 0.3},
        },
        "baseline": {
            "metrics_after": {"ece": 0.16, "brier_score": 0.26, "mce": 0.31},
        },
        "thresholds": {
            "min_samples": 30,
            "max_ece_increase": 0.02,
            "max_brier_increase": 0.02,
            "max_mce_increase": 0.03,
        },
        "failures": [],
        "warnings": [],
    }
    text = mod.build_markdown(report, "Hybrid Calibration Gate")
    assert "Hybrid Calibration Gate" in text
    assert "| status | `passed` |" in text
    assert "current_after" in text
