from __future__ import annotations

import json
from pathlib import Path


def test_build_hybrid_calibration_baseline_has_expected_fields() -> None:
    from scripts.ci.update_hybrid_confidence_calibration_baseline import build_baseline

    current = {
        "status": "ok",
        "method": "temperature_scaling",
        "per_source": True,
        "effective_per_source": True,
        "n_rows": 120,
        "n_samples": 90,
        "min_samples": 30,
        "min_samples_per_source": 10,
        "dropped_bad_confidence": 5,
        "dropped_no_correctness": 7,
        "pair_counts": {"match": 50, "mismatch": 40},
        "source_counts": {"filename": 45, "graph2d": 45},
        "temperature": 0.87,
        "source_temperatures": {
            "filename": {"temperature": 0.8, "n_samples": 45},
            "graph2d": {"temperature": 0.9, "n_samples": 45},
        },
        "metrics_before": {"ece": 0.2, "brier_score": 0.3, "mce": 0.4},
        "metrics_after": {"ece": 0.12, "brier_score": 0.22, "mce": 0.26},
    }
    payload = build_baseline(
        current=current,
        current_path="/tmp/current.json",
        snapshot_ref="reports/experiments/20260312/snapshot.json",
        generated_at="2026-03-12T12:00:00Z",
    )

    assert payload["source"]["current_report_json"] == "/tmp/current.json"
    assert payload["source"]["snapshot_ref"].endswith("snapshot.json")
    assert payload["calibration"]["status"] == "ok"
    assert payload["calibration"]["n_samples"] == 90
    assert payload["calibration"]["metrics_after"]["ece"] == 0.12
    integrity = payload.get("integrity") or {}
    assert integrity.get("algorithm") == "sha256-canonical-json"
    assert len(str(integrity.get("calibration_sha256", ""))) == 64
    assert len(str(integrity.get("payload_core_sha256", ""))) == 64


def test_update_hybrid_calibration_baseline_main_writes_outputs(tmp_path: Path) -> None:
    from scripts.ci import update_hybrid_confidence_calibration_baseline as mod

    current_json = tmp_path / "current.json"
    baseline_json = tmp_path / "config" / "baseline.json"
    snapshot_json = tmp_path / "reports" / "snapshot.json"

    current_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "method": "temperature_scaling",
                "per_source": True,
                "effective_per_source": True,
                "n_rows": 10,
                "n_samples": 8,
                "metrics_before": {"ece": 0.22, "brier_score": 0.28, "mce": 0.34},
                "metrics_after": {"ece": 0.15, "brier_score": 0.2, "mce": 0.25},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rc = mod.main(
        [
            "--current-json",
            str(current_json),
            "--output-baseline-json",
            str(baseline_json),
            "--snapshot-output-json",
            str(snapshot_json),
        ]
    )
    assert rc == 0
    assert baseline_json.exists()
    assert snapshot_json.exists()

    payload = json.loads(baseline_json.read_text(encoding="utf-8"))
    assert payload["calibration"]["status"] == "ok"
    assert payload["source"]["current_report_json"] == str(current_json)
    assert payload["source"]["snapshot_ref"] == str(snapshot_json)
