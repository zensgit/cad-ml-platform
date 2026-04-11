from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _hybrid_payload(
    timestamp: str,
    *,
    hybrid_accuracy: float,
    hybrid_gain: float,
    coverage: float,
    label_acc: float,
    label_gain: float,
    family_acc: float,
    family_gain: float,
) -> dict:
    return {
        "schema_version": "1.0.0",
        "timestamp": timestamp,
        "branch": "main",
        "commit": "abc1234",
        "type": "hybrid_blind",
        "metrics": {
            "hybrid_accuracy": hybrid_accuracy,
            "hybrid_gain_vs_graph2d": hybrid_gain,
            "weak_label_coverage": coverage,
            "label_slices": [
                {
                    "label": "人孔",
                    "support": 10,
                    "hybrid_accuracy": label_acc,
                    "hybrid_gain_vs_graph2d": label_gain,
                },
                {
                    "label": "捕集口",
                    "support": 10,
                    "hybrid_accuracy": max(0.0, label_acc - 0.05),
                    "hybrid_gain_vs_graph2d": max(0.0, label_gain - 0.05),
                },
            ],
            "family_slices": [
                {
                    "family": "人孔",
                    "support": 20,
                    "hybrid_accuracy": family_acc,
                    "hybrid_gain_vs_graph2d": family_gain,
                }
            ],
        },
    }


def test_suggest_hybrid_blind_drift_thresholds_ok(tmp_path: Path) -> None:
    from scripts.ci import suggest_hybrid_blind_drift_thresholds as mod

    history_dir = tmp_path / "eval_history"
    output_json = tmp_path / "suggest.json"
    output_md = tmp_path / "suggest.md"
    _write_json(
        history_dir / "r1.json",
        _hybrid_payload(
            "2026-03-10T00:00:00Z",
            hybrid_accuracy=0.90,
            hybrid_gain=0.25,
            coverage=0.95,
            label_acc=0.90,
            label_gain=0.60,
            family_acc=0.88,
            family_gain=0.55,
        ),
    )
    _write_json(
        history_dir / "r2.json",
        _hybrid_payload(
            "2026-03-11T00:00:00Z",
            hybrid_accuracy=0.86,
            hybrid_gain=0.20,
            coverage=0.93,
            label_acc=0.84,
            label_gain=0.50,
            family_acc=0.83,
            family_gain=0.45,
        ),
    )
    _write_json(
        history_dir / "r3.json",
        _hybrid_payload(
            "2026-03-12T00:00:00Z",
            hybrid_accuracy=0.83,
            hybrid_gain=0.18,
            coverage=0.90,
            label_acc=0.80,
            label_gain=0.42,
            family_acc=0.79,
            family_gain=0.38,
        ),
    )
    _write_json(
        history_dir / "r4.json",
        _hybrid_payload(
            "2026-03-13T00:00:00Z",
            hybrid_accuracy=0.82,
            hybrid_gain=0.17,
            coverage=0.88,
            label_acc=0.78,
            label_gain=0.40,
            family_acc=0.76,
            family_gain=0.34,
        ),
    )

    rc = mod.main(
        [
            "--eval-history-dir",
            str(history_dir),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--quantile",
            "0.9",
            "--min-reports",
            "4",
        ]
    )
    assert rc == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    md_text = output_md.read_text(encoding="utf-8")

    assert report["status"] == "ok"
    assert report["history"]["report_count"] == 4
    assert report["history"]["pair_count"] == 3
    thresholds = report["recommended_thresholds"]
    assert thresholds["max_hybrid_accuracy_drop"] >= 0.05
    assert thresholds["max_gain_drop"] >= 0.05
    assert thresholds["max_coverage_drop"] >= 0.10
    assert thresholds["label_slice_enable"] is True
    assert thresholds["family_slice_enable"] is True
    assert thresholds["label_slice_auto_cap_min_common"] is True
    assert thresholds["family_slice_auto_cap_min_common"] is True
    assert "Suggested Thresholds" in md_text


def test_suggest_hybrid_blind_drift_thresholds_insufficient(tmp_path: Path) -> None:
    from scripts.ci import suggest_hybrid_blind_drift_thresholds as mod

    history_dir = tmp_path / "eval_history"
    output_json = tmp_path / "suggest.json"
    output_md = tmp_path / "suggest.md"
    _write_json(
        history_dir / "r1.json",
        _hybrid_payload(
            "2026-03-10T00:00:00Z",
            hybrid_accuracy=0.90,
            hybrid_gain=0.25,
            coverage=0.95,
            label_acc=0.90,
            label_gain=0.60,
            family_acc=0.88,
            family_gain=0.55,
        ),
    )

    rc = mod.main(
        [
            "--eval-history-dir",
            str(history_dir),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--min-reports",
            "4",
        ]
    )
    assert rc == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    md_text = output_md.read_text(encoding="utf-8")

    assert report["status"] == "insufficient"
    assert report["recommended_thresholds"] == {}
    assert report["history"]["report_count"] == 1
    assert "insufficient reports" in "\n".join(report.get("notes", []))
    assert "Status: `insufficient`" in md_text
