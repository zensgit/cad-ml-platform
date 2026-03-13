from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_collect_metrics_and_markdown(tmp_path: Path) -> None:
    from scripts.ci.generate_eval_weekly_summary import build_weekly_markdown, collect_metrics

    now = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)
    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "a.json",
        {
            "timestamp": "2026-03-11T01:00:00Z",
            "type": "ocr",
            "metrics": {"dimension_recall": 0.9, "brier_score": 0.2, "edge_f1": 0.7},
        },
    )
    _write_json(
        history_dir / "b.json",
        {
            "timestamp": "2026-03-10T01:00:00Z",
            "type": "combined",
            "combined": {"combined_score": 0.8, "vision_score": 0.75, "ocr_score": 0.85},
        },
    )
    _write_json(
        history_dir / "c.json",
        {
            "timestamp": "2026-03-09T01:00:00Z",
            "type": "history_sequence",
            "metrics": {"coverage": 0.6, "accuracy_overall": 0.65, "macro_f1_overall": 0.62},
        },
    )
    _write_json(
        history_dir / "d.json",
        {
            "timestamp": "2026-03-08T01:00:00Z",
            "type": "hybrid_blind",
            "metrics": {
                "weak_label_coverage": 0.82,
                "hybrid_accuracy": 0.41,
                "graph2d_accuracy": 0.33,
                "hybrid_gain_vs_graph2d": 0.08,
                "label_slices": [
                    {"label": "人孔", "support": 10},
                    {"label": "捕集口", "support": 8},
                ],
                "family_slices": [
                    {"family": "人孔", "support": 10},
                    {"family": "捕集", "support": 8},
                ],
            },
        },
    )
    _write_json(
        history_dir / "old.json",
        {
            "timestamp": "2025-01-01T01:00:00Z",
            "type": "combined",
            "combined": {"combined_score": 0.1, "vision_score": 0.1, "ocr_score": 0.1},
        },
    )

    metrics = collect_metrics(history_dir, days=7, now=now)
    assert metrics.total_reports == 4
    assert metrics.ocr_reports == 1
    assert metrics.combined_reports == 1
    assert metrics.history_reports == 1
    assert metrics.hybrid_blind_reports == 1
    assert metrics.combined_score_mean == 0.8
    assert metrics.hybrid_blind_gain_mean == 0.08
    assert metrics.hybrid_blind_label_slice_count_mean == 2.0
    assert metrics.hybrid_blind_label_slice_count_latest == 2
    assert metrics.hybrid_blind_family_slice_count_mean == 2.0
    assert metrics.hybrid_blind_family_slice_count_latest == 2

    text = build_weekly_markdown(
        metrics=metrics,
        days=7,
        generated_at="2026-03-12T12:00:00Z",
        context={
            "graph2d_blind_status": "passed",
            "graph2d_blind_accuracy": "0.33",
            "hybrid_blind_status": "passed",
            "hybrid_blind_accuracy": "0.41",
            "hybrid_blind_gain": "0.08",
            "hybrid_calibration_status": "ok",
            "hybrid_calibration_gate_status": "passed",
        },
    )
    assert "Weekly Evaluation Summary" in text
    assert "Combined score mean" in text
    assert "Hybrid blind gain vs Graph2D" in text
    assert "Hybrid blind gain mean" in text
    assert "Hybrid blind label-slice count mean" in text
    assert "Hybrid blind family-slice count mean" in text


def test_main_writes_output(tmp_path: Path) -> None:
    from scripts.ci import generate_eval_weekly_summary as mod

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "a.json",
        {
            "timestamp": "2026-03-11T01:00:00Z",
            "type": "combined",
            "combined": {"combined_score": 0.8, "vision_score": 0.75, "ocr_score": 0.85},
        },
    )
    output_md = tmp_path / "weekly.md"
    rc = mod.main(
        [
            "--eval-history-dir",
            str(history_dir),
            "--output-md",
            str(output_md),
            "--days",
            "7",
            "--graph2d-blind-status",
            "passed",
            "--hybrid-blind-status",
            "failed",
            "--hybrid-blind-gain",
            "-0.03",
            "--hybrid-calibration-status",
            "ok",
            "--hybrid-calibration-gate-status",
            "passed",
        ]
    )
    assert rc == 0
    text = output_md.read_text(encoding="utf-8")
    assert "Graph2D blind gate" in text
    assert "Hybrid blind gate" in text
