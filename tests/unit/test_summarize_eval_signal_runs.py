from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_eval_signal_summary_groups_combined_ocr_and_hybrid_rows(tmp_path: Path) -> None:
    from scripts import summarize_eval_signal_runs as mod

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "combined.json",
        {
            "timestamp": "2026-03-29T08:00:00Z",
            "type": "combined",
            "branch": "feature/test",
            "commit": "abc1234",
            "combined": {"combined_score": 0.81, "vision_score": 0.78, "ocr_score": 0.84},
            "run_context": {"runner": "ci"},
        },
    )
    _write_json(
        history_dir / "ocr.json",
        {
            "timestamp": "2026-03-29T09:00:00Z",
            "type": "ocr",
            "metrics": {"dimension_recall": 0.91, "brier_score": 0.08, "edge_f1": 0.87},
        },
    )
    _write_json(
        history_dir / "hybrid.json",
        {
            "timestamp": "2026-03-29T10:00:00Z",
            "type": "hybrid_blind",
            "metrics": {
                "weak_label_coverage": 0.82,
                "hybrid_accuracy": 0.41,
                "graph2d_accuracy": 0.33,
                "hybrid_gain_vs_graph2d": 0.08,
                "label_slices": [{"label": "人孔"}, {"label": "捕集口"}],
                "family_slices": [{"family": "人孔"}],
            },
        },
    )

    rows = mod._collect_reports(history_dir, "*.json")
    summary = mod._build_summary(rows, eval_history_dir=history_dir, report_glob="*.json")

    assert summary["surface_kind"] == "eval_signal_experiment_summary"
    assert summary["report_count"] == 3
    assert summary["report_counts"] == {"combined": 1, "ocr": 1, "hybrid_blind": 1}
    assert summary["aggregate_metrics"]["combined_score_mean"] == 0.81
    assert summary["aggregate_metrics"]["ocr_dimension_recall_mean"] == 0.91
    assert summary["aggregate_metrics"]["hybrid_blind_gain_mean"] == 0.08
    assert len(summary["report_rows"]) == 3
    assert [row["report_type"] for row in summary["report_rows"]] == [
        "combined",
        "ocr",
        "hybrid_blind",
    ]
    assert summary["report_rows"][0]["branch"] == "feature/test"
    assert summary["report_rows"][0]["commit"] == "abc1234"
    assert summary["report_rows"][0]["run_context"]["runner"] == "ci"
    assert summary["latest_combined_run"]["timestamp"] == "2026-03-29T08:00:00Z"
    assert summary["latest_hybrid_blind_run"]["metrics"]["label_slice_count"] == 2
    window = mod._build_window_summary(rows)
    assert window["report_count"] == 3
    assert window["hybrid_blind_family_slice_count_latest"] == 1


def test_eval_signal_summary_main_writes_outputs(tmp_path: Path) -> None:
    from scripts import summarize_eval_signal_runs as mod

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "run.json",
        {
            "timestamp": "2026-03-29T12:00:00Z",
            "type": "combined",
            "combined": {"combined_score": 0.77, "vision_score": 0.73, "ocr_score": 0.81},
        },
    )
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"

    rc = mod.main(
        [
            "--eval-history-dir",
            str(history_dir),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["surface_kind"] == "eval_signal_experiment_summary"
    assert payload["latest_combined_run"]["combined"]["combined_score"] == 0.77
    text = output_md.read_text(encoding="utf-8")
    assert "Eval Signal Experiment Summary" in text
    assert "Combined score mean" in text
