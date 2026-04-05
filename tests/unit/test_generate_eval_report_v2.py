from __future__ import annotations

import base64
import json
from pathlib import Path


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wn7nS8AAAAASUVORK5CYII="
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG_1X1)


def test_generate_eval_report_v2_embeds_history_sequence_bundle_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import generate_eval_report_v2 as mod

    history_dir = tmp_path / "reports" / "eval_history"
    out_dir = history_dir / "report"
    plots_dir = history_dir / "plots"
    eval_signal_summary_path = history_dir / "eval_signal_experiment_summary.json"
    summary_path = history_dir / "history_sequence_experiment_summary.json"
    compare_path = history_dir / "history_sequence_surface_comparison_report.json"

    monkeypatch.chdir(tmp_path)

    _write_json(
        eval_signal_summary_path,
        {
            "status": "ok",
            "surface_kind": "eval_signal_experiment_summary",
            "report_count": 2,
            "report_counts": {"combined": 1, "ocr": 1, "hybrid_blind": 0},
            "report_rows": [
                {
                    "timestamp": "2026-03-29T01:00:00Z",
                    "report_type": "combined",
                    "report_path": str(history_dir / "combined.json"),
                    "branch": "feature/test",
                    "commit": "abc1234",
                    "combined": {
                        "combined_score": 0.81,
                        "vision_score": 0.78,
                        "ocr_score": 0.84,
                    },
                    "run_context": {"runner": "ci"},
                },
                {
                    "timestamp": "2026-03-29T02:00:00Z",
                    "report_type": "ocr",
                    "report_path": str(history_dir / "ocr.json"),
                    "branch": "feature/test",
                    "commit": "abc1234",
                    "metrics": {
                        "dimension_recall": 0.91,
                        "brier_score": 0.08,
                        "edge_f1": 0.87,
                    },
                },
            ],
            "latest_combined_run": {
                "timestamp": "2026-03-29T01:00:00Z",
                "report_type": "combined",
                "report_path": str(history_dir / "combined.json"),
                "branch": "feature/test",
                "commit": "abc1234",
                "combined": {
                    "combined_score": 0.81,
                    "vision_score": 0.78,
                    "ocr_score": 0.84,
                },
                "run_context": {"runner": "ci"},
            },
            "latest_ocr_run": {
                "timestamp": "2026-03-29T02:00:00Z",
                "report_type": "ocr",
                "report_path": str(history_dir / "ocr.json"),
                "branch": "feature/test",
                "commit": "abc1234",
                "metrics": {
                    "dimension_recall": 0.91,
                    "brier_score": 0.08,
                    "edge_f1": 0.87,
                },
            },
        },
    )
    _write_json(
        summary_path,
        {
            "status": "ok",
            "surface_kind": "history_sequence_experiment_summary",
            "report_count": 2,
            "aggregate_metrics": {
                "mean_accuracy_overall": 0.73,
                "mean_macro_f1_overall": 0.69,
                "mean_named_command_explainability_rate": 0.88,
            },
            "latest_run": {
                "timestamp": "2026-03-29T05:00:00Z",
                "surface_contract": {
                    "sequence_surface_kind": "typed_program_tensor_ir",
                    "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                },
            },
            "report_rows": [
                {
                    "timestamp": "2026-03-29T03:00:00Z",
                    "history_metrics": {"accuracy_overall": 0.70, "macro_f1_overall": 0.66},
                    "named_command_summary": {"named_command_explainability_rate": 0.80},
                    "surface_contract": {
                        "sequence_surface_kind": "typed_program_tensor_ir",
                        "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                    },
                },
                {
                    "timestamp": "2026-03-29T05:00:00Z",
                    "history_metrics": {"accuracy_overall": 0.76, "macro_f1_overall": 0.72},
                    "named_command_summary": {"named_command_explainability_rate": 0.96},
                    "surface_contract": {
                        "sequence_surface_kind": "typed_program_tensor_ir",
                        "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                    },
                },
            ],
        },
    )
    _write_json(
        compare_path,
        {
            "status": "ok",
            "surface_kind": "history_sequence_surface_comparison_report",
            "leaderboard": [
                {
                    "rank": 1,
                    "surface_key": (
                        "typed_program_tensor_ir::reference_derived_named_command_vocabulary"
                    ),
                    "report_count": 2,
                    "mean_accuracy_overall": 0.73,
                    "mean_macro_f1_overall": 0.69,
                    "mean_named_explainability_rate": 0.88,
                }
            ],
        },
    )
    _write_json(
        history_dir / "history_sequence_reporting_bundle.json",
        {
            "status": "ok",
            "surface_kind": "history_sequence_reporting_bundle",
            "summary_json": str(summary_path),
            "compare_json": str(compare_path),
            "best_surface_key_by_mean_accuracy_overall": (
                "typed_program_tensor_ir::reference_derived_named_command_vocabulary"
            ),
        },
    )

    _write_png(plots_dir / "combined_trend.png")
    _write_png(plots_dir / "ocr_trend.png")
    _write_png(plots_dir / "history_sequence_trend.png")
    _write_png(plots_dir / "history_sequence_surface_trend.png")

    mod.main(["--dir", str(history_dir), "--out", str(out_dir)])

    html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "History Sequence Reporting" in html
    assert "historySequenceChart" in html
    assert "Top History Sequence Surfaces" in html
    assert "typed_program_tensor_ir::reference_derived_named_command_vocabulary" in html
    assert "0.730" in html
    assert "0.880" in html
    assert html.count("data:image/png;base64,") >= 4


def test_generate_eval_report_v2_shows_history_sequence_bundle_notice_when_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import generate_eval_report_v2 as mod

    history_dir = tmp_path / "reports" / "eval_history"
    out_dir = history_dir / "report"
    monkeypatch.chdir(tmp_path)

    _write_json(
        history_dir / "combined.json",
        {
            "timestamp": "2026-03-29T01:00:00Z",
            "type": "combined",
            "combined": {
                "combined_score": 0.81,
                "vision_score": 0.78,
                "ocr_score": 0.84,
            },
        },
    )

    mod.main(["--dir", str(history_dir), "--out", str(out_dir)])

    html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "history-sequence reporting bundle" in html.lower()
    assert "make history-sequence-reporting-bundle" in html


def test_generate_eval_report_v2_uses_eval_signal_summary_rows(
    tmp_path: Path,
    capsys,
) -> None:
    from scripts import generate_eval_report_v2 as mod

    history_dir = tmp_path / "reports" / "eval_history"
    out_dir = history_dir / "report"

    _write_json(
        history_dir / "eval_signal_experiment_summary.json",
        {
            "status": "ok",
            "surface_kind": "eval_signal_experiment_summary",
            "report_count": 1,
            "report_counts": {"combined": 0, "ocr": 1, "hybrid_blind": 0},
            "report_rows": [
                {
                    "timestamp": "2026-03-29T02:00:00Z",
                    "report_type": "ocr",
                    "report_path": str(history_dir / "ocr.json"),
                    "metrics": {
                        "dimension_recall": 0.91,
                        "brier_score": 0.08,
                        "edge_f1": 0.87,
                    },
                }
            ],
            "latest_ocr_run": {
                "timestamp": "2026-03-29T02:00:00Z",
                "report_type": "ocr",
                "report_path": str(history_dir / "ocr.json"),
                "metrics": {
                    "dimension_recall": 0.91,
                    "brier_score": 0.08,
                    "edge_f1": 0.87,
                },
            },
        },
    )

    mod.main(["--dir", str(history_dir), "--out", str(out_dir)])

    captured = capsys.readouterr()
    assert "Loaded 0 combined and 1 OCR history entries" in captured.out


def test_generate_eval_report_v2_renders_hybrid_blind_block(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import generate_eval_report_v2 as mod

    history_dir = tmp_path / "reports" / "eval_history"
    out_dir = history_dir / "report"
    monkeypatch.chdir(tmp_path)

    _write_json(
        history_dir / "eval_signal_experiment_summary.json",
        {
            "status": "ok",
            "surface_kind": "eval_signal_experiment_summary",
            "report_count": 2,
            "report_counts": {"combined": 1, "ocr": 0, "hybrid_blind": 1},
            "aggregate_metrics": {
                "hybrid_blind_accuracy_mean": 0.75,
                "hybrid_blind_graph2d_accuracy_mean": 0.70,
                "hybrid_blind_gain_mean": 0.05,
                "hybrid_blind_coverage_mean": 0.88,
                "hybrid_blind_label_slice_count_latest": 10,
                "hybrid_blind_family_slice_count_latest": 5,
            },
            "report_rows": [
                {
                    "timestamp": "2026-03-29T01:00:00Z",
                    "report_type": "combined",
                    "report_path": str(history_dir / "combined.json"),
                    "combined": {"combined_score": 0.81, "vision_score": 0.78, "ocr_score": 0.84},
                },
                {
                    "timestamp": "2026-03-29T02:00:00Z",
                    "report_type": "hybrid_blind",
                    "report_path": str(history_dir / "hybrid.json"),
                    "metrics": {
                        "hybrid_accuracy": 0.75,
                        "graph2d_accuracy": 0.70,
                        "hybrid_gain_vs_graph2d": 0.05,
                        "weak_label_coverage": 0.88,
                        "label_slice_count": 10,
                        "family_slice_count": 5,
                    },
                },
            ],
            "latest_combined_run": {
                "timestamp": "2026-03-29T01:00:00Z",
                "report_type": "combined",
                "report_path": str(history_dir / "combined.json"),
                "combined": {"combined_score": 0.81, "vision_score": 0.78, "ocr_score": 0.84},
            },
        },
    )

    mod.main(["--dir", str(history_dir), "--out", str(out_dir)])

    html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "Hybrid Blind Reporting" in html
    assert "Mean Hybrid Accuracy" in html
    assert "0.750" in html
