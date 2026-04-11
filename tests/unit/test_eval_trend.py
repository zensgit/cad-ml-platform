from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_load_history_collects_history_sequence_reports(tmp_path: Path) -> None:
    from scripts.eval_trend import load_history

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "combined.json",
        {
            "timestamp": "2026-03-29T01:00:00Z",
            "type": "combined",
            "combined": {"combined_score": 0.8, "vision_score": 0.76, "ocr_score": 0.84},
        },
    )
    _write_json(
        history_dir / "ocr.json",
        {
            "timestamp": "2026-03-29T02:00:00Z",
            "type": "ocr",
            "metrics": {"dimension_recall": 0.91, "brier_score": 0.08, "edge_f1": 0.87},
        },
    )
    _write_json(
        history_dir / "history.json",
        {
            "timestamp": "2026-03-29T03:00:00Z",
            "type": "history_sequence",
            "metrics": {
                "coverage": 0.72,
                "accuracy_overall": 0.68,
                "macro_f1_overall": 0.64,
            },
            "history_metrics": {
                "coverage": 0.72,
                "accuracy_overall": 0.68,
                "macro_f1_overall": 0.64,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "prediction_surface_counts": {"typed_program_tensor_ir": 1},
                "named_command_explainability_rate": 1.0,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "available": True,
                "slice_fields": [
                    "primary_family",
                    "primary_reference_surface",
                    "primary_status",
                ],
                "low_conf_threshold": 0.5,
                "overall_incorrect_rate": 0.2,
                "overall_low_conf_rate": 0.1,
                "worst_primary_family": {
                    "value": "sketch_entity",
                    "incorrect_rate": 0.2,
                    "incorrect_rows": 1,
                    "total_rows": 5,
                },
                "worst_primary_reference_surface": {
                    "value": "deepcad_command_macro",
                    "incorrect_rate": 0.2,
                    "incorrect_rows": 1,
                    "total_rows": 5,
                },
                "worst_primary_status": {
                    "value": "heuristic_reference_alignment",
                    "incorrect_rate": 0.2,
                    "incorrect_rows": 1,
                    "total_rows": 5,
                },
            },
        },
    )

    combined, ocr_only, history_sequence = load_history(history_dir)

    assert len(combined) == 1
    assert len(ocr_only) == 1
    assert history_sequence == []


def test_eval_trend_main_emits_history_sequence_plot_and_metadata(tmp_path: Path) -> None:
    from scripts import eval_trend

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "combined.json",
        {
            "timestamp": "2026-03-29T01:00:00Z",
            "type": "combined",
            "combined": {"combined_score": 0.8, "vision_score": 0.76, "ocr_score": 0.84},
        },
    )
    _write_json(
        history_dir / "ocr.json",
        {
            "timestamp": "2026-03-29T02:00:00Z",
            "type": "ocr",
            "metrics": {"dimension_recall": 0.91, "brier_score": 0.08, "edge_f1": 0.87},
        },
    )
    _write_json(
        history_dir / "history_1.json",
        {
            "timestamp": "2026-03-29T03:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.72,
                "accuracy_overall": 0.68,
                "macro_f1_overall": 0.64,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "prediction_surface_counts": {"typed_program_tensor_ir": 1},
                "named_command_explainability_rate": 0.75,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "available": True,
                "slice_fields": [
                    "primary_family",
                    "primary_reference_surface",
                    "primary_status",
                ],
                "low_conf_threshold": 0.5,
                "overall_incorrect_rate": 0.4,
                "overall_low_conf_rate": 0.2,
                "worst_primary_family": {
                    "value": "sketch_entity",
                    "incorrect_rate": 0.4,
                    "incorrect_rows": 2,
                    "total_rows": 5,
                },
                "worst_primary_reference_surface": {
                    "value": "deepcad_command_macro",
                    "incorrect_rate": 0.4,
                    "incorrect_rows": 2,
                    "total_rows": 5,
                },
                "worst_primary_status": {
                    "value": "heuristic_reference_alignment",
                    "incorrect_rate": 0.4,
                    "incorrect_rows": 2,
                    "total_rows": 5,
                },
            },
        },
    )
    _write_json(
        history_dir / "history_2.json",
        {
            "timestamp": "2026-03-29T04:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.78,
                "accuracy_overall": 0.7,
                "macro_f1_overall": 0.67,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "prediction_surface_counts": {"typed_program_tensor_ir": 1},
                "named_command_explainability_rate": 1.0,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "available": True,
                "slice_fields": [
                    "primary_family",
                    "primary_reference_surface",
                    "primary_status",
                ],
                "low_conf_threshold": 0.5,
                "overall_incorrect_rate": 0.25,
                "overall_low_conf_rate": 0.15,
                "worst_primary_family": {
                    "value": "constraint",
                    "incorrect_rate": 0.25,
                    "incorrect_rows": 1,
                    "total_rows": 4,
                },
                "worst_primary_reference_surface": {
                    "value": "sketchgraphs_construction_step",
                    "incorrect_rate": 0.25,
                    "incorrect_rows": 1,
                    "total_rows": 4,
                },
                "worst_primary_status": {
                    "value": "heuristic_reference_alignment",
                    "incorrect_rate": 0.25,
                    "incorrect_rows": 1,
                    "total_rows": 4,
                },
            },
        },
    )
    _write_json(
        history_dir / "history_3.json",
        {
            "timestamp": "2026-03-29T05:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.74,
                "accuracy_overall": 0.66,
                "macro_f1_overall": 0.63,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_sequence_program_ir",
                "prediction_surface_counts": {"typed_sequence_program_ir": 1},
                "named_command_explainability_rate": 0.85,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "available": True,
                "slice_fields": [
                    "primary_family",
                    "primary_reference_surface",
                    "primary_status",
                ],
                "low_conf_threshold": 0.5,
                "overall_incorrect_rate": 0.30,
                "overall_low_conf_rate": 0.18,
                "worst_primary_family": {
                    "value": "constraint",
                    "incorrect_rate": 0.30,
                    "incorrect_rows": 2,
                    "total_rows": 6,
                },
                "worst_primary_reference_surface": {
                    "value": "sketchgraphs_construction_step",
                    "incorrect_rate": 0.30,
                    "incorrect_rows": 2,
                    "total_rows": 6,
                },
                "worst_primary_status": {
                    "value": "heuristic_reference_alignment",
                    "incorrect_rate": 0.30,
                    "incorrect_rows": 2,
                    "total_rows": 6,
                },
            },
        },
    )

    outdir = tmp_path / "plots"
    rc = eval_trend.main(
        [
            "--eval-history-dir",
            str(history_dir),
            "--out",
            str(outdir),
        ]
    )

    assert rc == 0
    assert (outdir / "combined_trend.png").exists()
    assert (outdir / "ocr_trend.png").exists()
    assert (outdir / "history_sequence_trend.png").exists()
    assert (outdir / "history_sequence_surface_trend.png").exists()
    metadata = json.loads(
        (outdir / "history_sequence_trend_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["report_count"] == 3
    assert metadata["latest_sequence_surface_kind"] == "typed_sequence_program_ir"
    assert metadata["latest_named_command_vocabulary_kind"] == (
        "reference_derived_named_command_vocabulary"
    )
    assert metadata["latest_named_command_explainability_rate"] == 0.85
    assert metadata["latest_named_command_error_rate"] == 0.30
    assert metadata["latest_named_command_low_conf_rate"] == 0.18
    assert metadata["latest_named_command_low_conf_threshold"] == 0.5
    assert metadata["latest_worst_primary_family"] == "constraint"
    assert metadata["latest_worst_primary_reference_surface"] == (
        "sketchgraphs_construction_step"
    )
    assert metadata["latest_worst_primary_status"] == "heuristic_reference_alignment"

    surface_metadata = json.loads(
        (outdir / "history_sequence_surface_trend_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert surface_metadata["surface_kind"] == "history_sequence_surface_trend_metadata"
    assert surface_metadata["report_count"] == 3
    assert surface_metadata["surface_count"] == 2
    assert surface_metadata["best_surface_key_by_mean_accuracy_overall"] == (
        "typed_program_tensor_ir::reference_derived_named_command_vocabulary"
    )
    tensor_group = next(
        group
        for group in surface_metadata["surface_groups"]
        if group["surface_key"]
        == "typed_program_tensor_ir::reference_derived_named_command_vocabulary"
    )
    assert tensor_group["report_count"] == 2
    assert tensor_group["latest_timestamp"] == "2026-03-29T04:00:00Z"
    assert tensor_group["mean_accuracy_overall"] == (0.68 + 0.70) / 2.0


def test_eval_trend_can_use_canonical_history_sequence_summary_json(tmp_path: Path) -> None:
    from scripts import eval_trend
    from scripts import summarize_history_sequence_runs as canonical

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "combined.json",
        {
            "timestamp": "2026-03-29T01:00:00Z",
            "type": "combined",
            "combined": {"combined_score": 0.8, "vision_score": 0.76, "ocr_score": 0.84},
        },
    )
    history_raw = history_dir / "history.json"
    _write_json(
        history_raw,
        {
            "timestamp": "2026-03-29T03:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.72,
                "accuracy_overall": 0.68,
                "macro_f1_overall": 0.64,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "prediction_surface_counts": {"typed_program_tensor_ir": 1},
                "named_command_explainability_rate": 1.0,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "overall_incorrect_rate": 0.2,
                "overall_low_conf_rate": 0.1,
                "low_conf_threshold": 0.5,
                "worst_primary_family": {"value": "sketch_entity"},
                "worst_primary_reference_surface": {"value": "deepcad_command_macro"},
                "worst_primary_status": {"value": "heuristic_reference_alignment"},
            },
        },
    )
    rows = canonical._collect_reports(history_dir, "*.json")
    summary = canonical._build_summary(rows, eval_history_dir=history_dir, report_glob="*.json")
    summary_json = history_dir / "history_sequence_experiment_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    history_raw.unlink()

    outdir = tmp_path / "plots"
    rc = eval_trend.main(
        [
            "--eval-history-dir",
            str(history_dir),
            "--history-sequence-summary-json",
            str(summary_json),
            "--out",
            str(outdir),
        ]
    )

    assert rc == 0
    assert (outdir / "history_sequence_trend.png").exists()
    metadata = json.loads(
        (outdir / "history_sequence_trend_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["report_count"] == 1
    assert metadata["latest_sequence_surface_kind"] == "typed_program_tensor_ir"


def test_load_history_sequence_rows_can_use_preloaded_summary(tmp_path: Path) -> None:
    from scripts import eval_trend
    from scripts import summarize_history_sequence_runs as canonical

    history_dir = tmp_path / "eval_history"
    raw_path = history_dir / "history.json"
    _write_json(
        raw_path,
        {
            "timestamp": "2026-03-29T03:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.72,
                "accuracy_overall": 0.68,
                "macro_f1_overall": 0.64,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "prediction_surface_counts": {"typed_program_tensor_ir": 1},
                "named_command_explainability_rate": 1.0,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "overall_incorrect_rate": 0.2,
                "overall_low_conf_rate": 0.1,
                "low_conf_threshold": 0.5,
                "worst_primary_family": {"value": "sketch_entity"},
                "worst_primary_reference_surface": {"value": "deepcad_command_macro"},
                "worst_primary_status": {"value": "heuristic_reference_alignment"},
            },
        },
    )
    rows = canonical._collect_reports(history_dir, "*.json")
    summary = canonical._build_summary(rows, eval_history_dir=history_dir, report_glob="*.json")
    raw_path.unlink()

    history_rows = eval_trend.load_history_sequence_rows(
        history_dir,
        history_sequence_summary=summary,
    )

    assert len(history_rows) == 1
    assert history_rows[0]["history_metrics"]["accuracy_overall"] == 0.68


def test_eval_trend_can_use_canonical_eval_signal_summary_json(tmp_path: Path) -> None:
    from scripts import eval_trend
    from scripts import summarize_eval_signal_runs as signal_canonical

    history_dir = tmp_path / "eval_history"
    combined_path = history_dir / "combined.json"
    ocr_path = history_dir / "ocr.json"
    _write_json(
        combined_path,
        {
            "timestamp": "2026-03-29T01:00:00Z",
            "type": "combined",
            "combined": {"combined_score": 0.8, "vision_score": 0.76, "ocr_score": 0.84},
        },
    )
    _write_json(
        ocr_path,
        {
            "timestamp": "2026-03-29T02:00:00Z",
            "type": "ocr",
            "metrics": {"dimension_recall": 0.91, "brier_score": 0.08, "edge_f1": 0.87},
        },
    )
    signal_rows = signal_canonical._collect_reports(history_dir, "*.json")
    signal_summary = signal_canonical._build_summary(
        signal_rows, eval_history_dir=history_dir, report_glob="*.json"
    )
    signal_summary_json = history_dir / "eval_signal_experiment_summary.json"
    signal_summary_json.write_text(
        json.dumps(signal_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    combined_path.unlink()
    ocr_path.unlink()

    combined, ocr_only, history_sequence = eval_trend.load_history(
        history_dir,
        eval_signal_summary_json=signal_summary_json,
    )

    assert len(combined) == 1
    assert combined[0]["combined"]["combined_score"] == 0.8
    assert len(ocr_only) == 1
    assert ocr_only[0]["metrics"]["dimension_recall"] == 0.91
    assert history_sequence == []
