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
            "history_metrics": {
                "coverage": 0.6,
                "accuracy_overall": 0.65,
                "macro_f1_overall": 0.62,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
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
                "overall_incorrect_rate": 0.35,
                "overall_low_conf_rate": 0.15,
                "worst_primary_family": {
                    "value": "sketch_entity",
                    "incorrect_rate": 0.35,
                    "incorrect_rows": 7,
                    "total_rows": 20,
                },
                "worst_primary_reference_surface": {
                    "value": "deepcad_command_macro",
                    "incorrect_rate": 0.35,
                    "incorrect_rows": 7,
                    "total_rows": 20,
                },
                "worst_primary_status": {
                    "value": "heuristic_reference_alignment",
                    "incorrect_rate": 0.35,
                    "incorrect_rows": 7,
                    "total_rows": 20,
                },
            },
        },
    )
    _write_json(
        history_dir / "c2.json",
        {
            "timestamp": "2026-03-09T02:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.7,
                "accuracy_overall": 0.72,
                "macro_f1_overall": 0.68,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_sequence_program_ir",
                "named_command_explainability_rate": 0.6,
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
                "overall_incorrect_rate": 0.28,
                "overall_low_conf_rate": 0.12,
                "worst_primary_family": {
                    "value": "constraint",
                    "incorrect_rate": 0.28,
                    "incorrect_rows": 4,
                    "total_rows": 14,
                },
                "worst_primary_reference_surface": {
                    "value": "sketchgraphs_construction_step",
                    "incorrect_rate": 0.28,
                    "incorrect_rows": 4,
                    "total_rows": 14,
                },
                "worst_primary_status": {
                    "value": "heuristic_reference_alignment",
                    "incorrect_rate": 0.28,
                    "incorrect_rows": 4,
                    "total_rows": 14,
                },
            },
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
    assert metrics.total_reports == 5
    assert metrics.ocr_reports == 1
    assert metrics.combined_reports == 1
    assert metrics.history_reports == 2
    assert metrics.history_named_explainability_rate_mean == (0.75 + 0.6) / 2.0
    assert metrics.history_named_error_rate_mean == (0.35 + 0.28) / 2.0
    assert metrics.history_named_low_conf_rate_mean == (0.15 + 0.12) / 2.0
    assert metrics.history_sequence_surface_kind_latest == "typed_sequence_program_ir"
    assert (
        metrics.history_named_vocabulary_kind_latest
        == "reference_derived_named_command_vocabulary"
    )
    assert metrics.history_named_worst_primary_family_latest == "constraint"
    assert (
        metrics.history_named_worst_primary_reference_surface_latest
        == "sketchgraphs_construction_step"
    )
    assert metrics.history_surface_group_count == 2
    assert metrics.history_best_surface_key_by_mean_accuracy_overall == (
        "typed_sequence_program_ir::reference_derived_named_command_vocabulary"
    )
    assert len(metrics.history_surface_groups) == 2
    assert metrics.history_named_worst_primary_status_latest == "heuristic_reference_alignment"
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
    assert "History named explainability rate mean" in text
    assert "History named error rate mean" in text
    assert "History named low-conf rate mean" in text
    assert "History sequence surface latest" in text
    assert "History worst family latest" in text
    assert "History worst reference surface latest" in text
    assert "History worst status latest" in text
    assert "History Surface Groups" in text
    assert "typed_sequence_program_ir::reference_derived_named_command_vocabulary" in text
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
    _write_json(
        history_dir / "b.json",
        {
            "timestamp": "2026-03-11T02:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.7,
                "accuracy_overall": 0.66,
                "macro_f1_overall": 0.61,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
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
    _write_json(
        history_dir / "c.json",
        {
            "timestamp": "2026-03-11T03:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.64,
                "accuracy_overall": 0.60,
                "macro_f1_overall": 0.58,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_sequence_program_ir",
                "named_command_explainability_rate": 0.8,
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
                "overall_incorrect_rate": 0.32,
                "overall_low_conf_rate": 0.18,
                "worst_primary_family": {
                    "value": "constraint",
                    "incorrect_rate": 0.32,
                    "incorrect_rows": 2,
                    "total_rows": 6,
                },
                "worst_primary_reference_surface": {
                    "value": "sketchgraphs_construction_step",
                    "incorrect_rate": 0.32,
                    "incorrect_rows": 2,
                    "total_rows": 6,
                },
                "worst_primary_status": {
                    "value": "heuristic_reference_alignment",
                    "incorrect_rate": 0.32,
                    "incorrect_rows": 2,
                    "total_rows": 6,
                },
            },
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
    assert "History sequence surface latest" in text
    assert "History named error rate mean" in text
    assert "History Surface Groups" in text


def test_collect_metrics_can_use_canonical_history_sequence_summary_json(tmp_path: Path) -> None:
    from scripts import summarize_history_sequence_runs as canonical
    from scripts.ci.generate_eval_weekly_summary import collect_metrics

    now = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)
    history_dir = tmp_path / "eval_history"
    raw_path = history_dir / "history.json"
    _write_json(
        raw_path,
        {
            "timestamp": "2026-03-11T02:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.7,
                "accuracy_overall": 0.66,
                "macro_f1_overall": 0.61,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "named_command_explainability_rate": 1.0,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "overall_incorrect_rate": 0.2,
                "overall_low_conf_rate": 0.1,
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
    raw_path.unlink()

    metrics = collect_metrics(
        history_dir,
        days=7,
        now=now,
        history_sequence_summary_json=summary_json,
    )

    assert metrics.history_reports == 1
    assert metrics.history_accuracy_mean == 0.66
    assert metrics.history_sequence_surface_kind_latest == "typed_program_tensor_ir"


def test_collect_metrics_can_use_preloaded_history_sequence_summary(tmp_path: Path) -> None:
    from scripts import summarize_history_sequence_runs as canonical
    from scripts.ci.generate_eval_weekly_summary import collect_metrics

    now = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)
    history_dir = tmp_path / "eval_history"
    raw_path = history_dir / "history.json"
    _write_json(
        raw_path,
        {
            "timestamp": "2026-03-11T02:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.7,
                "accuracy_overall": 0.66,
                "macro_f1_overall": 0.61,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "named_command_explainability_rate": 1.0,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "overall_incorrect_rate": 0.2,
                "overall_low_conf_rate": 0.1,
                "worst_primary_family": {"value": "sketch_entity"},
                "worst_primary_reference_surface": {"value": "deepcad_command_macro"},
                "worst_primary_status": {"value": "heuristic_reference_alignment"},
            },
        },
    )
    rows = canonical._collect_reports(history_dir, "*.json")
    summary = canonical._build_summary(rows, eval_history_dir=history_dir, report_glob="*.json")
    raw_path.unlink()

    metrics = collect_metrics(
        history_dir,
        days=7,
        now=now,
        history_sequence_summary=summary,
    )

    assert metrics.history_reports == 1
    assert metrics.history_accuracy_mean == 0.66
    assert metrics.history_named_explainability_rate_mean == 1.0
    assert metrics.history_sequence_surface_kind_latest == "typed_program_tensor_ir"


def test_collect_metrics_can_use_canonical_eval_signal_summary_json(tmp_path: Path) -> None:
    from scripts import summarize_eval_signal_runs as signal_canonical
    from scripts.ci.generate_eval_weekly_summary import collect_metrics

    now = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)
    history_dir = tmp_path / "eval_history"
    combined_path = history_dir / "combined.json"
    ocr_path = history_dir / "ocr.json"
    hybrid_path = history_dir / "hybrid.json"
    _write_json(
        combined_path,
        {
            "timestamp": "2026-03-11T01:00:00Z",
            "type": "combined",
            "combined": {"combined_score": 0.8, "vision_score": 0.75, "ocr_score": 0.85},
        },
    )
    _write_json(
        ocr_path,
        {
            "timestamp": "2026-03-11T02:00:00Z",
            "type": "ocr",
            "metrics": {"dimension_recall": 0.9, "brier_score": 0.2, "edge_f1": 0.7},
        },
    )
    _write_json(
        hybrid_path,
        {
            "timestamp": "2026-03-11T03:00:00Z",
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
    _write_json(
        history_dir / "history.json",
        {
            "timestamp": "2026-03-11T04:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.7,
                "accuracy_overall": 0.66,
                "macro_f1_overall": 0.61,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "named_command_explainability_rate": 1.0,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "overall_incorrect_rate": 0.2,
                "overall_low_conf_rate": 0.1,
                "worst_primary_family": {"value": "sketch_entity"},
                "worst_primary_reference_surface": {"value": "deepcad_command_macro"},
                "worst_primary_status": {"value": "heuristic_reference_alignment"},
            },
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
    hybrid_path.unlink()

    metrics = collect_metrics(
        history_dir,
        days=7,
        now=now,
        eval_signal_summary_json=signal_summary_json,
    )

    assert metrics.total_reports == 4
    assert metrics.combined_reports == 1
    assert metrics.ocr_reports == 1
    assert metrics.hybrid_blind_reports == 1
    assert metrics.combined_score_mean == 0.8
    assert metrics.hybrid_blind_label_slice_count_latest == 2


def test_collect_metrics_uses_canonical_eval_signal_summary_loader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts.ci import generate_eval_weekly_summary as mod

    now = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(
        mod.eval_signal_canonical,
        "_load_or_build_summary",
        lambda _summary_json, **_kwargs: {
            "status": "ok",
            "surface_kind": "eval_signal_experiment_summary",
            "report_rows": [
                {
                    "timestamp": "2026-03-11T01:00:00Z",
                    "report_type": "combined",
                    "combined": {
                        "combined_score": 0.8,
                        "vision_score": 0.75,
                        "ocr_score": 0.85,
                    },
                    "metrics": {},
                    "report_path": "combined.json",
                },
                {
                    "timestamp": "2026-03-11T02:00:00Z",
                    "report_type": "ocr",
                    "combined": {},
                    "metrics": {
                        "dimension_recall": 0.9,
                        "brier_score": 0.2,
                        "edge_f1": 0.7,
                    },
                    "report_path": "ocr.json",
                },
                {
                    "timestamp": "2026-03-11T03:00:00Z",
                    "report_type": "hybrid_blind",
                    "combined": {},
                    "metrics": {
                        "weak_label_coverage": 0.82,
                        "hybrid_accuracy": 0.41,
                        "graph2d_accuracy": 0.33,
                        "hybrid_gain_vs_graph2d": 0.08,
                        "label_slice_count": 2,
                        "family_slice_count": 1,
                    },
                    "report_path": "hybrid.json",
                },
            ],
        },
    )

    metrics = mod.collect_metrics(
        tmp_path / "eval_history",
        days=7,
        now=now,
        history_sequence_summary={
            "status": "ok",
            "surface_kind": "history_sequence_experiment_summary",
            "report_rows": [],
        },
    )

    assert metrics.total_reports == 3
    assert metrics.ocr_reports == 1
    assert metrics.combined_reports == 1
    assert metrics.hybrid_blind_reports == 1
    assert metrics.combined_score_mean == 0.8
    assert metrics.ocr_dimension_recall_mean == 0.9
    assert metrics.hybrid_blind_gain_mean == 0.08
