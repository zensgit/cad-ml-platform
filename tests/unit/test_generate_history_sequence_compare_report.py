from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_history_sequence_compare_report_groups_by_surface_key(tmp_path: Path) -> None:
    from scripts import summarize_history_sequence_runs as canonical
    from scripts.ci.generate_history_sequence_compare_report import _build_report

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "typed_a.json",
        {
            "timestamp": "2026-03-29T08:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.72,
                "accuracy_overall": 0.68,
                "macro_f1_overall": 0.64,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "named_command_explainability_rate": 0.75,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "overall_incorrect_rate": 0.40,
                "overall_low_conf_rate": 0.20,
            },
        },
    )
    _write_json(
        history_dir / "typed_b.json",
        {
            "timestamp": "2026-03-29T09:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.81,
                "accuracy_overall": 0.77,
                "macro_f1_overall": 0.73,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "named_command_explainability_rate": 1.0,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "overall_incorrect_rate": 0.20,
                "overall_low_conf_rate": 0.10,
            },
        },
    )
    _write_json(
        history_dir / "program_a.json",
        {
            "timestamp": "2026-03-29T10:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.60,
                "accuracy_overall": 0.55,
                "macro_f1_overall": 0.50,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_sequence_program_ir",
                "named_command_explainability_rate": 0.50,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "overall_incorrect_rate": 0.45,
                "overall_low_conf_rate": 0.30,
            },
        },
    )

    rows = canonical._collect_reports(history_dir, "*.json")
    report = _build_report(rows, eval_history_dir=history_dir, report_glob="*.json")

    assert report["surface_kind"] == "history_sequence_surface_comparison_report"
    assert report["total_reports"] == 3
    assert report["total_groups"] == 2
    assert report["best_surface_key"] == (
        "typed_program_tensor_ir::reference_derived_named_command_vocabulary"
    )
    assert report["leaderboard"][0]["report_count"] == 2
    assert report["leaderboard"][0]["mean_accuracy_overall"] == (0.68 + 0.77) / 2.0
    assert report["surface_groups"][0]["latest_run"]["timestamp"] == "2026-03-29T09:00:00Z"
    assert report["surface_groups"][0]["best_run"]["selection_metric_value"] == 0.77


def test_history_sequence_compare_report_main_writes_outputs(tmp_path: Path) -> None:
    from scripts.ci import generate_history_sequence_compare_report as mod

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "run.json",
        {
            "timestamp": "2026-03-29T12:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.81,
                "accuracy_overall": 0.77,
                "macro_f1_overall": 0.73,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "named_command_explainability_rate": 1.0,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "overall_incorrect_rate": 0.20,
                "overall_low_conf_rate": 0.10,
            },
        },
    )
    output_json = tmp_path / "compare.json"
    output_md = tmp_path / "compare.md"

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
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["surface_kind"] == "history_sequence_surface_comparison_report"
    assert report["leaderboard"][0]["surface_key"] == (
        "typed_program_tensor_ir::reference_derived_named_command_vocabulary"
    )
    text = output_md.read_text(encoding="utf-8")
    assert "History Sequence Surface Comparison Report" in text
    assert "typed_program_tensor_ir::reference_derived_named_command_vocabulary" in text


def test_history_sequence_compare_report_can_use_canonical_summary_json(tmp_path: Path) -> None:
    from scripts import summarize_history_sequence_runs as canonical
    from scripts.ci import generate_history_sequence_compare_report as mod

    history_dir = tmp_path / "eval_history"
    raw_path = history_dir / "run.json"
    _write_json(
        raw_path,
        {
            "timestamp": "2026-03-29T12:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.81,
                "accuracy_overall": 0.77,
                "macro_f1_overall": 0.73,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "named_command_explainability_rate": 1.0,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "overall_incorrect_rate": 0.20,
                "overall_low_conf_rate": 0.10,
            },
        },
    )
    rows = canonical._collect_reports(history_dir, "*.json")
    summary = canonical._build_summary(rows, eval_history_dir=history_dir, report_glob="*.json")
    summary_json = history_dir / "history_sequence_experiment_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_path.unlink()

    output_json = tmp_path / "compare_from_summary.json"
    output_md = tmp_path / "compare_from_summary.md"
    rc = mod.main(
        [
            "--eval-history-dir",
            str(history_dir),
            "--summary-json",
            str(summary_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert rc == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["total_reports"] == 1
    assert report["leaderboard"][0]["surface_key"] == (
        "typed_program_tensor_ir::reference_derived_named_command_vocabulary"
    )
