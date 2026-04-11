from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_history_sequence_summary_groups_named_surface_contract(tmp_path: Path) -> None:
    from scripts import summarize_history_sequence_runs as mod

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "a.json",
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
        history_dir / "b.json",
        {
            "timestamp": "2026-03-29T09:00:00Z",
            "type": "history_sequence",
            "history_metrics": {
                "coverage": 0.80,
                "accuracy_overall": 0.74,
                "macro_f1_overall": 0.70,
            },
            "named_command_summary": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "named_command_explainability_rate": 1.0,
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                "named_command_authoritative_names_known": False,
            },
            "named_command_error_summary": {
                "overall_incorrect_rate": 0.25,
                "overall_low_conf_rate": 0.10,
            },
        },
    )
    _write_json(
        history_dir / "c.json",
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

    rows = mod._collect_reports(history_dir, "*.json")
    summary = mod._build_summary(rows, eval_history_dir=history_dir, report_glob="*.json")

    assert summary["surface_kind"] == "history_sequence_experiment_summary"
    assert summary["report_count"] == 3
    assert summary["surface_counts"]["sequence_surface_kind"] == {
        "typed_program_tensor_ir": 2,
        "typed_sequence_program_ir": 1,
    }
    assert summary["surface_counts"]["named_command_vocabulary_kind"] == {
        "reference_derived_named_command_vocabulary": 3
    }
    assert summary["surface_counts"]["surface_vocabulary_matrix"] == {
        "typed_program_tensor_ir::reference_derived_named_command_vocabulary": 2,
        "typed_sequence_program_ir::reference_derived_named_command_vocabulary": 1,
    }
    assert summary["best_run"]["timestamp"] == "2026-03-29T09:00:00Z"
    assert summary["latest_run"]["timestamp"] == "2026-03-29T10:00:00Z"
    assert summary["aggregate_metrics"]["mean_accuracy_overall"] == (0.68 + 0.74 + 0.55) / 3.0
    assert len(summary["report_rows"]) == 3
    assert summary["report_rows"][0]["timestamp"] == "2026-03-29T08:00:00Z"
    assert len(summary["surface_groups"]) == 2
    window = mod._build_window_summary(rows)
    assert window["report_count"] == 3
    assert window["surface_group_count"] == 2
    assert window["latest_sequence_surface_kind"] == "typed_sequence_program_ir"
    assert window["best_surface_key_by_mean_accuracy_overall"] == (
        "typed_program_tensor_ir::reference_derived_named_command_vocabulary"
    )


def test_history_sequence_summary_main_writes_outputs(tmp_path: Path) -> None:
    from scripts import summarize_history_sequence_runs as mod

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
    assert payload["surface_kind"] == "history_sequence_experiment_summary"
    assert payload["best_run"]["surface_contract"]["sequence_surface_kind"] == (
        "typed_program_tensor_ir"
    )
    text = output_md.read_text(encoding="utf-8")
    assert "History Sequence Experiment Summary" in text
    assert "typed_program_tensor_ir" in text
