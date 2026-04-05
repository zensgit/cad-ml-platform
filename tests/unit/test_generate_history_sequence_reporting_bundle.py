from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_history_sequence_reporting_bundle_materializes_outputs(tmp_path: Path) -> None:
    from scripts.ci import generate_history_sequence_reporting_bundle as mod

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
        history_dir / "history.json",
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

    bundle_json = tmp_path / "bundle.json"
    bundle_md = tmp_path / "bundle.md"
    rc = mod.main(
        [
            "--eval-history-dir",
            str(history_dir),
            "--bundle-json",
            str(bundle_json),
            "--bundle-md",
            str(bundle_md),
            "--trend-out-dir",
            str(tmp_path / "plots"),
        ]
    )

    assert rc == 0
    manifest = json.loads(bundle_json.read_text(encoding="utf-8"))
    assert manifest["surface_kind"] == "history_sequence_reporting_bundle"
    assert manifest["summary_source_mode"] == "materialized_from_raw"
    assert manifest["report_count"] == 1
    assert Path(manifest["summary_json"]).exists()
    assert Path(manifest["compare_json"]).exists()
    assert Path(manifest["weekly_md"]).exists()
    assert any(path.endswith("history_sequence_trend.png") for path in manifest["trend_outputs"])
    assert "History Sequence Reporting Bundle" in bundle_md.read_text(encoding="utf-8")


def test_history_sequence_reporting_bundle_can_use_existing_summary_artifact(tmp_path: Path) -> None:
    from scripts import summarize_history_sequence_runs as canonical
    from scripts.ci import generate_history_sequence_reporting_bundle as mod

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
    summary_json = history_dir / "history_sequence_experiment_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_path.unlink()

    bundle_json = tmp_path / "bundle_from_summary.json"
    rc = mod.main(
        [
            "--eval-history-dir",
            str(history_dir),
            "--summary-json",
            str(summary_json),
            "--bundle-json",
            str(bundle_json),
            "--trend-out-dir",
            str(tmp_path / "plots"),
        ]
    )

    assert rc == 0
    manifest = json.loads(bundle_json.read_text(encoding="utf-8"))
    assert manifest["summary_source_mode"] == "loaded_from_existing_summary"
    assert manifest["report_count"] == 1
