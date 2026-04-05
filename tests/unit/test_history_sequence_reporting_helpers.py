from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_load_history_sequence_reporting_assets_returns_none_for_empty_history_dir(
    tmp_path: Path,
) -> None:
    from scripts.history_sequence_reporting_helpers import load_history_sequence_reporting_assets

    bundle, summary, compare = load_history_sequence_reporting_assets(tmp_path / "eval_history")

    assert bundle is None
    assert summary is None
    assert compare is None


def test_load_history_sequence_reporting_assets_prefers_bundle_referenced_artifacts(
    tmp_path: Path,
) -> None:
    from scripts.history_sequence_reporting_helpers import load_history_sequence_reporting_assets

    history_dir = tmp_path / "eval_history"
    external_summary = tmp_path / "external" / "summary.json"
    external_compare = tmp_path / "external" / "compare.json"

    _write_json(history_dir / "history_sequence_experiment_summary.json", {"surface_kind": "wrong"})
    _write_json(
        history_dir / "history_sequence_surface_comparison_report.json",
        {"surface_kind": "wrong_compare"},
    )
    _write_json(external_summary, {"surface_kind": "history_sequence_experiment_summary", "report_count": 2})
    _write_json(
        external_compare,
        {"surface_kind": "history_sequence_surface_comparison_report", "leaderboard": []},
    )
    _write_json(
        history_dir / "history_sequence_reporting_bundle.json",
        {"summary_json": str(external_summary), "compare_json": str(external_compare)},
    )

    bundle, summary, compare = load_history_sequence_reporting_assets(history_dir)

    assert bundle is not None
    assert summary == {"surface_kind": "history_sequence_experiment_summary", "report_count": 2}
    assert compare == {"surface_kind": "history_sequence_surface_comparison_report", "leaderboard": []}


def test_history_sequence_chart_rows_normalizes_report_rows(tmp_path: Path) -> None:
    from scripts.history_sequence_reporting_helpers import history_sequence_chart_rows

    rows = history_sequence_chart_rows(
        {
            "report_rows": [
                {
                    "timestamp": "2026-03-29T03:00:00Z",
                    "history_metrics": {"accuracy_overall": 0.7, "macro_f1_overall": 0.66},
                    "named_command_summary": {"named_command_explainability_rate": 0.8},
                    "surface_contract": {
                        "sequence_surface_kind": "typed_program_tensor_ir",
                        "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                    },
                }
            ]
        }
    )

    assert rows == [
        {
            "timestamp": "2026-03-29T03:00:00Z",
            "history_metrics": {"accuracy_overall": 0.7, "macro_f1_overall": 0.66},
            "named_command_summary": {"named_command_explainability_rate": 0.8},
            "surface_contract": {
                "sequence_surface_kind": "typed_program_tensor_ir",
                "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
            },
        }
    ]


def test_build_history_sequence_report_context_gracefully_handles_missing_summary() -> None:
    from scripts.history_sequence_reporting_helpers import build_history_sequence_report_context

    context = build_history_sequence_report_context(None, None, None)

    assert context["available"] is False
    assert context["report_count"] == 0
    assert context["best_surface_key"] == ""
    assert context["leaderboard_rows"] == []


def test_build_history_sequence_report_context_exposes_display_contract() -> None:
    from scripts.history_sequence_reporting_helpers import build_history_sequence_report_context

    context = build_history_sequence_report_context(
        {
            "best_surface_key_by_mean_accuracy_overall": (
                "typed_program_tensor_ir::reference_derived_named_command_vocabulary"
            )
        },
        {
            "report_count": 2,
            "aggregate_metrics": {
                "mean_accuracy_overall": 0.73,
                "mean_macro_f1_overall": 0.69,
                "mean_named_command_explainability_rate": 0.88,
            },
            "latest_run": {
                "surface_contract": {
                    "sequence_surface_kind": "typed_program_tensor_ir",
                    "named_command_vocabulary_kind": "reference_derived_named_command_vocabulary",
                }
            },
        },
        {
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
            ]
        },
    )

    assert context["available"] is True
    assert context["report_count"] == 2
    assert context["mean_accuracy_overall"] == 0.73
    assert context["mean_macro_f1_overall"] == 0.69
    assert context["mean_named_command_explainability_rate"] == 0.88
    assert context["latest_sequence_surface_kind"] == "typed_program_tensor_ir"
    assert (
        context["latest_named_command_vocabulary_kind"]
        == "reference_derived_named_command_vocabulary"
    )
    assert (
        context["best_surface_key"]
        == "typed_program_tensor_ir::reference_derived_named_command_vocabulary"
    )
    assert context["leaderboard_rows"][0]["rank"] == 1
