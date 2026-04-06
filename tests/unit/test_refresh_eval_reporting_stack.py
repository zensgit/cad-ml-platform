from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_refresh_materializes_bundle_health_and_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts.ci import refresh_eval_reporting_stack as mod

    history_dir = tmp_path / "eval_history"
    monkeypatch.chdir(tmp_path)

    # Seed raw data
    _write_json(
        history_dir / "combined_20260330.json",
        {
            "timestamp": "2026-03-30T01:00:00Z",
            "type": "combined",
            "combined": {"combined_score": 0.81, "vision_score": 0.78, "ocr_score": 0.84},
            "branch": "main",
            "commit": "abc1234",
        },
    )
    _write_json(
        history_dir / "history_20260330.json",
        {
            "timestamp": "2026-03-30T03:00:00Z",
            "type": "history_sequence",
            "history_metrics": {"coverage": 0.72, "accuracy_overall": 0.68, "macro_f1_overall": 0.64},
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

    rc = mod.main(["--eval-history-dir", str(history_dir)])

    assert rc == 0

    # bundle materialized
    bundle_path = history_dir / "eval_reporting_bundle.json"
    assert bundle_path.exists()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["surface_kind"] == "eval_reporting_bundle"

    # health report materialized
    health_path = history_dir / "eval_reporting_bundle_health_report.json"
    assert health_path.exists()
    health = json.loads(health_path.read_text(encoding="utf-8"))
    assert health["surface_kind"] == "eval_reporting_bundle_health_report"

    # index materialized
    index_path = history_dir / "eval_reporting_index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["surface_kind"] == "eval_reporting_index"
    assert index["status"] == "ok"

    # HTML reports exist
    assert (history_dir / "report_static" / "index.html").exists()
    assert (history_dir / "report_interactive" / "index.html").exists()

    # landing page materialized
    assert (history_dir / "index.html").exists()
    landing_html = (history_dir / "index.html").read_text(encoding="utf-8")
    assert "Eval Reporting Stack" in landing_html


def test_refresh_fails_closed_when_bundle_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts.ci import refresh_eval_reporting_stack as mod
    from scripts.ci import generate_eval_reporting_bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "main", lambda argv: 1)

    rc = mod.main(["--eval-history-dir", str(tmp_path / "nonexistent")])

    assert rc != 0
    assert not (tmp_path / "nonexistent" / "eval_reporting_index.json").exists()


def test_refresh_fails_closed_when_health_check_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts.ci import refresh_eval_reporting_stack as mod
    from scripts.ci import check_eval_reporting_bundle_health as health_mod

    history_dir = tmp_path / "eval_history"
    monkeypatch.chdir(tmp_path)

    # Seed raw data so bundle materializes successfully
    _write_json(
        history_dir / "combined_20260330.json",
        {
            "timestamp": "2026-03-30T01:00:00Z",
            "type": "combined",
            "combined": {"combined_score": 0.81, "vision_score": 0.78, "ocr_score": 0.84},
            "branch": "main",
            "commit": "abc1234",
        },
    )
    _write_json(
        history_dir / "history_20260330.json",
        {
            "timestamp": "2026-03-30T03:00:00Z",
            "type": "history_sequence",
            "history_metrics": {"coverage": 0.72, "accuracy_overall": 0.68, "macro_f1_overall": 0.64},
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

    # Monkeypatch health checker to return failure
    monkeypatch.setattr(health_mod, "main", lambda argv: 1)

    rc = mod.main(["--eval-history-dir", str(history_dir)])

    assert rc != 0
    assert not (history_dir / "eval_reporting_index.json").exists()


def test_refresh_does_not_own_metrics_logic() -> None:
    import ast
    from scripts.ci import refresh_eval_reporting_stack as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    forbidden_prefixes = (
        "_build_summary", "_build_window", "_collect_reports",
        "plot_", "_mean_from", "generate_html", "build_weekly",
    )
    for name in function_names:
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), (
                f"Refresh orchestrator must not own metrics/render logic, but defines {name}"
            )
