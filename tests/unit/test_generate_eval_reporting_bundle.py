from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_eval_reporting_bundle_materializes_sub_bundles_and_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts.ci import generate_eval_reporting_bundle as mod

    history_dir = tmp_path / "eval_history"
    monkeypatch.chdir(tmp_path)

    # Seed a combined eval-signal report
    _write_json(
        history_dir / "combined_20260329.json",
        {
            "timestamp": "2026-03-29T01:00:00Z",
            "type": "combined",
            "combined": {
                "combined_score": 0.81,
                "vision_score": 0.78,
                "ocr_score": 0.84,
            },
            "branch": "main",
            "commit": "abc1234",
        },
    )
    # Seed a history_sequence report
    _write_json(
        history_dir / "history_20260329.json",
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

    bundle_json = history_dir / "eval_reporting_bundle.json"
    bundle_md = history_dir / "eval_reporting_bundle.md"
    static_out = history_dir / "report_static"
    interactive_out = history_dir / "report_interactive"

    rc = mod.main([
        "--eval-history-dir", str(history_dir),
        "--bundle-json", str(bundle_json),
        "--bundle-md", str(bundle_md),
        "--static-report-out", str(static_out),
        "--interactive-report-out", str(interactive_out),
    ])

    assert rc == 0

    # Top-level bundle manifest
    assert bundle_json.exists()
    assert bundle_md.exists()
    manifest = json.loads(bundle_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["surface_kind"] == "eval_reporting_bundle"
    assert "generated_at" in manifest
    assert "eval_history_dir" in manifest
    assert "eval_signal_bundle_json" in manifest
    assert "history_sequence_bundle_json" in manifest
    assert "static_report_html" in manifest
    assert "interactive_report_html" in manifest
    assert "plots_dir" in manifest

    # Sub-bundles materialized
    assert Path(manifest["eval_signal_bundle_json"]).exists()
    assert Path(manifest["history_sequence_bundle_json"]).exists()

    # HTML reports at distinct paths
    static_html = Path(manifest["static_report_html"])
    interactive_html = Path(manifest["interactive_report_html"])
    assert static_html.exists()
    assert interactive_html.exists()
    assert str(static_html) != str(interactive_html)

    # Both are valid HTML
    assert "Evaluation Report" in static_html.read_text(encoding="utf-8")
    assert "Evaluation Report" in interactive_html.read_text(encoding="utf-8")

    # Bundle markdown
    md_text = bundle_md.read_text(encoding="utf-8")
    assert "Eval Reporting Bundle" in md_text
    assert "report_static" in md_text
    assert "report_interactive" in md_text


def test_eval_reporting_bundle_does_not_introduce_new_metrics_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The top-level bundle script must not contain any metrics computation."""
    import ast
    from scripts.ci import generate_eval_reporting_bundle as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    # Must not define any summary / metrics / trend building functions
    forbidden_prefixes = ("_build_summary", "_build_window", "_collect_reports", "plot_", "_mean_from")
    for name in function_names:
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), (
                f"Top-level bundle must not own metrics logic, but defines {name}"
            )


def test_static_and_interactive_reports_still_work_standalone(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Both report scripts must still be runnable independently."""
    from scripts import generate_eval_report as static_mod
    from scripts import generate_eval_report_v2 as interactive_mod

    history_dir = tmp_path / "eval_history"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        static_mod,
        "get_git_info",
        lambda: {"branch": "main", "commit": "abc1234", "tag": None},
    )

    _write_json(
        history_dir / "eval_signal_experiment_summary.json",
        {
            "status": "ok",
            "surface_kind": "eval_signal_experiment_summary",
            "report_count": 1,
            "report_counts": {"combined": 1, "ocr": 0, "hybrid_blind": 0},
            "report_rows": [
                {
                    "timestamp": "2026-03-29T01:00:00Z",
                    "report_type": "combined",
                    "report_path": str(history_dir / "combined.json"),
                    "combined": {"combined_score": 0.81, "vision_score": 0.78, "ocr_score": 0.84},
                }
            ],
            "latest_combined_run": {
                "timestamp": "2026-03-29T01:00:00Z",
                "report_type": "combined",
                "report_path": str(history_dir / "combined.json"),
                "combined": {"combined_score": 0.81, "vision_score": 0.78, "ocr_score": 0.84},
            },
        },
    )

    # static report standalone
    static_out = tmp_path / "standalone_static"
    rc = static_mod.main(["--history-dir", str(history_dir), "--out", str(static_out)])
    assert rc == 0
    assert (static_out / "index.html").exists()

    # interactive report standalone — now returns explicit int
    interactive_out = tmp_path / "standalone_interactive"
    rc_v2 = interactive_mod.main(["--dir", str(history_dir), "--out", str(interactive_out)])
    assert rc_v2 == 0
    assert (interactive_out / "index.html").exists()


def test_interactive_report_failure_causes_bundle_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """If the interactive report returns non-zero, the top-level bundle must fail-closed."""
    from scripts.ci import generate_eval_reporting_bundle as mod
    from scripts import generate_eval_report_v2 as v2_mod

    history_dir = tmp_path / "eval_history"
    monkeypatch.chdir(tmp_path)

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

    # Monkeypatch interactive report to return failure
    monkeypatch.setattr(v2_mod, "main", lambda argv: 1)

    bundle_json = history_dir / "eval_reporting_bundle.json"
    rc = mod.main([
        "--eval-history-dir", str(history_dir),
        "--bundle-json", str(bundle_json),
    ])

    assert rc != 0
    assert not bundle_json.exists()
