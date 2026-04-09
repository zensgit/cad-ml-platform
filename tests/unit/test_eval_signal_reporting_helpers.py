from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_eval_signal_report_rows_normalize_canonical_summary_payload(tmp_path: Path) -> None:
    from scripts.eval_signal_reporting_helpers import eval_signal_report_rows

    history_dir = tmp_path / "eval_history"
    summary = {
        "surface_kind": "eval_signal_experiment_summary",
        "report_rows": [
            {
                "timestamp": "2026-03-29T08:00:00Z",
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
                "timestamp": "2026-03-29T09:00:00Z",
                "report_type": "ocr",
                "report_path": str(history_dir / "ocr.json"),
                "metrics": {
                    "dimension_recall": 0.91,
                    "brier_score": 0.08,
                    "edge_f1": 0.87,
                },
            },
        ],
    }

    combined_rows = eval_signal_report_rows(
        summary,
        history_dir=history_dir,
        report_type="combined",
    )
    ocr_rows = eval_signal_report_rows(
        summary,
        history_dir=history_dir,
        report_type="ocr",
    )

    assert len(combined_rows) == 1
    assert combined_rows[0]["branch"] == "feature/test"
    assert combined_rows[0]["commit"] == "abc1234"
    assert combined_rows[0]["run_context"]["runner"] == "ci"
    assert combined_rows[0]["_file"] == "combined.json"
    assert len(ocr_rows) == 1
    assert ocr_rows[0]["metrics"]["dimension_recall"] == 0.91
    assert ocr_rows[0]["_file"] == "ocr.json"


def test_load_eval_signal_reporting_summary_uses_existing_summary_artifact(tmp_path: Path) -> None:
    from scripts.eval_signal_reporting_helpers import load_eval_signal_reporting_summary

    history_dir = tmp_path / "eval_history"
    summary_json = history_dir / "eval_signal_experiment_summary.json"
    _write_json(
        summary_json,
        {
            "status": "ok",
            "surface_kind": "eval_signal_experiment_summary",
            "report_count": 1,
            "report_counts": {"combined": 1, "ocr": 0, "hybrid_blind": 0},
            "report_rows": [
                {
                    "timestamp": "2026-03-29T08:00:00Z",
                    "report_type": "combined",
                    "report_path": str(history_dir / "combined.json"),
                    "combined": {"combined_score": 0.81},
                }
            ],
            "latest_combined_run": {
                "timestamp": "2026-03-29T08:00:00Z",
                "report_type": "combined",
                "report_path": str(history_dir / "combined.json"),
                "combined": {"combined_score": 0.81},
            },
        },
    )

    summary = load_eval_signal_reporting_summary(history_dir)

    assert isinstance(summary, dict)
    assert summary["surface_kind"] == "eval_signal_experiment_summary"
    assert summary["report_count"] == 1


def test_load_eval_signal_reporting_assets_prefers_bundle_summary(tmp_path: Path) -> None:
    from scripts.eval_signal_reporting_helpers import load_eval_signal_reporting_assets

    history_dir = tmp_path / "eval_history"
    summary_path = history_dir / "eval_signal_experiment_summary.json"
    bundle_path = history_dir / "eval_signal_reporting_bundle.json"

    _write_json(
        summary_path,
        {
            "status": "ok",
            "surface_kind": "eval_signal_experiment_summary",
            "report_count": 2,
            "report_counts": {"combined": 1, "ocr": 1, "hybrid_blind": 0},
            "report_rows": [],
        },
    )
    _write_json(
        bundle_path,
        {
            "status": "ok",
            "surface_kind": "eval_signal_reporting_bundle",
            "summary_json": str(summary_path),
        },
    )

    bundle, summary = load_eval_signal_reporting_assets(history_dir)

    assert isinstance(bundle, dict)
    assert bundle["surface_kind"] == "eval_signal_reporting_bundle"
    assert isinstance(summary, dict)
    assert summary["report_count"] == 2


def test_load_eval_signal_reporting_assets_falls_back_without_bundle(tmp_path: Path) -> None:
    from scripts.eval_signal_reporting_helpers import load_eval_signal_reporting_assets

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "eval_signal_experiment_summary.json",
        {
            "status": "ok",
            "surface_kind": "eval_signal_experiment_summary",
            "report_count": 1,
            "report_counts": {"combined": 1, "ocr": 0, "hybrid_blind": 0},
            "report_rows": [],
        },
    )

    bundle, summary = load_eval_signal_reporting_assets(history_dir)

    assert bundle is None
    assert isinstance(summary, dict)
    assert summary["report_count"] == 1


def test_build_eval_signal_report_context_includes_aggregate_metrics(tmp_path: Path) -> None:
    from scripts.eval_signal_reporting_helpers import build_eval_signal_report_context

    history_dir = tmp_path / "eval_history"
    summary = {
        "status": "ok",
        "surface_kind": "eval_signal_experiment_summary",
        "report_count": 3,
        "report_counts": {"combined": 1, "ocr": 1, "hybrid_blind": 1},
        "aggregate_metrics": {
            "hybrid_blind_accuracy_mean": 0.75,
            "hybrid_blind_graph2d_accuracy_mean": 0.70,
            "hybrid_blind_gain_mean": 0.05,
            "hybrid_blind_coverage_mean": 0.88,
            "hybrid_blind_label_slice_count_latest": 10,
            "hybrid_blind_family_slice_count_latest": 5,
        },
    }

    context = build_eval_signal_report_context(summary, history_dir=history_dir)

    assert context["available"] is True
    assert context["hybrid_blind_report_count"] == 1
    assert context["aggregate_metrics"]["hybrid_blind_accuracy_mean"] == 0.75
    assert context["aggregate_metrics"]["hybrid_blind_label_slice_count_latest"] == 10
