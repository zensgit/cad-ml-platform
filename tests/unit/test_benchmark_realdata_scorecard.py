from __future__ import annotations

import json
from pathlib import Path

from scripts.export_benchmark_artifact_bundle import build_bundle
from scripts.export_benchmark_companion_summary import build_companion_summary
from scripts.export_benchmark_realdata_scorecard import (
    build_realdata_scorecard_summary,
    main as realdata_scorecard_main,
)
from scripts.export_benchmark_release_decision import build_release_decision
from scripts.export_benchmark_release_runbook import build_release_runbook
from src.core.benchmark import build_realdata_scorecard_status


def _realdata_scorecard_input() -> dict:
    return {
        "hybrid_summary": {
            "sample_size": 110,
            "coarse_scores": {
                "hybrid_label": {"accuracy": 0.8727},
                "graph2d_label": {"accuracy": 0.1182},
            },
            "exact_scores": {"hybrid_label": {"accuracy": 0.5545}},
            "confidence_stats": {"hybrid_label": {"low_conf_rate": 0.1}},
        },
        "history_summary": {
            "total": 12,
            "ok_count": 12,
            "accuracy_overall": 0.8333,
            "coarse_accuracy_overall": 0.9167,
            "macro_f1_overall": 0.8,
            "coarse_macro_f1_overall": 0.9,
            "low_conf_rate": 0.05,
            "coarse_top_mismatches": [],
        },
        "online_example_report": {
            "h5_validation": {
                "status": "ok",
                "tokens_length": 5,
                "vec_shape": [5, 21],
                "prediction": {"label": "轴类", "confidence": 0.5},
            },
            "step_validation": {
                "status": "ok",
                "shape_loaded": True,
                "brep_graph": {
                    "valid_3d": True,
                    "graph_schema_version": "v2",
                    "node_count": 7,
                    "edge_count": 28,
                },
                "brep_features": {"valid_3d": True, "faces": 7},
            },
        },
        "step_dir_summary": {
            "sample_size": 3,
            "status_counts": {"ok": 3},
            "valid_3d_count": 3,
            "hint_coverage_count": 2,
            "graph_schema_version_counts": {"v2": 3},
        },
    }


def test_build_realdata_scorecard_status_summarizes_cross_surface_outcomes() -> None:
    payload = build_realdata_scorecard_status(**_realdata_scorecard_input())

    assert payload["status"] == "realdata_scorecard_ready"
    assert payload["best_surface"] == "history_h5"
    assert payload["component_statuses"]["hybrid_dxf"] == "ready"
    assert payload["components"]["hybrid_dxf"]["hybrid_minus_graph2d"] > 0.7
    assert payload["components"]["history_h5"]["coarse_accuracy"] == 0.9167
    assert payload["components"]["step_dir"]["coverage_ratio"] == 1.0


def test_export_benchmark_realdata_scorecard_outputs_files(
    tmp_path: Path, monkeypatch
) -> None:
    hybrid_summary = tmp_path / "hybrid.json"
    history_summary = tmp_path / "history.json"
    online_report = tmp_path / "online.json"
    step_dir_summary = tmp_path / "step.json"
    output_json = tmp_path / "scorecard.json"
    output_md = tmp_path / "scorecard.md"
    data = _realdata_scorecard_input()

    hybrid_summary.write_text(json.dumps(data["hybrid_summary"]), encoding="utf-8")
    history_summary.write_text(json.dumps(data["history_summary"]), encoding="utf-8")
    online_report.write_text(
        json.dumps(data["online_example_report"]), encoding="utf-8"
    )
    step_dir_summary.write_text(
        json.dumps(data["step_dir_summary"]), encoding="utf-8"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "export_benchmark_realdata_scorecard.py",
            "--hybrid-summary",
            str(hybrid_summary),
            "--history-summary",
            str(history_summary),
            "--online-example-report",
            str(online_report),
            "--step-dir-summary",
            str(step_dir_summary),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    realdata_scorecard_main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["realdata_scorecard"]["status"] == "realdata_scorecard_ready"
    assert payload["realdata_scorecard"]["best_surface"] == "history_h5"
    rendered = output_md.read_text(encoding="utf-8")
    assert "# Benchmark Real-Data Scorecard" in rendered
    assert "## Components" in rendered
    assert "hybrid_dxf" in rendered
    assert "history_h5" in rendered


def test_realdata_scorecard_surfaces_propagate_status_and_guidance() -> None:
    scorecard = build_realdata_scorecard_summary(
        title="Benchmark Real-Data Scorecard",
        artifact_paths={},
        **_realdata_scorecard_input(),
    )
    companion = build_companion_summary(
        title="Benchmark Companion",
        benchmark_scorecard={},
        benchmark_operational_summary={},
        benchmark_artifact_bundle={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_realdata_scorecard=scorecard,
        benchmark_operator_adoption={},
        artifact_paths={"benchmark_realdata_scorecard": "realdata_scorecard.json"},
    )
    bundle = build_bundle(
        title="Benchmark Artifact Bundle",
        benchmark_scorecard={},
        benchmark_operational_summary={},
        benchmark_companion_summary=companion,
        benchmark_release_decision={},
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_realdata_scorecard=scorecard,
        benchmark_operator_adoption={},
        benchmark_knowledge_application={},
        benchmark_knowledge_realdata_correlation={},
        benchmark_knowledge_domain_matrix={},
        feedback_flywheel={},
        assistant_evidence={},
        review_queue={},
        ocr_review={},
        artifact_paths={"benchmark_realdata_scorecard": "realdata_scorecard.json"},
    )
    decision = build_release_decision(
        title="Benchmark Release Decision",
        benchmark_scorecard={},
        benchmark_operational_summary={},
        benchmark_artifact_bundle=bundle,
        benchmark_companion_summary=companion,
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_realdata_scorecard=scorecard,
        benchmark_operator_adoption={},
        artifact_paths={"benchmark_realdata_scorecard": "realdata_scorecard.json"},
    )
    runbook = build_release_runbook(
        title="Benchmark Release Runbook",
        benchmark_release_decision=decision,
        benchmark_companion_summary=companion,
        benchmark_artifact_bundle=bundle,
        benchmark_knowledge_readiness={},
        benchmark_knowledge_drift={},
        benchmark_engineering_signals={},
        benchmark_realdata_signals={},
        benchmark_realdata_scorecard=scorecard,
        benchmark_operator_adoption={},
        benchmark_knowledge_application={},
        benchmark_knowledge_realdata_correlation={},
        benchmark_knowledge_domain_matrix={},
        artifact_paths={"benchmark_realdata_scorecard": "realdata_scorecard.json"},
    )

    assert companion["component_statuses"]["realdata_scorecard"] == (
        "realdata_scorecard_ready"
    )
    assert bundle["component_statuses"]["realdata_scorecard"] == "realdata_scorecard_ready"
    assert decision["realdata_scorecard_status"] == "realdata_scorecard_ready"
    assert runbook["realdata_scorecard_status"] == "realdata_scorecard_ready"
    assert "benchmark_realdata_scorecard" in bundle["artifacts"]
    assert "benchmark_realdata_scorecard" in decision["artifacts"]
    assert "benchmark_realdata_scorecard" in runbook["artifacts"]
    assert "benchmark_realdata_scorecard" not in runbook["missing_artifacts"]
