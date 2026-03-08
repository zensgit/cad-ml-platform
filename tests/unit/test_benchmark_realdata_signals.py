import json
from pathlib import Path

from src.core.benchmark import (
    build_realdata_signals_status,
    realdata_signals_recommendations,
    render_realdata_signals_markdown,
)
from scripts.export_benchmark_realdata_signals import build_realdata_summary, main


def test_build_realdata_signals_status_summarizes_hybrid_h5_and_step() -> None:
    payload = build_realdata_signals_status(
        hybrid_summary={
            "sample_size": 110,
            "coarse_scores": {"hybrid_label": {"accuracy": 0.8727}},
            "exact_scores": {"hybrid_label": {"accuracy": 0.5545}},
            "confidence_stats": {"hybrid_label": {"low_conf_rate": 0.1}},
        },
        online_example_report={
            "h5_validation": {
                "status": "ok",
                "tokens_length": 5,
                "vec_shape": [5, 21],
                "prediction": {
                    "label": "轴类",
                    "confidence": 0.5069,
                    "source": "history_sequence_prototype",
                },
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
                "brep_features": {
                    "valid_3d": True,
                    "faces": 7,
                },
            },
        },
        step_dir_summary={
            "sample_size": 3,
            "status_counts": {"ok": 3},
            "valid_3d_count": 3,
            "hint_coverage_count": 2,
            "graph_schema_version_counts": {"v2": 3},
        },
    )

    assert payload["status"] == "realdata_foundation_ready"
    assert payload["component_statuses"]["hybrid_dxf"] == "ready"
    assert payload["components"]["history_h5"]["status"] == "ready"
    assert payload["components"]["step_dir"]["status"] == "ready"


def test_realdata_signals_recommendations_flag_environment_gaps() -> None:
    component = build_realdata_signals_status(
        hybrid_summary={},
        online_example_report={
            "h5_validation": {"status": "skipped_no_h5py"},
            "step_validation": {"status": "skipped_no_occ"},
        },
        step_dir_summary={},
    )

    recommendations = realdata_signals_recommendations(component)

    assert any("h5py" in item for item in recommendations)
    assert any("OCC-enabled runtime" in item for item in recommendations)
    assert component["status"] == "realdata_foundation_missing"


def test_export_benchmark_realdata_signals_outputs_files(tmp_path: Path, monkeypatch) -> None:
    hybrid_summary = tmp_path / "hybrid.json"
    online_report = tmp_path / "online.json"
    step_dir_summary = tmp_path / "step.json"
    output_json = tmp_path / "realdata.json"
    output_md = tmp_path / "realdata.md"

    hybrid_summary.write_text(
        json.dumps(
            {
                "sample_size": 10,
                "coarse_scores": {"hybrid_label": {"accuracy": 0.8}},
                "confidence_stats": {"hybrid_label": {"low_conf_rate": 0.2}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    online_report.write_text(
        json.dumps(
            {
                "h5_validation": {
                    "status": "ok",
                    "tokens_length": 5,
                    "prediction": {"label": "轴类", "confidence": 0.5},
                },
                "step_validation": {"status": "skipped_no_occ"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    step_dir_summary.write_text(
        json.dumps(
            {
                "sample_size": 3,
                "status_counts": {"ok": 3},
                "valid_3d_count": 3,
                "hint_coverage_count": 2,
                "graph_schema_version_counts": {"v2": 3},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "export_benchmark_realdata_signals.py",
            "--hybrid-summary",
            str(hybrid_summary),
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

    main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["realdata_signals"]["components"]["hybrid_dxf"]["sample_size"] == 10
    assert payload["realdata_signals"]["components"]["step_dir"]["graph_schema_versions"] == {
        "v2": 3
    }
    assert "## Components" in output_md.read_text(encoding="utf-8")
    assert "history_h5" in output_md.read_text(encoding="utf-8")


def test_render_realdata_signals_markdown_includes_component_sections() -> None:
    rendered = render_realdata_signals_markdown(
        {
            "realdata_signals": {
                "status": "realdata_foundation_partial",
                "ready_component_count": 2,
                "partial_component_count": 1,
                "environment_blocked_count": 1,
                "components": {
                    "hybrid_dxf": {"status": "ready", "sample_size": 10},
                    "history_h5": {"status": "ready", "sequence_length": 5},
                    "step_smoke": {
                        "status": "environment_blocked",
                        "input_status": "skipped_no_occ",
                    },
                    "step_dir": {"status": "ready", "sample_size": 3},
                },
            },
            "recommendations": ["Provide an OCC-enabled runtime."],
        },
        "Benchmark Real-Data Signals",
    )

    assert "# Benchmark Real-Data Signals" in rendered
    assert "### hybrid_dxf" in rendered
    assert "### step_smoke" in rendered
    assert "Provide an OCC-enabled runtime." in rendered
