from __future__ import annotations

import json
from pathlib import Path

from scripts.export_benchmark_knowledge_source_coverage import (
    main as export_knowledge_source_coverage_main,
)
from src.core.benchmark.knowledge_source_coverage import (
    build_knowledge_source_coverage_status,
    knowledge_source_coverage_recommendations,
    render_knowledge_source_coverage_markdown,
)


def test_build_knowledge_source_coverage_status_marks_core_ready_and_expansion_ready() -> None:
    payload = build_knowledge_source_coverage_status(
        {
            "tolerance": {
                "source_tables": {
                    "tolerance_grades": 10,
                    "common_fits": 5,
                }
            },
            "standards": {
                "source_tables": {
                    "metric_threads": 10,
                    "bearing_database": 5,
                }
            },
            "design_standards": {
                "source_tables": {
                    "linear_tolerance_table": 10,
                    "standard_fillets": 4,
                }
            },
            "gdt": {
                "source_tables": {
                    "gdt_symbols": 10,
                    "tolerance_zone_shapes": 4,
                }
            },
            "machining": {
                "source_tables": {
                    "tool_database": 4,
                    "machinability_database": 5,
                }
            },
        }
    )

    assert payload["status"] == "knowledge_source_coverage_ready"
    assert payload["source_groups"]["tolerance"]["status"] == "ready"
    assert payload["source_groups"]["design_standards"]["source_tables"]["standard_fillets"] == 4
    assert payload["expansion_candidates"][0]["name"] == "machining"
    assert payload["expansion_candidates"][0]["status"] == "ready"
    assert knowledge_source_coverage_recommendations(payload) == [
        "Promote next-wave manufacturing knowledge into benchmark views: machining"
    ]


def test_build_knowledge_source_coverage_status_surfaces_missing_focus_areas() -> None:
    payload = build_knowledge_source_coverage_status(
        {
            "tolerance": {
                "source_tables": {
                    "tolerance_grades": 10,
                    "common_fits": 0,
                }
            },
            "standards": {"source_tables": {}},
            "design_standards": {"source_tables": {"surface_finish_table": 3}},
            "gdt": {"source_tables": {"gdt_symbols": 0}},
        }
    )

    assert payload["status"] == "knowledge_source_coverage_partial"
    assert payload["priority_domains"] == ["tolerance", "standards", "gdt"]
    assert payload["focus_areas"] == ["tolerance", "standards", "gdt"]
    recommendations = knowledge_source_coverage_recommendations(payload)
    assert "Restore tolerance source tables: common_fits" in recommendations
    assert any("reference standards non-zero" in item for item in recommendations)
    rendered = render_knowledge_source_coverage_markdown(
        {
            "knowledge_source_coverage": payload,
            "recommendations": recommendations,
        },
        "Benchmark Knowledge Source Coverage",
    )
    assert "## Expansion Candidates" in rendered
    assert "Restore tolerance source tables: common_fits" in rendered


def test_export_benchmark_knowledge_source_coverage_outputs_files(
    tmp_path: Path, monkeypatch
) -> None:
    output_json = tmp_path / "knowledge_source_coverage.json"
    output_md = tmp_path / "knowledge_source_coverage.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_benchmark_knowledge_source_coverage.py",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    export_knowledge_source_coverage_main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    component = payload["knowledge_source_coverage"]
    assert component["status"] in {
        "knowledge_source_coverage_ready",
        "knowledge_source_coverage_partial",
    }
    assert "domains" in component
    assert output_md.read_text(encoding="utf-8").startswith(
        "# Benchmark Knowledge Source Coverage"
    )
