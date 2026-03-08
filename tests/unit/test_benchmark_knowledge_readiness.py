from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.benchmark.knowledge_readiness import (
    build_knowledge_readiness_status,
    knowledge_readiness_recommendations,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_readiness.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_knowledge_readiness_status_ready() -> None:
    payload = build_knowledge_readiness_status(
        {
            "tolerance": {"it_grade_count": 20, "common_fit_count": 10},
            "standards": {
                "thread_count": 30,
                "bearing_count": 20,
                "oring_count": 10,
            },
            "design_standards": {
                "surface_finish_grade_count": 12,
                "linear_tolerance_range_count": 8,
                "preferred_diameter_count": 25,
                "standard_chamfer_count": 10,
                "standard_fillet_count": 9,
            },
            "gdt": {
                "symbol_count": 14,
                "application_count": 8,
                "datum_feature_type_count": 4,
                "tolerance_recommendation_count": 12,
            },
        }
    )
    assert payload["status"] == "knowledge_foundation_ready"
    assert payload["ready_component_count"] == 4
    assert payload["total_reference_items"] == 192
    assert payload["focus_areas"] == []
    assert payload["focus_areas_detail"] == []


def test_build_knowledge_readiness_status_partial_and_recommendations() -> None:
    payload = build_knowledge_readiness_status(
        {
            "tolerance": {"it_grade_count": 20, "common_fit_count": 0},
            "standards": {},
            "design_standards": {
                "surface_finish_grade_count": 12,
                "linear_tolerance_range_count": 8,
                "preferred_diameter_count": 0,
                "standard_chamfer_count": 0,
                "standard_fillet_count": 0,
            },
            "gdt": {
                "symbol_count": 14,
                "application_count": 8,
                "datum_feature_type_count": 4,
                "tolerance_recommendation_count": 12,
            },
        }
    )
    assert payload["status"] == "knowledge_foundation_partial"
    assert payload["focus_areas"] == ["tolerance", "standards", "design_standards"]
    assert payload["focus_areas_detail"][0]["component"] == "tolerance"
    assert payload["focus_areas_detail"][0]["missing_metrics"] == ["common_fit_count"]
    assert payload["focus_areas_detail"][0]["priority"] == "medium"
    assert payload["focus_areas_detail"][1]["component"] == "standards"
    assert payload["focus_areas_detail"][1]["priority"] == "high"
    recs = knowledge_readiness_recommendations(payload)
    assert any("ISO 286" in item for item in recs)
    assert any("standard-part coverage" in item for item in recs)
    assert any("Lift knowledge readiness" in item for item in recs)
    assert any("Prioritize knowledge gaps" in item for item in recs)


def test_export_benchmark_knowledge_readiness_cli(tmp_path: Path) -> None:
    snapshot = _write_json(
        tmp_path / "snapshot.json",
        {
            "tolerance": {"it_grade_count": 20, "common_fit_count": 10},
            "standards": {"thread_count": 30, "bearing_count": 20, "oring_count": 10},
            "design_standards": {
                "surface_finish_grade_count": 12,
                "linear_tolerance_range_count": 8,
                "preferred_diameter_count": 25,
                "standard_chamfer_count": 10,
                "standard_fillet_count": 9,
            },
            "gdt": {
                "symbol_count": 14,
                "application_count": 8,
                "datum_feature_type_count": 4,
                "tolerance_recommendation_count": 12,
            },
        },
    )
    output_json = tmp_path / "knowledge.json"
    output_md = tmp_path / "knowledge.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--knowledge-snapshot",
            str(snapshot),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["knowledge_readiness"]["status"] == "knowledge_foundation_ready"
    assert payload["knowledge_readiness"]["focus_areas_detail"] == []
    assert output_json.exists()
    assert output_md.exists()
    assert "Benchmark Knowledge Readiness" in output_md.read_text(encoding="utf-8")
