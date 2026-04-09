from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.benchmark.engineering_signals import (
    build_engineering_signals_status,
    engineering_signals_recommendations,
)


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "export_benchmark_engineering_signals.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_engineering_signals_status_ready() -> None:
    payload = build_engineering_signals_status(
        hybrid_summary={
            "sample_size": 20,
            "knowledge_signals": {
                "rows_with_checks": 10,
                "rows_with_violations": 6,
                "rows_with_standards_candidates": 8,
                "rows_with_hints": 9,
                "total_checks": 15,
                "total_violations": 6,
                "total_standards_candidates": 8,
                "total_hints": 12,
                "top_violation_categories": {"tolerance": 4},
                "top_standard_types": {"gdt": 5},
                "top_hint_labels": {"hole": 4},
            },
        },
        ocr_review_summary={
            "review_candidate_count": 2,
            "exported_records": 2,
            "automation_ready_count": 1,
            "top_standards_candidates": [{"name": "GB/T 1800.2-2020", "count": 2}],
            "primary_gap_counts": [{"name": "missing_gdt", "count": 1}],
        },
    )
    assert payload["status"] == "engineering_semantics_ready"
    assert payload["coverage_ratio"] == 0.5
    assert payload["ocr_standard_signal_count"] == 2


def test_build_engineering_signals_status_partial_and_recommendations() -> None:
    payload = build_engineering_signals_status(
        hybrid_summary={
            "sample_size": 10,
            "knowledge_signals": {
                "rows_with_checks": 1,
                "rows_with_violations": 0,
                "rows_with_standards_candidates": 0,
                "rows_with_hints": 1,
                "total_checks": 1,
                "total_violations": 0,
                "total_standards_candidates": 0,
                "total_hints": 1,
            },
        },
        ocr_review_summary={},
    )
    assert payload["status"] == "partial_engineering_semantics"
    recs = engineering_signals_recommendations(payload)
    assert any("violation" in item for item in recs)
    assert any("standards/tolerance/GD&T" in item for item in recs)


def test_export_benchmark_engineering_signals_cli(tmp_path: Path) -> None:
    hybrid = _write_json(
        tmp_path / "hybrid.json",
        {
            "sample_size": 12,
            "knowledge_signals": {
                "rows_with_checks": 4,
                "rows_with_violations": 2,
                "rows_with_standards_candidates": 3,
                "rows_with_hints": 4,
                "total_checks": 6,
                "total_violations": 2,
                "total_standards_candidates": 3,
                "total_hints": 5,
                "top_violation_categories": {"standard_conflict": 2},
                "top_standard_types": {"tolerance": 3},
                "top_hint_labels": {"flange": 2},
            },
        },
    )
    ocr = _write_json(
        tmp_path / "ocr.json",
        {
            "review_candidate_count": 2,
            "exported_records": 2,
            "automation_ready_count": 1,
            "top_standards_candidates": [{"name": "GB/T 1800.2-2020", "count": 1}],
        },
    )
    output_json = tmp_path / "engineering.json"
    output_md = tmp_path / "engineering.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--title",
            "Benchmark Engineering Signals",
            "--hybrid-summary",
            str(hybrid),
            "--ocr-review-summary",
            str(ocr),
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
    assert payload["engineering_signals"]["status"] == "partial_engineering_semantics"
    assert output_json.exists()
    assert output_md.exists()
    assert "Benchmark Engineering Signals" in output_md.read_text(encoding="utf-8")
