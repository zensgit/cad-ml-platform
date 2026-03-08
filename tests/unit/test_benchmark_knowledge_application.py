from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.benchmark.knowledge_application import (
    build_knowledge_application_status,
    knowledge_application_recommendations,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_application.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_knowledge_application_status_ready() -> None:
    payload = build_knowledge_application_status(
        engineering_signals_summary={
            "engineering_signals": {
                "top_check_categories": {
                    "general_tolerance": 4,
                    "thread_standard": 3,
                    "gdt": 3,
                },
                "top_violation_categories": {
                    "general_tolerance": 1,
                    "thread_standard": 1,
                    "gdt": 1,
                },
                "top_standard_types": {
                    "general_tolerance": 3,
                    "metric_thread": 2,
                    "gdt": 2,
                },
                "ocr_top_standards_candidates": {
                    "GB/T 1800.2-2020": 2,
                    "ISO 2768-m": 1,
                    "GDT datum frame": 1,
                },
            }
        },
        knowledge_readiness_summary={
            "knowledge_readiness": {
                "domains": {
                    "tolerance": {"status": "ready", "missing_metrics": []},
                    "standards": {"status": "ready", "missing_metrics": []},
                    "gdt": {"status": "ready", "missing_metrics": []},
                }
            }
        },
    )
    assert payload["status"] == "knowledge_application_ready"
    assert payload["ready_domain_count"] == 3
    assert payload["domains"]["tolerance"]["status"] == "ready"
    assert payload["domains"]["standards"]["signal_count"] >= 3


def test_build_knowledge_application_status_partial_and_recommendations() -> None:
    payload = build_knowledge_application_status(
        engineering_signals_summary={
            "engineering_signals": {
                "top_check_categories": {"general_tolerance": 1},
                "top_violation_categories": {},
                "top_standard_types": {},
                "ocr_top_standards_candidates": {},
            }
        },
        knowledge_readiness_summary={
            "knowledge_readiness": {
                "domains": {
                    "tolerance": {
                        "status": "partial",
                        "missing_metrics": ["common_fit_count"],
                        "action": "Backfill tolerance fit tables.",
                    },
                    "standards": {
                        "status": "missing",
                        "missing_metrics": ["thread_count"],
                        "action": "Expand standards dictionaries.",
                    },
                    "gdt": {
                        "status": "ready",
                        "missing_metrics": [],
                        "action": "Keep GD&T baseline healthy.",
                    },
                }
            }
        },
    )
    assert payload["status"] == "knowledge_application_partial"
    assert payload["domains"]["tolerance"]["status"] == "partial"
    assert payload["domains"]["standards"]["priority"] == "high"
    assert payload["domains"]["gdt"]["gap_reason"] == "missing_application_evidence"
    recs = knowledge_application_recommendations(payload)
    assert any("Backfill standards foundation" in item for item in recs)
    assert any("promote gdt application evidence" in item.lower() for item in recs)


def test_export_benchmark_knowledge_application_cli(tmp_path: Path) -> None:
    engineering = _write_json(
        tmp_path / "engineering.json",
        {
            "engineering_signals": {
                "top_check_categories": {
                    "general_tolerance": 2,
                    "thread_standard": 2,
                    "gdt": 1,
                },
                "top_violation_categories": {"gdt": 1},
                "top_standard_types": {
                    "general_tolerance": 1,
                    "metric_thread": 2,
                },
                "ocr_top_standards_candidates": {"GB/T 1800.2-2020": 1},
            }
        },
    )
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "knowledge_readiness": {
                "domains": {
                    "tolerance": {"status": "ready", "missing_metrics": []},
                    "standards": {"status": "ready", "missing_metrics": []},
                    "gdt": {"status": "partial", "missing_metrics": ["symbol_count"]},
                }
            }
        },
    )
    output_json = tmp_path / "knowledge_application.json"
    output_md = tmp_path / "knowledge_application.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--engineering-signals-summary",
            str(engineering),
            "--knowledge-readiness-summary",
            str(readiness),
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
    assert payload["knowledge_application"]["status"] == "knowledge_application_partial"
    assert output_json.exists()
    assert output_md.exists()
    rendered = output_md.read_text(encoding="utf-8")
    assert "Benchmark Knowledge Application" in rendered
    assert "## Domains" in rendered
