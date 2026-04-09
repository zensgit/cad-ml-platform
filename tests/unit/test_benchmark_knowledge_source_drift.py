from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.benchmark.knowledge_source_drift import (
    build_knowledge_source_drift_status,
    knowledge_source_drift_recommendations,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_source_drift.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_knowledge_source_drift_status_improved() -> None:
    payload = build_knowledge_source_drift_status(
        {
            "knowledge_source_coverage": {
                "status": "knowledge_source_coverage_ready",
                "ready_source_group_count": 4,
                "missing_source_group_count": 0,
                "focus_areas": [],
                "priority_domains": [],
                "expansion_candidates": [{"name": "machining"}],
                "source_groups": {
                    "standards": {
                        "status": "ready",
                        "domain": "standards",
                        "source_item_count": 12,
                        "missing_source_tables": [],
                    }
                },
            }
        },
        {
            "knowledge_source_coverage": {
                "status": "knowledge_source_coverage_partial",
                "ready_source_group_count": 3,
                "missing_source_group_count": 1,
                "focus_areas": ["standards"],
                "priority_domains": ["standards"],
                "expansion_candidates": [],
                "source_groups": {
                    "standards": {
                        "status": "partial",
                        "domain": "standards",
                        "source_item_count": 4,
                        "missing_source_tables": ["metric_threads"],
                    }
                },
            }
        },
    )

    assert payload["status"] == "improved"
    assert payload["ready_source_group_delta"] == 1
    assert payload["missing_source_group_delta"] == -1
    assert payload["source_group_improvements"] == ["standards"]
    assert payload["resolved_focus_areas"] == ["standards"]
    assert payload["resolved_priority_domains"] == ["standards"]
    recs = knowledge_source_drift_recommendations(payload)
    assert any("Promote the improved knowledge source coverage" in item for item in recs)


def test_build_knowledge_source_drift_status_regressed() -> None:
    payload = build_knowledge_source_drift_status(
        {
            "knowledge_source_coverage": {
                "status": "knowledge_source_coverage_partial",
                "ready_source_group_count": 2,
                "missing_source_group_count": 1,
                "focus_areas": ["gdt"],
                "priority_domains": ["gdt"],
                "expansion_candidates": [{"name": "welding"}],
                "source_groups": {
                    "gdt": {
                        "status": "missing",
                        "domain": "gdt",
                        "source_item_count": 0,
                        "missing_source_tables": ["gdt_symbols"],
                    }
                },
            }
        },
        {
            "knowledge_source_coverage": {
                "status": "knowledge_source_coverage_ready",
                "ready_source_group_count": 3,
                "missing_source_group_count": 0,
                "focus_areas": [],
                "priority_domains": [],
                "expansion_candidates": [],
                "source_groups": {
                    "gdt": {
                        "status": "ready",
                        "domain": "gdt",
                        "source_item_count": 7,
                        "missing_source_tables": [],
                    }
                },
            }
        },
    )

    assert payload["status"] == "regressed"
    assert payload["source_group_regressions"] == ["gdt"]
    assert payload["new_focus_areas"] == ["gdt"]
    assert payload["new_priority_domains"] == ["gdt"]
    recs = knowledge_source_drift_recommendations(payload)
    assert any("Restore regressed knowledge source groups" in item for item in recs)


def test_export_benchmark_knowledge_source_drift_cli(tmp_path: Path) -> None:
    current = _write_json(
        tmp_path / "current.json",
        {
            "knowledge_source_coverage": {
                "status": "knowledge_source_coverage_ready",
                "ready_source_group_count": 4,
                "missing_source_group_count": 0,
                "focus_areas": [],
                "priority_domains": [],
                "source_groups": {
                    "standards": {
                        "status": "ready",
                        "source_item_count": 12,
                        "missing_source_tables": [],
                    }
                },
            }
        },
    )
    previous = _write_json(
        tmp_path / "previous.json",
        {
            "knowledge_source_coverage": {
                "status": "knowledge_source_coverage_partial",
                "ready_source_group_count": 3,
                "missing_source_group_count": 1,
                "focus_areas": ["standards"],
                "priority_domains": ["standards"],
                "source_groups": {
                    "standards": {
                        "status": "partial",
                        "source_item_count": 4,
                        "missing_source_tables": ["metric_threads"],
                    }
                },
            }
        },
    )
    output_json = tmp_path / "source_drift.json"
    output_md = tmp_path / "source_drift.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--current-summary",
            str(current),
            "--previous-summary",
            str(previous),
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
    assert payload["knowledge_source_drift"]["status"] == "improved"
    assert payload["knowledge_source_drift"]["source_group_improvements"] == ["standards"]
    assert output_json.exists()
    assert output_md.exists()
    rendered = output_md.read_text(encoding="utf-8")
    assert "Benchmark Knowledge Source Drift" in rendered
    assert "Source Group Changes" in rendered
