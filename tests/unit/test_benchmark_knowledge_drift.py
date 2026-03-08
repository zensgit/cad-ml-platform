from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.benchmark.knowledge_drift import (
    build_knowledge_drift_status,
    knowledge_drift_recommendations,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_drift.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_knowledge_drift_status_improved() -> None:
    payload = build_knowledge_drift_status(
        {
            "knowledge_readiness": {
                "status": "knowledge_foundation_ready",
                "total_reference_items": 200,
                "focus_areas": [],
                "components": {
                    "tolerance": {"status": "ready", "total_reference_items": 20},
                    "standards": {"status": "ready", "total_reference_items": 60},
                },
            }
        },
        {
            "knowledge_readiness": {
                "status": "knowledge_foundation_partial",
                "total_reference_items": 120,
                "focus_areas": ["standards"],
                "components": {
                    "tolerance": {"status": "ready", "total_reference_items": 20},
                    "standards": {"status": "partial", "total_reference_items": 30},
                },
            }
        },
    )

    assert payload["status"] == "improved"
    assert payload["reference_item_delta"] == 80
    assert payload["improvements"] == ["standards"]
    assert payload["resolved_focus_areas"] == ["standards"]
    recs = knowledge_drift_recommendations(payload)
    assert any("Promote the improved knowledge baseline" in item for item in recs)


def test_build_knowledge_drift_status_regressed() -> None:
    payload = build_knowledge_drift_status(
        {
            "knowledge_readiness": {
                "status": "knowledge_foundation_partial",
                "total_reference_items": 90,
                "focus_areas": ["gdt"],
                "components": {
                    "gdt": {"status": "missing", "total_reference_items": 0},
                },
            }
        },
        {
            "knowledge_readiness": {
                "status": "knowledge_foundation_ready",
                "total_reference_items": 130,
                "focus_areas": [],
                "components": {
                    "gdt": {"status": "ready", "total_reference_items": 40},
                },
            }
        },
    )

    assert payload["status"] == "regressed"
    assert payload["regressions"] == ["gdt"]
    assert payload["new_focus_areas"] == ["gdt"]
    recs = knowledge_drift_recommendations(payload)
    assert any("Resolve knowledge regressions" in item for item in recs)


def test_export_benchmark_knowledge_drift_cli(tmp_path: Path) -> None:
    current = _write_json(
        tmp_path / "current.json",
        {
            "knowledge_readiness": {
                "status": "knowledge_foundation_ready",
                "total_reference_items": 200,
                "focus_areas": [],
                "components": {
                    "tolerance": {"status": "ready", "total_reference_items": 20},
                    "standards": {"status": "ready", "total_reference_items": 60},
                },
            }
        },
    )
    previous = _write_json(
        tmp_path / "previous.json",
        {
            "knowledge_readiness": {
                "status": "knowledge_foundation_partial",
                "total_reference_items": 120,
                "focus_areas": ["standards"],
                "components": {
                    "tolerance": {"status": "ready", "total_reference_items": 20},
                    "standards": {"status": "partial", "total_reference_items": 30},
                },
            }
        },
    )
    output_json = tmp_path / "drift.json"
    output_md = tmp_path / "drift.md"

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
    assert payload["knowledge_drift"]["status"] == "improved"
    assert output_json.exists()
    assert output_md.exists()
    assert "Benchmark Knowledge Drift" in output_md.read_text(encoding="utf-8")
