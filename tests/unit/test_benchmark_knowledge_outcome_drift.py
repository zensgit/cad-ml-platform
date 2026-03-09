from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.benchmark.knowledge_outcome_drift import (
    build_knowledge_outcome_drift_status,
    knowledge_outcome_drift_recommendations,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_outcome_drift.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_knowledge_outcome_drift_status_improved() -> None:
    payload = build_knowledge_outcome_drift_status(
        {
            "knowledge_outcome_correlation": {
                "status": "knowledge_outcome_correlation_ready",
                "ready_domain_count": 3,
                "blocked_domain_count": 0,
                "focus_areas": [],
                "priority_domains": [],
                "domains": {
                    "tolerance": {
                        "status": "ready",
                        "best_surface": "hybrid_dxf",
                        "best_surface_score": 0.92,
                    },
                    "standards": {
                        "status": "ready",
                        "best_surface": "history_h5",
                        "best_surface_score": 0.84,
                    },
                },
            }
        },
        {
            "knowledge_outcome_correlation": {
                "status": "knowledge_outcome_correlation_partial",
                "ready_domain_count": 1,
                "blocked_domain_count": 1,
                "focus_areas": ["standards"],
                "priority_domains": ["standards"],
                "domains": {
                    "tolerance": {
                        "status": "ready",
                        "best_surface": "hybrid_dxf",
                        "best_surface_score": 0.88,
                    },
                    "standards": {
                        "status": "partial",
                        "best_surface": "history_h5",
                        "best_surface_score": 0.61,
                    },
                },
            }
        },
    )

    assert payload["status"] == "improved"
    assert payload["ready_domain_delta"] == 2
    assert payload["blocked_domain_delta"] == -1
    assert payload["domain_improvements"] == ["standards"]
    assert payload["resolved_focus_areas"] == ["standards"]
    assert payload["resolved_priority_domains"] == ["standards"]
    recs = knowledge_outcome_drift_recommendations(payload)
    assert any("Promote the improved knowledge outcome correlation" in item for item in recs)


def test_build_knowledge_outcome_drift_status_regressed() -> None:
    payload = build_knowledge_outcome_drift_status(
        {
            "knowledge_outcome_correlation": {
                "status": "knowledge_outcome_correlation_partial",
                "ready_domain_count": 0,
                "blocked_domain_count": 1,
                "focus_areas": ["gdt"],
                "priority_domains": ["gdt"],
                "domains": {
                    "gdt": {
                        "status": "blocked",
                        "best_surface": "step_smoke",
                        "best_surface_score": 0.0,
                    }
                },
            }
        },
        {
            "knowledge_outcome_correlation": {
                "status": "knowledge_outcome_correlation_ready",
                "ready_domain_count": 1,
                "blocked_domain_count": 0,
                "focus_areas": [],
                "priority_domains": [],
                "domains": {
                    "gdt": {
                        "status": "ready",
                        "best_surface": "step_dir",
                        "best_surface_score": 0.83,
                    }
                },
            }
        },
    )

    assert payload["status"] == "regressed"
    assert payload["domain_regressions"] == ["gdt"]
    assert payload["new_focus_areas"] == ["gdt"]
    assert payload["new_priority_domains"] == ["gdt"]
    recs = knowledge_outcome_drift_recommendations(payload)
    assert any("Resolve knowledge outcome regressions" in item for item in recs)


def test_export_benchmark_knowledge_outcome_drift_cli(tmp_path: Path) -> None:
    current = _write_json(
        tmp_path / "current.json",
        {
            "knowledge_outcome_correlation": {
                "status": "knowledge_outcome_correlation_ready",
                "ready_domain_count": 2,
                "blocked_domain_count": 0,
                "focus_areas": [],
                "priority_domains": [],
                "domains": {
                    "standards": {
                        "status": "ready",
                        "best_surface": "history_h5",
                        "best_surface_score": 0.84,
                    }
                },
            }
        },
    )
    previous = _write_json(
        tmp_path / "previous.json",
        {
            "knowledge_outcome_correlation": {
                "status": "knowledge_outcome_correlation_partial",
                "ready_domain_count": 1,
                "blocked_domain_count": 0,
                "focus_areas": ["standards"],
                "priority_domains": ["standards"],
                "domains": {
                    "standards": {
                        "status": "partial",
                        "best_surface": "history_h5",
                        "best_surface_score": 0.58,
                    }
                },
            }
        },
    )
    output_json = tmp_path / "outcome_drift.json"
    output_md = tmp_path / "outcome_drift.md"

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
    assert payload["knowledge_outcome_drift"]["status"] == "improved"
    assert payload["knowledge_outcome_drift"]["domain_improvements"] == ["standards"]
    assert output_json.exists()
    assert output_md.exists()
    rendered = output_md.read_text(encoding="utf-8")
    assert "Benchmark Knowledge Outcome Drift" in rendered
    assert "Domain Changes" in rendered
