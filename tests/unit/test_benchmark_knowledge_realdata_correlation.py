from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.benchmark import (
    build_knowledge_realdata_correlation_status,
    knowledge_realdata_correlation_recommendations,
    render_knowledge_realdata_correlation_markdown,
)
from scripts.export_benchmark_knowledge_realdata_correlation import build_summary


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_realdata_correlation.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_knowledge_realdata_correlation_status_ready() -> None:
    payload = build_knowledge_realdata_correlation_status(
        knowledge_readiness_summary={
            "knowledge_readiness": {
                "domains": {
                    "tolerance": {"status": "ready", "missing_metrics": []},
                    "standards": {"status": "ready", "missing_metrics": []},
                    "gdt": {"status": "ready", "missing_metrics": []},
                }
            }
        },
        knowledge_application_summary={
            "knowledge_application": {
                "domains": {
                    "tolerance": {"status": "ready", "signal_count": 4},
                    "standards": {"status": "ready", "signal_count": 4},
                    "gdt": {"status": "ready", "signal_count": 4},
                }
            }
        },
        realdata_signals_summary={
            "realdata_signals": {
                "component_statuses": {
                    "hybrid_dxf": "ready",
                    "history_h5": "ready",
                    "step_smoke": "ready",
                    "step_dir": "ready",
                }
            }
        },
    )
    assert payload["status"] == "knowledge_realdata_ready"
    assert payload["ready_domain_count"] == 3
    assert payload["domains"]["gdt"]["realdata_status"] == "ready"


def test_build_knowledge_realdata_correlation_status_partial_and_blocked() -> None:
    payload = build_knowledge_realdata_correlation_status(
        knowledge_readiness_summary={
            "knowledge_readiness": {
                "domains": {
                    "tolerance": {
                        "status": "partial",
                        "focus_components": ["tolerance"],
                        "missing_metrics": ["common_fit_count"],
                    },
                    "standards": {
                        "status": "missing",
                        "focus_components": ["standards"],
                        "missing_metrics": ["thread_count"],
                    },
                    "gdt": {"status": "ready", "missing_metrics": []},
                }
            }
        },
        knowledge_application_summary={
            "knowledge_application": {
                "domains": {
                    "tolerance": {
                        "status": "partial",
                        "signal_count": 1,
                        "action": "Raise tolerance application coverage.",
                    },
                    "standards": {
                        "status": "missing",
                        "signal_count": 0,
                        "action": "Promote standards evidence.",
                    },
                    "gdt": {
                        "status": "ready",
                        "signal_count": 3,
                        "action": "Keep GD&T evidence healthy.",
                    },
                }
            }
        },
        realdata_signals_summary={
            "realdata_signals": {
                "component_statuses": {
                    "hybrid_dxf": "ready",
                    "history_h5": "missing",
                    "step_smoke": "environment_blocked",
                    "step_dir": "partial",
                }
            }
        },
    )
    assert payload["status"] == "knowledge_realdata_partial"
    assert payload["domains"]["tolerance"]["status"] == "partial"
    assert payload["domains"]["standards"]["status"] == "blocked"
    assert payload["domains"]["standards"]["priority"] == "high"
    recommendations = knowledge_realdata_correlation_recommendations(payload)
    assert any("Backfill standards knowledge foundation" in item for item in recommendations)
    assert any("Raise standards real-data depth" in item for item in recommendations)
    assert any("Raise tolerance real-data depth" in item for item in recommendations)


def test_export_benchmark_knowledge_realdata_correlation_cli(tmp_path: Path) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "knowledge_readiness": {
                "domains": {
                    "tolerance": {"status": "ready", "missing_metrics": []},
                    "standards": {"status": "partial", "missing_metrics": ["thread_count"]},
                    "gdt": {"status": "ready", "missing_metrics": []},
                }
            }
        },
    )
    application = _write_json(
        tmp_path / "application.json",
        {
            "knowledge_application": {
                "domains": {
                    "tolerance": {"status": "ready", "signal_count": 3},
                    "standards": {"status": "partial", "signal_count": 1},
                    "gdt": {"status": "ready", "signal_count": 3},
                }
            }
        },
    )
    realdata = _write_json(
        tmp_path / "realdata.json",
        {
            "realdata_signals": {
                "component_statuses": {
                    "hybrid_dxf": "ready",
                    "history_h5": "ready",
                    "step_smoke": "ready",
                    "step_dir": "partial",
                }
            }
        },
    )
    output_json = tmp_path / "knowledge_realdata.json"
    output_md = tmp_path / "knowledge_realdata.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--knowledge-readiness",
            str(readiness),
            "--knowledge-application",
            str(application),
            "--realdata-signals",
            str(realdata),
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
    assert payload["knowledge_realdata_correlation"]["status"] == "knowledge_realdata_partial"
    assert payload["knowledge_realdata_correlation"]["domains"]["standards"]["status"] == (
        "partial"
    )
    assert output_json.exists()
    assert output_md.exists()


def test_render_knowledge_realdata_correlation_markdown() -> None:
    rendered = render_knowledge_realdata_correlation_markdown(
        build_summary(
            title="Benchmark Knowledge Real-Data Correlation",
            knowledge_readiness_summary={
                "knowledge_readiness": {
                    "domains": {
                        "tolerance": {"status": "ready", "missing_metrics": []},
                        "standards": {
                            "status": "partial",
                            "missing_metrics": ["thread_count"],
                        },
                        "gdt": {"status": "ready", "missing_metrics": []},
                    }
                }
            },
            knowledge_application_summary={
                "knowledge_application": {
                    "domains": {
                        "tolerance": {"status": "ready", "signal_count": 3},
                        "standards": {"status": "partial", "signal_count": 1},
                        "gdt": {"status": "ready", "signal_count": 2},
                    }
                }
            },
            realdata_signals_summary={
                "realdata_signals": {
                    "component_statuses": {
                        "hybrid_dxf": "ready",
                        "history_h5": "partial",
                        "step_smoke": "ready",
                        "step_dir": "partial",
                    }
                }
            },
            artifact_paths={},
        ),
        "Benchmark Knowledge Real-Data Correlation",
    )
    assert "# Benchmark Knowledge Real-Data Correlation" in rendered
    assert "## Domains" in rendered
    assert "### standards" in rendered
