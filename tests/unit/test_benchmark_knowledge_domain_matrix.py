from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.export_benchmark_knowledge_domain_matrix import build_summary
from src.core.benchmark import (
    build_knowledge_domain_matrix_status,
    knowledge_domain_matrix_recommendations,
    render_knowledge_domain_matrix_markdown,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_knowledge_domain_matrix.py"
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_knowledge_domain_matrix_status_ready() -> None:
    payload = build_knowledge_domain_matrix_status(
        knowledge_readiness_summary={
            "knowledge_readiness": {
                "domains": {
                    "tolerance": {"status": "ready"},
                    "standards": {"status": "ready"},
                    "gdt": {"status": "ready"},
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
        knowledge_realdata_correlation_summary={
            "knowledge_realdata_correlation": {
                "domains": {
                    "tolerance": {"status": "ready"},
                    "standards": {"status": "ready"},
                    "gdt": {"status": "ready"},
                }
            }
        },
    )
    assert payload["status"] == "knowledge_domain_matrix_ready"
    assert payload["ready_domain_count"] == 3
    assert payload["domains"]["tolerance"]["status"] == "ready"


def test_build_knowledge_domain_matrix_status_partial_and_blocked() -> None:
    payload = build_knowledge_domain_matrix_status(
        knowledge_readiness_summary={
            "knowledge_readiness": {
                "domains": {
                    "tolerance": {
                        "status": "partial",
                        "focus_components": ["tolerance"],
                        "missing_metrics": ["common_fit_count"],
                        "action": "Backfill tolerance coverage.",
                    },
                    "standards": {
                        "status": "missing",
                        "focus_components": ["standards"],
                        "missing_metrics": ["thread_count"],
                        "action": "Expand standards coverage.",
                    },
                    "gdt": {"status": "ready"},
                }
            }
        },
        knowledge_application_summary={
            "knowledge_application": {
                "domains": {
                    "tolerance": {
                        "status": "partial",
                        "domain": "tolerance",
                        "signal_count": 1,
                        "action": "Promote tolerance evidence.",
                    },
                    "standards": {
                        "status": "missing",
                        "domain": "standards",
                        "signal_count": 0,
                        "action": "Promote standards evidence.",
                    },
                    "gdt": {
                        "status": "ready",
                        "domain": "gdt",
                        "signal_count": 3,
                        "action": "Keep GD&T evidence healthy.",
                    },
                }
            }
        },
        knowledge_realdata_correlation_summary={
            "knowledge_realdata_correlation": {
                "domains": {
                    "tolerance": {
                        "status": "partial",
                        "partial_realdata_components": ["step_dir"],
                        "action": "Raise tolerance real-data coverage.",
                    },
                    "standards": {
                        "status": "blocked",
                        "missing_realdata_components": ["history_h5"],
                        "blocked_realdata_components": ["step_smoke"],
                        "action": "Raise standards real-data coverage.",
                    },
                    "gdt": {
                        "status": "ready",
                        "ready_realdata_components": ["hybrid_dxf", "step_dir"],
                    },
                }
            }
        },
    )
    assert payload["status"] == "knowledge_domain_matrix_partial"
    assert payload["domains"]["tolerance"]["status"] == "partial"
    assert payload["domains"]["standards"]["status"] == "blocked"
    assert payload["domains"]["standards"]["priority"] == "high"
    assert "realdata:step_smoke" in payload["domains"]["standards"]["missing_metrics"]
    recommendations = knowledge_domain_matrix_recommendations(payload)
    assert any("Backfill standards foundation" in item for item in recommendations)
    assert any("Promote standards application evidence" in item for item in recommendations)
    assert any("Expand standards real-data coverage" in item for item in recommendations)


def test_export_benchmark_knowledge_domain_matrix_cli(tmp_path: Path) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "knowledge_readiness": {
                "domains": {
                    "tolerance": {"status": "ready"},
                    "standards": {"status": "partial", "missing_metrics": ["thread_count"]},
                    "gdt": {"status": "ready"},
                }
            }
        },
    )
    application = _write_json(
        tmp_path / "application.json",
        {
            "knowledge_application": {
                "domains": {
                    "tolerance": {"status": "ready", "signal_count": 4},
                    "standards": {"status": "partial", "signal_count": 1},
                    "gdt": {"status": "ready", "signal_count": 4},
                }
            }
        },
    )
    realdata = _write_json(
        tmp_path / "realdata.json",
        {
            "knowledge_realdata_correlation": {
                "domains": {
                    "tolerance": {"status": "ready"},
                    "standards": {
                        "status": "partial",
                        "partial_realdata_components": ["history_h5"],
                    },
                    "gdt": {"status": "ready"},
                }
            }
        },
    )
    output_json = tmp_path / "knowledge_domain_matrix.json"
    output_md = tmp_path / "knowledge_domain_matrix.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--knowledge-readiness",
            str(readiness),
            "--knowledge-application",
            str(application),
            "--knowledge-realdata-correlation",
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
    assert (
        payload["knowledge_domain_matrix"]["status"]
        == "knowledge_domain_matrix_partial"
    )
    assert (
        payload["knowledge_domain_matrix"]["domains"]["standards"]["status"]
        == "partial"
    )
    assert output_json.exists()
    assert output_md.exists()


def test_render_knowledge_domain_matrix_markdown() -> None:
    rendered = render_knowledge_domain_matrix_markdown(
        build_summary(
            title="Benchmark Knowledge Domain Matrix",
            knowledge_readiness_summary={
                "knowledge_readiness": {
                    "domains": {
                        "tolerance": {"status": "ready"},
                        "standards": {"status": "partial", "missing_metrics": ["thread_count"]},
                        "gdt": {"status": "ready"},
                    }
                }
            },
            knowledge_application_summary={
                "knowledge_application": {
                    "domains": {
                        "tolerance": {"status": "ready", "signal_count": 4},
                        "standards": {"status": "partial", "signal_count": 1},
                        "gdt": {"status": "ready", "signal_count": 4},
                    }
                }
            },
            knowledge_realdata_correlation_summary={
                "knowledge_realdata_correlation": {
                    "domains": {
                        "tolerance": {"status": "ready"},
                        "standards": {
                            "status": "partial",
                            "partial_realdata_components": ["history_h5"],
                        },
                        "gdt": {"status": "ready"},
                    }
                }
            },
            artifact_paths={},
        ),
        "Benchmark Knowledge Domain Matrix",
    )
    assert "# Benchmark Knowledge Domain Matrix" in rendered
    assert "## Domains" in rendered
    assert "### standards" in rendered
