from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.benchmark.competitive_surpass_trend import (
    build_competitive_surpass_trend_status,
    competitive_surpass_trend_recommendations,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "export_benchmark_competitive_surpass_trend.py"
)


def test_build_competitive_surpass_trend_status_improved() -> None:
    payload = build_competitive_surpass_trend_status(
        current_summary={
            "competitive_surpass_index": {
                "status": "competitive_surpass_ready",
                "score": 92,
                "ready_pillars": ["engineering", "knowledge", "realdata"],
                "partial_pillars": ["operator_adoption"],
                "blocked_pillars": [],
                "primary_gaps": ["operator_adoption"],
            }
        },
        previous_summary={
            "competitive_surpass_index": {
                "status": "competitive_surpass_partial",
                "score": 78,
                "ready_pillars": ["engineering", "knowledge"],
                "partial_pillars": ["realdata", "operator_adoption"],
                "blocked_pillars": [],
                "primary_gaps": ["realdata", "operator_adoption"],
            }
        },
    )
    assert payload["status"] == "improved"
    assert payload["score_delta"] == 14
    assert payload["pillar_improvements"] == ["realdata"]
    assert payload["resolved_primary_gaps"] == ["realdata"]
    assert competitive_surpass_trend_recommendations(payload)[0].startswith(
        "Promote the improved competitive surpass posture"
    )


def test_build_competitive_surpass_trend_status_regressed() -> None:
    payload = build_competitive_surpass_trend_status(
        current_summary={
            "competitive_surpass_index": {
                "status": "competitive_surpass_partial",
                "score": 70,
                "ready_pillars": ["engineering"],
                "partial_pillars": ["knowledge"],
                "blocked_pillars": ["realdata", "operator_adoption"],
                "primary_gaps": ["realdata", "operator_adoption"],
            }
        },
        previous_summary={
            "competitive_surpass_index": {
                "status": "competitive_surpass_ready",
                "score": 90,
                "ready_pillars": ["engineering", "knowledge", "realdata"],
                "partial_pillars": ["operator_adoption"],
                "blocked_pillars": [],
                "primary_gaps": ["operator_adoption"],
            }
        },
    )
    assert payload["status"] == "regressed"
    assert payload["score_delta"] == -20
    assert payload["pillar_regressions"] == [
        "knowledge",
        "operator_adoption",
        "realdata",
    ]
    assert payload["new_primary_gaps"] == ["realdata"]


def test_export_benchmark_competitive_surpass_trend_cli(tmp_path: Path) -> None:
    current_summary = tmp_path / "current.json"
    previous_summary = tmp_path / "previous.json"
    output_json = tmp_path / "trend.json"
    output_md = tmp_path / "trend.md"

    current_summary.write_text(
        json.dumps(
            {
                "competitive_surpass_index": {
                    "status": "competitive_surpass_ready",
                    "score": 90,
                    "ready_pillars": ["engineering", "knowledge", "realdata"],
                    "partial_pillars": ["operator_adoption"],
                    "blocked_pillars": [],
                    "primary_gaps": ["operator_adoption"],
                }
            }
        ),
        encoding="utf-8",
    )
    previous_summary.write_text(
        json.dumps(
            {
                "competitive_surpass_index": {
                    "status": "competitive_surpass_partial",
                    "score": 80,
                    "ready_pillars": ["engineering", "knowledge"],
                    "partial_pillars": ["realdata", "operator_adoption"],
                    "blocked_pillars": [],
                    "primary_gaps": ["realdata", "operator_adoption"],
                }
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--current-summary",
            str(current_summary),
            "--previous-summary",
            str(previous_summary),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["competitive_surpass_trend"]["status"] == "improved"
    assert payload["competitive_surpass_trend"]["score_delta"] == 10
    assert output_md.exists()
