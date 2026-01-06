from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "vision_cad_feature_baseline_report.py"
)


def test_baseline_report_generation(tmp_path: Path) -> None:
    payload = {
        "results": [
            {
                "thresholds": {"min_area": 12},
                "summary": {
                    "total_lines": 5,
                    "total_circles": 1,
                    "total_arcs": 0,
                    "avg_ink_ratio": 0.01,
                    "avg_components": 2.5,
                },
            }
        ]
    }
    input_json = tmp_path / "input.json"
    output_md = tmp_path / "report.md"
    input_json.write_text(json.dumps(payload))

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-json",
            str(input_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    content = output_md.read_text()
    assert "# CAD Feature Benchmark Baseline Summary" in content
    assert "## Combo 1" in content
    assert "min_area=12" in content
    assert "| total_lines | 5 |" in content


def test_baseline_report_missing_results(tmp_path: Path) -> None:
    payload = {"base_thresholds": {"min_area": 12}}
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(payload))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-json",
            str(input_json),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "results" in result.stderr.lower()


def test_baseline_report_multi_combo(tmp_path: Path) -> None:
    payload = {
        "results": [
            {"thresholds": {"min_area": 12}, "summary": {"total_lines": 1}},
            {"thresholds": {"min_area": 24}, "summary": {"total_lines": 2}},
        ]
    }
    input_json = tmp_path / "input.json"
    output_md = tmp_path / "report.md"
    input_json.write_text(json.dumps(payload))

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-json",
            str(input_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    content = output_md.read_text()
    assert "## Combo 1" in content
    assert "## Combo 2" in content
    assert "min_area=24" in content
