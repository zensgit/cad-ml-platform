from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "vision_cad_feature_benchmark.py"
)


def _run_benchmark(tmp_path: Path, payload: dict) -> dict:
    threshold_file = tmp_path / "thresholds.json"
    threshold_file.write_text(json.dumps(payload))
    output_json = tmp_path / "out.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--no-clients",
            "--max-samples",
            "1",
            "--threshold-file",
            str(threshold_file),
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output_json.read_text())


def test_threshold_file_grid(tmp_path: Path) -> None:
    payload = {
        "thresholds": {"min_area": 12},
        "grid": {"arc_fill_min": [0.05, 0.08], "circle_fill_min": [0.3, 0.4]},
    }
    data = _run_benchmark(tmp_path, payload)

    assert len(data["results"]) == 4
    assert data["results"][0]["thresholds"]["min_area"] == 12


def test_threshold_file_variants(tmp_path: Path) -> None:
    payload = {
        "variants": [
            {"min_area": 12, "line_aspect": 4},
            {"min_area": 24, "line_aspect": 6},
        ]
    }
    data = _run_benchmark(tmp_path, payload)

    assert len(data["results"]) == 2
    assert data["results"][1]["thresholds"]["min_area"] == 24
