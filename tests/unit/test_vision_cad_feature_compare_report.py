from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "vision_cad_feature_compare_report.py"
)


def test_compare_report_generation(tmp_path: Path) -> None:
    payload = {
        "results": [
            {
                "thresholds": {"min_area": 12},
            }
        ],
        "comparison": {
            "combo_deltas": [
                {
                    "summary_delta": {
                        "total_lines": -5,
                        "total_circles": 1,
                        "total_arcs": 0,
                        "avg_ink_ratio": 0.0,
                        "avg_components": -2.0,
                    },
                    "sample_deltas": [
                        {
                            "name": "sample-a",
                            "lines_delta": -2,
                            "circles_delta": 0,
                            "arcs_delta": 0,
                            "ink_ratio_delta": 0.0,
                            "components_delta": -2,
                        }
                    ],
                }
            ]
        },
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
            "--top-samples",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    content = output_md.read_text()
    assert "# CAD Feature Benchmark Comparison Summary" in content
    assert "## Combo 1" in content
    assert "| total_lines | -5 |" in content
    assert "| sample-a | -2 | 0 | 0 | -2 |" in content
