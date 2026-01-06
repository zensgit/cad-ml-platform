from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "vision_cad_feature_compare_export.py"
)


def test_compare_export_json_csv(tmp_path: Path) -> None:
    payload = {
        "results": [{"thresholds": {"min_area": 12}}],
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
                            "components_delta": -5,
                        },
                        {
                            "name": "sample-b",
                            "lines_delta": -1,
                            "circles_delta": 1,
                            "arcs_delta": 0,
                            "ink_ratio_delta": 0.0,
                            "components_delta": -1,
                        },
                    ],
                }
            ]
        },
    }
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "out.json"
    output_csv = tmp_path / "out.csv"
    input_json.write_text(json.dumps(payload))

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--top-samples",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(output_json.read_text())
    assert data["top_samples"] == 1
    assert len(data["combo_exports"]) == 1
    export = data["combo_exports"][0]
    assert export["status"] == "ok"
    assert export["combo_index"] == 1
    assert len(export["top_samples"]) == 1
    assert export["top_samples"][0]["name"] == "sample-a"

    with output_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "combo_index",
        "status",
        "sample",
        "lines_delta",
        "circles_delta",
        "arcs_delta",
        "ink_ratio_delta",
        "components_delta",
        "thresholds",
    ]
    assert len(rows) == 1
    assert rows[0]["sample"] == "sample-a"
    assert rows[0]["status"] == "ok"


def test_compare_export_combo_out_of_range(tmp_path: Path) -> None:
    payload = {
        "results": [{"thresholds": {"min_area": 12}}],
        "comparison": {"combo_deltas": [{"summary_delta": {}, "sample_deltas": []}]},
    }
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(payload))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-json",
            str(input_json),
            "--combo-index",
            "2",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "combo index out of range" in result.stderr.lower()


def test_compare_export_missing_baseline(tmp_path: Path) -> None:
    payload = {"results": [], "comparison": {"combo_deltas": [{"missing_baseline": True}]}}
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "out.json"
    output_csv = tmp_path / "out.csv"
    input_json.write_text(json.dumps(payload))

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(output_json.read_text())
    assert len(data["combo_exports"]) == 1
    export = data["combo_exports"][0]
    assert export["status"] == "missing_baseline"
    assert export["top_samples"] == []

    with output_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["status"] == "missing_baseline"
    assert rows[0]["sample"] == ""


def test_compare_export_invalid_combo_index(tmp_path: Path) -> None:
    payload = {
        "results": [{"thresholds": {"min_area": 12}}],
        "comparison": {"combo_deltas": [{"summary_delta": {}, "sample_deltas": []}]},
    }
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(payload))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-json",
            str(input_json),
            "--combo-index",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "combo index must be >= 1" in result.stderr.lower()


def test_compare_export_combo_filter(tmp_path: Path) -> None:
    payload = {
        "results": [{"thresholds": {"min_area": 12}}, {"thresholds": {"min_area": 24}}],
        "comparison": {
            "combo_deltas": [
                {
                    "summary_delta": {"total_lines": -1},
                    "sample_deltas": [
                        {
                            "name": "sample-a",
                            "lines_delta": -1,
                            "circles_delta": 0,
                            "arcs_delta": 0,
                            "ink_ratio_delta": 0.0,
                            "components_delta": -1,
                        }
                    ],
                },
                {
                    "summary_delta": {"total_lines": -2},
                    "sample_deltas": [
                        {
                            "name": "sample-b",
                            "lines_delta": -2,
                            "circles_delta": 0,
                            "arcs_delta": 0,
                            "ink_ratio_delta": 0.0,
                            "components_delta": -2,
                        }
                    ],
                },
            ]
        },
    }
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "out.json"
    output_csv = tmp_path / "out.csv"
    input_json.write_text(json.dumps(payload))

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--combo-index",
            "2",
            "--top-samples",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(output_json.read_text())
    assert len(data["combo_exports"]) == 1
    export = data["combo_exports"][0]
    assert export["combo_index"] == 2
    assert export["top_samples"][0]["name"] == "sample-b"

    with output_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["sample"] == "sample-b"


def test_compare_export_stdout_output(tmp_path: Path) -> None:
    payload = {
        "results": [{"thresholds": {"min_area": 12}}],
        "comparison": {
            "combo_deltas": [
                {
                    "summary_delta": {"total_lines": -1},
                    "sample_deltas": [
                        {
                            "name": "sample-a",
                            "lines_delta": -1,
                            "circles_delta": 0,
                            "arcs_delta": 0,
                            "ink_ratio_delta": 0.0,
                            "components_delta": -1,
                        }
                    ],
                }
            ]
        },
    }
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(payload))

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-json",
            str(input_json),
            "--top-samples",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout)
    assert output["top_samples"] == 1
    assert len(output["combo_exports"]) == 1
