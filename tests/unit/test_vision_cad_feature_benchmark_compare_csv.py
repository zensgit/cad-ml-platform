from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "vision_cad_feature_benchmark.py"
)


def _run_benchmark(output_json: Path, extra_args: list[str]) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--no-clients",
            "--max-samples",
            "1",
            *extra_args,
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_compare_csv_output(tmp_path: Path) -> None:
    baseline_json = tmp_path / "baseline.json"
    tuned_json = tmp_path / "tuned.json"
    compare_csv = tmp_path / "compare.csv"

    _run_benchmark(baseline_json, [])
    _run_benchmark(
        tuned_json,
        ["--compare-json", str(baseline_json), "--output-compare-csv", str(compare_csv)],
    )

    assert compare_csv.exists()
    with compare_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "combo_index",
        "status",
        "total_lines_delta",
        "total_circles_delta",
        "total_arcs_delta",
        "avg_ink_ratio_delta",
        "avg_components_delta",
    ]
    assert len(rows) == 1
    assert rows[0]["combo_index"] == "1"
    assert rows[0]["status"] == "ok"


def test_compare_csv_requires_baseline(tmp_path: Path) -> None:
    compare_csv = tmp_path / "compare.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--no-clients",
            "--max-samples",
            "1",
            "--output-compare-csv",
            str(compare_csv),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "requires --compare-json" in result.stderr
