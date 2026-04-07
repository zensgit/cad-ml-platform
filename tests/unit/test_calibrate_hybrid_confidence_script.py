from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_calibrate_hybrid_confidence_ok_with_per_source(tmp_path: Path) -> None:
    from scripts import calibrate_hybrid_confidence as mod
    from src.ml.hybrid.calibration import CalibrationMethod

    rows: list[dict[str, object]] = []
    for i in range(40):
        conf = 0.9 if i < 20 else 0.4
        rows.append(
            {
                "confidence": conf,
                "is_correct": 1 if i < 20 else 0,
                "source": "filename" if i % 2 == 0 else "graph2d",
                "predicted_label": "人孔" if i < 20 else "传动件",
                "correct_label": "人孔" if i < 20 else "人孔",
            }
        )

    result = mod.calibrate(
        rows,
        method=CalibrationMethod.TEMPERATURE_SCALING,
        per_source=True,
        confidence_col="confidence",
        correct_col="is_correct",
        pred_label_col="predicted_label",
        truth_label_col="correct_label",
        source_col="source",
        min_samples=20,
        min_samples_per_source=5,
    )
    assert result["status"] == "ok"
    assert int(result["n_samples"]) == 40
    assert isinstance(result.get("metrics_before"), dict)
    assert isinstance(result.get("metrics_after"), dict)
    assert "ece" in result["metrics_after"]
    assert isinstance(result.get("source_temperatures"), dict)


def test_calibrate_hybrid_confidence_insufficient_samples(tmp_path: Path) -> None:
    from scripts import calibrate_hybrid_confidence as mod
    from src.ml.hybrid.calibration import CalibrationMethod

    rows = [
        {"confidence": 0.8, "is_correct": 1, "source": "filename"},
        {"confidence": 0.3, "is_correct": 0, "source": "filename"},
    ]
    result = mod.calibrate(
        rows,
        method=CalibrationMethod.TEMPERATURE_SCALING,
        per_source=True,
        confidence_col="confidence",
        correct_col="is_correct",
        pred_label_col="predicted_label",
        truth_label_col="correct_label",
        source_col="source",
        min_samples=10,
        min_samples_per_source=5,
    )
    assert result["status"] == "insufficient_samples"
    assert int(result["n_samples"]) == 2


def test_calibrate_hybrid_confidence_main_writes_output(tmp_path: Path) -> None:
    from scripts import calibrate_hybrid_confidence as mod

    input_csv = tmp_path / "input.csv"
    output_json = tmp_path / "out.json"
    _write_rows(
        input_csv,
        [
            {"confidence": 0.9, "is_correct": 1, "source": "filename"},
            {"confidence": 0.85, "is_correct": 1, "source": "filename"},
            {"confidence": 0.4, "is_correct": 0, "source": "graph2d"},
            {"confidence": 0.35, "is_correct": 0, "source": "graph2d"},
            {"confidence": 0.8, "is_correct": 1, "source": "filename"},
            {"confidence": 0.2, "is_correct": 0, "source": "graph2d"},
        ],
    )

    rc = mod.main(
        [
            "--input-csv",
            str(input_csv),
            "--output-json",
            str(output_json),
            "--min-samples",
            "4",
            "--min-samples-per-source",
            "2",
            "--per-source",
            "--include-fit-data",
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["method"] == "temperature_scaling"
    assert payload["n_samples"] >= 4
    assert isinstance(payload.get("fit_confidences"), list)


def test_calibrate_hybrid_confidence_main_auto_resolves_review_template_columns(
    tmp_path: Path,
) -> None:
    from scripts import calibrate_hybrid_confidence as mod

    input_csv = tmp_path / "review.csv"
    output_json = tmp_path / "out.json"
    _write_rows(
        input_csv,
        [
            {
                "confidence": 0.55,
                "graph2d_label": "传动件",
                "agree_with_graph2d": "N",
                "correct_label": "人孔",
            },
            {
                "confidence": 0.60,
                "graph2d_label": "传动件",
                "agree_with_graph2d": "Y",
                "correct_label": "传动件",
            },
            {
                "confidence": 0.45,
                "graph2d_label": "传动件",
                "agree_with_graph2d": "N",
                "correct_label": "捕集口",
            },
            {
                "confidence": 0.62,
                "graph2d_label": "人孔",
                "agree_with_graph2d": "Y",
                "correct_label": "人孔",
            },
        ],
    )

    rc = mod.main(
        [
            "--input-csv",
            str(input_csv),
            "--output-json",
            str(output_json),
            "--min-samples",
            "2",
            "--min-samples-per-source",
            "1",
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    resolved = payload.get("resolved_columns") or {}
    assert resolved.get("correct_col") == "agree_with_graph2d"
    assert resolved.get("pred_label_col") == "graph2d_label"
    assert resolved.get("truth_label_col") == "correct_label"
