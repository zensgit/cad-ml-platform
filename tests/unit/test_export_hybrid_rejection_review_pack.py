from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(keys))
        writer.writeheader()
        writer.writerows(rows)


def test_to_bool_variants() -> None:
    from scripts.export_hybrid_rejection_review_pack import _to_bool

    assert _to_bool(True) is True
    assert _to_bool("true") is True
    assert _to_bool("YES") is True
    assert _to_bool("1") is True
    assert _to_bool("false") is False
    assert _to_bool("0") is False
    assert _to_bool("") is False
    assert _to_bool(None) is False


def test_export_review_pack_filters_and_ranks(tmp_path: Path) -> None:
    from scripts.export_hybrid_rejection_review_pack import main

    input_csv = tmp_path / "batch_results.csv"
    output_csv = tmp_path / "review_pack.csv"
    summary_json = tmp_path / "review_pack_summary.json"
    _write_rows(
        input_csv,
        [
            {
                "status": "ok",
                "file": "a.dxf",
                "confidence": "0.88",
                "graph2d_label": "传动件",
                "hybrid_label": "",
                "hybrid_rejected": "true",
                "hybrid_rejection_reason": "below_min_confidence",
            },
            {
                "status": "ok",
                "file": "b.dxf",
                "confidence": "0.42",
                "graph2d_label": "壳体类",
                "hybrid_label": "壳体类",
            },
            {
                "status": "ok",
                "file": "c.dxf",
                "confidence": "0.92",
                "graph2d_label": "壳体类",
                "hybrid_label": "人孔",
            },
            {
                "status": "ok",
                "file": "d.dxf",
                "confidence": "0.95",
                "graph2d_label": "人孔",
                "hybrid_label": "人孔",
            },
        ],
    )

    exit_code = main(
        [
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--low-confidence-threshold",
            "0.6",
        ]
    )
    assert exit_code == 0

    with output_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["file"] == "a.dxf"
    assert rows[0]["review_has_hybrid_rejection"] == "True"
    assert "hybrid_rejected:below_min_confidence" in rows[0]["review_reasons"]
    assert any(r["file"] == "b.dxf" and r["review_is_low_confidence"] == "True" for r in rows)
    assert any(
        r["file"] == "c.dxf" and r["review_has_hybrid_graph2d_conflict"] == "True"
        for r in rows
    )
    assert not any(r["file"] == "d.dxf" for r in rows)

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["total_rows"] == 4
    assert summary["candidate_rows"] == 3
    assert summary["hybrid_rejected_count"] == 1
