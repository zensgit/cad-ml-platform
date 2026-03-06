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


def test_export_review_pack_preserves_explanation_context(tmp_path: Path) -> None:
    from scripts.export_hybrid_rejection_review_pack import main

    input_csv = tmp_path / "batch_results.csv"
    output_csv = tmp_path / "review_pack.csv"
    summary_json = tmp_path / "review_pack_summary.json"
    _write_rows(
        input_csv,
        [
            {
                "status": "ok",
                "file": "sample.dxf",
                "confidence": "0.39",
                "graph2d_label": "传动件",
                "hybrid_label": "人孔",
                "hybrid_rejected": "true",
                "hybrid_rejection_reason": "below_min_confidence",
                "hybrid_path": "filename_extracted;fusion_scored;fusion_engine_weighted_average",
                "hybrid_fusion_strategy": "weighted_average",
                "hybrid_source_contributions": json.dumps(
                    {"filename": 0.61, "titleblock": 0.22, "history_sequence": 0.11},
                    ensure_ascii=False,
                ),
                "hybrid_explanation_summary": "综合 文件名, 标题栏, 历史序列 多源信息，融合得出 人孔",
            }
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

    assert len(rows) == 1
    assert rows[0]["review_primary_sources"] == "filename;titleblock;history_sequence"
    assert rows[0]["review_explanation_summary"].startswith("综合 文件名")
    assert rows[0]["review_decision_path"].endswith("fusion_engine_weighted_average")
    assert rows[0]["review_fusion_strategy"] == "weighted_average"

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["top_primary_sources"][0] == {"name": "filename", "count": 1}
    assert summary["sample_explanations"][0].startswith("综合 文件名")
    assert summary["sample_candidates"][0]["file"] == "sample.dxf"
