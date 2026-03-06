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
                "hybrid_shadow_predictions": json.dumps(
                    {
                        "history_sequence": {
                            "label": "人孔",
                            "confidence": 0.58,
                            "status": "ok",
                        }
                    },
                    ensure_ascii=False,
                ),
                "history_label": "人孔",
                "history_confidence": "0.58",
                "history_shadow_only": "true",
                "hybrid_explanation_summary": "综合 文件名, 标题栏, 历史序列 多源信息，融合得出 人孔",
                "fusion_consistency_check": "high",
                "fusion_consistency_notes": "知识规则与细分类预测冲突",
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
    assert rows[0]["review_shadow_sources"] == "history_sequence"
    assert rows[0]["review_history_shadow_only"] == "True"
    assert rows[0]["review_history_shadow_label"] == "人孔"
    assert rows[0]["review_explanation_summary"].startswith("综合 文件名")
    assert rows[0]["review_decision_path"].endswith("fusion_engine_weighted_average")
    assert rows[0]["review_fusion_strategy"] == "weighted_average"
    assert rows[0]["review_coarse_label"] == "传动件"
    assert rows[0]["review_fine_label"] == "人孔"
    assert rows[0]["review_rejection_reason"] == "below_min_confidence"
    assert rows[0]["review_has_knowledge_conflict"] == "True"
    assert rows[0]["review_knowledge_conflict"] == "high"
    assert rows[0]["review_knowledge_conflict_note"] == "知识规则与细分类预测冲突"

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["knowledge_conflict_count"] == 1
    assert summary["top_coarse_labels"][0] == {"name": "传动件", "count": 1}
    assert summary["top_fine_labels"][0] == {"name": "人孔", "count": 1}
    assert summary["top_rejection_reasons"][0] == {
        "name": "below_min_confidence",
        "count": 1,
    }
    assert summary["top_knowledge_conflicts"][0] == {"name": "high", "count": 1}
    assert summary["top_primary_sources"][0] == {"name": "filename", "count": 1}
    assert summary["top_shadow_sources"][0] == {
        "name": "history_sequence",
        "count": 1,
    }
    assert summary["sample_explanations"][0].startswith("综合 文件名")
    assert summary["sample_candidates"][0]["file"] == "sample.dxf"
    assert summary["sample_candidates"][0]["coarse_label"] == "传动件"
    assert summary["sample_candidates"][0]["fine_label"] == "人孔"
    assert summary["sample_candidates"][0]["rejection_reason"] == "below_min_confidence"
    assert summary["sample_candidates"][0]["knowledge_conflict"] == "high"
    assert summary["sample_candidates"][0]["shadow_sources"] == "history_sequence"
