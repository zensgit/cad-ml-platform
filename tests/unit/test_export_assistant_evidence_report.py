from __future__ import annotations

import csv
import json
from pathlib import Path


def test_main_exports_report_for_mixed_inputs(tmp_path: Path) -> None:
    from scripts.export_assistant_evidence_report import main

    input_dir = tmp_path / "inputs"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)

    assistant_json = input_dir / "assistant_response.json"
    assistant_json.write_text(
        json.dumps(
            {
                "id": "assistant-1",
                "query": "M10 螺纹底孔尺寸?",
                "answer": "推荐底孔 8.5 mm。",
                "sources": [
                    "threads: M10 coarse thread",
                    "standards: GB/T thread table",
                ],
                "evidence": [
                    {
                        "reference_id": "E1",
                        "source": "threads",
                        "summary": "M10 coarse thread spec",
                        "match_type": "direct",
                        "key_facts": ["攻丝底孔: 8.5 mm"],
                    },
                    {
                        "reference_id": "E2",
                        "source": "standards",
                        "summary": "",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    analyze_json = nested_dir / "analyze_output.json"
    analyze_json.write_text(
        json.dumps(
            {
                "analysis_id": "analysis-1",
                "results": {
                    "classification": {
                        "decision_path": [
                            "filename_extracted",
                            "fusion_scored",
                        ],
                        "source_contributions": {
                            "filename": 0.61,
                            "titleblock": 0.22,
                        },
                        "hybrid_explanation": {
                            "summary": "综合 文件名, 标题栏 多源信息"
                        },
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    jsonl_path = input_dir / "assistant_events.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": "doc-1",
                        "score_breakdown": {
                            "decision_path": ["history_shadow_only"],
                            "source_contributions": {"history_sequence": 0.58},
                            "hybrid_explanation": {
                                "summary": "history shadow only"
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "assistant-2",
                        "answer": "当前回答缺少可用出处。",
                        "evidence": [
                            {
                                "reference_id": "E1",
                                "match_type": "keyword",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "assistant_evidence_report.csv"
    summary_json = tmp_path / "assistant_evidence_report.summary.json"
    exit_code = main(
        [
            "--input-path",
            str(input_dir),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--top-k",
            "5",
        ]
    )
    assert exit_code == 0

    with output_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    by_id = {row["record_id"]: row for row in rows}
    assert by_id["assistant-1"]["record_kind"] == "assistant"
    assert by_id["assistant-1"]["evidence_count"] == "2"
    assert by_id["assistant-1"]["sources_count"] == "2"
    assert by_id["assistant-1"]["evidence_types"] == "direct;unknown"
    assert (
        by_id["assistant-1"]["evidence_missing_fields"]
        == "evidence[].match_type;evidence[].summary"
    )
    assert by_id["assistant-1"]["missing_fields"] == "decision_path;source_contributions"
    assert by_id["analysis-1"]["record_kind"] == "analyze"
    assert by_id["analysis-1"]["source_contribution_sources"] == "filename;titleblock"
    assert by_id["analysis-1"]["decision_path"] == "filename_extracted;fusion_scored"
    assert by_id["doc-1"]["decision_path"] == "history_shadow_only"
    assert by_id["assistant-2"]["evidence_sources"] == ""
    assert "evidence[].source" in by_id["assistant-2"]["evidence_missing_fields"]

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["scanned_file_count"] == 3
    assert summary["total_records"] == 4
    assert summary["total_evidence_items"] == 3
    assert summary["average_evidence_count"] == 0.75
    assert summary["average_sources_count"] == 0.5
    assert summary["average_decision_path_count"] == 0.75
    assert summary["coverage"]["records_with_evidence"] == {
        "count": 2,
        "total": 4,
        "missing_count": 2,
        "coverage_pct": 0.5,
    }
    assert summary["coverage"]["records_with_sources"] == {
        "count": 1,
        "total": 4,
        "missing_count": 3,
        "coverage_pct": 0.25,
    }
    assert summary["coverage"]["records_with_source_contributions"] == {
        "count": 2,
        "total": 4,
        "missing_count": 2,
        "coverage_pct": 0.5,
    }
    assert summary["coverage"]["records_with_any_source_signal"] == {
        "count": 3,
        "total": 4,
        "missing_count": 1,
        "coverage_pct": 0.75,
    }
    assert summary["coverage"]["records_with_decision_path"] == {
        "count": 2,
        "total": 4,
        "missing_count": 2,
        "coverage_pct": 0.5,
    }
    assert summary["evidence_item_coverage"]["evidence[].source"] == {
        "count": 2,
        "total": 3,
        "missing_count": 1,
        "coverage_pct": 0.6667,
    }
    assert summary["evidence_item_coverage"]["evidence[].summary"] == {
        "count": 1,
        "total": 3,
        "missing_count": 2,
        "coverage_pct": 0.3333,
    }
    assert summary["evidence_item_coverage"]["evidence[].match_type"] == {
        "count": 2,
        "total": 3,
        "missing_count": 1,
        "coverage_pct": 0.6667,
    }
    assert summary["top_record_kinds"] == [
        {"name": "analyze", "count": 2},
        {"name": "assistant", "count": 2},
    ]
    assert summary["top_evidence_types"] == [
        {"name": "direct", "count": 1},
        {"name": "keyword", "count": 1},
        {"name": "unknown", "count": 1},
    ]
    assert summary["top_structured_sources"] == [
        {"name": "filename", "count": 1},
        {"name": "history_sequence", "count": 1},
        {"name": "standards", "count": 1},
        {"name": "threads", "count": 1},
        {"name": "titleblock", "count": 1},
    ]
    assert summary["top_decision_steps"] == [
        {"name": "filename_extracted", "count": 1},
        {"name": "fusion_scored", "count": 1},
        {"name": "history_shadow_only", "count": 1},
    ]
    top_missing = {item["name"]: item["count"] for item in summary["top_missing_fields"]}
    assert top_missing["sources"] == 3
    assert top_missing["decision_path"] == 2
    assert top_missing["source_contributions"] == 2
    assert top_missing["evidence"] == 2
    assert top_missing["evidence[].summary"] == 2


def test_discover_input_files_rejects_missing_path(tmp_path: Path) -> None:
    from scripts.export_assistant_evidence_report import _discover_input_files

    missing_path = tmp_path / "missing"
    try:
        _discover_input_files([str(missing_path)])
    except SystemExit as exc:
        assert str(exc) == f"Input path not found: {missing_path}"
    else:
        raise AssertionError("expected SystemExit for missing input path")
