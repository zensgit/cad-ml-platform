from __future__ import annotations

import csv
import json
from pathlib import Path


def test_main_exports_summary_from_raw_samples_jsonl(tmp_path: Path) -> None:
    from scripts.export_active_learning_review_queue_report import main

    input_dir = tmp_path / "active_learning"
    input_dir.mkdir(parents=True)
    samples_jsonl = input_dir / "samples.jsonl"
    samples_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "sample-1",
                        "doc_id": "doc-1",
                        "predicted_type": "bolt",
                        "predicted_fine_type": "bolt",
                        "predicted_coarse_type": "紧固件",
                        "predicted_is_coarse_label": False,
                        "confidence": 0.22,
                        "score_breakdown": {
                            "decision_source": "graph2d",
                            "review_automation_ready": True,
                        },
                        "uncertainty_reason": "low_confidence",
                        "status": "pending",
                        "created_at": "2026-03-08T01:00:00Z",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "sample-2",
                        "doc_id": "doc-2",
                        "predicted_type": "人孔",
                        "predicted_fine_type": "人孔",
                        "predicted_coarse_type": "开孔件",
                        "predicted_is_coarse_label": False,
                        "confidence": 0.41,
                        "score_breakdown": {
                            "decision_source": "hybrid",
                            "review_reasons": ["branch_conflict"],
                        },
                        "uncertainty_reason": "hybrid_rejected:below_min_confidence",
                        "status": "pending",
                        "created_at": "2026-03-08T01:01:00Z",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_json = tmp_path / "review_queue.summary.json"
    output_csv = tmp_path / "review_queue.csv"
    exit_code = main(
        [
            "--input-path",
            str(input_dir),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--top-k",
            "5",
        ]
    )
    assert exit_code == 0

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["total"] == 2
    assert summary["by_sample_type"]["low_confidence"] == 1
    assert summary["by_sample_type"]["hybrid_rejection"] == 1
    assert summary["by_feedback_priority"]["medium"] == 1
    assert summary["by_feedback_priority"]["high"] == 1
    assert summary["by_decision_source"]["graph2d"] == 1
    assert summary["by_decision_source"]["hybrid"] == 1
    assert summary["by_review_reason"]["branch_conflict"] == 1
    assert summary["automation_ready_count"] == 1
    assert summary["high_count"] == 1
    assert summary["operational_status"] == "managed_backlog"
    assert summary["top_feedback_priorities"][0] == {"name": "medium", "count": 1}

    with output_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["review_reasons"] == "[\"low_confidence\"]"


def test_main_exports_summary_from_review_queue_csv(tmp_path: Path) -> None:
    from scripts.export_active_learning_review_queue_report import main

    export_csv = tmp_path / "review_queue_20260308.csv"
    with export_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "doc_id",
                "status",
                "confidence",
                "predicted_type",
                "predicted_fine_type",
                "predicted_coarse_type",
                "sample_type",
                "feedback_priority",
                "uncertainty_reason",
                "decision_source",
                "review_reasons",
                "automation_ready",
                "score_breakdown",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "sample-3",
                "doc_id": "doc-3",
                "status": "pending",
                "confidence": "0.88",
                "predicted_type": "法兰",
                "predicted_fine_type": "法兰",
                "predicted_coarse_type": "连接件",
                "sample_type": "knowledge_conflict",
                "feedback_priority": "critical",
                "uncertainty_reason": "knowledge_conflict",
                "decision_source": "hybrid",
                "review_reasons": json.dumps(["knowledge_conflict"], ensure_ascii=False),
                "automation_ready": "false",
                "score_breakdown": json.dumps({"violations": ["rule-1"]}, ensure_ascii=False),
            }
        )

    output_json = tmp_path / "review_queue_csv.summary.json"
    exit_code = main(
        [
            "--input-path",
            str(export_csv),
            "--output-json",
            str(output_json),
        ]
    )
    assert exit_code == 0

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["total"] == 1
    assert summary["critical_count"] == 1
    assert summary["critical_ratio"] == 1.0
    assert summary["operational_status"] == "critical_backlog"
    assert summary["top_review_reasons"] == [
        {"name": "knowledge_conflict", "count": 1}
    ]
