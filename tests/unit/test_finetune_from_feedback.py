from __future__ import annotations

import json
from pathlib import Path

from scripts import finetune_from_feedback as module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_samples_prefers_true_fine_type(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    _write_jsonl(
        path,
        [
            {
                "doc_id": "doc-1",
                "true_type": "开孔件",
                "true_fine_type": "人孔",
                "true_coarse_type": "开孔件",
            }
        ],
    )

    doc_ids, labels = module._load_samples(str(path), label_field="true_fine_type")

    assert doc_ids == ["doc-1"]
    assert labels == ["人孔"]


def test_load_samples_can_train_on_coarse_labels(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    _write_jsonl(
        path,
        [
            {
                "doc_id": "doc-2",
                "true_type": "人孔",
                "true_fine_type": "人孔",
                "true_coarse_type": "开孔件",
            }
        ],
    )

    doc_ids, labels = module._load_samples(str(path), label_field="true_coarse_type")

    assert doc_ids == ["doc-2"]
    assert labels == ["开孔件"]


def test_load_samples_falls_back_to_legacy_true_type(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    _write_jsonl(path, [{"doc_id": "doc-3", "true_type": "法兰"}])

    doc_ids, labels = module._load_samples(str(path), label_field="true_fine_type")

    assert doc_ids == ["doc-3"]
    assert labels == ["法兰"]


def test_load_samples_accepts_feedback_aliases_and_analysis_id(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    _write_jsonl(
        path,
        [
            {
                "analysis_id": "analysis-4",
                "correct_label": "人孔",
                "correct_coarse_label": "开孔件",
            }
        ],
    )

    doc_ids, labels = module._load_samples(str(path), label_field="true_fine_type")

    assert doc_ids == ["analysis-4"]
    assert labels == ["人孔"]


def test_build_training_summary_tracks_coarse_distribution() -> None:
    summary = module._build_training_summary(
        ["人孔", "捕集口", "法兰"],
        label_field="true_fine_type",
        vector_count=3,
    )

    assert summary["label_field"] == "true_fine_type"
    assert summary["sample_count"] == 3
    assert summary["vector_count"] == 3
    assert summary["unique_label_count"] == 3
    assert summary["unique_coarse_label_count"] == 2
    assert summary["label_distribution"]["人孔"] == 1
    assert summary["coarse_label_distribution"]["开孔件"] == 2
    assert summary["coarse_label_distribution"]["法兰"] == 1


def test_write_training_summary_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    summary = module._build_training_summary(["法兰"], label_field="true_type")

    module._write_training_summary(str(output), summary)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1
    assert payload["label_distribution"] == {"法兰": 1}
