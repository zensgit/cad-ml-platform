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
