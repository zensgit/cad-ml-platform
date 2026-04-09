from __future__ import annotations

import json
from pathlib import Path

from scripts import train_metric_model as module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_iter_feedback_paths_includes_default_feedback_log(tmp_path: Path, monkeypatch) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir(parents=True)
    default_log = tmp_path / "feedback_log.jsonl"
    _write_jsonl(default_log, [{"analysis_id": "a-1", "correct_label": "人孔"}])
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(default_log))

    paths = module._iter_feedback_paths(str(feedback_dir))

    assert str(default_log) in paths


def test_load_feedback_data_reads_default_feedback_log(tmp_path: Path, monkeypatch) -> None:
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir(parents=True)
    default_log = tmp_path / "feedback_log.jsonl"
    row = {"analysis_id": "a-2", "correct_label": "法兰"}
    _write_jsonl(default_log, [row])
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(default_log))

    dataset = module.load_feedback_data(str(feedback_dir))

    assert dataset == [row]


def test_prepare_training_triplets_accepts_alias_fields(monkeypatch) -> None:
    class DummyStore:
        def __init__(self) -> None:
            self._store = {
                "anchor-1": [0.1] * 4,
                "pos-1": [0.2] * 4,
                "neg-1": [0.3] * 4,
            }

        def exists(self, doc_id: str) -> bool:
            return doc_id in self._store

    meta = {
        "anchor-1": {"type": "人孔"},
        "pos-1": {"type": "人孔"},
        "neg-1": {"type": "法兰"},
    }

    monkeypatch.setattr("src.core.similarity.get_vector_store", lambda: DummyStore())
    monkeypatch.setattr(
        "src.core.similarity.get_vector_meta",
        lambda doc_id: meta.get(doc_id),
        raising=False,
    )

    triplets = module.prepare_training_triplets(
        [{"analysis_id": "anchor-1", "correct_label": "人孔"}]
    )

    assert len(triplets) == 1
    assert triplets[0]["anchor_id"] == "anchor-1"
    assert triplets[0]["anchor_label"] == "人孔"


def test_build_training_summary_tracks_distributions() -> None:
    summary = module._build_training_summary(
        [{"analysis_id": "a-1"}, {"analysis_id": "a-2"}],
        [
            {
                "anchor_id": "a-1",
                "anchor_label": "人孔",
                "negative_label": "法兰",
            },
            {
                "anchor_id": "a-2",
                "anchor_label": "法兰",
                "negative_label": "人孔",
            },
        ],
    )

    assert summary["feedback_entry_count"] == 2
    assert summary["triplet_count"] == 2
    assert summary["unique_anchor_count"] == 2
    assert summary["anchor_label_distribution"] == {"人孔": 1, "法兰": 1}
    assert summary["negative_label_distribution"] == {"人孔": 1, "法兰": 1}


def test_write_training_summary_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "metric_summary.json"
    module._write_training_summary(
        str(output),
        {
            "feedback_entry_count": 1,
            "triplet_count": 1,
            "unique_anchor_count": 1,
            "anchor_label_distribution": {"人孔": 1},
            "negative_label_distribution": {"法兰": 1},
        },
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["triplet_count"] == 1
    assert payload["anchor_label_distribution"] == {"人孔": 1}
