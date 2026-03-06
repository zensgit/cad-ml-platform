from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

pytest.importorskip("torch")

hpsketch_dataset = importlib.import_module("src.ml.train.hpsketch_dataset")
HPSketchSequenceDataset = hpsketch_dataset.HPSketchSequenceDataset


def _touch(path: Path) -> None:
    path.write_bytes(b"placeholder")


def test_hpsketch_dataset_from_csv_manifest_and_collate(
    tmp_path: Path, monkeypatch
) -> None:
    sample_a = tmp_path / "sample_a.h5"
    sample_b = tmp_path / "sample_b.h5"
    _touch(sample_a)
    _touch(sample_b)
    token_map = {
        str(sample_a): [1, 2, 3, 4],
        str(sample_b): [9, 8],
    }
    monkeypatch.setattr(
        hpsketch_dataset,
        "load_command_tokens_from_h5",
        lambda path, **_kwargs: token_map[str(path)],
    )

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_path", "label"])
        writer.writeheader()
        writer.writerow({"file_path": sample_a.name, "label": "bracket"})
        writer.writerow({"file_path": sample_b.name, "label": "plate"})

    dataset = HPSketchSequenceDataset(
        manifest_path=str(manifest),
        max_sequence_length=3,
    )

    sample, label = dataset[0]
    assert sample["file_path"] == str(sample_a)
    assert sample["input_ids"].tolist() == [2, 3, 4]
    assert sample["length"] == 3
    assert sample["label_name"] == "bracket"
    assert int(label.item()) == dataset.label_map["bracket"]
    assert dataset.num_classes() == 2

    merged, labels = HPSketchSequenceDataset.collate_fn([dataset[0], dataset[1]])
    assert merged["input_ids"].shape == (2, 3)
    assert merged["lengths"].tolist() == [3, 2]
    assert labels.tolist() == [
        dataset.label_map["bracket"],
        dataset.label_map["plate"],
    ]


def test_hpsketch_dataset_from_root_dir_unlabeled(tmp_path: Path, monkeypatch) -> None:
    sample = tmp_path / "standalone.h5"
    _touch(sample)
    monkeypatch.setattr(
        hpsketch_dataset,
        "load_command_tokens_from_h5",
        lambda path, **_kwargs: [5, 6, 7] if str(path) == str(sample) else [],
    )

    dataset = HPSketchSequenceDataset(root_dir=str(tmp_path))

    payload, label = dataset[0]
    assert payload["file_path"] == str(sample)
    assert payload["input_ids"].tolist() == [5, 6, 7]
    assert int(label.item()) == -1
    assert dataset.num_classes() == 0
