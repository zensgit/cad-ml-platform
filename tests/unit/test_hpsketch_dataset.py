from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.ml.train.hpsketch_dataset import HPSketchSequenceDataset


def _write_h5(path: Path, rows: list[list[int]]) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("vec", data=rows)


def test_hpsketch_dataset_from_csv_manifest_and_collate(tmp_path: Path) -> None:
    sample_a = tmp_path / "sample_a.h5"
    sample_b = tmp_path / "sample_b.h5"
    _write_h5(sample_a, [[1, 0], [2, 0], [3, 0], [4, 0]])
    _write_h5(sample_b, [[9, 0], [8, 0]])

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


def test_hpsketch_dataset_from_root_dir_unlabeled(tmp_path: Path) -> None:
    sample = tmp_path / "standalone.h5"
    _write_h5(sample, [[5, 0], [6, 0], [7, 0]])

    dataset = HPSketchSequenceDataset(root_dir=str(tmp_path))

    payload, label = dataset[0]
    assert payload["file_path"] == str(sample)
    assert payload["input_ids"].tolist() == [5, 6, 7]
    assert int(label.item()) == -1
    assert dataset.num_classes() == 0
