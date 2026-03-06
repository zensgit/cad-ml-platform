from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

pytest.importorskip("torch")

hpsketch_dataset = importlib.import_module("src.ml.train.hpsketch_dataset")
HPSketchDataset = hpsketch_dataset.HPSketchDataset
HPSketchNextTokenDataset = hpsketch_dataset.HPSketchNextTokenDataset
HPSketchSequenceDataset = hpsketch_dataset.HPSketchSequenceDataset
collate_hpsketch_next_token_batch = hpsketch_dataset.collate_hpsketch_next_token_batch
collate_hpsketch_sequences = hpsketch_dataset.collate_hpsketch_sequences


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def test_hpsketch_dataset_reads_tokens_and_collates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample_a = tmp_path / "data" / "0001" / "a.h5"
    sample_b = tmp_path / "data" / "0002" / "b.h5"
    _touch(sample_a)
    _touch(sample_b)
    token_map = {
        str(sample_a): [1, 2, 3],
        str(sample_b): [9],
    }
    monkeypatch.setattr(
        hpsketch_dataset,
        "load_command_tokens_from_h5",
        lambda path, **_kwargs: token_map[str(path)],
    )

    dataset = HPSketchDataset(str(tmp_path), min_seq_len=1)
    batch = [dataset[0], dataset[1]]
    payload = collate_hpsketch_sequences(batch, pad_token=0)

    assert dataset[0]["tokens"].tolist() == [1, 2, 3]
    assert dataset[0]["length"] == 3
    assert payload["tokens"].shape == (2, 3)
    assert payload["lengths"].tolist() == [3, 1]
    assert payload["tokens"][1].tolist() == [9, 0, 0]


def test_hpsketch_next_token_dataset_builds_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample = tmp_path / "x.h5"
    _touch(sample)
    monkeypatch.setattr(
        hpsketch_dataset,
        "load_command_tokens_from_h5",
        lambda path, **_kwargs: [4, 5, 6] if str(path) == str(sample) else [],
    )

    dataset = HPSketchDataset(str(tmp_path), min_seq_len=1)
    next_dataset = HPSketchNextTokenDataset(dataset)
    features, target = next_dataset[0]

    assert features["tokens"].tolist() == [4, 5]
    assert features["length"] == 2
    assert int(target.item()) == 6

    batch_payload, batch_targets = collate_hpsketch_next_token_batch(
        [next_dataset[0], next_dataset[0]],
        pad_token=0,
    )
    assert batch_payload["tokens"].shape == (2, 2)
    assert batch_targets.tolist() == [6, 6]


def test_hpsketch_dataset_preserves_true_length_when_min_seq_len_pads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample = tmp_path / "pad.h5"
    _touch(sample)
    monkeypatch.setattr(
        hpsketch_dataset,
        "load_command_tokens_from_h5",
        lambda path, **_kwargs: [7] if str(path) == str(sample) else [],
    )

    dataset = HPSketchDataset(str(tmp_path), min_seq_len=4)
    sample_payload = dataset[0]
    assert sample_payload["tokens"].tolist() == [7, 0, 0, 0]
    assert sample_payload["length"] == 1

    next_dataset = HPSketchNextTokenDataset(dataset)
    features, target = next_dataset[0]
    assert features["tokens"].tolist() == [7]
    assert features["length"] == 1
    assert int(target.item()) == 7


def test_hpsketch_dataset_seq_max_len_keeps_tail_tokens(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample = tmp_path / "tail.h5"
    _touch(sample)
    monkeypatch.setattr(
        hpsketch_dataset,
        "load_command_tokens_from_h5",
        lambda path, **_kwargs: [1, 2, 3, 4, 5, 6] if str(path) == str(sample) else [],
    )

    dataset = HPSketchDataset(str(tmp_path), seq_max_len=3, min_seq_len=1)
    sample_payload = dataset[0]
    assert sample_payload["tokens"].tolist() == [4, 5, 6]
    assert sample_payload["length"] == 3
