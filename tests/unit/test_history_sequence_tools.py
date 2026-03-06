from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.ml.history_sequence_tools import (
    build_label_map,
    discover_h5_files,
    extract_command_tokens,
    load_command_tokens_from_h5,
    sequence_statistics,
    truncate_sequence,
)


def test_extract_command_tokens_from_2d_tensor() -> None:
    sequence = torch.tensor(
        [
            [9, 100],
            [3, 200],
            [-1, 300],
            [7, 400],
        ],
        dtype=torch.long,
    )

    tokens = extract_command_tokens(sequence, command_col=0, min_token=0)

    assert tokens == [9, 3, 7]


def test_sequence_statistics_and_label_map() -> None:
    stats = sequence_statistics([1, 2, 1, 3, 1], top_k=2)

    assert stats["length"] == 5
    assert stats["unique_commands"] == 3
    assert stats["top_commands"][0] == (1, 3)
    assert stats["top_bigrams"][0] == ((1, 2), 1)

    label_map = build_label_map([" bracket ", "plate", "bracket", ""])
    assert label_map == {"bracket": 0, "plate": 1}


def test_truncate_discover_and_load_h5(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")

    nested = tmp_path / "nested"
    nested.mkdir()
    file_path = nested / "sample.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("vec", data=[[1, 10], [2, 20], [3, 30]])

    assert discover_h5_files(str(tmp_path)) == [str(file_path)]
    assert load_command_tokens_from_h5(str(file_path)) == [1, 2, 3]
    assert truncate_sequence([1, 2, 3, 4], 2) == [3, 4]
