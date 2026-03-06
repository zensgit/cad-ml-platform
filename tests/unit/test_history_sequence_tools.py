from __future__ import annotations

from pathlib import Path

import src.ml.history_sequence_tools as history_sequence_tools
from src.ml.history_sequence_tools import (
    build_label_map,
    discover_h5_files,
    extract_command_tokens,
    load_command_tokens_from_h5,
    sequence_statistics,
    truncate_sequence,
)


class FakeArray:
    def __init__(self, values):
        self._values = values
        if values and isinstance(values[0], list):
            self.ndim = 2
            self.shape = (len(values), len(values[0]))
        else:
            self.ndim = 1
            self.shape = (len(values),)

    def tolist(self):
        return self._values

    def __getitem__(self, item):
        if isinstance(item, tuple):
            rows, col = item
            if rows != slice(None):
                raise TypeError("FakeArray only supports full-row slicing")
            return FakeArray([row[col] for row in self._values])
        return self._values[item]


def test_extract_command_tokens_from_2d_tensor() -> None:
    sequence = FakeArray(
        [[9, 100], [3, 200], [-1, 300], [7, 400]],
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


def test_truncate_discover_and_load_h5(tmp_path: Path, monkeypatch) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    file_path = nested / "sample.h5"
    file_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        history_sequence_tools,
        "load_h5_sequence_array",
        lambda *_args, **_kwargs: FakeArray([[1, 10], [2, 20], [3, 30]]),
    )

    assert discover_h5_files(str(tmp_path)) == [str(file_path)]
    assert load_command_tokens_from_h5(str(file_path)) == [1, 2, 3]
    assert truncate_sequence([1, 2, 3, 4], 2) == [3, 4]
