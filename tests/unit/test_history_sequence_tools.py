from __future__ import annotations

from pathlib import Path

import pytest

import src.ml.history_sequence_tools as history_sequence_tools
from src.ml.history_sequence_tools import (
    build_label_map,
    build_prototype_payload,
    discover_h5_files,
    extract_command_tokens,
    iter_h5_files,
    load_h5_label_pairs_from_manifest,
    load_command_tokens_from_h5,
    macro_f1,
    read_command_tokens_from_h5,
    sequence_statistics,
    truncate_sequence,
)

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None


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


def _write_h5(path: Path, rows: list[list[int]]) -> None:
    if h5py is None:
        pytest.skip("h5py not installed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("vec", data=rows, dtype="i4")


def test_iter_h5_files_recursive(tmp_path: Path) -> None:
    _write_h5(tmp_path / "a.h5", [[1, 0]])
    _write_h5(tmp_path / "sub" / "b.h5", [[2, 0]])
    files = iter_h5_files(tmp_path, recursive=True)
    assert [path.name for path in files] == ["a.h5", "b.h5"]


def test_load_h5_label_pairs_from_manifest_json_and_csv(tmp_path: Path) -> None:
    h5_a = tmp_path / "a.h5"
    h5_b = tmp_path / "b.h5"
    _write_h5(h5_a, [[1, 0]])
    _write_h5(h5_b, [[2, 0]])

    json_manifest = tmp_path / "manifest.json"
    json_manifest.write_text(
        '[{"h5_path": "a.h5", "label": "shaft"}, {"h5_path": "b.h5", "label": "link"}]',
        encoding="utf-8",
    )
    json_pairs = load_h5_label_pairs_from_manifest(json_manifest)
    assert json_pairs == [(h5_a.resolve(), "shaft"), (h5_b.resolve(), "link")]

    csv_manifest = tmp_path / "manifest.csv"
    csv_manifest.write_text("h5_path,label\na.h5,shaft\nb.h5,link\n", encoding="utf-8")
    csv_pairs = load_h5_label_pairs_from_manifest(csv_manifest)
    assert csv_pairs == [(h5_a.resolve(), "shaft"), (h5_b.resolve(), "link")]


def test_read_command_tokens_from_h5_first_column(tmp_path: Path) -> None:
    h5_path = tmp_path / "sample.h5"
    _write_h5(
        h5_path,
        [
            [6, -1, -1],
            [10, 20, 30],
            [-1, 11, 22],
            [15, 0, 0],
        ],
    )
    tokens = read_command_tokens_from_h5(h5_path, vec_key="vec", command_col=0)
    assert tokens == [6, 10, 15]


def test_build_prototype_payload_filters_low_sample_labels() -> None:
    samples = [
        ("shaft", [6, 10, 10, 15]),
        ("shaft", [6, 10, 15]),
        ("link", [1, 2, 10]),
    ]
    payload = build_prototype_payload(samples, top_k=4, min_samples_per_label=2)
    labels = payload.get("labels") or {}
    assert set(labels.keys()) == {"shaft"}
    assert abs(float(labels["shaft"]["bias"])) < 1e-3
    assert "10" in labels["shaft"]["token_weights"]
    assert "6,10" in labels["shaft"]["bigram_weights"]


def test_build_prototype_payload_ignores_invalid_tokens() -> None:
    payload = build_prototype_payload(
        [
            ("shaft", [6, "x", None, -1, 10]),  # type: ignore[list-item]
            ("shaft", [6, 10]),
        ],
        top_k=4,
        min_samples_per_label=2,
    )
    labels = payload.get("labels") or {}
    assert "shaft" in labels
    assert set(labels["shaft"]["token_weights"].keys()) <= {"6", "10"}


def test_macro_f1_basic() -> None:
    expected = ["A", "A", "B", "B"]
    predicted = ["A", "B", "B", "B"]
    value = macro_f1(expected, predicted)
    assert round(value, 6) == round(11 / 15, 6)
