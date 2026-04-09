from pathlib import Path

import numpy as np
from scripts import validate_online_example_ai_inputs as module
from scripts.validate_online_example_ai_inputs import inspect_h5_input


def test_inspect_h5_input_skips_when_h5py_missing(tmp_path: Path, monkeypatch) -> None:
    sample = tmp_path / "sample.h5"
    sample.write_bytes(b"placeholder")

    monkeypatch.setattr(module, "h5py", None)

    payload = inspect_h5_input(sample)

    assert payload["status"] == "skipped_no_h5py"
    assert payload["has_h5py"] is False


def test_inspect_h5_input_reads_vec_and_prediction(tmp_path: Path) -> None:
    import pytest

    h5py = pytest.importorskip("h5py")
    sample = tmp_path / "sample.h5"
    with h5py.File(sample, "w") as handle:
        handle.create_dataset(
            "vec",
            data=np.array(
                [
                    [6] + [-1] * 20,
                    [2] + [-1] * 20,
                    [10] + [-1] * 20,
                    [15] + [-1] * 20,
                    [5] + [-1] * 20,
                ],
                dtype=np.int32,
            ),
        )

    payload = inspect_h5_input(sample)

    assert payload["status"] == "ok"
    assert payload["keys"] == ["vec"]
    assert payload["vec_shape"] == [5, 21]
    assert payload["tokens_length"] == 5
    assert payload["prediction"]["status"] == "ok"
