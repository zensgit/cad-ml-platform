"""Tests hash mismatch handling in model reload logic.

Ensures _MODEL_LAST_ERROR is set and response structure is correct when
whitelist validation fails.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import pytest


class DummyModel:
    def __init__(self, marker: str):
        self.marker = marker
    def predict(self, X):
        return [self.marker] * len(X)


@pytest.fixture
def temp_model_file():
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(DummyModel("ok"), f, protocol=4)
        path = f.name
    yield Path(path)
    Path(path).unlink(missing_ok=True)


def test_model_reload_hash_mismatch_sets_last_error(temp_model_file, monkeypatch):
    from src.ml import classifier
    from src.ml.classifier import reload_model, get_model_info
    classifier._MODEL = None  # type: ignore
    classifier._MODEL_HASH = None  # type: ignore
    classifier._MODEL_LAST_ERROR = None  # type: ignore
    classifier._MODEL_LOAD_SEQ = 0  # type: ignore
    result1 = reload_model(str(temp_model_file), expected_version="vA")
    assert result1["status"] == "success"
    baseline_seq = get_model_info()["load_seq"]
    monkeypatch.setenv("ALLOWED_MODEL_HASHES", "deadbeefcafefeed")
    result2 = reload_model(str(temp_model_file), expected_version="vB")
    assert result2["status"] == "hash_mismatch"
    assert "Hash whitelist validation failed" in result2.get("message", "")
    info2 = get_model_info()
    assert info2["last_error"] is not None
    assert "whitelist" in info2["last_error"].lower()
    assert info2["load_seq"] == baseline_seq
    monkeypatch.delenv("ALLOWED_MODEL_HASHES", raising=False)
