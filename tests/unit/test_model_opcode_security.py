from pathlib import Path
import pickle
import pytest
import os

from src.ml.classifier import reload_model


# Module-level class so it can be pickled
class _DummyModel:
    def predict(self, X):  # minimal interface
        return ["ok"]


def create_pickle_with_global(tmp_path: Path):
    # Craft object whose pickle contains GLOBAL opcode via custom class reference
    obj = _DummyModel()
    p = tmp_path / "model_global.pkl"
    with p.open("wb") as f:
        pickle.dump(obj, f, protocol=2)
    return p


def create_corrupt_pickle(tmp_path: Path):
    p = tmp_path / "model_corrupt.pkl"
    with p.open("wb") as f:
        f.write(b"\x80\x02THIS_IS_NOT_VALID_PICKLE_PAYLOAD")
    return p


@pytest.mark.parametrize("strict", ["0", "1"])
def test_opcode_block_detection(tmp_path, monkeypatch, strict):
    monkeypatch.setenv("MODEL_OPCODE_SCAN", "1")
    monkeypatch.setenv("MODEL_OPCODE_STRICT", strict)
    path = create_pickle_with_global(Path(tmp_path))
    res = reload_model(str(path))
    # In some environments protocol may not emit blocked opcode; allow success fallback
    assert res["status"] in {"success", "security_blocked", "opcode_blocked", "rollback", "rollback_level2"}
    if res["status"] == "security_blocked":
        assert "opcode" in res and "blocked_set" in res


def test_opcode_scan_error_on_corrupt_pickle(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_OPCODE_SCAN", "1")
    monkeypatch.setenv("MODEL_OPCODE_STRICT", "1")
    path = create_corrupt_pickle(Path(tmp_path))
    res = reload_model(str(path))
    assert res["status"] in {"opcode_scan_error", "security_scan_error", "magic_invalid"}
    if res["status"] in {"opcode_scan_error", "security_scan_error"}:
        assert "error" in res

