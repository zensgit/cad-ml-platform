import os
import pickle
import io
import pytest
from src.ml.classifier import reload_model


def forge_pickle_with_global():
    # Create a pickle stream containing a GLOBAL opcode by pickling a function reference
    import math
    data = pickle.dumps(math.sin)  # contains GLOBAL opcode
    return data


def test_model_reload_opcode_block(tmp_path, monkeypatch):
    # Enable opcode scan strict
    monkeypatch.setenv("MODEL_OPCODE_SCAN", "1")
    monkeypatch.setenv("MODEL_OPCODE_STRICT", "1")
    test_file = tmp_path / "malicious.pkl"
    test_file.write_bytes(forge_pickle_with_global())
    resp = reload_model(str(test_file), expected_version="vX", force=True)
    assert resp["status"] in {"opcode_block", "opcode_scan_error"}
    if resp["status"] == "opcode_block":
        assert resp.get("opcode") in {"GLOBAL", "STACK_GLOBAL"}
