import json
import os
import time
from pathlib import Path

from src.core import similarity


def test_recovery_state_persistence(tmp_path, monkeypatch):
    # Arrange: use a temporary state file
    state_path = tmp_path / "faiss_recovery_state.json"
    monkeypatch.setenv("FAISS_RECOVERY_STATE_PATH", str(state_path))

    # Set globals to simulated degraded with scheduled recovery and suppression
    now = time.time()
    similarity._FAISS_NEXT_RECOVERY_TS = now + 120
    similarity._FAISS_SUPPRESS_UNTIL_TS = now + 300
    similarity._VECTOR_DEGRADED = True
    similarity._VECTOR_DEGRADED_REASON = "init_fail"
    similarity._VECTOR_DEGRADED_AT = now - 60

    # Act: persist then reset and reload
    similarity._persist_recovery_state()

    # Sanity check file exists and contains expected keys
    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert "next_recovery_ts" in data and "suppress_until_ts" in data

    # Reset globals
    similarity._FAISS_NEXT_RECOVERY_TS = None
    similarity._FAISS_SUPPRESS_UNTIL_TS = None
    similarity._VECTOR_DEGRADED = False
    similarity._VECTOR_DEGRADED_REASON = None
    similarity._VECTOR_DEGRADED_AT = None

    # Reload and assert values restored
    loaded = similarity.load_recovery_state()
    assert loaded is True
    assert similarity._FAISS_NEXT_RECOVERY_TS == data["next_recovery_ts"]
    assert similarity._FAISS_SUPPRESS_UNTIL_TS == data["suppress_until_ts"]
    # Degraded flags are only restored if recent (within 24h)
    assert similarity._VECTOR_DEGRADED is True
    assert similarity._VECTOR_DEGRADED_REASON == "init_fail"
    assert isinstance(similarity._VECTOR_DEGRADED_AT, float)
