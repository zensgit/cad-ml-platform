import json
import os
import time

from src.core import similarity


def test_faiss_recovery_persistence_reload(tmp_path, monkeypatch):
    state_path = tmp_path / "faiss_recovery_state.json"
    monkeypatch.setenv("FAISS_RECOVERY_STATE_PATH", str(state_path))

    # Prepare degraded state with suppression and next recovery
    similarity._VECTOR_DEGRADED = True  # type: ignore
    similarity._VECTOR_DEGRADED_REASON = "init_failure"  # type: ignore
    similarity._VECTOR_DEGRADED_AT = time.time() - 30  # type: ignore
    similarity._FAISS_NEXT_RECOVERY_TS = time.time() + 120  # type: ignore
    similarity._FAISS_SUPPRESS_UNTIL_TS = time.time() + 60  # type: ignore

    # Persist state
    similarity._persist_recovery_state()  # type: ignore
    assert state_path.exists()

    # Reset globals to ensure load actually repopulates
    similarity._VECTOR_DEGRADED = False  # type: ignore
    similarity._VECTOR_DEGRADED_REASON = None  # type: ignore
    similarity._VECTOR_DEGRADED_AT = None  # type: ignore
    similarity._FAISS_NEXT_RECOVERY_TS = None  # type: ignore
    similarity._FAISS_SUPPRESS_UNTIL_TS = None  # type: ignore

    loaded = similarity.load_recovery_state()  # type: ignore
    assert loaded is True
    assert similarity._VECTOR_DEGRADED is True  # type: ignore
    assert similarity._VECTOR_DEGRADED_REASON == "init_failure"  # type: ignore
    assert similarity._FAISS_NEXT_RECOVERY_TS is not None  # type: ignore
    assert similarity._FAISS_SUPPRESS_UNTIL_TS is not None  # type: ignore
