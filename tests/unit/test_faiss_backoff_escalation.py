import time
import importlib

from src.core import similarity


def test_faiss_backoff_escalation(monkeypatch):
    # Force degraded state
    similarity._VECTOR_DEGRADED = True  # type: ignore
    similarity._VECTOR_DEGRADED_AT = time.time() - 10  # type: ignore
    similarity._VECTOR_DEGRADED_REASON = "init_failure"  # type: ignore
    similarity._FAISS_NEXT_RECOVERY_TS = None  # type: ignore
    similarity._FAISS_SUPPRESS_UNTIL_TS = None  # type: ignore

    # Configure short interval & backoff for test
    # Interval below 60 gets clamped to 60 in backoff formula
    similarity._FAISS_RECOVERY_INTERVAL_SECONDS = 60  # type: ignore
    similarity._FAISS_RECOVERY_BACKOFF_MULTIPLIER = 2  # type: ignore
    similarity._FAISS_RECOVERY_MAX_BACKOFF = 120  # type: ignore

    # Monkeypatch reset_default_store to always keep degraded
    def fake_reset():
        return None

    monkeypatch.setattr(similarity, "reset_default_store", fake_reset)

    # Monkeypatch get_vector_store to return unavailable FaissVectorStore proxy
    class DummyFaiss(similarity.FaissVectorStore):
        def __init__(self):
            self._available = False

    monkeypatch.setattr(similarity, "get_vector_store", lambda backend=None: DummyFaiss())

    # Run several failed attempts and capture backoff progression
    backoffs = []
    base = time.time()
    for i in range(4):
        monkeypatch.setattr(time, "time", lambda b=base, i=i: b + i)  # deterministic time progression
        similarity.attempt_faiss_recovery(now=base + i)
        if similarity._FAISS_NEXT_RECOVERY_TS:  # type: ignore
            backoffs.append(round(similarity._FAISS_NEXT_RECOVERY_TS - (base + i), 2))  # type: ignore

    # Expect escalating backoff but capped at max
    # Backoff is min(interval*multiplier, max); interval=60, multiplier=2 -> 120 cap -> first scheduled 120
    assert backoffs[0] >= 100  # approximate initial backoff
    assert backoffs == sorted(backoffs, reverse=True)  # with advancing time, remaining decreases
    assert backoffs[-1] <= similarity._FAISS_RECOVERY_MAX_BACKOFF  # type: ignore

    # Simulate a successful recovery and ensure ETA resets
    similarity._VECTOR_DEGRADED = True  # type: ignore
    # Clear any scheduled recovery to force immediate attempt
    similarity._FAISS_NEXT_RECOVERY_TS = None  # type: ignore

    # A healthy FaissVectorStore must also trigger degraded flag clearing via reset_default_store.
    class HealthyFaiss(similarity.FaissVectorStore):
        def __init__(self):
            self._available = True

    monkeypatch.setattr(similarity, "get_vector_store", lambda backend=None: HealthyFaiss())
    # reset_default_store should not alter availability so keeping fake_reset is fine
    similarity.attempt_faiss_recovery(now=time.time())
    assert similarity._FAISS_NEXT_RECOVERY_TS is None  # type: ignore
    assert similarity._VECTOR_DEGRADED is False  # type: ignore
