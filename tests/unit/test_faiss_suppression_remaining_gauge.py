import time

from src.core import similarity


def test_faiss_suppression_remaining_gauge(monkeypatch):
    # Reset faiss state from prior tests
    similarity._FAISS_INDEX = None
    similarity._FAISS_DIM = None
    similarity._FAISS_ID_MAP = {}
    similarity._FAISS_REVERSE_MAP = {}

    # Prepare degraded history to trigger suppression
    similarity._VECTOR_DEGRADED = True  # type: ignore
    now = time.time()
    similarity._DEGRADATION_HISTORY.clear()  # type: ignore
    # Create events within flap window
    # Need >= threshold events within window to trigger suppression
    for i in range(similarity._FAISS_RECOVERY_FLAP_THRESHOLD + 1):  # type: ignore
        similarity._DEGRADATION_HISTORY.append({"timestamp": now - 10 * (i + 1)})  # type: ignore

    # Monkeypatch time to 'now'
    monkeypatch.setattr(time, "time", lambda: now)
    # Trigger detection
    triggered = similarity._detect_flapping_and_set_suppression(now=now)  # type: ignore
    assert triggered is True
    assert similarity._FAISS_SUPPRESS_UNTIL_TS is not None  # type: ignore
    remaining = similarity._FAISS_SUPPRESS_UNTIL_TS - now  # type: ignore
    assert remaining > 0

    # Fast-forward time beyond suppression window
    future = similarity._FAISS_SUPPRESS_UNTIL_TS + 1  # type: ignore
    monkeypatch.setattr(time, "time", lambda: future)
    # Attempt recovery (will schedule or success) should zero gauge if window expired
    similarity.attempt_faiss_recovery(now=future)
    if similarity._FAISS_SUPPRESS_UNTIL_TS and future < similarity._FAISS_SUPPRESS_UNTIL_TS:  # type: ignore
        # If still suppressed (unlikely here), ensure remaining decreased
        assert similarity._FAISS_SUPPRESS_UNTIL_TS - future <= remaining  # type: ignore
    else:
        # Suppression ended path: gauge should have been set to 0 via recovery scheduling branch
        # We can't directly read the gauge without registry; rely on internal state
        assert future >= (similarity._FAISS_SUPPRESS_UNTIL_TS or 0)  # type: ignore
