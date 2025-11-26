import time
from src.core import similarity


def test_faiss_suppression_window_stuck_simulation(monkeypatch):
    # Simulate suppression window set and time not advancing (stuck condition basis)
    similarity._VECTOR_DEGRADED = True  # type: ignore
    now = time.time()
    similarity._FAISS_SUPPRESS_UNTIL_TS = now + 300  # type: ignore
    similarity._FAISS_NEXT_RECOVERY_TS = now + 120  # type: ignore

    # Monkeypatch time to fixed value to emulate stalled loop
    monkeypatch.setattr(time, "time", lambda: now)
    # Call attempt to update remaining seconds gauge without time change
    similarity.attempt_faiss_recovery(now=now)
    # The gauge set path occurs when suppression active; remaining seconds should equal window length initially
    remaining_initial = similarity._FAISS_SUPPRESS_UNTIL_TS - now  # type: ignore
    assert remaining_initial > 0

    # After another attempt with same time, remaining seconds unchanged (stuck scenario basis for alert)
    similarity.attempt_faiss_recovery(now=now)
    remaining_second = similarity._FAISS_SUPPRESS_UNTIL_TS - now  # type: ignore
    assert remaining_second == remaining_initial
    # This mirrors alert condition: increase(faiss_recovery_suppression_remaining_seconds[10m]) == 0 while value >0

