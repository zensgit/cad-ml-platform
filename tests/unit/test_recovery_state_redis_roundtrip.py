import json
import types


class FakeRedis:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, val):
        self._store[key] = val


def test_recovery_state_roundtrip_with_fake_redis(monkeypatch):
    # Force backend to redis and inject fake client
    monkeypatch.setenv("FAISS_RECOVERY_STATE_BACKEND", "redis")
    from src.core import similarity

    fake = FakeRedis()
    monkeypatch.setattr(similarity, "get_client", lambda: fake)

    # Set globals to known values and persist
    similarity._FAISS_NEXT_RECOVERY_TS = 123.0
    similarity._FAISS_SUPPRESS_UNTIL_TS = 456.0
    similarity._VECTOR_DEGRADED = True
    similarity._VECTOR_DEGRADED_REASON = "test"
    import time

    similarity._VECTOR_DEGRADED_AT = time.time()

    similarity._persist_recovery_state()

    # Clear and load back
    similarity._FAISS_NEXT_RECOVERY_TS = None
    similarity._FAISS_SUPPRESS_UNTIL_TS = None
    similarity._VECTOR_DEGRADED = False
    similarity._VECTOR_DEGRADED_REASON = None
    similarity._VECTOR_DEGRADED_AT = None

    ok = similarity.load_recovery_state()
    assert ok is True
    assert similarity._FAISS_NEXT_RECOVERY_TS == 123.0
    assert similarity._FAISS_SUPPRESS_UNTIL_TS == 456.0
    assert similarity._VECTOR_DEGRADED is True
    assert similarity._VECTOR_DEGRADED_REASON == "test"
