import os
import pytest


@pytest.fixture(autouse=True)
def vector_store_isolation():
    """Isolate global in-memory vector store between tests.

    Backs up and restores similarity globals to avoid cross-test interference.
    """
    from src.core import similarity  # type: ignore
    backup_store = dict(similarity._VECTOR_STORE)
    backup_meta = dict(similarity._VECTOR_META)
    backup_ts = dict(similarity._VECTOR_TS)
    try:
        yield
    finally:
        similarity._VECTOR_STORE.clear()
        similarity._VECTOR_META.clear()
        similarity._VECTOR_TS.clear()
        similarity._VECTOR_STORE.update(backup_store)
        similarity._VECTOR_META.update(backup_meta)
        similarity._VECTOR_TS.update(backup_ts)


@pytest.fixture
def disable_dimension_enforcement(monkeypatch):
    monkeypatch.setenv("ANALYSIS_VECTOR_DIM_CHECK", "0")
    yield

