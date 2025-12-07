import os

from src.core.twin.ingest import get_ingestor, reset_ingestor_for_tests, get_store
from src.core.storage.timeseries import InMemoryTimeSeriesStore, NullTimeSeriesStore


def test_store_factory_falls_back_to_memory(monkeypatch):
    monkeypatch.setenv("TELEMETRY_STORE_BACKEND", "unknown")
    # reset singleton
    import asyncio
    asyncio.run(reset_ingestor_for_tests())

    ingestor = get_ingestor()
    store = get_store()
    assert isinstance(store, InMemoryTimeSeriesStore)
    # basic enqueue works
    assert ingestor.queue.qsize() == 0


def test_store_factory_can_disable(monkeypatch):
    monkeypatch.setenv("TELEMETRY_STORE_BACKEND", "none")
    import src.core.config as cfg
    cfg._settings_cache = None  # reset settings cache to pick up env
    import asyncio
    asyncio.run(reset_ingestor_for_tests())

    get_ingestor()
    store = get_store()
    assert isinstance(store, NullTimeSeriesStore)
