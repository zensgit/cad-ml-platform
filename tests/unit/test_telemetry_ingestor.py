import asyncio

import pytest

from src.core.storage.timeseries import InMemoryTimeSeriesStore
from src.core.twin.connectivity import TelemetryFrame
from src.core.twin.ingest import TelemetryIngestor


@pytest.mark.asyncio
async def test_ingestor_backpressure_and_persistence():
    store = InMemoryTimeSeriesStore(max_per_device=5)
    ingestor = TelemetryIngestor(store, max_queue=2)
    await ingestor.ensure_started()

    frames = [
        TelemetryFrame(timestamp=1.0, device_id="dev", sensors={"t": 1.0}, metrics={}, status={}),
        TelemetryFrame(timestamp=2.0, device_id="dev", sensors={"t": 2.0}, metrics={}, status={}),
        TelemetryFrame(timestamp=3.0, device_id="dev", sensors={"t": 3.0}, metrics={}, status={}),
    ]

    results = []
    for frame in frames:
        results.append(await ingestor.handle_payload(frame))

    await asyncio.wait_for(ingestor.queue.join(), timeout=1)

    hist = await store.history("dev", limit=10)
    assert len(hist) == 2  # third should be dropped due to backpressure
    assert ingestor.drop_count == 1
    assert results[-1]["status"] == "dropped"
