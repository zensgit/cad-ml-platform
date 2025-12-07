import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1 import twin as twin_router
from src.core.twin.connectivity import TelemetryFrame
from src.core.twin.ingest import get_ingestor, reset_ingestor_for_tests


@pytest.mark.asyncio
async def test_history_endpoint_returns_recent_frames():
    await reset_ingestor_for_tests()
    app = FastAPI()
    app.include_router(twin_router.router, prefix="/api/v1/twin")

    ingestor = get_ingestor()
    await ingestor.ensure_started()

    for ts in [10.0, 20.0, 30.0]:
        await ingestor.handle_payload(
            TelemetryFrame(
                timestamp=ts,
                device_id="asset-1",
                sensors={"temp": ts},
                metrics={},
                status={},
            )
        )
    await asyncio.wait_for(ingestor.queue.join(), timeout=1)

    client = TestClient(app)
    resp = client.get(
        "/api/v1/twin/history",
        params={"device_id": "asset-1", "limit": 2},
        headers={"X-API-Key": "test"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 2
    # Newest first
    timestamps = [f["timestamp"] for f in payload["frames"]]
    assert timestamps == [30.0, 20.0]

