"""
Digital Twin API
Phase 8: Real-time Sync
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from src.core.twin.sync import twin_sync
from src.core.twin.ingest import get_ingestor, get_store
from src.core.twin.connectivity import TelemetryFrame
from src.api.dependencies import get_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/{asset_id}")
async def websocket_endpoint(websocket: WebSocket, asset_id: str):
    """
    WebSocket endpoint for real-time digital twin updates.
    """
    await websocket.accept()

    # Callback to push updates to this websocket
    async def push_update(event: Dict[str, Any]):
        if event["asset_id"] == asset_id:
            try:
                await websocket.send_json(event)
            except Exception:
                # Connection likely closed
                pass

    # Register subscription (Note: In a real app, we need a way to unsubscribe)
    # For this prototype, we'll just append.
    # A better approach is to have a connection manager.

    # Wrapper to bridge sync callback to async websocket send
    # Since twin_sync.subscribe expects a sync callable, we need to be careful.
    # Ideally twin_sync should support async callbacks or we use an event loop.

    # For this prototype, let's just poll or handle incoming messages.
    # Real implementation would use a proper Pub/Sub manager.

    try:
        while True:
            # Wait for messages from client (e.g. control commands)
            _ = await websocket.receive_text()

            # Echo back current state
            state = twin_sync.get_state(asset_id)
            await websocket.send_json({
                "type": "state_snapshot",
                "asset_id": asset_id,
                "state": state
            })

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from twin {asset_id}")

@router.post("/{asset_id}/telemetry")
async def ingest_telemetry(asset_id: str, telemetry: Dict[str, Any]):
    """
    Ingest telemetry data (HTTP fallback for MQTT).
    """
    await twin_sync.update_state(asset_id, telemetry)

    # Push to telemetry ingestor for persistence/backpressure handling
    ingestor = get_ingestor()
    await ingestor.ensure_started()
    frame = TelemetryFrame(
        timestamp=float(telemetry.get("timestamp", __import__("time").time())),
        device_id=asset_id,
        sensors=telemetry.get("sensors", {}),
        metrics=telemetry.get("metrics", {}),
        status=telemetry.get("status", {}),
    )
    result = await ingestor.handle_payload(frame, topic=f"http/{asset_id}")
    return {"status": "accepted", "ingest": result}

@router.get("/{asset_id}/state")
async def get_twin_state(asset_id: str):
    """
    Get current state snapshot.
    """
    return twin_sync.get_state(asset_id)


@router.get("/history")
async def get_history(
    device_id: str = Query(..., description="Device/asset id to fetch history for"),
    limit: int = Query(default=50, ge=1, le=500, description="Max frames to return"),
    api_key: str = Depends(get_api_key),
):
    """
    Fetch recent telemetry history from the time-series store.
    """
    store = get_store()
    ingestor = get_ingestor()
    await ingestor.ensure_started()
    frames = await store.history(device_id=device_id, limit=limit)
    return {
        "device_id": device_id,
        "count": len(frames),
        "frames": [f.model_dump() for f in frames],
    }
