"""
Digital Twin Sync module.

Provides real-time state synchronization for digital twins.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TwinSync:
    """Manages real-time state synchronization for digital twins."""

    def __init__(self) -> None:
        self._states: Dict[str, Dict[str, Any]] = {}
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []

    async def update_state(self, asset_id: str, data: Dict[str, Any]) -> None:
        """Update the state for a given asset."""
        if asset_id not in self._states:
            self._states[asset_id] = {}
        self._states[asset_id].update(data)

        event = {
            "type": "state_update",
            "asset_id": asset_id,
            "data": data,
        }

        for callback in self._subscribers:
            try:
                callback(event)
            except Exception:
                logger.exception("Error in subscriber callback")

    def get_state(self, asset_id: str) -> Dict[str, Any]:
        """Get the current state for a given asset."""
        return self._states.get(asset_id, {})

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to state updates."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Unsubscribe from state updates."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def reset(self) -> None:
        """Reset all states and subscribers."""
        self._states.clear()
        self._subscribers.clear()


# Singleton instance
twin_sync = TwinSync()
