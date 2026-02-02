"""WebSocket Room Management.

Provides room-based grouping for WebSocket connections.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Room:
    """A room for grouping WebSocket connections."""

    room_id: str
    name: str
    room_type: str = "default"  # user, tenant, topic, custom

    # Members
    members: Set[str] = field(default_factory=set)  # connection_ids

    # Settings
    max_members: Optional[int] = None
    is_private: bool = False
    password_hash: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    owner_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    def add_member(self, connection_id: str) -> bool:
        """Add a member to the room.

        Returns:
            True if added, False if room is full
        """
        if self.max_members and len(self.members) >= self.max_members:
            return False

        self.members.add(connection_id)
        self.last_activity = datetime.utcnow()
        return True

    def remove_member(self, connection_id: str) -> bool:
        """Remove a member from the room.

        Returns:
            True if removed
        """
        if connection_id in self.members:
            self.members.discard(connection_id)
            self.last_activity = datetime.utcnow()
            return True
        return False

    def has_member(self, connection_id: str) -> bool:
        """Check if connection is a member."""
        return connection_id in self.members

    def is_empty(self) -> bool:
        """Check if room has no members."""
        return len(self.members) == 0

    def member_count(self) -> int:
        """Get number of members."""
        return len(self.members)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "room_id": self.room_id,
            "name": self.name,
            "room_type": self.room_type,
            "member_count": len(self.members),
            "max_members": self.max_members,
            "is_private": self.is_private,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class RoomManager:
    """Manager for WebSocket rooms.

    Handles room creation, membership, and messaging.
    """

    def __init__(
        self,
        auto_cleanup: bool = True,
        cleanup_interval: float = 300.0,  # 5 minutes
    ):
        self._rooms: Dict[str, Room] = {}
        self._connection_rooms: Dict[str, Set[str]] = {}  # connection_id -> room_ids

        self._auto_cleanup = auto_cleanup
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock: Optional[asyncio.Lock] = None

        # Callbacks
        self._on_join: List[Callable[[str, str], Any]] = []  # room_id, connection_id
        self._on_leave: List[Callable[[str, str], Any]] = []

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def create_room(
        self,
        room_id: str,
        name: Optional[str] = None,
        room_type: str = "default",
        max_members: Optional[int] = None,
        is_private: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        owner_id: Optional[str] = None,
    ) -> Room:
        """Create a new room.

        Args:
            room_id: Unique room identifier
            name: Room display name
            room_type: Type of room
            max_members: Maximum members allowed
            is_private: Whether room requires password
            metadata: Additional metadata
            owner_id: Owner connection ID

        Returns:
            Created Room
        """
        async with self._get_lock():
            if room_id in self._rooms:
                return self._rooms[room_id]

            room = Room(
                room_id=room_id,
                name=name or room_id,
                room_type=room_type,
                max_members=max_members,
                is_private=is_private,
                metadata=metadata or {},
                owner_id=owner_id,
            )

            self._rooms[room_id] = room
            logger.info(f"Created room: {room_id}")
            return room

    async def delete_room(self, room_id: str) -> bool:
        """Delete a room.

        Args:
            room_id: Room ID

        Returns:
            True if deleted
        """
        async with self._get_lock():
            room = self._rooms.pop(room_id, None)
            if not room:
                return False

            # Remove from connection indices
            for conn_id in list(room.members):
                if conn_id in self._connection_rooms:
                    self._connection_rooms[conn_id].discard(room_id)

            logger.info(f"Deleted room: {room_id}")
            return True

    async def join_room(
        self,
        room_id: str,
        connection_id: str,
        create_if_missing: bool = True,
    ) -> bool:
        """Join a room.

        Args:
            room_id: Room ID
            connection_id: Connection ID
            create_if_missing: Create room if it doesn't exist

        Returns:
            True if joined
        """
        async with self._get_lock():
            room = self._rooms.get(room_id)

            if not room:
                if create_if_missing:
                    room = Room(room_id=room_id, name=room_id)
                    self._rooms[room_id] = room
                else:
                    return False

            if not room.add_member(connection_id):
                return False

            # Update connection index
            if connection_id not in self._connection_rooms:
                self._connection_rooms[connection_id] = set()
            self._connection_rooms[connection_id].add(room_id)

        # Call join callbacks
        for callback in self._on_join:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(room_id, connection_id)
                else:
                    callback(room_id, connection_id)
            except Exception as e:
                logger.error(f"Join callback error: {e}")

        logger.debug(f"Connection {connection_id} joined room {room_id}")
        return True

    async def leave_room(self, room_id: str, connection_id: str) -> bool:
        """Leave a room.

        Args:
            room_id: Room ID
            connection_id: Connection ID

        Returns:
            True if left
        """
        async with self._get_lock():
            room = self._rooms.get(room_id)
            if not room:
                return False

            if not room.remove_member(connection_id):
                return False

            # Update connection index
            if connection_id in self._connection_rooms:
                self._connection_rooms[connection_id].discard(room_id)

        # Call leave callbacks
        for callback in self._on_leave:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(room_id, connection_id)
                else:
                    callback(room_id, connection_id)
            except Exception as e:
                logger.error(f"Leave callback error: {e}")

        logger.debug(f"Connection {connection_id} left room {room_id}")
        return True

    async def leave_all_rooms(self, connection_id: str) -> int:
        """Remove connection from all rooms.

        Args:
            connection_id: Connection ID

        Returns:
            Number of rooms left
        """
        room_ids = list(self._connection_rooms.get(connection_id, set()))
        count = 0

        for room_id in room_ids:
            if await self.leave_room(room_id, connection_id):
                count += 1

        return count

    def get_room(self, room_id: str) -> Optional[Room]:
        """Get a room by ID."""
        return self._rooms.get(room_id)

    def get_room_members(self, room_id: str) -> Set[str]:
        """Get all members of a room."""
        room = self._rooms.get(room_id)
        if room:
            return room.members.copy()
        return set()

    def get_connection_rooms(self, connection_id: str) -> Set[str]:
        """Get all rooms a connection is in."""
        return self._connection_rooms.get(connection_id, set()).copy()

    def list_rooms(
        self,
        room_type: Optional[str] = None,
        include_empty: bool = False,
    ) -> List[Room]:
        """List all rooms.

        Args:
            room_type: Filter by room type
            include_empty: Include empty rooms

        Returns:
            List of rooms
        """
        rooms = list(self._rooms.values())

        if room_type:
            rooms = [r for r in rooms if r.room_type == room_type]

        if not include_empty:
            rooms = [r for r in rooms if not r.is_empty()]

        return rooms

    def on_join(self, callback: Callable[[str, str], Any]) -> None:
        """Register join callback."""
        self._on_join.append(callback)

    def on_leave(self, callback: Callable[[str, str], Any]) -> None:
        """Register leave callback."""
        self._on_leave.append(callback)

    async def start_cleanup(self) -> None:
        """Start automatic cleanup of empty rooms."""
        if not self._auto_cleanup or self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self._cleanup_interval)
                    await self._cleanup_empty_rooms()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Room cleanup error: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_cleanup(self) -> None:
        """Stop automatic cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_empty_rooms(self) -> int:
        """Remove empty rooms.

        Returns:
            Number of rooms removed
        """
        empty_rooms = [
            room_id for room_id, room in self._rooms.items()
            if room.is_empty() and room.room_type != "persistent"
        ]

        for room_id in empty_rooms:
            await self.delete_room(room_id)

        if empty_rooms:
            logger.info(f"Cleaned up {len(empty_rooms)} empty rooms")

        return len(empty_rooms)

    def get_stats(self) -> Dict[str, Any]:
        """Get room statistics."""
        total_members = sum(room.member_count() for room in self._rooms.values())

        return {
            "total_rooms": len(self._rooms),
            "total_members": total_members,
            "rooms_by_type": self._count_by_type(),
            "empty_rooms": sum(1 for r in self._rooms.values() if r.is_empty()),
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count rooms by type."""
        counts: Dict[str, int] = {}
        for room in self._rooms.values():
            counts[room.room_type] = counts.get(room.room_type, 0) + 1
        return counts


# Global room manager
_room_manager: Optional[RoomManager] = None


def get_room_manager() -> RoomManager:
    """Get global room manager."""
    global _room_manager
    if _room_manager is None:
        _room_manager = RoomManager()
    return _room_manager
