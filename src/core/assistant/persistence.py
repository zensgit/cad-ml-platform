"""
Conversation Persistence Module.

Provides storage and retrieval of conversation history
with support for JSON and SQLite backends.
"""

import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .conversation import (
    Conversation,
    ConversationContext,
    Message,
    MessageRole,
)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save_conversation(self, conversation: Conversation) -> bool:
        """Save a conversation to storage."""
        pass

    @abstractmethod
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load a conversation from storage."""
        pass

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation from storage."""
        pass

    @abstractmethod
    def list_conversations(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored conversations with metadata."""
        pass

    @abstractmethod
    def cleanup_expired(self, max_age_days: int = 30) -> int:
        """Remove conversations older than max_age_days."""
        pass


class JSONStorageBackend(StorageBackend):
    """JSON file-based storage backend."""

    def __init__(self, storage_dir: str = ".cad_assistant/conversations"):
        """
        Initialize JSON storage backend.

        Args:
            storage_dir: Directory to store conversation files
        """
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_dir / "index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load or create the conversation index."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._index = {"conversations": {}}
        else:
            self._index = {"conversations": {}}

    def _save_index(self) -> None:
        """Save the conversation index."""
        with open(self._index_file, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)

    def _get_file_path(self, conversation_id: str) -> Path:
        """Get the file path for a conversation."""
        return self.storage_dir / f"{conversation_id}.json"

    def _serialize_conversation(self, conv: Conversation) -> Dict[str, Any]:
        """Serialize a conversation to a dictionary."""
        return {
            "id": conv.id,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata,
                }
                for msg in conv.messages
            ],
            "context": {
                "materials": conv.context.materials,
                "dimensions": conv.context.dimensions,
                "tolerances": conv.context.tolerances,
                "thread_specs": conv.context.thread_specs,
                "welding_processes": conv.context.welding_processes,
                "heat_treatments": conv.context.heat_treatments,
                "surface_treatments": conv.context.surface_treatments,
                "gdt_characteristics": conv.context.gdt_characteristics,
                "current_topic": conv.context.current_topic,
                "topic_history": conv.context.topic_history,
                "last_query_intent": conv.context.last_query_intent,
                "pending_clarifications": conv.context.pending_clarifications,
                "last_mentioned_material": conv.context.last_mentioned_material,
                "last_mentioned_dimension": conv.context.last_mentioned_dimension,
                "last_mentioned_process": conv.context.last_mentioned_process,
            },
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "metadata": conv.metadata,
        }

    def _deserialize_conversation(self, data: Dict[str, Any]) -> Conversation:
        """Deserialize a dictionary to a conversation."""
        context_data = data.get("context", {})
        context = ConversationContext(
            materials=context_data.get("materials", []),
            dimensions=context_data.get("dimensions", []),
            tolerances=context_data.get("tolerances", []),
            thread_specs=context_data.get("thread_specs", []),
            welding_processes=context_data.get("welding_processes", []),
            heat_treatments=context_data.get("heat_treatments", []),
            surface_treatments=context_data.get("surface_treatments", []),
            gdt_characteristics=context_data.get("gdt_characteristics", []),
            current_topic=context_data.get("current_topic"),
            topic_history=context_data.get("topic_history", []),
            last_query_intent=context_data.get("last_query_intent"),
            pending_clarifications=context_data.get("pending_clarifications", []),
            last_mentioned_material=context_data.get("last_mentioned_material"),
            last_mentioned_dimension=context_data.get("last_mentioned_dimension"),
            last_mentioned_process=context_data.get("last_mentioned_process"),
        )

        conv = Conversation(
            id=data["id"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
        )
        conv.context = context

        for msg_data in data.get("messages", []):
            msg = Message(
                role=MessageRole(msg_data["role"]),
                content=msg_data["content"],
                timestamp=msg_data.get("timestamp", time.time()),
                metadata=msg_data.get("metadata", {}),
            )
            conv.messages.append(msg)

        return conv

    def save_conversation(self, conversation: Conversation) -> bool:
        """Save a conversation to a JSON file."""
        try:
            file_path = self._get_file_path(conversation.id)
            data = self._serialize_conversation(conversation)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Update index
            self._index["conversations"][conversation.id] = {
                "id": conversation.id,
                "message_count": len(conversation.messages),
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "current_topic": conversation.context.current_topic,
            }
            self._save_index()

            return True
        except IOError:
            return False

    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load a conversation from a JSON file."""
        file_path = self._get_file_path(conversation_id)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._deserialize_conversation(data)
        except (json.JSONDecodeError, IOError, KeyError):
            return None

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation file."""
        file_path = self._get_file_path(conversation_id)
        try:
            if file_path.exists():
                file_path.unlink()

            if conversation_id in self._index["conversations"]:
                del self._index["conversations"][conversation_id]
                self._save_index()

            return True
        except IOError:
            return False

    def list_conversations(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored conversations with metadata."""
        conversations = list(self._index["conversations"].values())
        # Sort by updated_at descending
        conversations.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return conversations[offset : offset + limit]

    def cleanup_expired(self, max_age_days: int = 30) -> int:
        """Remove conversations older than max_age_days."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        deleted_count = 0

        for conv_id, conv_meta in list(self._index["conversations"].items()):
            if conv_meta.get("updated_at", 0) < cutoff_time:
                if self.delete_conversation(conv_id):
                    deleted_count += 1

        return deleted_count


class SQLiteStorageBackend(StorageBackend):
    """SQLite database storage backend."""

    def __init__(self, db_path: str = ".cad_assistant/conversations.db"):
        """
        Initialize SQLite storage backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    current_topic TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_updated_at
                ON conversations(updated_at DESC)
            """)
            conn.commit()

    def save_conversation(self, conversation: Conversation) -> bool:
        """Save a conversation to SQLite."""
        try:
            data = {
                "id": conversation.id,
                "messages": [
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "metadata": msg.metadata,
                    }
                    for msg in conversation.messages
                ],
                "context": {
                    "materials": conversation.context.materials,
                    "dimensions": conversation.context.dimensions,
                    "tolerances": conversation.context.tolerances,
                    "thread_specs": conversation.context.thread_specs,
                    "welding_processes": conversation.context.welding_processes,
                    "heat_treatments": conversation.context.heat_treatments,
                    "surface_treatments": conversation.context.surface_treatments,
                    "gdt_characteristics": conversation.context.gdt_characteristics,
                    "current_topic": conversation.context.current_topic,
                    "topic_history": conversation.context.topic_history,
                    "last_query_intent": conversation.context.last_query_intent,
                    "pending_clarifications": conversation.context.pending_clarifications,
                    "last_mentioned_material": conversation.context.last_mentioned_material,
                    "last_mentioned_dimension": conversation.context.last_mentioned_dimension,
                    "last_mentioned_process": conversation.context.last_mentioned_process,
                },
                "metadata": conversation.metadata,
            }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO conversations
                    (id, data, created_at, updated_at, message_count, current_topic)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation.id,
                        json.dumps(data, ensure_ascii=False),
                        conversation.created_at,
                        conversation.updated_at,
                        len(conversation.messages),
                        conversation.context.current_topic,
                    ),
                )
                conn.commit()

            return True
        except sqlite3.Error:
            return False

    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load a conversation from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data, created_at, updated_at FROM conversations WHERE id = ?",
                    (conversation_id,),
                )
                row = cursor.fetchone()

            if not row:
                return None

            data = json.loads(row[0])
            context_data = data.get("context", {})

            context = ConversationContext(
                materials=context_data.get("materials", []),
                dimensions=context_data.get("dimensions", []),
                tolerances=context_data.get("tolerances", []),
                thread_specs=context_data.get("thread_specs", []),
                welding_processes=context_data.get("welding_processes", []),
                heat_treatments=context_data.get("heat_treatments", []),
                surface_treatments=context_data.get("surface_treatments", []),
                gdt_characteristics=context_data.get("gdt_characteristics", []),
                current_topic=context_data.get("current_topic"),
                topic_history=context_data.get("topic_history", []),
                last_query_intent=context_data.get("last_query_intent"),
                pending_clarifications=context_data.get("pending_clarifications", []),
                last_mentioned_material=context_data.get("last_mentioned_material"),
                last_mentioned_dimension=context_data.get("last_mentioned_dimension"),
                last_mentioned_process=context_data.get("last_mentioned_process"),
            )

            conv = Conversation(
                id=data["id"],
                created_at=row[1],
                updated_at=row[2],
                metadata=data.get("metadata", {}),
            )
            conv.context = context

            for msg_data in data.get("messages", []):
                msg = Message(
                    role=MessageRole(msg_data["role"]),
                    content=msg_data["content"],
                    timestamp=msg_data.get("timestamp", time.time()),
                    metadata=msg_data.get("metadata", {}),
                )
                conv.messages.append(msg)

            return conv
        except (sqlite3.Error, json.JSONDecodeError, KeyError):
            return None

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (conversation_id,),
                )
                conn.commit()
            return True
        except sqlite3.Error:
            return False

    def list_conversations(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored conversations with metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, created_at, updated_at, message_count, current_topic
                    FROM conversations
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
                rows = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "created_at": row[1],
                    "updated_at": row[2],
                    "message_count": row[3],
                    "current_topic": row[4],
                }
                for row in rows
            ]
        except sqlite3.Error:
            return []

    def cleanup_expired(self, max_age_days: int = 30) -> int:
        """Remove conversations older than max_age_days."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE updated_at < ?",
                    (cutoff_time,),
                )
                deleted_count = cursor.rowcount
                conn.commit()
            return deleted_count
        except sqlite3.Error:
            return 0


class ConversationPersistence:
    """
    High-level interface for conversation persistence.

    Provides auto-save, restore, and management capabilities.

    Example:
        >>> persistence = ConversationPersistence()
        >>> persistence.save(conversation)
        >>> restored = persistence.load(conversation_id)
    """

    def __init__(
        self,
        backend: str = "json",
        storage_path: Optional[str] = None,
        auto_save: bool = True,
        auto_save_interval: int = 5,  # Save after every N messages
    ):
        """
        Initialize conversation persistence.

        Args:
            backend: Storage backend type ("json" or "sqlite")
            storage_path: Custom storage path (uses default if None)
            auto_save: Enable automatic saving
            auto_save_interval: Save after every N messages
        """
        self.auto_save = auto_save
        self.auto_save_interval = auto_save_interval
        self._message_counts: Dict[str, int] = {}

        if backend == "sqlite":
            path = storage_path or ".cad_assistant/conversations.db"
            self._backend = SQLiteStorageBackend(path)
        else:
            path = storage_path or ".cad_assistant/conversations"
            self._backend = JSONStorageBackend(path)

    def save(self, conversation: Conversation) -> bool:
        """
        Save a conversation to storage.

        Args:
            conversation: Conversation to save

        Returns:
            True if successful
        """
        return self._backend.save_conversation(conversation)

    def load(self, conversation_id: str) -> Optional[Conversation]:
        """
        Load a conversation from storage.

        Args:
            conversation_id: ID of conversation to load

        Returns:
            Conversation if found, None otherwise
        """
        return self._backend.load_conversation(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        """
        Delete a conversation from storage.

        Args:
            conversation_id: ID of conversation to delete

        Returns:
            True if successful
        """
        if conversation_id in self._message_counts:
            del self._message_counts[conversation_id]
        return self._backend.delete_conversation(conversation_id)

    def list_conversations(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List stored conversations.

        Args:
            limit: Maximum number to return
            offset: Number to skip

        Returns:
            List of conversation metadata dicts
        """
        return self._backend.list_conversations(limit, offset)

    def cleanup(self, max_age_days: int = 30) -> int:
        """
        Remove old conversations.

        Args:
            max_age_days: Remove conversations older than this

        Returns:
            Number of conversations deleted
        """
        return self._backend.cleanup_expired(max_age_days)

    def should_auto_save(self, conversation: Conversation) -> bool:
        """
        Check if conversation should be auto-saved.

        Args:
            conversation: Conversation to check

        Returns:
            True if auto-save should trigger
        """
        if not self.auto_save:
            return False

        current_count = len(conversation.messages)
        last_count = self._message_counts.get(conversation.id, 0)

        if current_count - last_count >= self.auto_save_interval:
            self._message_counts[conversation.id] = current_count
            return True

        return False

    def auto_save_if_needed(self, conversation: Conversation) -> bool:
        """
        Auto-save conversation if needed.

        Args:
            conversation: Conversation to potentially save

        Returns:
            True if saved, False otherwise
        """
        if self.should_auto_save(conversation):
            return self.save(conversation)
        return False

    def export_conversation(
        self,
        conversation_id: str,
        format: str = "json",
    ) -> Optional[str]:
        """
        Export a conversation to a string format.

        Args:
            conversation_id: ID of conversation to export
            format: Export format ("json" or "markdown")

        Returns:
            Exported string, or None if not found
        """
        conv = self.load(conversation_id)
        if not conv:
            return None

        if format == "markdown":
            lines = [f"# 对话记录 {conv.id}\n"]
            lines.append(f"创建时间: {time.ctime(conv.created_at)}\n")
            lines.append(f"话题: {conv.context.current_topic or '未分类'}\n\n")

            for msg in conv.messages:
                role = "**用户**" if msg.role == MessageRole.USER else "**助手**"
                lines.append(f"{role}: {msg.content}\n\n")

            return "".join(lines)
        else:
            # JSON format
            data = {
                "id": conv.id,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "topic": conv.context.current_topic,
                "messages": [
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                    }
                    for msg in conv.messages
                ],
            }
            return json.dumps(data, ensure_ascii=False, indent=2)

    def import_conversation(
        self,
        data: str,
        format: str = "json",
    ) -> Optional[Conversation]:
        """
        Import a conversation from a string.

        Args:
            data: Conversation data string
            format: Import format ("json")

        Returns:
            Imported conversation, or None if failed
        """
        if format != "json":
            return None

        try:
            conv_data = json.loads(data)

            conv = Conversation(
                id=conv_data.get("id"),
                created_at=conv_data.get("created_at", time.time()),
                updated_at=conv_data.get("updated_at", time.time()),
            )

            if "topic" in conv_data:
                conv.context.current_topic = conv_data["topic"]

            for msg_data in conv_data.get("messages", []):
                msg = Message(
                    role=MessageRole(msg_data["role"]),
                    content=msg_data["content"],
                    timestamp=msg_data.get("timestamp", time.time()),
                )
                conv.messages.append(msg)

            # Save imported conversation
            self.save(conv)
            return conv
        except (json.JSONDecodeError, KeyError):
            return None
