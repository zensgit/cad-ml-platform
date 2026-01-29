"""Tests for conversation persistence module."""

import json
import os
import shutil
import tempfile
import time

import pytest

from src.core.assistant.conversation import (
    Conversation,
    ConversationContext,
    Message,
    MessageRole,
)
from src.core.assistant.persistence import (
    ConversationPersistence,
    JSONStorageBackend,
    SQLiteStorageBackend,
)


class TestJSONStorageBackend:
    """Tests for JSON storage backend."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = JSONStorageBackend(self.temp_dir)

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_conversation(self) -> Conversation:
        """Create a test conversation."""
        conv = Conversation()
        conv.add_message(MessageRole.USER, "304不锈钢的强度是多少？")
        conv.add_message(MessageRole.ASSISTANT, "304不锈钢的抗拉强度约为520MPa。")
        conv.context.materials = ["304", "不锈钢"]
        conv.context.current_topic = "material_properties"
        conv.context.last_mentioned_material = "304"
        return conv

    def test_save_and_load(self):
        """Test saving and loading a conversation."""
        conv = self._create_test_conversation()

        # Save
        result = self.backend.save_conversation(conv)
        assert result is True

        # Load
        loaded = self.backend.load_conversation(conv.id)
        assert loaded is not None
        assert loaded.id == conv.id
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "304不锈钢的强度是多少？"
        assert loaded.context.materials == ["304", "不锈钢"]
        assert loaded.context.current_topic == "material_properties"
        assert loaded.context.last_mentioned_material == "304"

    def test_load_nonexistent(self):
        """Test loading a nonexistent conversation."""
        loaded = self.backend.load_conversation("nonexistent-id")
        assert loaded is None

    def test_delete(self):
        """Test deleting a conversation."""
        conv = self._create_test_conversation()
        self.backend.save_conversation(conv)

        # Delete
        result = self.backend.delete_conversation(conv.id)
        assert result is True

        # Verify deleted
        loaded = self.backend.load_conversation(conv.id)
        assert loaded is None

    def test_list_conversations(self):
        """Test listing conversations."""
        # Create multiple conversations
        for i in range(5):
            conv = Conversation()
            conv.add_message(MessageRole.USER, f"问题 {i}")
            self.backend.save_conversation(conv)

        # List
        conversations = self.backend.list_conversations()
        assert len(conversations) == 5

    def test_list_with_pagination(self):
        """Test listing conversations with pagination."""
        for i in range(10):
            conv = Conversation()
            conv.add_message(MessageRole.USER, f"问题 {i}")
            self.backend.save_conversation(conv)

        # Paginate
        page1 = self.backend.list_conversations(limit=3, offset=0)
        page2 = self.backend.list_conversations(limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3

    def test_cleanup_expired(self):
        """Test cleaning up expired conversations."""
        # Create old conversation
        old_conv = Conversation()
        old_conv.add_message(MessageRole.USER, "旧对话")
        old_conv.updated_at = time.time() - (40 * 24 * 60 * 60)  # 40 days ago
        self.backend.save_conversation(old_conv)

        # Create new conversation
        new_conv = Conversation()
        new_conv.add_message(MessageRole.USER, "新对话")
        self.backend.save_conversation(new_conv)

        # Cleanup (30 days)
        deleted = self.backend.cleanup_expired(30)
        assert deleted == 1

        # Verify old is deleted
        assert self.backend.load_conversation(old_conv.id) is None
        # Verify new still exists
        assert self.backend.load_conversation(new_conv.id) is not None

    def test_context_serialization(self):
        """Test full context serialization."""
        conv = Conversation()
        conv.context.materials = ["304"]
        conv.context.dimensions = [50.0, 100.0]
        conv.context.welding_processes = ["TIG"]
        conv.context.heat_treatments = ["淬火"]
        conv.context.surface_treatments = ["电镀"]
        conv.context.gdt_characteristics = ["平面度"]
        conv.context.topic_history = ["welding", "heat_treatment"]
        conv.context.last_mentioned_dimension = 50.0
        conv.context.last_mentioned_process = "TIG"

        self.backend.save_conversation(conv)
        loaded = self.backend.load_conversation(conv.id)

        assert loaded.context.materials == ["304"]
        assert loaded.context.dimensions == [50.0, 100.0]
        assert loaded.context.welding_processes == ["TIG"]
        assert loaded.context.heat_treatments == ["淬火"]
        assert loaded.context.surface_treatments == ["电镀"]
        assert loaded.context.gdt_characteristics == ["平面度"]
        assert loaded.context.topic_history == ["welding", "heat_treatment"]
        assert loaded.context.last_mentioned_dimension == 50.0
        assert loaded.context.last_mentioned_process == "TIG"


class TestSQLiteStorageBackend:
    """Tests for SQLite storage backend."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.backend = SQLiteStorageBackend(self.db_path)

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_conversation(self) -> Conversation:
        """Create a test conversation."""
        conv = Conversation()
        conv.add_message(MessageRole.USER, "TIG焊接参数是什么？")
        conv.add_message(MessageRole.ASSISTANT, "TIG焊接主要参数包括电流、电压、送丝速度等。")
        conv.context.welding_processes = ["TIG"]
        conv.context.current_topic = "welding"
        return conv

    def test_save_and_load(self):
        """Test saving and loading a conversation."""
        conv = self._create_test_conversation()

        # Save
        result = self.backend.save_conversation(conv)
        assert result is True

        # Load
        loaded = self.backend.load_conversation(conv.id)
        assert loaded is not None
        assert loaded.id == conv.id
        assert len(loaded.messages) == 2
        assert loaded.context.welding_processes == ["TIG"]

    def test_update_existing(self):
        """Test updating an existing conversation."""
        conv = self._create_test_conversation()
        self.backend.save_conversation(conv)

        # Add message and update
        conv.add_message(MessageRole.USER, "追问")
        self.backend.save_conversation(conv)

        # Load and verify
        loaded = self.backend.load_conversation(conv.id)
        assert len(loaded.messages) == 3

    def test_delete(self):
        """Test deleting a conversation."""
        conv = self._create_test_conversation()
        self.backend.save_conversation(conv)

        result = self.backend.delete_conversation(conv.id)
        assert result is True

        loaded = self.backend.load_conversation(conv.id)
        assert loaded is None

    def test_list_conversations(self):
        """Test listing conversations."""
        for i in range(5):
            conv = Conversation()
            conv.add_message(MessageRole.USER, f"问题 {i}")
            self.backend.save_conversation(conv)

        conversations = self.backend.list_conversations()
        assert len(conversations) == 5
        # Should be sorted by updated_at descending
        assert all("id" in c for c in conversations)

    def test_cleanup_expired(self):
        """Test cleaning up expired conversations."""
        # Create old conversation
        old_conv = Conversation()
        old_conv.add_message(MessageRole.USER, "旧对话")
        old_conv.updated_at = time.time() - (40 * 24 * 60 * 60)
        self.backend.save_conversation(old_conv)

        # Create new conversation
        new_conv = Conversation()
        new_conv.add_message(MessageRole.USER, "新对话")
        self.backend.save_conversation(new_conv)

        deleted = self.backend.cleanup_expired(30)
        assert deleted == 1


class TestConversationPersistence:
    """Tests for ConversationPersistence class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_json_backend_default(self):
        """Test default JSON backend."""
        storage_path = os.path.join(self.temp_dir, "json_test")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
        )

        conv = Conversation()
        conv.add_message(MessageRole.USER, "测试")

        assert persistence.save(conv) is True
        loaded = persistence.load(conv.id)
        assert loaded is not None

    def test_sqlite_backend(self):
        """Test SQLite backend."""
        db_path = os.path.join(self.temp_dir, "sqlite_test.db")
        persistence = ConversationPersistence(
            backend="sqlite",
            storage_path=db_path,
        )

        conv = Conversation()
        conv.add_message(MessageRole.USER, "测试")

        assert persistence.save(conv) is True
        loaded = persistence.load(conv.id)
        assert loaded is not None

    def test_delete(self):
        """Test delete operation."""
        storage_path = os.path.join(self.temp_dir, "delete_test")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
        )

        conv = Conversation()
        conv.add_message(MessageRole.USER, "测试")
        persistence.save(conv)

        assert persistence.delete(conv.id) is True
        assert persistence.load(conv.id) is None

    def test_list_conversations(self):
        """Test listing conversations."""
        storage_path = os.path.join(self.temp_dir, "list_test")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
        )

        for i in range(3):
            conv = Conversation()
            conv.add_message(MessageRole.USER, f"问题 {i}")
            persistence.save(conv)

        conversations = persistence.list_conversations()
        assert len(conversations) == 3

    def test_auto_save_disabled(self):
        """Test with auto-save disabled."""
        storage_path = os.path.join(self.temp_dir, "auto_save_test")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
            auto_save=False,
        )

        conv = Conversation()
        conv.add_message(MessageRole.USER, "测试")

        assert persistence.should_auto_save(conv) is False

    def test_auto_save_trigger(self):
        """Test auto-save trigger."""
        storage_path = os.path.join(self.temp_dir, "auto_save_trigger")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
            auto_save=True,
            auto_save_interval=3,
        )

        conv = Conversation()

        # Should not trigger with < 3 messages
        conv.add_message(MessageRole.USER, "1")
        conv.add_message(MessageRole.ASSISTANT, "1")
        assert persistence.should_auto_save(conv) is False

        # Should trigger at 3 messages
        conv.add_message(MessageRole.USER, "2")
        assert persistence.should_auto_save(conv) is True

        # Should not trigger immediately after
        assert persistence.should_auto_save(conv) is False

    def test_auto_save_if_needed(self):
        """Test auto_save_if_needed method."""
        storage_path = os.path.join(self.temp_dir, "auto_save_needed")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
            auto_save=True,
            auto_save_interval=2,
        )

        conv = Conversation()
        conv.add_message(MessageRole.USER, "1")
        conv.add_message(MessageRole.ASSISTANT, "1")

        # Should save
        result = persistence.auto_save_if_needed(conv)
        assert result is True

        # Verify saved
        loaded = persistence.load(conv.id)
        assert loaded is not None

    def test_cleanup(self):
        """Test cleanup operation."""
        storage_path = os.path.join(self.temp_dir, "cleanup_test")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
        )

        # Create old conversation
        old_conv = Conversation()
        old_conv.add_message(MessageRole.USER, "旧")
        old_conv.updated_at = time.time() - (40 * 24 * 60 * 60)
        persistence.save(old_conv)

        deleted = persistence.cleanup(30)
        assert deleted == 1

    def test_export_json(self):
        """Test exporting conversation as JSON."""
        storage_path = os.path.join(self.temp_dir, "export_test")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
        )

        conv = Conversation()
        conv.add_message(MessageRole.USER, "问题")
        conv.add_message(MessageRole.ASSISTANT, "回答")
        conv.context.current_topic = "test_topic"
        persistence.save(conv)

        exported = persistence.export_conversation(conv.id, format="json")
        assert exported is not None

        data = json.loads(exported)
        assert data["id"] == conv.id
        assert len(data["messages"]) == 2
        assert data["topic"] == "test_topic"

    def test_export_markdown(self):
        """Test exporting conversation as Markdown."""
        storage_path = os.path.join(self.temp_dir, "export_md_test")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
        )

        conv = Conversation()
        conv.add_message(MessageRole.USER, "问题")
        conv.add_message(MessageRole.ASSISTANT, "回答")
        persistence.save(conv)

        exported = persistence.export_conversation(conv.id, format="markdown")
        assert exported is not None
        assert "# 对话记录" in exported
        assert "**用户**" in exported
        assert "**助手**" in exported

    def test_export_nonexistent(self):
        """Test exporting nonexistent conversation."""
        storage_path = os.path.join(self.temp_dir, "export_none_test")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
        )

        exported = persistence.export_conversation("nonexistent", format="json")
        assert exported is None

    def test_import_json(self):
        """Test importing conversation from JSON."""
        storage_path = os.path.join(self.temp_dir, "import_test")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
        )

        json_data = json.dumps({
            "id": "imported-conv-123",
            "created_at": time.time(),
            "updated_at": time.time(),
            "topic": "imported_topic",
            "messages": [
                {"role": "user", "content": "导入的问题"},
                {"role": "assistant", "content": "导入的回答"},
            ],
        })

        conv = persistence.import_conversation(json_data, format="json")
        assert conv is not None
        assert conv.id == "imported-conv-123"
        assert len(conv.messages) == 2
        assert conv.context.current_topic == "imported_topic"

        # Verify saved
        loaded = persistence.load("imported-conv-123")
        assert loaded is not None

    def test_import_invalid_json(self):
        """Test importing invalid JSON."""
        storage_path = os.path.join(self.temp_dir, "import_invalid_test")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
        )

        conv = persistence.import_conversation("invalid json", format="json")
        assert conv is None


class TestPersistenceIntegration:
    """Integration tests for persistence functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_conversation_lifecycle(self):
        """Test complete conversation lifecycle with persistence."""
        storage_path = os.path.join(self.temp_dir, "lifecycle_test")
        persistence = ConversationPersistence(
            backend="json",
            storage_path=storage_path,
            auto_save=True,
            auto_save_interval=2,
        )

        # Create conversation
        conv = Conversation()
        conv.add_message(MessageRole.USER, "304不锈钢的强度？")
        conv.add_message(MessageRole.ASSISTANT, "抗拉强度约520MPa")
        conv.context.materials = ["304"]
        conv.context.current_topic = "material_properties"

        # Auto-save should trigger
        saved = persistence.auto_save_if_needed(conv)
        assert saved is True

        # Simulate session end and restart
        conv_id = conv.id

        # Load in new "session"
        restored = persistence.load(conv_id)
        assert restored is not None
        assert len(restored.messages) == 2
        assert restored.context.materials == ["304"]

        # Continue conversation
        restored.add_message(MessageRole.USER, "它的密度呢？")
        restored.add_message(MessageRole.ASSISTANT, "密度约7.93g/cm³")
        persistence.save(restored)

        # Verify final state
        final = persistence.load(conv_id)
        assert len(final.messages) == 4

    def test_multiple_backend_compatibility(self):
        """Test data compatibility between backends."""
        json_path = os.path.join(self.temp_dir, "json_compat")
        sqlite_path = os.path.join(self.temp_dir, "sqlite_compat.db")

        json_persistence = ConversationPersistence(
            backend="json",
            storage_path=json_path,
        )
        sqlite_persistence = ConversationPersistence(
            backend="sqlite",
            storage_path=sqlite_path,
        )

        # Create in JSON
        conv = Conversation()
        conv.add_message(MessageRole.USER, "测试兼容性")
        conv.context.materials = ["304"]
        conv.context.welding_processes = ["TIG"]
        json_persistence.save(conv)

        # Export from JSON
        exported = json_persistence.export_conversation(conv.id, format="json")

        # Import to SQLite
        imported = sqlite_persistence.import_conversation(exported, format="json")
        assert imported is not None

        # Verify data integrity
        loaded = sqlite_persistence.load(imported.id)
        assert loaded.messages[0].content == "测试兼容性"
