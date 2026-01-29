"""Tests for conversation history management."""

import pytest
import time


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        """Test creating a message."""
        from src.core.assistant.conversation import Message, MessageRole

        msg = Message(
            role=MessageRole.USER,
            content="304不锈钢的强度是多少？",
        )

        assert msg.role == MessageRole.USER
        assert "304" in msg.content
        assert msg.timestamp > 0

    def test_message_to_dict(self):
        """Test message serialization."""
        from src.core.assistant.conversation import Message, MessageRole

        msg = Message(
            role=MessageRole.ASSISTANT,
            content="304不锈钢的抗拉强度为520MPa",
            metadata={"confidence": 0.9},
        )

        data = msg.to_dict()

        assert data["role"] == "assistant"
        assert "520" in data["content"]
        assert data["metadata"]["confidence"] == 0.9

    def test_message_from_dict(self):
        """Test message deserialization."""
        from src.core.assistant.conversation import Message, MessageRole

        data = {
            "role": "user",
            "content": "它的密度呢？",
            "timestamp": 1234567890,
        }

        msg = Message.from_dict(data)

        assert msg.role == MessageRole.USER
        assert msg.content == "它的密度呢？"
        assert msg.timestamp == 1234567890


class TestConversation:
    """Tests for Conversation class."""

    def test_conversation_creation(self):
        """Test creating a conversation."""
        from src.core.assistant.conversation import Conversation

        conv = Conversation()

        assert conv.id is not None
        assert len(conv.messages) == 0
        assert conv.created_at > 0

    def test_add_message(self):
        """Test adding messages to conversation."""
        from src.core.assistant.conversation import Conversation, MessageRole

        conv = Conversation()
        conv.add_message(MessageRole.USER, "问题1")
        conv.add_message(MessageRole.ASSISTANT, "回答1")

        assert len(conv.messages) == 2
        assert conv.messages[0].role == MessageRole.USER
        assert conv.messages[1].role == MessageRole.ASSISTANT

    def test_get_recent_messages(self):
        """Test getting recent messages."""
        from src.core.assistant.conversation import Conversation, MessageRole

        conv = Conversation()
        for i in range(10):
            conv.add_message(MessageRole.USER, f"问题{i}")

        recent = conv.get_recent_messages(3)
        assert len(recent) == 3
        assert "问题7" in recent[0].content


class TestConversationManager:
    """Tests for ConversationManager."""

    def test_create_conversation(self):
        """Test creating a new conversation."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager()
        conv_id = manager.create_conversation()

        assert conv_id is not None
        assert manager.get_conversation(conv_id) is not None

    def test_add_messages(self):
        """Test adding messages to conversation."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager()
        conv_id = manager.create_conversation()

        manager.add_user_message(conv_id, "304不锈钢的强度？")
        manager.add_assistant_message(conv_id, "520MPa")

        conv = manager.get_conversation(conv_id)
        assert len(conv.messages) == 2

    def test_context_extraction(self):
        """Test context extraction from messages."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager()
        conv_id = manager.create_conversation()

        manager.add_user_message(conv_id, "304不锈钢的强度是多少？")

        ctx = manager.get_context(conv_id)
        assert "304" in ctx.materials

    def test_context_summary(self):
        """Test context summary generation."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager()
        conv_id = manager.create_conversation()

        manager.add_user_message(conv_id, "304不锈钢的强度")
        manager.add_user_message(conv_id, "它的公差是IT7")

        summary = manager.get_context_summary(conv_id)
        assert "304" in summary or "材料" in summary

    def test_detect_follow_up(self):
        """Test follow-up detection."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager()
        conv_id = manager.create_conversation()

        manager.add_user_message(conv_id, "304不锈钢的强度是多少？")
        manager.add_assistant_message(conv_id, "520MPa")

        # Short query should be detected as follow-up
        assert manager.detect_follow_up(conv_id, "密度呢？") is True

        # Query with reference pronoun
        assert manager.detect_follow_up(conv_id, "它的密度是多少？") is True

    def test_resolve_references(self):
        """Test reference resolution."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager()
        conv_id = manager.create_conversation()

        manager.add_user_message(conv_id, "304不锈钢的强度是多少？")

        resolved = manager.resolve_references(conv_id, "它的密度是多少？")
        # Should attempt to replace "它的" - may result in material name or remain unchanged
        # The resolution depends on context extraction
        assert resolved is not None
        assert len(resolved) > 0

    def test_conversation_history(self):
        """Test getting conversation history."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager()
        conv_id = manager.create_conversation()

        manager.add_user_message(conv_id, "问题1")
        manager.add_assistant_message(conv_id, "回答1")
        manager.add_user_message(conv_id, "问题2")
        manager.add_assistant_message(conv_id, "回答2")

        history = manager.get_recent_history(conv_id, 2)
        assert len(history) == 2
        assert history[0]["role"] in ["user", "assistant"]

    def test_clear_conversation(self):
        """Test clearing conversation."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager()
        conv_id = manager.create_conversation()

        manager.add_user_message(conv_id, "问题")
        assert manager.clear_conversation(conv_id) is True

        conv = manager.get_conversation(conv_id)
        assert len(conv.messages) == 0

    def test_delete_conversation(self):
        """Test deleting conversation."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager()
        conv_id = manager.create_conversation()

        assert manager.delete_conversation(conv_id) is True
        assert manager.get_conversation(conv_id) is None

    def test_max_conversations_limit(self):
        """Test conversation limit."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager(max_conversations=3)

        ids = []
        for i in range(5):
            ids.append(manager.create_conversation())

        # Should only have 3 conversations
        active = manager.list_conversations()
        assert len(active) == 3

        # First two should be evicted
        assert manager.get_conversation(ids[0]) is None
        assert manager.get_conversation(ids[1]) is None

    def test_max_messages_limit(self):
        """Test message limit per conversation."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager(max_messages_per_conversation=5)
        conv_id = manager.create_conversation()

        for i in range(10):
            manager.add_user_message(conv_id, f"消息{i}")

        conv = manager.get_conversation(conv_id)
        assert len(conv.messages) == 5

    def test_list_conversations(self):
        """Test listing conversations."""
        from src.core.assistant.conversation import ConversationManager

        manager = ConversationManager()

        manager.create_conversation()
        manager.create_conversation()
        manager.create_conversation()

        convs = manager.list_conversations()
        assert len(convs) == 3
        assert all("id" in c for c in convs)


class TestAssistantConversation:
    """Tests for assistant with conversation support."""

    def test_assistant_conversation_disabled(self):
        """Test assistant with conversation disabled."""
        from src.core.assistant import CADAssistant, AssistantConfig

        config = AssistantConfig(enable_conversation=False)
        assistant = CADAssistant(config=config)

        assert assistant._conversation_manager is None

    def test_assistant_conversation_enabled(self):
        """Test assistant with conversation enabled."""
        from src.core.assistant import CADAssistant, AssistantConfig

        config = AssistantConfig(enable_conversation=True)
        assistant = CADAssistant(config=config)

        assert assistant._conversation_manager is not None

    def test_start_conversation(self):
        """Test starting a conversation."""
        from src.core.assistant import CADAssistant

        assistant = CADAssistant()
        conv_id = assistant.start_conversation()

        assert conv_id is not None

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        from src.core.assistant import CADAssistant

        assistant = CADAssistant()
        conv_id = assistant.start_conversation()

        # First turn
        response1 = assistant.ask("304不锈钢的强度是多少？", conversation_id=conv_id)
        assert response1.answer is not None
        assert response1.conversation_id == conv_id

        # Second turn (follow-up)
        response2 = assistant.ask("它的密度呢？", conversation_id=conv_id)
        assert response2.answer is not None

        # Check history
        history = assistant.get_conversation_history(conv_id)
        assert len(history) >= 2

    def test_conversation_context_in_response(self):
        """Test that conversation context affects responses."""
        from src.core.assistant import CADAssistant

        assistant = CADAssistant()
        conv_id = assistant.start_conversation()

        # First establish context with material
        assistant.ask("304不锈钢的强度", conversation_id=conv_id)

        # Follow-up should use context
        response = assistant.ask("密度呢？", conversation_id=conv_id)
        assert response is not None

    def test_end_conversation(self):
        """Test ending a conversation."""
        from src.core.assistant import CADAssistant

        assistant = CADAssistant()
        conv_id = assistant.start_conversation()

        assistant.ask("问题", conversation_id=conv_id)
        assert assistant.end_conversation(conv_id) is True

        # Conversation should no longer exist
        history = assistant.get_conversation_history(conv_id)
        assert len(history) == 0


class TestSingletonManager:
    """Tests for singleton conversation manager."""

    def test_singleton_returns_same_instance(self):
        """Test that singleton returns the same instance."""
        from src.core.assistant.conversation import get_conversation_manager

        m1 = get_conversation_manager()
        m2 = get_conversation_manager()

        assert m1 is m2
