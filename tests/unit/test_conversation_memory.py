"""Unit tests for ConversationMemory."""

import pytest

from src.core.assistant.memory import ConversationMemory


class TestConversationMemory:
    """Tests for the sliding-window conversation memory."""

    def test_add_and_get_messages(self):
        """Adding 3 messages returns all 3 via get_messages."""
        mem = ConversationMemory()
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi, how can I help?")
        mem.add_message("user", "What is 304 stainless steel?")

        msgs = mem.get_messages()
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[2]["content"] == "What is 304 stainless steel?"

    def test_trim_at_max_turns(self):
        """History is trimmed when messages exceed max_turns."""
        mem = ConversationMemory(max_turns=20)
        for i in range(25):
            role = "user" if i % 2 == 0 else "assistant"
            mem.add_message(role, f"Message number {i} with some content to process")

        # After trimming, raw history should be < max_turns
        assert len(mem._history) <= mem.max_turns

    def test_summary_created_after_trim(self):
        """A summary exists after trimming takes place."""
        mem = ConversationMemory(max_turns=5)
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            mem.add_message(role, f"Substantial message {i} with enough content")

        assert mem._summary != ""

    def test_summary_prepended(self):
        """get_messages prepends summary pair when summary exists."""
        mem = ConversationMemory(max_turns=5)
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            mem.add_message(role, f"Substantial message {i} with enough content here")

        msgs = mem.get_messages()
        # First two messages should be the summary pair
        assert msgs[0]["role"] == "user"
        assert "之前对话摘要" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"
        assert "了解" in msgs[1]["content"]

    def test_stats(self):
        """get_stats returns expected keys and values."""
        mem = ConversationMemory(max_turns=5)
        mem.add_message("user", "Test question")
        mem.add_message("assistant", "Test answer with some detail")

        stats = mem.get_stats()
        assert "current_turns" in stats
        assert "total_turns" in stats
        assert "has_summary" in stats
        assert stats["current_turns"] == 2
        assert stats["total_turns"] == 2
        assert stats["has_summary"] is False

    def test_clear(self):
        """clear resets all internal state."""
        mem = ConversationMemory(max_turns=3)
        for i in range(6):
            mem.add_message("user", f"Message {i} with enough content to summarise")

        mem.clear()
        assert mem.get_messages() == []
        assert mem._summary == ""
        assert mem._total_turns == 0

    def test_estimate_tokens(self):
        """Token estimate is positive for non-empty history."""
        mem = ConversationMemory()
        mem.add_message("user", "Some Chinese text: 你好世界")

        assert mem._estimate_tokens() > 0
