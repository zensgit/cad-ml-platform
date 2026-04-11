"""
Conversation Memory Management.

Provides a sliding-window conversation history with rule-based
summarisation so that long-running sessions stay within LLM token
budgets without losing important context.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation history with sliding window and summarization.

    When the number of stored turns exceeds *max_turns*, the oldest
    messages are compressed into a textual summary that is prepended to
    the messages returned by :meth:`get_messages`.

    Parameters
    ----------
    max_turns : int
        Maximum number of raw messages to retain.
    max_tokens_estimate : int
        Soft token budget used by :meth:`_estimate_tokens`.
    summarize_after : int
        (Reserved) Number of turns before proactive summarisation.
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_tokens_estimate: int = 8000,
        summarize_after: int = 10,
    ):
        self.max_turns = max_turns
        self.max_tokens_estimate = max_tokens_estimate
        self.summarize_after = summarize_after
        self._history: List[Dict[str, str]] = []
        self._summary: str = ""
        self._total_turns: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Add a message to history."""
        self._history.append({"role": role, "content": content})
        self._total_turns += 1
        self._trim_if_needed()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages for LLM, prepending summary if available."""
        messages: List[Dict[str, str]] = []
        if self._summary:
            messages.append({
                "role": "user",
                "content": f"[之前对话摘要] {self._summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "好的，我了解之前的对话内容。请继续。",
            })
        messages.extend(self._history)
        return messages

    def get_stats(self) -> Dict[str, object]:
        """Return diagnostic statistics about the memory buffer."""
        return {
            "current_turns": len(self._history),
            "total_turns": self._total_turns,
            "has_summary": bool(self._summary),
            "estimated_tokens": self._estimate_tokens(),
        }

    def clear(self) -> None:
        """Reset all state."""
        self._history.clear()
        self._summary = ""
        self._total_turns = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_if_needed(self) -> None:
        """Trim history if too long, creating summary of removed messages."""
        if len(self._history) <= self.max_turns:
            return

        # Number of messages to remove (keep max_turns - 2 to make room
        # for the summary pair injected by get_messages).
        keep = self.max_turns - 2
        if keep < 1:
            keep = 1
        to_summarize = self._history[: len(self._history) - keep]
        self._summary = self._create_summary(to_summarize)
        self._history = self._history[len(self._history) - keep:]
        logger.debug(
            "Trimmed conversation: kept %d turns, summarised %d",
            len(self._history),
            len(to_summarize),
        )

    def _create_summary(self, messages: List[Dict[str, str]]) -> str:
        """Create a concise rule-based summary of *messages*.

        Extracts the main user questions and the first line of each
        assistant response.  No LLM call is required.
        """
        topics: List[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if msg["role"] == "user" and len(content) > 10:
                topics.append(f"用户问: {content[:100]}")
            elif msg["role"] == "assistant" and len(content) > 20:
                first_line = content.split("\n")[0][:100]
                topics.append(f"回答: {first_line}")

        combined = "\n".join(topics[-5:])
        if self._summary:
            return f"{self._summary}\n{combined}"
        return combined

    def _estimate_tokens(self) -> int:
        """Rough token estimate (Chinese ~ 1.5 tokens per char)."""
        total_chars = sum(len(m.get("content", "")) for m in self._history)
        return int(total_chars * 1.5)
