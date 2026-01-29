"""
Streaming Response Module for CAD Assistant.

Provides Server-Sent Events (SSE) streaming for real-time responses.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional


class StreamEventType(Enum):
    """Types of streaming events."""

    START = "start"
    TOKEN = "token"
    CHUNK = "chunk"
    SOURCE = "source"
    METADATA = "metadata"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamEvent:
    """A single streaming event."""

    event_type: StreamEventType
    data: Any
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        event_data = {
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class StreamingResponse:
    """
    Manages streaming response generation.

    Supports both SSE and WebSocket formats.

    Example:
        >>> async def generate():
        ...     streamer = StreamingResponse()
        ...     async for event in streamer.stream("304不锈钢的强度"):
        ...         yield event.to_sse()
    """

    def __init__(
        self,
        chunk_size: int = 50,
        delay_ms: int = 50,
    ):
        """
        Initialize streaming response.

        Args:
            chunk_size: Characters per chunk
            delay_ms: Delay between chunks (for simulation)
        """
        self.chunk_size = chunk_size
        self.delay_ms = delay_ms
        self._cancelled = False

    async def stream_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream text response in chunks.

        Args:
            text: Full text to stream
            metadata: Optional metadata

        Yields:
            Stream events
        """
        # Start event
        yield StreamEvent(
            event_type=StreamEventType.START,
            data={"message": "Starting response"},
        )

        # Stream text chunks
        for i in range(0, len(text), self.chunk_size):
            if self._cancelled:
                break

            chunk = text[i:i + self.chunk_size]
            yield StreamEvent(
                event_type=StreamEventType.CHUNK,
                data={"text": chunk, "index": i // self.chunk_size},
            )

            if self.delay_ms > 0:
                await asyncio.sleep(self.delay_ms / 1000)

        # Metadata event
        if metadata:
            yield StreamEvent(
                event_type=StreamEventType.METADATA,
                data=metadata,
            )

        # Done event
        yield StreamEvent(
            event_type=StreamEventType.DONE,
            data={"message": "Response complete", "total_length": len(text)},
        )

    async def stream_tokens(
        self,
        tokens: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream response token by token.

        Args:
            tokens: List of tokens
            metadata: Optional metadata

        Yields:
            Stream events
        """
        yield StreamEvent(
            event_type=StreamEventType.START,
            data={"message": "Starting token stream"},
        )

        for i, token in enumerate(tokens):
            if self._cancelled:
                break

            yield StreamEvent(
                event_type=StreamEventType.TOKEN,
                data={"token": token, "index": i},
            )

            if self.delay_ms > 0:
                await asyncio.sleep(self.delay_ms / 1000)

        if metadata:
            yield StreamEvent(
                event_type=StreamEventType.METADATA,
                data=metadata,
            )

        yield StreamEvent(
            event_type=StreamEventType.DONE,
            data={"total_tokens": len(tokens)},
        )

    def cancel(self) -> None:
        """Cancel ongoing stream."""
        self._cancelled = True


class StreamingAssistant:
    """
    Streaming wrapper for CAD Assistant.

    Provides streaming interface for assistant responses.
    """

    def __init__(self, assistant=None):
        """
        Initialize streaming assistant.

        Args:
            assistant: CAD Assistant instance (lazy loaded if None)
        """
        self._assistant = assistant

    def _get_assistant(self):
        """Lazy load assistant."""
        if self._assistant is None:
            from .assistant import CADAssistant
            self._assistant = CADAssistant()
        return self._assistant

    async def stream_ask(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        chunk_size: int = 30,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream assistant response.

        Args:
            query: User query
            conversation_id: Optional conversation ID
            chunk_size: Characters per chunk

        Yields:
            Stream events
        """
        streamer = StreamingResponse(chunk_size=chunk_size, delay_ms=30)

        try:
            # Get full response (in real implementation, integrate with LLM streaming)
            assistant = self._get_assistant()

            if not conversation_id:
                conversation_id = assistant.start_conversation()

            result = assistant.ask(query, conversation_id=conversation_id)

            # Stream the response
            async for event in streamer.stream_text(
                result.answer,
                metadata={
                    "confidence": result.confidence,
                    "conversation_id": conversation_id,
                    "intent": result.intent.value if result.intent else None,
                    "sources": [s.to_dict() if hasattr(s, 'to_dict') else str(s)
                               for s in (result.sources or [])],
                },
            ):
                yield event

        except Exception as e:
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": str(e)},
            )


def create_sse_response(
    generator: AsyncGenerator[StreamEvent, None],
) -> AsyncGenerator[str, None]:
    """
    Convert stream events to SSE format.

    Args:
        generator: Stream event generator

    Yields:
        SSE formatted strings
    """

    async def sse_generator():
        async for event in generator:
            yield event.to_sse()

    return sse_generator()
