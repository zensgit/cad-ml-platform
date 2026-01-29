"""
Unit tests for streaming.py - SSE Streaming Response Module.

Tests cover:
- StreamEvent creation and serialization
- StreamingResponse text chunking
- StreamingResponse token streaming
- Stream cancellation
- SSE format compliance
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.core.assistant.streaming import (
    StreamEventType,
    StreamEvent,
    StreamingResponse,
    StreamingAssistant,
    create_sse_response,
)


class TestStreamEvent:
    """Tests for StreamEvent class."""

    def test_stream_event_creation(self):
        """Test basic event creation."""
        event = StreamEvent(
            event_type=StreamEventType.TOKEN,
            data={"token": "hello"},
        )
        assert event.event_type == StreamEventType.TOKEN
        assert event.data == {"token": "hello"}
        assert event.timestamp > 0

    def test_stream_event_to_dict(self):
        """Test conversion to dictionary."""
        event = StreamEvent(
            event_type=StreamEventType.CHUNK,
            data={"text": "test chunk", "index": 0},
        )
        result = event.to_dict()

        assert result["type"] == "chunk"
        assert result["data"]["text"] == "test chunk"
        assert result["data"]["index"] == 0
        assert "timestamp" in result

    def test_stream_event_to_sse(self):
        """Test SSE format output."""
        event = StreamEvent(
            event_type=StreamEventType.START,
            data={"message": "Starting"},
        )
        sse_output = event.to_sse()

        assert sse_output.startswith("data: ")
        assert sse_output.endswith("\n\n")

        # Parse JSON from SSE
        json_str = sse_output[6:-2]  # Remove "data: " prefix and "\n\n" suffix
        parsed = json.loads(json_str)
        assert parsed["type"] == "start"
        assert parsed["data"]["message"] == "Starting"

    def test_all_event_types(self):
        """Test all event types serialize correctly."""
        event_types = [
            StreamEventType.START,
            StreamEventType.TOKEN,
            StreamEventType.CHUNK,
            StreamEventType.SOURCE,
            StreamEventType.METADATA,
            StreamEventType.ERROR,
            StreamEventType.DONE,
        ]

        for event_type in event_types:
            event = StreamEvent(event_type=event_type, data={})
            sse = event.to_sse()
            assert f'"type": "{event_type.value}"' in sse


class TestStreamingResponse:
    """Tests for StreamingResponse class."""

    @pytest.fixture
    def streamer(self):
        """Create a streaming response instance."""
        return StreamingResponse(chunk_size=10, delay_ms=0)

    @pytest.mark.asyncio
    async def test_stream_text_basic(self, streamer):
        """Test basic text streaming."""
        text = "Hello, World!"
        events = []

        async for event in streamer.stream_text(text):
            events.append(event)

        # Should have: START, CHUNK(s), DONE
        assert events[0].event_type == StreamEventType.START
        assert events[-1].event_type == StreamEventType.DONE

        # Check chunks
        chunks = [e for e in events if e.event_type == StreamEventType.CHUNK]
        assert len(chunks) >= 1

        # Reconstruct text from chunks
        reconstructed = "".join(c.data["text"] for c in chunks)
        assert reconstructed == text

    @pytest.mark.asyncio
    async def test_stream_text_with_metadata(self, streamer):
        """Test streaming with metadata."""
        text = "Test"
        metadata = {"confidence": 0.95, "sources": ["doc1"]}

        events = []
        async for event in streamer.stream_text(text, metadata=metadata):
            events.append(event)

        # Should have metadata event
        metadata_events = [e for e in events if e.event_type == StreamEventType.METADATA]
        assert len(metadata_events) == 1
        assert metadata_events[0].data["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_stream_text_chunking(self):
        """Test text is properly chunked."""
        streamer = StreamingResponse(chunk_size=5, delay_ms=0)
        text = "ABCDEFGHIJ"  # 10 characters

        events = []
        async for event in streamer.stream_text(text):
            events.append(event)

        chunks = [e for e in events if e.event_type == StreamEventType.CHUNK]
        assert len(chunks) == 2  # 10 chars / 5 chunk_size = 2 chunks
        assert chunks[0].data["text"] == "ABCDE"
        assert chunks[1].data["text"] == "FGHIJ"

    @pytest.mark.asyncio
    async def test_stream_text_empty(self, streamer):
        """Test streaming empty text."""
        events = []
        async for event in streamer.stream_text(""):
            events.append(event)

        assert events[0].event_type == StreamEventType.START
        assert events[-1].event_type == StreamEventType.DONE
        assert events[-1].data["total_length"] == 0

    @pytest.mark.asyncio
    async def test_stream_tokens(self, streamer):
        """Test token streaming."""
        tokens = ["Hello", " ", "World", "!"]

        events = []
        async for event in streamer.stream_tokens(tokens):
            events.append(event)

        # Check structure
        assert events[0].event_type == StreamEventType.START
        assert events[-1].event_type == StreamEventType.DONE

        # Check tokens
        token_events = [e for e in events if e.event_type == StreamEventType.TOKEN]
        assert len(token_events) == 4
        assert token_events[0].data["token"] == "Hello"
        assert token_events[0].data["index"] == 0

    @pytest.mark.asyncio
    async def test_stream_cancel(self, streamer):
        """Test stream cancellation."""
        text = "A" * 100  # Long text

        events = []
        async for event in streamer.stream_text(text):
            events.append(event)
            if len(events) >= 3:
                streamer.cancel()

        # Should be cancelled before all chunks sent
        chunks = [e for e in events if e.event_type == StreamEventType.CHUNK]
        assert len(chunks) < 10  # Would be 10 without cancellation

    @pytest.mark.asyncio
    async def test_stream_with_delay(self):
        """Test streaming with delay."""
        streamer = StreamingResponse(chunk_size=5, delay_ms=10)
        text = "ABCDEFGHIJ"

        import time
        start = time.time()

        events = []
        async for event in streamer.stream_text(text):
            events.append(event)

        elapsed = time.time() - start
        # Should take at least 2 * 10ms for 2 chunks
        assert elapsed >= 0.015  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_done_event_contains_total_length(self, streamer):
        """Test DONE event contains total length."""
        text = "Hello World"
        events = []

        async for event in streamer.stream_text(text):
            events.append(event)

        done_event = events[-1]
        assert done_event.event_type == StreamEventType.DONE
        assert done_event.data["total_length"] == len(text)

    @pytest.mark.asyncio
    async def test_chunk_index_tracking(self, streamer):
        """Test chunk index is tracked correctly."""
        streamer = StreamingResponse(chunk_size=3, delay_ms=0)
        text = "ABCDEFGHI"  # 9 chars = 3 chunks

        events = []
        async for event in streamer.stream_text(text):
            events.append(event)

        chunks = [e for e in events if e.event_type == StreamEventType.CHUNK]
        assert chunks[0].data["index"] == 0
        assert chunks[1].data["index"] == 1
        assert chunks[2].data["index"] == 2


class TestStreamingAssistant:
    """Tests for StreamingAssistant class."""

    @pytest.fixture
    def mock_assistant(self):
        """Create mock CAD assistant."""
        assistant = MagicMock()
        assistant.start_conversation.return_value = "conv-123"

        result = MagicMock()
        result.answer = "This is the answer"
        result.confidence = 0.85
        result.intent = MagicMock(value="material_query")
        result.sources = []
        assistant.ask.return_value = result

        return assistant

    @pytest.mark.asyncio
    async def test_stream_ask_basic(self, mock_assistant):
        """Test basic streaming ask."""
        streaming_assistant = StreamingAssistant(mock_assistant)

        events = []
        async for event in streaming_assistant.stream_ask("What is steel?"):
            events.append(event)

        # Verify structure
        assert events[0].event_type == StreamEventType.START
        assert events[-1].event_type == StreamEventType.DONE

        # Verify assistant was called
        mock_assistant.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_ask_with_conversation_id(self, mock_assistant):
        """Test streaming with existing conversation."""
        streaming_assistant = StreamingAssistant(mock_assistant)

        events = []
        async for event in streaming_assistant.stream_ask(
            "Follow up question",
            conversation_id="existing-conv"
        ):
            events.append(event)

        # Should use existing conversation
        mock_assistant.ask.assert_called_with(
            "Follow up question",
            conversation_id="existing-conv"
        )

    @pytest.mark.asyncio
    async def test_stream_ask_error_handling(self, mock_assistant):
        """Test error handling in streaming."""
        mock_assistant.ask.side_effect = Exception("API Error")
        streaming_assistant = StreamingAssistant(mock_assistant)

        events = []
        async for event in streaming_assistant.stream_ask("Test"):
            events.append(event)

        # Should have error event
        error_events = [e for e in events if e.event_type == StreamEventType.ERROR]
        assert len(error_events) == 1
        assert "API Error" in error_events[0].data["error"]

    @pytest.mark.asyncio
    async def test_stream_ask_metadata(self, mock_assistant):
        """Test metadata is included in stream."""
        streaming_assistant = StreamingAssistant(mock_assistant)

        events = []
        async for event in streaming_assistant.stream_ask("Query"):
            events.append(event)

        metadata_events = [e for e in events if e.event_type == StreamEventType.METADATA]
        assert len(metadata_events) == 1

        metadata = metadata_events[0].data
        assert "confidence" in metadata
        assert "conversation_id" in metadata
        assert metadata["confidence"] == 0.85


class TestCreateSSEResponse:
    """Tests for create_sse_response helper."""

    @pytest.mark.asyncio
    async def test_sse_response_generator(self):
        """Test SSE response generator."""
        async def mock_events():
            yield StreamEvent(StreamEventType.START, {"msg": "start"})
            yield StreamEvent(StreamEventType.CHUNK, {"text": "hello"})
            yield StreamEvent(StreamEventType.DONE, {"msg": "done"})

        sse_gen = create_sse_response(mock_events())

        responses = []
        async for sse in sse_gen:
            responses.append(sse)

        assert len(responses) == 3
        assert all(r.startswith("data: ") for r in responses)
        assert all(r.endswith("\n\n") for r in responses)

    @pytest.mark.asyncio
    async def test_sse_response_valid_json(self):
        """Test SSE responses contain valid JSON."""
        async def mock_events():
            yield StreamEvent(StreamEventType.TOKEN, {"token": "test"})

        sse_gen = create_sse_response(mock_events())

        async for sse in sse_gen:
            json_str = sse[6:-2]  # Remove "data: " and "\n\n"
            parsed = json.loads(json_str)  # Should not raise
            assert parsed["type"] == "token"


class TestStreamEventTypes:
    """Tests for stream event type coverage."""

    def test_start_event(self):
        """Test START event structure."""
        event = StreamEvent(StreamEventType.START, {"message": "Starting"})
        assert event.event_type.value == "start"

    def test_token_event(self):
        """Test TOKEN event structure."""
        event = StreamEvent(StreamEventType.TOKEN, {"token": "word", "index": 5})
        assert event.event_type.value == "token"
        assert event.data["index"] == 5

    def test_chunk_event(self):
        """Test CHUNK event structure."""
        event = StreamEvent(StreamEventType.CHUNK, {"text": "chunk", "index": 0})
        assert event.event_type.value == "chunk"

    def test_source_event(self):
        """Test SOURCE event structure."""
        event = StreamEvent(StreamEventType.SOURCE, {"source": "doc.pdf", "relevance": 0.9})
        assert event.event_type.value == "source"

    def test_metadata_event(self):
        """Test METADATA event structure."""
        event = StreamEvent(StreamEventType.METADATA, {"confidence": 0.85})
        assert event.event_type.value == "metadata"

    def test_error_event(self):
        """Test ERROR event structure."""
        event = StreamEvent(StreamEventType.ERROR, {"error": "Something failed"})
        assert event.event_type.value == "error"

    def test_done_event(self):
        """Test DONE event structure."""
        event = StreamEvent(StreamEventType.DONE, {"total_length": 100})
        assert event.event_type.value == "done"
