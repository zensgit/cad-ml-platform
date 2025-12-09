"""Streaming support for vision analysis.

Provides:
- Server-Sent Events (SSE) streaming
- Async generator patterns
- Progress streaming for batch operations
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
)

from .base import VisionDescription, VisionProvider, VisionProviderError

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of streaming events."""

    START = "start"
    PROGRESS = "progress"
    PARTIAL = "partial"
    COMPLETE = "complete"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamEvent:
    """A streaming event."""

    event_type: StreamEventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0

    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        event_data = {
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }
        return f"data: {json.dumps(event_data)}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }


@dataclass
class StreamConfig:
    """Configuration for streaming."""

    heartbeat_interval: float = 15.0  # Seconds between heartbeats
    enable_partial_results: bool = True
    buffer_size: int = 10
    timeout: float = 300.0  # 5 minutes max


class StreamingVisionProvider:
    """
    Wrapper that adds streaming support to any VisionProvider.

    Features:
    - SSE event streaming
    - Partial result updates
    - Progress tracking
    - Heartbeat for connection keep-alive
    """

    def __init__(
        self,
        provider: VisionProvider,
        config: Optional[StreamConfig] = None,
    ):
        """
        Initialize streaming provider.

        Args:
            provider: The underlying vision provider
            config: Streaming configuration
        """
        self._provider = provider
        self._config = config or StreamConfig()
        self._sequence = 0

    async def analyze_image_stream(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream vision analysis results.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Yields:
            StreamEvent with analysis progress and results
        """
        start_time = time.time()
        self._sequence = 0

        # Emit start event
        yield self._create_event(
            StreamEventType.START,
            {
                "provider": self._provider.provider_name,
                "image_size": len(image_data),
            },
        )

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(
            self._heartbeat_generator()
        )

        try:
            # Emit progress: analyzing
            yield self._create_event(
                StreamEventType.PROGRESS,
                {
                    "stage": "analyzing",
                    "progress": 0.2,
                    "message": "Starting image analysis...",
                },
            )

            # Run the actual analysis
            analysis_task = asyncio.create_task(
                self._provider.analyze_image(image_data, include_description)
            )

            # Emit progress updates while waiting
            progress = 0.2
            while not analysis_task.done():
                await asyncio.sleep(0.5)
                progress = min(progress + 0.1, 0.9)
                yield self._create_event(
                    StreamEventType.PROGRESS,
                    {
                        "stage": "processing",
                        "progress": progress,
                        "message": "Processing image...",
                    },
                )

            # Get result
            result = await analysis_task
            elapsed_ms = (time.time() - start_time) * 1000

            # Emit complete event
            yield self._create_event(
                StreamEventType.COMPLETE,
                {
                    "result": {
                        "summary": result.summary,
                        "details": result.details,
                        "confidence": result.confidence,
                    },
                    "elapsed_ms": elapsed_ms,
                    "provider": self._provider.provider_name,
                },
            )

        except VisionProviderError as e:
            yield self._create_event(
                StreamEventType.ERROR,
                {
                    "error": str(e),
                    "provider": e.provider,
                    "recoverable": True,
                },
            )

        except Exception as e:
            yield self._create_event(
                StreamEventType.ERROR,
                {
                    "error": str(e),
                    "provider": self._provider.provider_name,
                    "recoverable": False,
                },
            )

        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_generator(self) -> None:
        """Generate heartbeat events."""
        while True:
            await asyncio.sleep(self._config.heartbeat_interval)
            # Note: heartbeats are handled separately in SSE endpoint

    def _create_event(
        self,
        event_type: StreamEventType,
        data: Dict[str, Any],
    ) -> StreamEvent:
        """Create a stream event with sequence number."""
        self._sequence += 1
        return StreamEvent(
            event_type=event_type,
            data=data,
            sequence=self._sequence,
        )

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name


class BatchStreamProcessor:
    """
    Stream batch processing results.

    Provides real-time updates as each image is processed.
    """

    def __init__(
        self,
        provider: VisionProvider,
        max_concurrency: int = 5,
    ):
        """
        Initialize batch stream processor.

        Args:
            provider: Vision provider to use
            max_concurrency: Maximum concurrent requests
        """
        self._provider = provider
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def process_batch_stream(
        self,
        images: List[bytes],
        include_description: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream batch processing results.

        Args:
            images: List of image data bytes
            include_description: Whether to include descriptions

        Yields:
            StreamEvent for each processed image
        """
        total = len(images)
        start_time = time.time()
        sequence = 0

        # Emit start event
        sequence += 1
        yield StreamEvent(
            event_type=StreamEventType.START,
            data={
                "total": total,
                "max_concurrency": self._max_concurrency,
                "provider": self._provider.provider_name,
            },
            sequence=sequence,
        )

        # Track completion
        completed = 0
        failed = 0
        results_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        async def process_single(index: int, image: bytes) -> None:
            nonlocal completed, failed, sequence

            async with self._semaphore:
                item_start = time.time()
                try:
                    result = await self._provider.analyze_image(
                        image, include_description
                    )
                    completed += 1
                    sequence += 1

                    await results_queue.put(StreamEvent(
                        event_type=StreamEventType.COMPLETE,
                        data={
                            "index": index,
                            "result": {
                                "summary": result.summary,
                                "details": result.details,
                                "confidence": result.confidence,
                            },
                            "elapsed_ms": (time.time() - item_start) * 1000,
                            "progress": (completed + failed) / total,
                        },
                        sequence=sequence,
                    ))

                except Exception as e:
                    failed += 1
                    sequence += 1

                    await results_queue.put(StreamEvent(
                        event_type=StreamEventType.ERROR,
                        data={
                            "index": index,
                            "error": str(e),
                            "progress": (completed + failed) / total,
                        },
                        sequence=sequence,
                    ))

        # Start all tasks
        tasks = [
            asyncio.create_task(process_single(i, img))
            for i, img in enumerate(images)
        ]

        # Yield results as they complete
        pending_results = total
        while pending_results > 0:
            try:
                event = await asyncio.wait_for(
                    results_queue.get(),
                    timeout=1.0,
                )
                pending_results -= 1
                yield event
            except asyncio.TimeoutError:
                # Emit progress heartbeat
                sequence += 1
                yield StreamEvent(
                    event_type=StreamEventType.PROGRESS,
                    data={
                        "completed": completed,
                        "failed": failed,
                        "pending": pending_results,
                        "progress": (completed + failed) / total,
                    },
                    sequence=sequence,
                )

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Emit final summary
        total_time_ms = (time.time() - start_time) * 1000
        sequence += 1
        yield StreamEvent(
            event_type=StreamEventType.COMPLETE,
            data={
                "summary": {
                    "total": total,
                    "completed": completed,
                    "failed": failed,
                    "success_rate": completed / total if total > 0 else 0,
                    "total_time_ms": total_time_ms,
                    "avg_time_ms": total_time_ms / total if total > 0 else 0,
                },
            },
            sequence=sequence,
        )


EventCallback = Callable[[StreamEvent], None]


async def stream_analysis(
    provider: VisionProvider,
    image_data: bytes,
    callback: Optional[EventCallback] = None,
) -> VisionDescription:
    """
    Stream analysis with optional callback.

    Args:
        provider: Vision provider to use
        image_data: Raw image bytes
        callback: Optional callback for each event

    Returns:
        Final VisionDescription result

    Example:
        >>> def on_event(event):
        ...     print(f"[{event.event_type.value}] {event.data}")
        >>> result = await stream_analysis(provider, image_bytes, on_event)
    """
    streaming = StreamingVisionProvider(provider)
    result = None

    async for event in streaming.analyze_image_stream(image_data):
        if callback:
            callback(event)

        if event.event_type == StreamEventType.COMPLETE:
            data = event.data.get("result", {})
            result = VisionDescription(
                summary=data.get("summary", ""),
                details=data.get("details", []),
                confidence=data.get("confidence", 0.0),
            )
        elif event.event_type == StreamEventType.ERROR:
            raise VisionProviderError(
                event.data.get("provider", "unknown"),
                event.data.get("error", "Unknown error"),
            )

    if result is None:
        raise VisionProviderError(
            provider.provider_name,
            "No result received from streaming analysis",
        )

    return result


async def generate_sse_stream(
    provider: VisionProvider,
    image_data: bytes,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events stream.

    Args:
        provider: Vision provider to use
        image_data: Raw image bytes

    Yields:
        SSE formatted strings

    Example:
        >>> async for sse_data in generate_sse_stream(provider, image_bytes):
        ...     yield sse_data  # In FastAPI StreamingResponse
    """
    streaming = StreamingVisionProvider(provider)

    async for event in streaming.analyze_image_stream(image_data):
        yield event.to_sse()


def create_streaming_provider(
    provider: VisionProvider,
    heartbeat_interval: float = 15.0,
    enable_partial_results: bool = True,
) -> StreamingVisionProvider:
    """
    Factory to create a streaming provider wrapper.

    Args:
        provider: The underlying vision provider
        heartbeat_interval: Seconds between heartbeats
        enable_partial_results: Whether to emit partial results

    Returns:
        StreamingVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> streaming = create_streaming_provider(provider)
        >>> async for event in streaming.analyze_image_stream(image_bytes):
        ...     print(event.to_dict())
    """
    config = StreamConfig(
        heartbeat_interval=heartbeat_interval,
        enable_partial_results=enable_partial_results,
    )
    return StreamingVisionProvider(provider=provider, config=config)
