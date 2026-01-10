"""Batch image processing for vision analysis.

Provides:
- Concurrent batch processing
- Progress tracking
- Error aggregation
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from .base import VisionDescription, VisionProvider, VisionProviderError

logger = logging.getLogger(__name__)


class BatchItemStatus(Enum):
    """Status of individual batch item."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchItem:
    """Individual item in a batch."""

    image_data: bytes
    index: int
    status: BatchItemStatus = BatchItemStatus.PENDING
    result: Optional[VisionDescription] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class BatchResult:
    """Result of batch processing."""

    total: int
    completed: int
    failed: int
    results: List[Optional[VisionDescription]]
    errors: Dict[int, str]
    total_time_ms: float

    @property
    def success_rate(self) -> float:
        """Calculate batch success rate."""
        return self.completed / self.total if self.total > 0 else 0.0


@dataclass
class BatchProgress:
    """Progress tracking for batch processing."""

    total: int
    completed: int = 0
    failed: int = 0
    current_index: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        return (self.completed + self.failed) / self.total * 100 if self.total > 0 else 0.0

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time."""
        return time.time() - self.start_time

    @property
    def estimated_remaining_seconds(self) -> float:
        """Estimate remaining time."""
        processed = self.completed + self.failed
        if processed == 0:
            return 0.0
        rate = processed / self.elapsed_seconds
        remaining = self.total - processed
        return remaining / rate if rate > 0 else 0.0


ProgressCallback = Callable[[BatchProgress], None]


class BatchProcessor:
    """
    Batch processor for vision analysis.

    Features:
    - Concurrent processing with configurable concurrency
    - Progress tracking with callbacks
    - Error isolation (one failure doesn't stop batch)
    - Result aggregation
    """

    def __init__(
        self,
        provider: VisionProvider,
        max_concurrency: int = 5,
        continue_on_error: bool = True,
    ):
        """
        Initialize batch processor.

        Args:
            provider: Vision provider to use
            max_concurrency: Maximum concurrent requests
            continue_on_error: Continue processing on individual failures
        """
        self._provider = provider
        self._max_concurrency = max_concurrency
        self._continue_on_error = continue_on_error
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def process_batch(
        self,
        images: List[bytes],
        include_description: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> BatchResult:
        """
        Process multiple images concurrently.

        Args:
            images: List of image data bytes
            include_description: Whether to include descriptions
            progress_callback: Optional callback for progress updates

        Returns:
            BatchResult with all results and errors
        """
        start_time = time.time()
        total = len(images)

        if total == 0:
            return BatchResult(
                total=0,
                completed=0,
                failed=0,
                results=[],
                errors={},
                total_time_ms=0.0,
            )

        # Initialize progress
        progress = BatchProgress(total=total)

        # Create batch items
        items = [BatchItem(image_data=img, index=i) for i, img in enumerate(images)]

        # Process concurrently
        tasks = [
            self._process_item(item, include_description, progress, progress_callback)
            for item in items
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        results = [None] * total
        errors = {}
        completed = 0
        failed = 0

        for item in items:
            if item.status == BatchItemStatus.COMPLETED:
                results[item.index] = item.result
                completed += 1
            else:
                errors[item.index] = item.error or "Unknown error"
                failed += 1

        total_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Batch completed: {completed}/{total} successful, "
            f"{failed} failed, {total_time_ms:.0f}ms total"
        )

        return BatchResult(
            total=total,
            completed=completed,
            failed=failed,
            results=results,
            errors=errors,
            total_time_ms=total_time_ms,
        )

    async def _process_item(
        self,
        item: BatchItem,
        include_description: bool,
        progress: BatchProgress,
        progress_callback: Optional[ProgressCallback],
    ) -> None:
        """Process single batch item with semaphore."""
        async with self._semaphore:
            item.status = BatchItemStatus.PROCESSING
            progress.current_index = item.index
            start_time = time.time()

            try:
                result = await self._provider.analyze_image(
                    item.image_data,
                    include_description,
                )
                item.result = result
                item.status = BatchItemStatus.COMPLETED
                progress.completed += 1

            except VisionProviderError as e:
                item.error = str(e)
                item.status = BatchItemStatus.FAILED
                progress.failed += 1
                logger.warning(f"Batch item {item.index} failed: {e}")

                if not self._continue_on_error:
                    raise

            except Exception as e:
                item.error = str(e)
                item.status = BatchItemStatus.FAILED
                progress.failed += 1
                logger.error(f"Batch item {item.index} unexpected error: {e}")

                if not self._continue_on_error:
                    raise

            finally:
                item.processing_time_ms = (time.time() - start_time) * 1000

                # Call progress callback
                if progress_callback:
                    try:
                        progress_callback(progress)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")


async def process_images_batch(
    provider: VisionProvider,
    images: List[bytes],
    max_concurrency: int = 5,
    include_description: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> BatchResult:
    """
    Convenience function for batch processing.

    Args:
        provider: Vision provider to use
        images: List of image data bytes
        max_concurrency: Maximum concurrent requests
        include_description: Whether to include descriptions
        progress_callback: Optional callback for progress updates

    Returns:
        BatchResult with all results and errors

    Example:
        >>> provider = create_vision_provider("openai")
        >>> images = [image1_bytes, image2_bytes, image3_bytes]
        >>> result = await process_images_batch(provider, images)
        >>> print(f"Success rate: {result.success_rate:.0%}")
    """
    processor = BatchProcessor(
        provider=provider,
        max_concurrency=max_concurrency,
        continue_on_error=True,
    )
    return await processor.process_batch(
        images=images,
        include_description=include_description,
        progress_callback=progress_callback,
    )
