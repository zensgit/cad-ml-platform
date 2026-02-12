"""
Dynamic batching for model serving.

Provides:
- Request batching
- Adaptive batch sizes
- Timeout handling
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from src.ml.serving.request import InferenceRequest, InferenceResponse, Prediction, RequestStatus

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for dynamic batching."""
    max_batch_size: int = 32
    max_wait_time: float = 0.05  # 50ms max wait
    min_batch_size: int = 1
    adaptive: bool = True
    target_latency_ms: float = 100.0


@dataclass
class BatchRequest:
    """A batch of requests."""
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    requests: List[InferenceRequest] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    model_name: str = ""

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def total_inputs(self) -> int:
        return sum(len(r.inputs) for r in self.requests)

    @property
    def age_ms(self) -> float:
        return (time.time() - self.created_at) * 1000

    def add(self, request: InferenceRequest) -> None:
        """Add request to batch."""
        request.batch_id = self.batch_id
        self.requests.append(request)

    def get_all_inputs(self) -> Tuple[List[Any], List[int]]:
        """
        Get all inputs flattened with boundaries.

        Returns:
            (inputs, boundaries) where boundaries[i] is the start index for request i
        """
        inputs = []
        boundaries = []

        for request in self.requests:
            boundaries.append(len(inputs))
            inputs.extend(request.inputs)

        return inputs, boundaries


@dataclass
class BatchResult:
    """Result of batch inference."""
    batch_id: str
    predictions: List[Prediction]
    boundaries: List[int]
    latency_ms: float
    model_name: str

    def get_predictions_for_request(self, request_index: int) -> List[Prediction]:
        """Get predictions for a specific request."""
        if request_index >= len(self.boundaries):
            return []

        start = self.boundaries[request_index]
        end = self.boundaries[request_index + 1] if request_index + 1 < len(self.boundaries) else len(self.predictions)
        return self.predictions[start:end]


class DynamicBatcher:
    """
    Dynamic batching for model inference.

    Collects requests and batches them for efficient processing.
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize dynamic batcher.

        Args:
            config: Batching configuration
        """
        self._config = config or BatchConfig()
        self._pending: Dict[str, List[InferenceRequest]] = defaultdict(list)  # model -> requests
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._batch_handlers: Dict[str, Callable] = {}

        # Adaptive batching state
        self._recent_latencies: Dict[str, List[float]] = defaultdict(list)
        self._current_batch_size: Dict[str, int] = {}

    def register_handler(
        self,
        model_name: str,
        handler: Callable[[BatchRequest], BatchResult],
    ) -> None:
        """
        Register batch handler for a model.

        Args:
            model_name: Model name
            handler: Function that processes BatchRequest and returns BatchResult
        """
        self._batch_handlers[model_name] = handler
        self._current_batch_size[model_name] = self._config.max_batch_size

    def submit(self, request: InferenceRequest) -> None:
        """
        Submit a request for batching.

        Args:
            request: Inference request
        """
        with self._lock:
            self._pending[request.model_name].append(request)
            self._condition.notify_all()

    def get_batch(
        self,
        model_name: str,
        timeout: Optional[float] = None,
    ) -> Optional[BatchRequest]:
        """
        Get a batch of requests for processing.

        Args:
            model_name: Model to get batch for
            timeout: Max time to wait

        Returns:
            BatchRequest or None
        """
        timeout = timeout or self._config.max_wait_time
        batch_size = self._current_batch_size.get(model_name, self._config.max_batch_size)
        start_time = time.time()

        with self._condition:
            while True:
                pending = self._pending.get(model_name, [])

                # Check if we have enough for a batch
                if len(pending) >= batch_size:
                    break

                # Check if we've waited long enough
                elapsed = time.time() - start_time
                if elapsed >= timeout and pending:
                    break

                # Wait for more requests
                remaining = timeout - elapsed
                if remaining <= 0:
                    break

                self._condition.wait(timeout=remaining)

            # Build batch from pending requests
            if not self._pending.get(model_name):
                return None

            requests = self._pending[model_name][:batch_size]
            self._pending[model_name] = self._pending[model_name][batch_size:]

            batch = BatchRequest(model_name=model_name)
            for req in requests:
                batch.add(req)

            return batch

    def process_batch(self, batch: BatchRequest) -> List[InferenceResponse]:
        """
        Process a batch and return responses.

        Args:
            batch: Batch to process

        Returns:
            List of InferenceResponse for each request
        """
        handler = self._batch_handlers.get(batch.model_name)
        if handler is None:
            logger.error(f"No handler for model {batch.model_name}")
            return [
                InferenceResponse.error_response(
                    req.request_id,
                    batch.model_name,
                    "No batch handler registered",
                )
                for req in batch.requests
            ]

        try:
            # Process batch
            start_time = time.time()
            result = handler(batch)
            latency = time.time() - start_time

            # Record latency for adaptive batching
            if self._config.adaptive:
                self._record_latency(batch.model_name, latency * 1000)

            # Build responses
            responses = []
            for i, request in enumerate(batch.requests):
                predictions = result.get_predictions_for_request(i)
                response = InferenceResponse(
                    request_id=request.request_id,
                    predictions=predictions,
                    model_name=batch.model_name,
                    latency_ms=latency * 1000 / batch.size,  # Per-request latency
                )
                responses.append(response)

            return responses

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [
                InferenceResponse.error_response(
                    req.request_id,
                    batch.model_name,
                    str(e),
                )
                for req in batch.requests
            ]

    def _record_latency(self, model_name: str, latency_ms: float) -> None:
        """Record latency and adjust batch size."""
        self._recent_latencies[model_name].append(latency_ms)

        # Keep last 100 latencies
        if len(self._recent_latencies[model_name]) > 100:
            self._recent_latencies[model_name].pop(0)

        # Adjust batch size
        if len(self._recent_latencies[model_name]) >= 10:
            avg_latency = sum(self._recent_latencies[model_name][-10:]) / 10
            current_size = self._current_batch_size.get(model_name, self._config.max_batch_size)

            if avg_latency > self._config.target_latency_ms * 1.2:
                # Latency too high, reduce batch size
                new_size = max(self._config.min_batch_size, current_size - 4)
                if new_size != current_size:
                    self._current_batch_size[model_name] = new_size
                    logger.debug(f"Reduced batch size for {model_name}: {current_size} -> {new_size}")

            elif avg_latency < self._config.target_latency_ms * 0.8:
                # Latency low, can increase batch size
                new_size = min(self._config.max_batch_size, current_size + 4)
                if new_size != current_size:
                    self._current_batch_size[model_name] = new_size
                    logger.debug(f"Increased batch size for {model_name}: {current_size} -> {new_size}")

    def get_pending_count(self, model_name: Optional[str] = None) -> int:
        """Get number of pending requests."""
        with self._lock:
            if model_name:
                return len(self._pending.get(model_name, []))
            return sum(len(reqs) for reqs in self._pending.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        with self._lock:
            stats = {
                "config": {
                    "max_batch_size": self._config.max_batch_size,
                    "max_wait_time_ms": self._config.max_wait_time * 1000,
                    "adaptive": self._config.adaptive,
                    "target_latency_ms": self._config.target_latency_ms,
                },
                "pending": {model: len(reqs) for model, reqs in self._pending.items()},
                "current_batch_sizes": dict(self._current_batch_size),
                "avg_latencies_ms": {
                    model: round(sum(lats) / len(lats), 2) if lats else 0.0
                    for model, lats in self._recent_latencies.items()
                },
            }
        return stats


class AsyncDynamicBatcher:
    """
    Async version of dynamic batcher.
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize async batcher."""
        self._config = config or BatchConfig()
        self._pending: Dict[str, List[Tuple[InferenceRequest, asyncio.Future]]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._batch_handlers: Dict[str, Callable] = {}

    def register_handler(
        self,
        model_name: str,
        handler: Callable[[BatchRequest], BatchResult],
    ) -> None:
        """Register batch handler."""
        self._batch_handlers[model_name] = handler

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """
        Submit request and wait for response.

        Args:
            request: Inference request

        Returns:
            InferenceResponse
        """
        future: asyncio.Future[InferenceResponse] = asyncio.get_event_loop().create_future()

        async with self._lock:
            self._pending[request.model_name].append((request, future))

            # Check if we should process batch
            pending = self._pending[request.model_name]
            if len(pending) >= self._config.max_batch_size:
                await self._process_pending(request.model_name)

        # Wait for result with timeout
        try:
            return await asyncio.wait_for(future, timeout=request.timeout)
        except asyncio.TimeoutError:
            return InferenceResponse.error_response(
                request.request_id,
                request.model_name,
                "Request timeout",
            )

    async def _process_pending(self, model_name: str) -> None:
        """Process pending requests for a model."""
        pending = self._pending[model_name]
        if not pending:
            return

        # Take batch
        batch_items = pending[:self._config.max_batch_size]
        self._pending[model_name] = pending[self._config.max_batch_size:]

        # Build batch
        batch = BatchRequest(model_name=model_name)
        futures = []
        for request, future in batch_items:
            batch.add(request)
            futures.append(future)

        # Process
        handler = self._batch_handlers.get(model_name)
        if handler is None:
            for future in futures:
                if not future.done():
                    future.set_result(InferenceResponse.error_response(
                        "", model_name, "No handler"
                    ))
            return

        try:
            result = handler(batch)

            for i, future in enumerate(futures):
                if not future.done():
                    predictions = result.get_predictions_for_request(i)
                    response = InferenceResponse(
                        request_id=batch.requests[i].request_id,
                        predictions=predictions,
                        model_name=model_name,
                        latency_ms=result.latency_ms / batch.size,
                    )
                    future.set_result(response)

        except Exception as e:
            for i, future in enumerate(futures):
                if not future.done():
                    future.set_result(InferenceResponse.error_response(
                        batch.requests[i].request_id, model_name, str(e)
                    ))
