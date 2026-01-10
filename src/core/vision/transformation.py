"""Request/response transformation pipeline for Vision Provider system.

This module provides transformation capabilities including:
- Request preprocessing transformations
- Response post-processing transformations
- Chainable transformation pipelines
- Custom transformer registration
- Transformation metrics and logging
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from .base import VisionDescription, VisionProvider


class TransformStage(Enum):
    """Stage in the transformation pipeline."""

    PRE_REQUEST = "pre_request"  # Before sending to provider
    POST_RESPONSE = "post_response"  # After receiving from provider
    ERROR = "error"  # On error transformation


class TransformPriority(Enum):
    """Priority for transformation ordering."""

    FIRST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LAST = 100


@dataclass
class TransformContext:
    """Context passed through transformation pipeline."""

    request_id: str
    stage: TransformStage
    provider_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to context."""
        self.metadata[key] = value

    def add_error(self, error: str) -> None:
        """Add error to context."""
        self.errors.append(error)


@dataclass
class TransformResult:
    """Result of a transformation."""

    success: bool
    data: Any
    transform_name: str
    duration_ms: float
    error: Optional[str] = None


T = TypeVar("T")


class Transformer(ABC, Generic[T]):
    """Abstract base class for transformers."""

    def __init__(
        self,
        name: str,
        stage: TransformStage,
        priority: TransformPriority = TransformPriority.NORMAL,
        enabled: bool = True,
    ) -> None:
        """Initialize transformer.

        Args:
            name: Transformer name
            stage: When to apply transformation
            priority: Execution priority
            enabled: Whether transformer is enabled
        """
        self._name = name
        self._stage = stage
        self._priority = priority
        self._enabled = enabled

    @property
    def name(self) -> str:
        """Return transformer name."""
        return self._name

    @property
    def stage(self) -> TransformStage:
        """Return transformation stage."""
        return self._stage

    @property
    def priority(self) -> TransformPriority:
        """Return transformer priority."""
        return self._priority

    @property
    def enabled(self) -> bool:
        """Return whether transformer is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        self._enabled = value

    @abstractmethod
    def transform(self, data: T, context: TransformContext) -> T:
        """Transform data.

        Args:
            data: Data to transform
            context: Transformation context

        Returns:
            Transformed data
        """
        pass


class RequestTransformer(Transformer[bytes]):
    """Transformer for request data (image bytes)."""

    pass


class ResponseTransformer(Transformer[VisionDescription]):
    """Transformer for response data (VisionDescription)."""

    pass


class LambdaRequestTransformer(RequestTransformer):
    """Request transformer using a lambda function."""

    def __init__(
        self,
        name: str,
        transform_fn: Callable[[bytes, TransformContext], bytes],
        priority: TransformPriority = TransformPriority.NORMAL,
    ) -> None:
        """Initialize lambda transformer."""
        super().__init__(name, TransformStage.PRE_REQUEST, priority)
        self._transform_fn = transform_fn

    def transform(self, data: bytes, context: TransformContext) -> bytes:
        """Apply transformation function."""
        return self._transform_fn(data, context)


class LambdaResponseTransformer(ResponseTransformer):
    """Response transformer using a lambda function."""

    def __init__(
        self,
        name: str,
        transform_fn: Callable[[VisionDescription, TransformContext], VisionDescription],
        priority: TransformPriority = TransformPriority.NORMAL,
    ) -> None:
        """Initialize lambda transformer."""
        super().__init__(name, TransformStage.POST_RESPONSE, priority)
        self._transform_fn = transform_fn

    def transform(self, data: VisionDescription, context: TransformContext) -> VisionDescription:
        """Apply transformation function."""
        return self._transform_fn(data, context)


@dataclass
class PipelineStats:
    """Statistics for transformation pipeline."""

    total_transforms: int = 0
    successful_transforms: int = 0
    failed_transforms: int = 0
    total_duration_ms: float = 0.0
    transforms_by_name: Dict[str, int] = field(default_factory=dict)
    errors_by_transform: Dict[str, int] = field(default_factory=dict)

    def record_transform(self, name: str, success: bool, duration_ms: float) -> None:
        """Record a transformation."""
        self.total_transforms += 1
        self.total_duration_ms += duration_ms

        if success:
            self.successful_transforms += 1
        else:
            self.failed_transforms += 1
            self.errors_by_transform[name] = self.errors_by_transform.get(name, 0) + 1

        self.transforms_by_name[name] = self.transforms_by_name.get(name, 0) + 1


class TransformationPipeline:
    """Pipeline for chaining transformations."""

    def __init__(
        self,
        name: str = "default",
        fail_fast: bool = False,
    ) -> None:
        """Initialize pipeline.

        Args:
            name: Pipeline name
            fail_fast: Stop on first error
        """
        self._name = name
        self._fail_fast = fail_fast
        self._request_transformers: List[RequestTransformer] = []
        self._response_transformers: List[ResponseTransformer] = []
        self._stats = PipelineStats()

    @property
    def name(self) -> str:
        """Return pipeline name."""
        return self._name

    @property
    def stats(self) -> PipelineStats:
        """Return pipeline stats."""
        return self._stats

    def add_request_transformer(self, transformer: RequestTransformer) -> None:
        """Add a request transformer."""
        self._request_transformers.append(transformer)
        self._request_transformers.sort(key=lambda t: t.priority.value)

    def add_response_transformer(self, transformer: ResponseTransformer) -> None:
        """Add a response transformer."""
        self._response_transformers.append(transformer)
        self._response_transformers.sort(key=lambda t: t.priority.value)

    def remove_transformer(self, name: str) -> bool:
        """Remove a transformer by name."""
        for lst in [self._request_transformers, self._response_transformers]:
            for i, t in enumerate(lst):
                if t.name == name:
                    lst.pop(i)
                    return True
        return False

    def get_transformer(
        self, name: str
    ) -> Optional[Union[RequestTransformer, ResponseTransformer]]:
        """Get a transformer by name."""
        for t in self._request_transformers + self._response_transformers:
            if t.name == name:
                return t
        return None

    def transform_request(
        self, data: bytes, context: TransformContext
    ) -> tuple[bytes, List[TransformResult]]:
        """Transform request data.

        Args:
            data: Request data
            context: Transformation context

        Returns:
            Tuple of (transformed data, list of results)
        """
        results: List[TransformResult] = []
        current_data = data

        for transformer in self._request_transformers:
            if not transformer.enabled:
                continue

            start_time = time.time()
            try:
                current_data = transformer.transform(current_data, context)
                duration_ms = (time.time() - start_time) * 1000

                result = TransformResult(
                    success=True,
                    data=current_data,
                    transform_name=transformer.name,
                    duration_ms=duration_ms,
                )
                results.append(result)
                self._stats.record_transform(transformer.name, True, duration_ms)

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                error_msg = str(e)

                result = TransformResult(
                    success=False,
                    data=current_data,
                    transform_name=transformer.name,
                    duration_ms=duration_ms,
                    error=error_msg,
                )
                results.append(result)
                context.add_error(f"{transformer.name}: {error_msg}")
                self._stats.record_transform(transformer.name, False, duration_ms)

                if self._fail_fast:
                    break

        return current_data, results

    def transform_response(
        self, data: VisionDescription, context: TransformContext
    ) -> tuple[VisionDescription, List[TransformResult]]:
        """Transform response data.

        Args:
            data: Response data
            context: Transformation context

        Returns:
            Tuple of (transformed data, list of results)
        """
        results: List[TransformResult] = []
        current_data = data

        for transformer in self._response_transformers:
            if not transformer.enabled:
                continue

            start_time = time.time()
            try:
                current_data = transformer.transform(current_data, context)
                duration_ms = (time.time() - start_time) * 1000

                result = TransformResult(
                    success=True,
                    data=current_data,
                    transform_name=transformer.name,
                    duration_ms=duration_ms,
                )
                results.append(result)
                self._stats.record_transform(transformer.name, True, duration_ms)

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                error_msg = str(e)

                result = TransformResult(
                    success=False,
                    data=current_data,
                    transform_name=transformer.name,
                    duration_ms=duration_ms,
                    error=error_msg,
                )
                results.append(result)
                context.add_error(f"{transformer.name}: {error_msg}")
                self._stats.record_transform(transformer.name, False, duration_ms)

                if self._fail_fast:
                    break

        return current_data, results


class TransformingVisionProvider(VisionProvider):
    """Vision provider with transformation pipeline support."""

    def __init__(
        self,
        provider: VisionProvider,
        pipeline: TransformationPipeline,
    ) -> None:
        """Initialize transforming provider.

        Args:
            provider: Underlying vision provider
            pipeline: Transformation pipeline
        """
        self._provider = provider
        self._pipeline = pipeline
        self._request_count = 0

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"transforming_{self._provider.provider_name}"

    @property
    def pipeline(self) -> TransformationPipeline:
        """Return the pipeline."""
        return self._pipeline

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with transformation pipeline.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        import uuid

        self._request_count += 1
        request_id = str(uuid.uuid4())

        # Pre-request transformation
        pre_context = TransformContext(
            request_id=request_id,
            stage=TransformStage.PRE_REQUEST,
            provider_name=self._provider.provider_name,
        )
        transformed_data, _ = self._pipeline.transform_request(image_data, pre_context)

        # Call provider
        result = await self._provider.analyze_image(transformed_data, include_description)

        # Post-response transformation
        post_context = TransformContext(
            request_id=request_id,
            stage=TransformStage.POST_RESPONSE,
            provider_name=self._provider.provider_name,
            metadata=pre_context.metadata,
        )
        transformed_result, _ = self._pipeline.transform_response(result, post_context)

        return transformed_result


# Built-in transformers


class ConfidenceBoostTransformer(ResponseTransformer):
    """Boost confidence scores by a factor."""

    def __init__(
        self,
        boost_factor: float = 1.1,
        max_confidence: float = 1.0,
    ) -> None:
        """Initialize confidence booster."""
        super().__init__(
            name="confidence_boost",
            stage=TransformStage.POST_RESPONSE,
            priority=TransformPriority.LOW,
        )
        self._boost_factor = boost_factor
        self._max_confidence = max_confidence

    def transform(self, data: VisionDescription, context: TransformContext) -> VisionDescription:
        """Boost confidence."""
        new_confidence = min(
            data.confidence * self._boost_factor,
            self._max_confidence,
        )
        return VisionDescription(
            summary=data.summary,
            details=data.details,
            confidence=new_confidence,
        )


class SummaryPrefixTransformer(ResponseTransformer):
    """Add prefix to summary."""

    def __init__(self, prefix: str = "[Analyzed] ") -> None:
        """Initialize prefix transformer."""
        super().__init__(
            name="summary_prefix",
            stage=TransformStage.POST_RESPONSE,
            priority=TransformPriority.LAST,
        )
        self._prefix = prefix

    def transform(self, data: VisionDescription, context: TransformContext) -> VisionDescription:
        """Add prefix to summary."""
        return VisionDescription(
            summary=self._prefix + data.summary,
            details=data.details,
            confidence=data.confidence,
        )


class DetailFilterTransformer(ResponseTransformer):
    """Filter details based on criteria."""

    def __init__(
        self,
        min_length: int = 0,
        max_length: Optional[int] = None,
        keywords: Optional[List[str]] = None,
    ) -> None:
        """Initialize detail filter."""
        super().__init__(
            name="detail_filter",
            stage=TransformStage.POST_RESPONSE,
            priority=TransformPriority.NORMAL,
        )
        self._min_length = min_length
        self._max_length = max_length
        self._keywords = keywords

    def transform(self, data: VisionDescription, context: TransformContext) -> VisionDescription:
        """Filter details."""
        filtered = []
        for detail in data.details:
            if len(detail) < self._min_length:
                continue
            if self._max_length and len(detail) > self._max_length:
                continue
            if self._keywords:
                if not any(kw.lower() in detail.lower() for kw in self._keywords):
                    continue
            filtered.append(detail)

        return VisionDescription(
            summary=data.summary,
            details=filtered,
            confidence=data.confidence,
        )


def create_transforming_provider(
    provider: VisionProvider,
    pipeline: Optional[TransformationPipeline] = None,
) -> TransformingVisionProvider:
    """Create a transforming vision provider.

    Args:
        provider: Underlying vision provider
        pipeline: Optional transformation pipeline

    Returns:
        TransformingVisionProvider instance
    """
    if pipeline is None:
        pipeline = TransformationPipeline()

    return TransformingVisionProvider(provider=provider, pipeline=pipeline)
