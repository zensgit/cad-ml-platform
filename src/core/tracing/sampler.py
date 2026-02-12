"""Sampling Strategies.

Provides sampling strategies for controlling trace collection:
- Always on/off sampling
- Probability-based sampling
- Rate limiting sampling
- Parent-based sampling
"""

from __future__ import annotations

import hashlib
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.core.tracing.context import SpanContext, TraceFlags, TraceId


@dataclass
class SamplingResult:
    """Result of a sampling decision."""
    decision: bool
    attributes: Dict[str, Any]
    trace_state: Dict[str, str]

    @classmethod
    def drop(cls) -> "SamplingResult":
        """Create a drop decision."""
        return cls(decision=False, attributes={}, trace_state={})

    @classmethod
    def record(cls, attributes: Optional[Dict[str, Any]] = None) -> "SamplingResult":
        """Create a record and sample decision."""
        return cls(decision=True, attributes=attributes or {}, trace_state={})


class Sampler(ABC):
    """Base class for samplers."""

    @abstractmethod
    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        kind: Any,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[Sequence[Any]] = None,
    ) -> SamplingResult:
        """Decide whether to sample a span."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get a description of this sampler."""
        pass


class AlwaysOnSampler(Sampler):
    """Always sample."""

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        kind: Any,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[Sequence[Any]] = None,
    ) -> SamplingResult:
        return SamplingResult.record()

    @property
    def description(self) -> str:
        return "AlwaysOnSampler"


class AlwaysOffSampler(Sampler):
    """Never sample."""

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        kind: Any,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[Sequence[Any]] = None,
    ) -> SamplingResult:
        return SamplingResult.drop()

    @property
    def description(self) -> str:
        return "AlwaysOffSampler"


class TraceIdRatioSampler(Sampler):
    """Sample based on trace ID ratio.

    Uses the trace ID to make deterministic sampling decisions.
    """

    def __init__(self, ratio: float):
        """Initialize with sampling ratio.

        Args:
            ratio: Probability of sampling (0.0 to 1.0)
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"Ratio must be between 0.0 and 1.0, got {ratio}")

        self.ratio = ratio
        # Calculate threshold for 64-bit comparison
        self._threshold = int(ratio * (2 ** 64 - 1))

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        kind: Any,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[Sequence[Any]] = None,
    ) -> SamplingResult:
        # Use low bits of trace ID for deterministic sampling
        if trace_id.low < self._threshold:
            return SamplingResult.record({"sampling.probability": self.ratio})
        return SamplingResult.drop()

    @property
    def description(self) -> str:
        return f"TraceIdRatioSampler({self.ratio})"


class ParentBasedSampler(Sampler):
    """Sample based on parent span's sampling decision."""

    def __init__(
        self,
        root_sampler: Sampler,
        remote_parent_sampled: Optional[Sampler] = None,
        remote_parent_not_sampled: Optional[Sampler] = None,
        local_parent_sampled: Optional[Sampler] = None,
        local_parent_not_sampled: Optional[Sampler] = None,
    ):
        """Initialize parent-based sampler.

        Args:
            root_sampler: Sampler for root spans
            remote_parent_sampled: Sampler when remote parent is sampled
            remote_parent_not_sampled: Sampler when remote parent is not sampled
            local_parent_sampled: Sampler when local parent is sampled
            local_parent_not_sampled: Sampler when local parent is not sampled
        """
        self.root_sampler = root_sampler
        self.remote_parent_sampled = remote_parent_sampled or AlwaysOnSampler()
        self.remote_parent_not_sampled = remote_parent_not_sampled or AlwaysOffSampler()
        self.local_parent_sampled = local_parent_sampled or AlwaysOnSampler()
        self.local_parent_not_sampled = local_parent_not_sampled or AlwaysOffSampler()

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        kind: Any,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[Sequence[Any]] = None,
    ) -> SamplingResult:
        if parent_context is None or not parent_context.is_valid():
            # Root span
            return self.root_sampler.should_sample(
                parent_context, trace_id, name, kind, attributes, links
            )

        if parent_context.is_remote:
            if parent_context.is_sampled():
                return self.remote_parent_sampled.should_sample(
                    parent_context, trace_id, name, kind, attributes, links
                )
            else:
                return self.remote_parent_not_sampled.should_sample(
                    parent_context, trace_id, name, kind, attributes, links
                )
        else:
            if parent_context.is_sampled():
                return self.local_parent_sampled.should_sample(
                    parent_context, trace_id, name, kind, attributes, links
                )
            else:
                return self.local_parent_not_sampled.should_sample(
                    parent_context, trace_id, name, kind, attributes, links
                )

    @property
    def description(self) -> str:
        return f"ParentBasedSampler(root={self.root_sampler.description})"


class RateLimitingSampler(Sampler):
    """Sample at a maximum rate."""

    def __init__(self, max_traces_per_second: float):
        """Initialize rate limiting sampler.

        Args:
            max_traces_per_second: Maximum traces per second
        """
        self.max_traces_per_second = max_traces_per_second
        self._tokens = max_traces_per_second
        self._last_refill = time.time()
        self._refill_rate = max_traces_per_second

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        kind: Any,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[Sequence[Any]] = None,
    ) -> SamplingResult:
        # Refill tokens
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.max_traces_per_second,
            self._tokens + elapsed * self._refill_rate
        )
        self._last_refill = now

        # Try to consume a token
        if self._tokens >= 1:
            self._tokens -= 1
            return SamplingResult.record({
                "sampling.rate_limit": self.max_traces_per_second
            })
        return SamplingResult.drop()

    @property
    def description(self) -> str:
        return f"RateLimitingSampler({self.max_traces_per_second}/s)"


class RuleBasedSampler(Sampler):
    """Sample based on configurable rules."""

    @dataclass
    class Rule:
        """A sampling rule."""
        name: str
        sampler: Sampler
        span_name_pattern: Optional[str] = None
        attribute_patterns: Optional[Dict[str, str]] = None
        span_kinds: Optional[List[Any]] = None

        def matches(
            self,
            name: str,
            kind: Any,
            attributes: Optional[Dict[str, Any]],
        ) -> bool:
            """Check if this rule matches."""
            # Check span name
            if self.span_name_pattern:
                import re
                if not re.match(self.span_name_pattern, name):
                    return False

            # Check kind
            if self.span_kinds and kind not in self.span_kinds:
                return False

            # Check attributes
            if self.attribute_patterns and attributes:
                import re
                for key, pattern in self.attribute_patterns.items():
                    value = attributes.get(key)
                    if value is None or not re.match(pattern, str(value)):
                        return False

            return True

    def __init__(
        self,
        rules: List[Rule],
        default_sampler: Optional[Sampler] = None,
    ):
        """Initialize rule-based sampler.

        Args:
            rules: List of sampling rules (evaluated in order)
            default_sampler: Fallback sampler if no rules match
        """
        self.rules = rules
        self.default_sampler = default_sampler or AlwaysOnSampler()

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        kind: Any,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[Sequence[Any]] = None,
    ) -> SamplingResult:
        for rule in self.rules:
            if rule.matches(name, kind, attributes):
                result = rule.sampler.should_sample(
                    parent_context, trace_id, name, kind, attributes, links
                )
                # Add rule name to attributes
                result.attributes["sampling.rule"] = rule.name
                return result

        return self.default_sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links
        )

    @property
    def description(self) -> str:
        return f"RuleBasedSampler({len(self.rules)} rules)"


class CompositeSampler(Sampler):
    """Combine multiple samplers."""

    def __init__(self, samplers: List[Sampler], require_all: bool = False):
        """Initialize composite sampler.

        Args:
            samplers: List of samplers
            require_all: If True, all must agree to sample. If False, any can trigger sampling.
        """
        self.samplers = samplers
        self.require_all = require_all

    def should_sample(
        self,
        parent_context: Optional[SpanContext],
        trace_id: TraceId,
        name: str,
        kind: Any,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[Sequence[Any]] = None,
    ) -> SamplingResult:
        results = [
            s.should_sample(parent_context, trace_id, name, kind, attributes, links)
            for s in self.samplers
        ]

        if self.require_all:
            # All must agree to sample
            decision = all(r.decision for r in results)
        else:
            # Any can trigger sampling
            decision = any(r.decision for r in results)

        if decision:
            # Merge attributes from all positive results
            merged_attrs = {}
            for r in results:
                if r.decision:
                    merged_attrs.update(r.attributes)
            return SamplingResult.record(merged_attrs)
        return SamplingResult.drop()

    @property
    def description(self) -> str:
        mode = "AND" if self.require_all else "OR"
        return f"CompositeSampler({mode}, {len(self.samplers)} samplers)"
