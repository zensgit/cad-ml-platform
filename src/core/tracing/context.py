"""OpenTelemetry Tracing Context.

Provides trace context management:
- Trace and span IDs
- Context propagation
- W3C Trace Context support
"""

from __future__ import annotations

import os
import random
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TraceFlags(Enum):
    """Trace flags for sampling decisions."""
    NOT_SAMPLED = 0x00
    SAMPLED = 0x01


@dataclass
class TraceId:
    """128-bit trace identifier."""
    high: int  # Upper 64 bits
    low: int   # Lower 64 bits

    @classmethod
    def generate(cls) -> "TraceId":
        """Generate a new random trace ID."""
        return cls(
            high=random.getrandbits(64),
            low=random.getrandbits(64),
        )

    @classmethod
    def from_hex(cls, hex_str: str) -> "TraceId":
        """Parse trace ID from 32-character hex string."""
        if len(hex_str) != 32:
            raise ValueError(f"Invalid trace ID length: {len(hex_str)}")
        return cls(
            high=int(hex_str[:16], 16),
            low=int(hex_str[16:], 16),
        )

    def to_hex(self) -> str:
        """Convert to 32-character hex string."""
        return f"{self.high:016x}{self.low:016x}"

    def is_valid(self) -> bool:
        """Check if trace ID is valid (non-zero)."""
        return self.high != 0 or self.low != 0

    def __str__(self) -> str:
        return self.to_hex()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TraceId):
            return False
        return self.high == other.high and self.low == other.low

    def __hash__(self) -> int:
        return hash((self.high, self.low))


@dataclass
class SpanId:
    """64-bit span identifier."""
    value: int

    @classmethod
    def generate(cls) -> "SpanId":
        """Generate a new random span ID."""
        return cls(value=random.getrandbits(64))

    @classmethod
    def from_hex(cls, hex_str: str) -> "SpanId":
        """Parse span ID from 16-character hex string."""
        if len(hex_str) != 16:
            raise ValueError(f"Invalid span ID length: {len(hex_str)}")
        return cls(value=int(hex_str, 16))

    def to_hex(self) -> str:
        """Convert to 16-character hex string."""
        return f"{self.value:016x}"

    def is_valid(self) -> bool:
        """Check if span ID is valid (non-zero)."""
        return self.value != 0

    def __str__(self) -> str:
        return self.to_hex()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpanId):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass
class SpanContext:
    """Immutable context for a span."""
    trace_id: TraceId
    span_id: SpanId
    trace_flags: TraceFlags = TraceFlags.SAMPLED
    trace_state: Dict[str, str] = field(default_factory=dict)
    is_remote: bool = False

    def is_valid(self) -> bool:
        """Check if context is valid."""
        return self.trace_id.is_valid() and self.span_id.is_valid()

    def is_sampled(self) -> bool:
        """Check if this context is sampled."""
        return self.trace_flags == TraceFlags.SAMPLED

    @classmethod
    def create_root(cls, sampled: bool = True) -> "SpanContext":
        """Create a new root span context."""
        return cls(
            trace_id=TraceId.generate(),
            span_id=SpanId.generate(),
            trace_flags=TraceFlags.SAMPLED if sampled else TraceFlags.NOT_SAMPLED,
        )

    def create_child(self) -> "SpanContext":
        """Create a child span context."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=SpanId.generate(),
            trace_flags=self.trace_flags,
            trace_state=self.trace_state.copy(),
        )


# Context variable for current span context
_current_context: ContextVar[Optional[SpanContext]] = ContextVar(
    'current_span_context',
    default=None,
)


def get_current_context() -> Optional[SpanContext]:
    """Get the current span context."""
    return _current_context.get()


def set_current_context(context: Optional[SpanContext]) -> None:
    """Set the current span context."""
    _current_context.set(context)


class ContextToken:
    """Token for restoring previous context."""

    def __init__(self, previous: Optional[SpanContext]):
        self.previous = previous


def attach_context(context: SpanContext) -> ContextToken:
    """Attach a span context, returning a token to restore previous."""
    previous = get_current_context()
    set_current_context(context)
    return ContextToken(previous)


def detach_context(token: ContextToken) -> None:
    """Restore previous context using token."""
    set_current_context(token.previous)


class W3CTraceContextPropagator:
    """W3C Trace Context propagation format.

    Implements the W3C Trace Context specification:
    https://www.w3.org/TR/trace-context/
    """

    TRACEPARENT_HEADER = "traceparent"
    TRACESTATE_HEADER = "tracestate"
    VERSION = "00"

    def inject(self, context: SpanContext, carrier: Dict[str, str]) -> None:
        """Inject trace context into carrier (headers)."""
        if not context.is_valid():
            return

        # Format: {version}-{trace-id}-{parent-id}-{trace-flags}
        traceparent = (
            f"{self.VERSION}-"
            f"{context.trace_id.to_hex()}-"
            f"{context.span_id.to_hex()}-"
            f"{context.trace_flags.value:02x}"
        )
        carrier[self.TRACEPARENT_HEADER] = traceparent

        # Inject tracestate if present
        if context.trace_state:
            tracestate = ",".join(
                f"{k}={v}" for k, v in context.trace_state.items()
            )
            carrier[self.TRACESTATE_HEADER] = tracestate

    def extract(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extract trace context from carrier (headers)."""
        traceparent = carrier.get(self.TRACEPARENT_HEADER)
        if not traceparent:
            return None

        try:
            parts = traceparent.split("-")
            if len(parts) != 4:
                return None

            version, trace_id_hex, span_id_hex, flags_hex = parts

            # Validate version
            if version != self.VERSION:
                return None

            trace_id = TraceId.from_hex(trace_id_hex)
            span_id = SpanId.from_hex(span_id_hex)
            flags = TraceFlags(int(flags_hex, 16) & 0x01)

            # Parse tracestate
            trace_state: Dict[str, str] = {}
            tracestate = carrier.get(self.TRACESTATE_HEADER)
            if tracestate:
                for pair in tracestate.split(","):
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        trace_state[key.strip()] = value.strip()

            return SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=flags,
                trace_state=trace_state,
                is_remote=True,
            )

        except (ValueError, IndexError):
            return None


class B3Propagator:
    """B3 propagation format (Zipkin-style).

    Supports both single-header and multi-header formats.
    """

    X_B3_TRACEID = "X-B3-TraceId"
    X_B3_SPANID = "X-B3-SpanId"
    X_B3_PARENTSPANID = "X-B3-ParentSpanId"
    X_B3_SAMPLED = "X-B3-Sampled"
    X_B3_FLAGS = "X-B3-Flags"
    B3_SINGLE = "b3"

    def inject(self, context: SpanContext, carrier: Dict[str, str]) -> None:
        """Inject trace context using B3 multi-header format."""
        if not context.is_valid():
            return

        carrier[self.X_B3_TRACEID] = context.trace_id.to_hex()
        carrier[self.X_B3_SPANID] = context.span_id.to_hex()
        carrier[self.X_B3_SAMPLED] = "1" if context.is_sampled() else "0"

    def inject_single(self, context: SpanContext, carrier: Dict[str, str]) -> None:
        """Inject trace context using B3 single-header format."""
        if not context.is_valid():
            return

        # Format: {trace_id}-{span_id}-{sampling_state}
        sampled = "1" if context.is_sampled() else "0"
        carrier[self.B3_SINGLE] = (
            f"{context.trace_id.to_hex()}-"
            f"{context.span_id.to_hex()}-"
            f"{sampled}"
        )

    def extract(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extract trace context from B3 headers."""
        # Try single header first
        b3_single = carrier.get(self.B3_SINGLE)
        if b3_single:
            return self._extract_single(b3_single)

        # Fall back to multi-header
        trace_id_hex = carrier.get(self.X_B3_TRACEID)
        span_id_hex = carrier.get(self.X_B3_SPANID)

        if not trace_id_hex or not span_id_hex:
            return None

        try:
            # Handle 64-bit trace IDs (pad to 128 bits)
            if len(trace_id_hex) == 16:
                trace_id_hex = "0" * 16 + trace_id_hex

            trace_id = TraceId.from_hex(trace_id_hex)
            span_id = SpanId.from_hex(span_id_hex)

            # Determine sampling
            sampled_str = carrier.get(self.X_B3_SAMPLED, "")
            flags_str = carrier.get(self.X_B3_FLAGS, "")

            if flags_str == "1":  # Debug flag means sampled
                sampled = True
            elif sampled_str in ("1", "true", "True"):
                sampled = True
            elif sampled_str in ("0", "false", "False"):
                sampled = False
            else:
                sampled = True  # Default to sampled

            return SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=TraceFlags.SAMPLED if sampled else TraceFlags.NOT_SAMPLED,
                is_remote=True,
            )

        except (ValueError, IndexError):
            return None

    def _extract_single(self, b3_value: str) -> Optional[SpanContext]:
        """Extract from single B3 header."""
        if b3_value == "0":
            # Deny sampling
            return None

        parts = b3_value.split("-")
        if len(parts) < 2:
            return None

        try:
            trace_id_hex = parts[0]
            span_id_hex = parts[1]

            if len(trace_id_hex) == 16:
                trace_id_hex = "0" * 16 + trace_id_hex

            trace_id = TraceId.from_hex(trace_id_hex)
            span_id = SpanId.from_hex(span_id_hex)

            sampled = True
            if len(parts) > 2:
                sampled = parts[2] in ("1", "d")  # 'd' is debug

            return SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=TraceFlags.SAMPLED if sampled else TraceFlags.NOT_SAMPLED,
                is_remote=True,
            )

        except (ValueError, IndexError):
            return None


class CompositePropagator:
    """Composite propagator that supports multiple formats."""

    def __init__(self, propagators: Optional[List] = None):
        self.propagators = propagators or [
            W3CTraceContextPropagator(),
            B3Propagator(),
        ]

    def inject(self, context: SpanContext, carrier: Dict[str, str]) -> None:
        """Inject using all propagators."""
        for propagator in self.propagators:
            propagator.inject(context, carrier)

    def extract(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extract using first successful propagator."""
        for propagator in self.propagators:
            context = propagator.extract(carrier)
            if context is not None:
                return context
        return None


# Global propagator
_propagator = CompositePropagator()


def get_propagator() -> CompositePropagator:
    """Get the global propagator."""
    return _propagator


def set_propagator(propagator: CompositePropagator) -> None:
    """Set the global propagator."""
    global _propagator
    _propagator = propagator
