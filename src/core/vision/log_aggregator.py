"""Log Aggregator Module for Vision System.

This module provides log aggregation capabilities including:
- Log collection from multiple sources
- Log parsing and structured extraction
- Log correlation and tracing
- Log filtering and search
- Log retention and archival
- Log analytics and patterns

Phase 17: Advanced Observability & Monitoring
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class LogLevel(str, Enum):
    """Log severity levels."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"


class LogSource(str, Enum):
    """Log source types."""

    APPLICATION = "application"
    SYSTEM = "system"
    SECURITY = "security"
    AUDIT = "audit"
    ACCESS = "access"
    NETWORK = "network"
    DATABASE = "database"
    CONTAINER = "container"


class ParserType(str, Enum):
    """Log parser types."""

    JSON = "json"
    REGEX = "regex"
    GROK = "grok"
    CSV = "csv"
    SYSLOG = "syslog"
    APACHE = "apache"
    NGINX = "nginx"
    CUSTOM = "custom"


class AggregationMode(str, Enum):
    """Log aggregation modes."""

    COUNT = "count"
    RATE = "rate"
    UNIQUE = "unique"
    TOP_VALUES = "top_values"
    HISTOGRAM = "histogram"


# ========================
# Data Classes
# ========================


@dataclass
class LogEntry:
    """A single log entry."""

    log_id: str
    timestamp: datetime
    level: LogLevel
    message: str
    source: LogSource = LogSource.APPLICATION
    service: str = ""
    host: str = ""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class LogPattern:
    """A detected log pattern."""

    pattern_id: str
    pattern: str
    sample_message: str
    count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    level: Optional[LogLevel] = None
    services: Set[str] = field(default_factory=set)


@dataclass
class ParserConfig:
    """Configuration for a log parser."""

    parser_id: str
    parser_type: ParserType
    name: str
    pattern: str = ""
    field_mappings: Dict[str, str] = field(default_factory=dict)
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    level_mapping: Dict[str, LogLevel] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class LogFilter:
    """Filter for log queries."""

    levels: List[LogLevel] = field(default_factory=list)
    sources: List[LogSource] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    hosts: List[str] = field(default_factory=list)
    trace_id: Optional[str] = None
    search_query: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    field_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogAggregation:
    """Log aggregation result."""

    mode: AggregationMode
    field: str
    value: Any
    count: int = 0
    buckets: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CorrelatedLogs:
    """Correlated log entries."""

    correlation_id: str
    trace_id: str
    entries: List[LogEntry] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    services: Set[str] = field(default_factory=set)
    error_count: int = 0


@dataclass
class LogStats:
    """Log statistics."""

    total_count: int = 0
    by_level: Dict[LogLevel, int] = field(default_factory=dict)
    by_source: Dict[LogSource, int] = field(default_factory=dict)
    by_service: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    logs_per_second: float = 0.0


@dataclass
class RetentionPolicy:
    """Log retention policy."""

    policy_id: str
    name: str
    max_age: timedelta
    max_size_mb: int = 0
    archive_enabled: bool = False
    archive_location: str = ""
    filters: LogFilter = field(default_factory=LogFilter)


# ========================
# Log Parsers
# ========================


class LogParser(ABC):
    """Abstract base class for log parsers."""

    @abstractmethod
    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """Parse a raw log line."""
        pass


class JSONLogParser(LogParser):
    """JSON log parser."""

    def __init__(self, config: ParserConfig):
        self._config = config

    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """Parse JSON log."""
        try:
            data = json.loads(raw_log)

            # Extract timestamp
            timestamp_field = self._config.field_mappings.get("timestamp", "timestamp")
            timestamp_str = data.get(timestamp_field, "")
            try:
                timestamp = datetime.strptime(timestamp_str, self._config.timestamp_format)
            except (ValueError, TypeError):
                timestamp = datetime.now()

            # Extract level
            level_field = self._config.field_mappings.get("level", "level")
            level_str = str(data.get(level_field, "info")).lower()
            level = self._config.level_mapping.get(level_str, LogLevel.INFO)

            # Extract message
            message_field = self._config.field_mappings.get("message", "message")
            message = str(data.get(message_field, ""))

            return LogEntry(
                log_id=f"log_{int(time.time() * 1000000)}",
                timestamp=timestamp,
                level=level,
                message=message,
                service=data.get("service", ""),
                host=data.get("host", ""),
                trace_id=data.get("trace_id"),
                span_id=data.get("span_id"),
                fields={
                    k: v
                    for k, v in data.items()
                    if k not in [timestamp_field, level_field, message_field]
                },
                raw=raw_log,
            )

        except json.JSONDecodeError:
            return None


class RegexLogParser(LogParser):
    """Regex-based log parser."""

    def __init__(self, config: ParserConfig):
        self._config = config
        self._pattern = re.compile(config.pattern) if config.pattern else None

    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """Parse log using regex."""
        if not self._pattern:
            return None

        match = self._pattern.match(raw_log)
        if not match:
            return None

        groups = match.groupdict()

        # Extract timestamp
        timestamp_str = groups.get("timestamp", "")
        try:
            timestamp = datetime.strptime(timestamp_str, self._config.timestamp_format)
        except (ValueError, TypeError):
            timestamp = datetime.now()

        # Extract level
        level_str = groups.get("level", "info").lower()
        level = self._config.level_mapping.get(level_str, LogLevel.INFO)

        return LogEntry(
            log_id=f"log_{int(time.time() * 1000000)}",
            timestamp=timestamp,
            level=level,
            message=groups.get("message", raw_log),
            service=groups.get("service", ""),
            host=groups.get("host", ""),
            fields={k: v for k, v in groups.items() if k not in ["timestamp", "level", "message"]},
            raw=raw_log,
        )


class SyslogParser(LogParser):
    """Syslog format parser."""

    SYSLOG_PATTERN = re.compile(
        r"<(?P<priority>\d+)>(?P<timestamp>\w{3}\s+\d+\s+\d+:\d+:\d+)\s+"
        r"(?P<host>\S+)\s+(?P<tag>\S+):\s+(?P<message>.*)"
    )

    def __init__(self, config: ParserConfig):
        self._config = config

    def parse(self, raw_log: str) -> Optional[LogEntry]:
        """Parse syslog format."""
        match = self.SYSLOG_PATTERN.match(raw_log)
        if not match:
            return None

        groups = match.groupdict()
        priority = int(groups.get("priority", 14))
        severity = priority % 8

        # Map syslog severity to LogLevel
        level_map = {
            0: LogLevel.FATAL,
            1: LogLevel.FATAL,
            2: LogLevel.FATAL,
            3: LogLevel.ERROR,
            4: LogLevel.WARN,
            5: LogLevel.INFO,
            6: LogLevel.INFO,
            7: LogLevel.DEBUG,
        }
        level = level_map.get(severity, LogLevel.INFO)

        return LogEntry(
            log_id=f"log_{int(time.time() * 1000000)}",
            timestamp=datetime.now(),  # Simplified
            level=level,
            message=groups.get("message", ""),
            source=LogSource.SYSTEM,
            host=groups.get("host", ""),
            fields={"tag": groups.get("tag", "")},
            raw=raw_log,
        )


# ========================
# Log Store
# ========================


class LogStore:
    """In-memory log storage with indexing."""

    def __init__(self, max_entries: int = 100000):
        self._entries: deque = deque(maxlen=max_entries)
        self._by_trace: Dict[str, List[LogEntry]] = defaultdict(list)
        self._by_service: Dict[str, List[LogEntry]] = defaultdict(list)
        self._by_level: Dict[LogLevel, List[LogEntry]] = defaultdict(list)
        self._lock = threading.Lock()

    def add(self, entry: LogEntry) -> None:
        """Add a log entry."""
        with self._lock:
            self._entries.append(entry)

            if entry.trace_id:
                self._by_trace[entry.trace_id].append(entry)

            if entry.service:
                self._by_service[entry.service].append(entry)

            self._by_level[entry.level].append(entry)

    def query(self, filter: LogFilter, limit: int = 1000) -> List[LogEntry]:
        """Query logs with filter."""
        with self._lock:
            results = list(self._entries)

        # Apply filters
        if filter.trace_id:
            results = [e for e in results if e.trace_id == filter.trace_id]

        if filter.levels:
            results = [e for e in results if e.level in filter.levels]

        if filter.sources:
            results = [e for e in results if e.source in filter.sources]

        if filter.services:
            results = [e for e in results if e.service in filter.services]

        if filter.hosts:
            results = [e for e in results if e.host in filter.hosts]

        if filter.start_time:
            results = [e for e in results if e.timestamp >= filter.start_time]

        if filter.end_time:
            results = [e for e in results if e.timestamp <= filter.end_time]

        if filter.search_query:
            query_lower = filter.search_query.lower()
            results = [e for e in results if query_lower in e.message.lower()]

        if filter.tags:
            results = [e for e in results if any(t in e.tags for t in filter.tags)]

        # Sort by timestamp descending
        results.sort(key=lambda x: x.timestamp, reverse=True)

        return results[:limit]

    def get_by_trace(self, trace_id: str) -> List[LogEntry]:
        """Get logs by trace ID."""
        with self._lock:
            entries = self._by_trace.get(trace_id, [])
            return sorted(entries, key=lambda x: x.timestamp)

    def get_stats(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> LogStats:
        """Get log statistics."""
        with self._lock:
            entries = list(self._entries)

        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        stats = LogStats(total_count=len(entries))

        for entry in entries:
            stats.by_level[entry.level] = stats.by_level.get(entry.level, 0) + 1
            stats.by_source[entry.source] = stats.by_source.get(entry.source, 0) + 1
            if entry.service:
                stats.by_service[entry.service] = stats.by_service.get(entry.service, 0) + 1

        error_count = stats.by_level.get(LogLevel.ERROR, 0) + stats.by_level.get(LogLevel.FATAL, 0)
        stats.error_rate = error_count / len(entries) if entries else 0.0

        if entries and len(entries) >= 2:
            time_span = (entries[-1].timestamp - entries[0].timestamp).total_seconds()
            if time_span > 0:
                stats.logs_per_second = len(entries) / time_span

        return stats

    def cleanup(self, max_age: timedelta) -> int:
        """Remove logs older than max_age."""
        cutoff = datetime.now() - max_age
        removed = 0

        with self._lock:
            while self._entries and self._entries[0].timestamp < cutoff:
                entry = self._entries.popleft()
                removed += 1

                # Clean up indices
                if entry.trace_id and entry in self._by_trace.get(entry.trace_id, []):
                    self._by_trace[entry.trace_id].remove(entry)
                if entry.service and entry in self._by_service.get(entry.service, []):
                    self._by_service[entry.service].remove(entry)
                if entry in self._by_level.get(entry.level, []):
                    self._by_level[entry.level].remove(entry)

        return removed


# ========================
# Log Correlator
# ========================


class LogCorrelator:
    """Correlates logs across services using trace IDs."""

    def __init__(self, store: LogStore):
        self._store = store

    def correlate_by_trace(self, trace_id: str) -> CorrelatedLogs:
        """Correlate logs by trace ID."""
        entries = self._store.get_by_trace(trace_id)

        services = set()
        error_count = 0
        start_time = None
        end_time = None

        for entry in entries:
            if entry.service:
                services.add(entry.service)
            if entry.level in (LogLevel.ERROR, LogLevel.FATAL):
                error_count += 1
            if start_time is None or entry.timestamp < start_time:
                start_time = entry.timestamp
            if end_time is None or entry.timestamp > end_time:
                end_time = entry.timestamp

        return CorrelatedLogs(
            correlation_id=f"corr_{trace_id}",
            trace_id=trace_id,
            entries=entries,
            start_time=start_time,
            end_time=end_time,
            services=services,
            error_count=error_count,
        )

    def find_related_errors(
        self, error_entry: LogEntry, time_window: timedelta = timedelta(minutes=5)
    ) -> List[LogEntry]:
        """Find related errors within a time window."""
        filter = LogFilter(
            levels=[LogLevel.ERROR, LogLevel.FATAL],
            start_time=error_entry.timestamp - time_window,
            end_time=error_entry.timestamp + time_window,
        )

        return self._store.query(filter)


# ========================
# Pattern Detector
# ========================


class PatternDetector:
    """Detects patterns in log messages."""

    def __init__(self):
        self._patterns: Dict[str, LogPattern] = {}
        self._lock = threading.Lock()

    def process(self, entry: LogEntry) -> Optional[LogPattern]:
        """Process a log entry and detect patterns."""
        # Create a simplified pattern by replacing variables
        pattern = self._simplify_message(entry.message)
        pattern_id = hashlib.md5(pattern.encode()).hexdigest()[:12]

        with self._lock:
            if pattern_id in self._patterns:
                existing = self._patterns[pattern_id]
                existing.count += 1
                existing.last_seen = datetime.now()
                if entry.service:
                    existing.services.add(entry.service)
                return existing
            else:
                new_pattern = LogPattern(
                    pattern_id=pattern_id,
                    pattern=pattern,
                    sample_message=entry.message,
                    count=1,
                    level=entry.level,
                    services={entry.service} if entry.service else set(),
                )
                self._patterns[pattern_id] = new_pattern
                return new_pattern

    def _simplify_message(self, message: str) -> str:
        """Simplify message to detect patterns."""
        # Replace numbers
        pattern = re.sub(r"\d+", "<NUM>", message)
        # Replace UUIDs
        pattern = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "<UUID>",
            pattern,
            flags=re.IGNORECASE,
        )
        # Replace IPs
        pattern = re.sub(r"\d+\.\d+\.\d+\.\d+", "<IP>", pattern)
        # Replace timestamps
        pattern = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "<TIMESTAMP>", pattern)
        return pattern

    def get_top_patterns(self, limit: int = 10) -> List[LogPattern]:
        """Get most frequent patterns."""
        with self._lock:
            patterns = list(self._patterns.values())
        return sorted(patterns, key=lambda x: x.count, reverse=True)[:limit]

    def get_error_patterns(self) -> List[LogPattern]:
        """Get error patterns."""
        with self._lock:
            return [
                p for p in self._patterns.values() if p.level in (LogLevel.ERROR, LogLevel.FATAL)
            ]


# ========================
# Log Aggregator
# ========================


class LogAggregator:
    """Main log aggregator coordinating all logging operations."""

    def __init__(self, max_entries: int = 100000):
        self._store = LogStore(max_entries)
        self._parsers: Dict[str, LogParser] = {}
        self._correlator = LogCorrelator(self._store)
        self._pattern_detector = PatternDetector()
        self._retention_policies: Dict[str, RetentionPolicy] = {}
        self._lock = threading.Lock()

    def register_parser(self, config: ParserConfig) -> bool:
        """Register a log parser."""
        parser: LogParser

        if config.parser_type == ParserType.JSON:
            parser = JSONLogParser(config)
        elif config.parser_type == ParserType.REGEX:
            parser = RegexLogParser(config)
        elif config.parser_type == ParserType.SYSLOG:
            parser = SyslogParser(config)
        else:
            return False

        with self._lock:
            self._parsers[config.parser_id] = parser
        return True

    def ingest(
        self,
        raw_log: str,
        parser_id: Optional[str] = None,
        source: LogSource = LogSource.APPLICATION,
    ) -> Optional[LogEntry]:
        """Ingest a raw log line."""
        entry = None

        # Try specific parser first
        if parser_id:
            with self._lock:
                parser = self._parsers.get(parser_id)
            if parser:
                entry = parser.parse(raw_log)

        # Try all parsers
        if not entry:
            with self._lock:
                parsers = list(self._parsers.values())
            for parser in parsers:
                entry = parser.parse(raw_log)
                if entry:
                    break

        # Fallback to raw entry
        if not entry:
            entry = LogEntry(
                log_id=f"log_{int(time.time() * 1000000)}",
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=raw_log,
                source=source,
                raw=raw_log,
            )

        entry.source = source
        self._store.add(entry)
        self._pattern_detector.process(entry)

        return entry

    def ingest_entry(self, entry: LogEntry) -> None:
        """Ingest a pre-parsed log entry."""
        self._store.add(entry)
        self._pattern_detector.process(entry)

    def query(self, filter: LogFilter, limit: int = 1000) -> List[LogEntry]:
        """Query logs."""
        return self._store.query(filter, limit)

    def correlate(self, trace_id: str) -> CorrelatedLogs:
        """Correlate logs by trace ID."""
        return self._correlator.correlate_by_trace(trace_id)

    def get_stats(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> LogStats:
        """Get log statistics."""
        return self._store.get_stats(start_time, end_time)

    def get_patterns(self, limit: int = 10) -> List[LogPattern]:
        """Get top log patterns."""
        return self._pattern_detector.get_top_patterns(limit)

    def get_error_patterns(self) -> List[LogPattern]:
        """Get error patterns."""
        return self._pattern_detector.get_error_patterns()

    def aggregate(
        self, field: str, mode: AggregationMode, filter: Optional[LogFilter] = None, limit: int = 10
    ) -> LogAggregation:
        """Aggregate logs by field."""
        entries = self._store.query(filter or LogFilter(), limit=100000)

        if mode == AggregationMode.COUNT:
            return LogAggregation(mode=mode, field=field, value=len(entries), count=len(entries))

        elif mode == AggregationMode.TOP_VALUES:
            counter: Dict[str, int] = defaultdict(int)
            for entry in entries:
                value = getattr(entry, field, None) or entry.fields.get(field)
                if value:
                    counter[str(value)] += 1

            buckets = [
                {"value": k, "count": v}
                for k, v in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:limit]
            ]

            return LogAggregation(
                mode=mode,
                field=field,
                value=buckets[0]["value"] if buckets else None,
                count=len(entries),
                buckets=buckets,
            )

        return LogAggregation(mode=mode, field=field, value=None)

    def add_retention_policy(self, policy: RetentionPolicy) -> bool:
        """Add a retention policy."""
        with self._lock:
            if policy.policy_id in self._retention_policies:
                return False
            self._retention_policies[policy.policy_id] = policy
            return True

    def apply_retention(self) -> int:
        """Apply retention policies."""
        total_removed = 0

        with self._lock:
            policies = list(self._retention_policies.values())

        for policy in policies:
            removed = self._store.cleanup(policy.max_age)
            total_removed += removed

        return total_removed


# ========================
# Log Aggregator Provider
# ========================


class LogAggregatorVisionProvider(VisionProvider):
    """Vision provider with log aggregation integration."""

    def __init__(self, base_provider: VisionProvider, aggregator: Optional[LogAggregator] = None):
        self._base_provider = base_provider
        self._aggregator = aggregator or LogAggregator()
        self._request_id = 0

    @property
    def provider_name(self) -> str:
        return f"log_aggregator_{self._base_provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, context: Optional[Dict[str, Any]] = None
    ) -> VisionDescription:
        """Analyze image with logging."""
        self._request_id += 1
        trace_id = context.get("trace_id") if context else None
        trace_id = trace_id or f"trace_{self._request_id}"

        # Log request start
        self._aggregator.ingest_entry(
            LogEntry(
                log_id=f"log_{int(time.time() * 1000000)}",
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message=f"Starting vision analysis request",
                service=self.provider_name,
                trace_id=trace_id,
                fields={"image_size": len(image_data), "request_id": self._request_id},
            )
        )

        try:
            result = await self._base_provider.analyze_image(image_data, context)

            # Log success
            self._aggregator.ingest_entry(
                LogEntry(
                    log_id=f"log_{int(time.time() * 1000000)}",
                    timestamp=datetime.now(),
                    level=LogLevel.INFO,
                    message=f"Vision analysis completed successfully",
                    service=self.provider_name,
                    trace_id=trace_id,
                    fields={"confidence": result.confidence, "request_id": self._request_id},
                )
            )

            return result

        except Exception as e:
            # Log error
            self._aggregator.ingest_entry(
                LogEntry(
                    log_id=f"log_{int(time.time() * 1000000)}",
                    timestamp=datetime.now(),
                    level=LogLevel.ERROR,
                    message=f"Vision analysis failed: {str(e)}",
                    service=self.provider_name,
                    trace_id=trace_id,
                    fields={"error_type": type(e).__name__, "request_id": self._request_id},
                )
            )
            raise

    def get_aggregator(self) -> LogAggregator:
        """Get the log aggregator."""
        return self._aggregator


# ========================
# Factory Functions
# ========================


def create_log_aggregator(max_entries: int = 100000) -> LogAggregator:
    """Create a new log aggregator."""
    return LogAggregator(max_entries)


def create_json_parser(
    parser_id: str, name: str, timestamp_format: str = "%Y-%m-%dT%H:%M:%S"
) -> ParserConfig:
    """Create a JSON parser configuration."""
    return ParserConfig(
        parser_id=parser_id,
        parser_type=ParserType.JSON,
        name=name,
        timestamp_format=timestamp_format,
    )


def create_regex_parser(
    parser_id: str, name: str, pattern: str, timestamp_format: str = "%Y-%m-%d %H:%M:%S"
) -> ParserConfig:
    """Create a regex parser configuration."""
    return ParserConfig(
        parser_id=parser_id,
        parser_type=ParserType.REGEX,
        name=name,
        pattern=pattern,
        timestamp_format=timestamp_format,
    )


def create_retention_policy(
    policy_id: str, name: str, max_age_days: int = 30, archive_enabled: bool = False
) -> RetentionPolicy:
    """Create a retention policy."""
    return RetentionPolicy(
        policy_id=policy_id,
        name=name,
        max_age=timedelta(days=max_age_days),
        archive_enabled=archive_enabled,
    )


def create_log_aggregator_provider(
    base_provider: VisionProvider, aggregator: Optional[LogAggregator] = None
) -> LogAggregatorVisionProvider:
    """Create a log aggregator vision provider."""
    return LogAggregatorVisionProvider(base_provider, aggregator)
