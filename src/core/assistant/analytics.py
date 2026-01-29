"""
Analytics Module for CAD Assistant.

Provides analytics, metrics, and dashboard data for
monitoring conversation quality and usage patterns.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TimeGranularity(Enum):
    """Time granularity for analytics."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class ConversationMetrics:
    """Metrics for a single conversation."""

    conversation_id: str
    message_count: int
    avg_response_quality: float
    total_tokens: int
    duration_seconds: float
    topics: List[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class UsageMetrics:
    """Usage metrics over a time period."""

    total_conversations: int
    total_messages: int
    unique_users: int
    avg_messages_per_conversation: float
    avg_quality_score: float
    peak_hour: int
    most_common_topics: List[Tuple[str, int]]
    error_rate: float


@dataclass
class QualityTrend:
    """Quality metrics over time."""

    timestamps: List[float]
    scores: List[float]
    grades: List[str]
    dimension_scores: Dict[str, List[float]]


class AnalyticsCollector:
    """
    Collects and aggregates analytics data.

    Example:
        >>> collector = AnalyticsCollector()
        >>> collector.record_conversation_start("conv-123", "user-1")
        >>> collector.record_message("conv-123", "user", "问题")
        >>> collector.record_message("conv-123", "assistant", "回答", quality_score=0.85)
        >>> metrics = collector.get_daily_metrics()
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize analytics collector.

        Args:
            storage_path: Path for analytics persistence
        """
        self.storage_path = Path(storage_path) if storage_path else None

        # In-memory storage
        self._conversations: Dict[str, Dict[str, Any]] = {}
        self._messages: List[Dict[str, Any]] = []
        self._quality_scores: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []

        # Load existing data
        if self.storage_path and self.storage_path.exists():
            self._load()

    # Event recording

    def record_conversation_start(
        self,
        conversation_id: str,
        user_id: str = "anonymous",
    ) -> None:
        """Record conversation start."""
        self._conversations[conversation_id] = {
            "id": conversation_id,
            "user_id": user_id,
            "started_at": time.time(),
            "ended_at": None,
            "message_count": 0,
            "quality_scores": [],
            "topics": [],
        }

    def record_conversation_end(self, conversation_id: str) -> None:
        """Record conversation end."""
        if conversation_id in self._conversations:
            self._conversations[conversation_id]["ended_at"] = time.time()

    def record_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        quality_score: Optional[float] = None,
        tokens: int = 0,
        topics: Optional[List[str]] = None,
    ) -> None:
        """Record a message."""
        timestamp = time.time()

        self._messages.append({
            "conversation_id": conversation_id,
            "role": role,
            "content_length": len(content),
            "tokens": tokens,
            "timestamp": timestamp,
        })

        if conversation_id in self._conversations:
            self._conversations[conversation_id]["message_count"] += 1
            if topics:
                self._conversations[conversation_id]["topics"].extend(topics)

        if quality_score is not None:
            self._quality_scores.append({
                "conversation_id": conversation_id,
                "score": quality_score,
                "timestamp": timestamp,
            })
            if conversation_id in self._conversations:
                self._conversations[conversation_id]["quality_scores"].append(quality_score)

    def record_quality_evaluation(
        self,
        conversation_id: str,
        overall_score: float,
        grade: str,
        dimension_scores: Dict[str, float],
    ) -> None:
        """Record quality evaluation result."""
        self._quality_scores.append({
            "conversation_id": conversation_id,
            "score": overall_score,
            "grade": grade,
            "dimensions": dimension_scores,
            "timestamp": time.time(),
        })

    def record_error(
        self,
        error_type: str,
        message: str,
        conversation_id: Optional[str] = None,
    ) -> None:
        """Record an error."""
        self._errors.append({
            "type": error_type,
            "message": message,
            "conversation_id": conversation_id,
            "timestamp": time.time(),
        })

    # Metrics retrieval

    def get_usage_metrics(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> UsageMetrics:
        """
        Get usage metrics for a time period.

        Args:
            start_time: Start timestamp (default: 24 hours ago)
            end_time: End timestamp (default: now)

        Returns:
            Usage metrics
        """
        if start_time is None:
            start_time = time.time() - 86400  # 24 hours ago
        if end_time is None:
            end_time = time.time()

        # Filter conversations
        convs = [
            c for c in self._conversations.values()
            if start_time <= c["started_at"] <= end_time
        ]

        # Filter messages
        msgs = [
            m for m in self._messages
            if start_time <= m["timestamp"] <= end_time
        ]

        # Filter quality scores
        scores = [
            q["score"] for q in self._quality_scores
            if start_time <= q["timestamp"] <= end_time
        ]

        # Filter errors
        errors = [
            e for e in self._errors
            if start_time <= e["timestamp"] <= end_time
        ]

        # Calculate metrics
        total_conversations = len(convs)
        total_messages = len(msgs)
        unique_users = len(set(c.get("user_id", "anonymous") for c in convs))

        avg_messages = total_messages / total_conversations if total_conversations > 0 else 0
        avg_quality = sum(scores) / len(scores) if scores else 0

        # Peak hour
        hour_counts = defaultdict(int)
        for msg in msgs:
            hour = datetime.fromtimestamp(msg["timestamp"]).hour
            hour_counts[hour] += 1
        peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 0

        # Topics
        all_topics = []
        for conv in convs:
            all_topics.extend(conv.get("topics", []))
        topic_counts = defaultdict(int)
        for topic in all_topics:
            topic_counts[topic] += 1
        most_common_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Error rate
        error_rate = len(errors) / total_messages if total_messages > 0 else 0

        return UsageMetrics(
            total_conversations=total_conversations,
            total_messages=total_messages,
            unique_users=unique_users,
            avg_messages_per_conversation=avg_messages,
            avg_quality_score=avg_quality,
            peak_hour=peak_hour,
            most_common_topics=most_common_topics,
            error_rate=error_rate,
        )

    def get_quality_trend(
        self,
        granularity: TimeGranularity = TimeGranularity.HOUR,
        limit: int = 24,
    ) -> QualityTrend:
        """
        Get quality score trend over time.

        Args:
            granularity: Time granularity
            limit: Number of periods to return

        Returns:
            Quality trend data
        """
        # Group scores by time bucket
        buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for score_data in self._quality_scores:
            ts = score_data["timestamp"]
            bucket = self._get_time_bucket(ts, granularity)
            buckets[bucket].append(score_data)

        # Sort and limit
        sorted_buckets = sorted(buckets.keys(), reverse=True)[:limit]
        sorted_buckets.reverse()  # Oldest first

        timestamps = []
        scores = []
        grades = []
        dimension_scores: Dict[str, List[float]] = defaultdict(list)

        for bucket in sorted_buckets:
            bucket_data = buckets[bucket]

            timestamps.append(float(bucket))

            # Average score
            avg_score = sum(d["score"] for d in bucket_data) / len(bucket_data)
            scores.append(avg_score)

            # Most common grade
            grade_counts = defaultdict(int)
            for d in bucket_data:
                grade_counts[d.get("grade", "C")] += 1
            most_common_grade = max(grade_counts, key=grade_counts.get)
            grades.append(most_common_grade)

            # Dimension averages
            dim_totals: Dict[str, List[float]] = defaultdict(list)
            for d in bucket_data:
                for dim, score in d.get("dimensions", {}).items():
                    dim_totals[dim].append(score)

            for dim, values in dim_totals.items():
                dimension_scores[dim].append(sum(values) / len(values))

        return QualityTrend(
            timestamps=timestamps,
            scores=scores,
            grades=grades,
            dimension_scores=dict(dimension_scores),
        )

    def get_conversation_metrics(
        self,
        conversation_id: str,
    ) -> Optional[ConversationMetrics]:
        """Get metrics for a specific conversation."""
        if conversation_id not in self._conversations:
            return None

        conv = self._conversations[conversation_id]

        # Calculate duration
        ended_at = conv.get("ended_at") or time.time()
        duration = ended_at - conv["started_at"]

        # Average quality
        scores = conv.get("quality_scores", [])
        avg_quality = sum(scores) / len(scores) if scores else 0

        # Count tokens
        conv_messages = [
            m for m in self._messages
            if m["conversation_id"] == conversation_id
        ]
        total_tokens = sum(m.get("tokens", 0) for m in conv_messages)

        return ConversationMetrics(
            conversation_id=conversation_id,
            message_count=conv["message_count"],
            avg_response_quality=avg_quality,
            total_tokens=total_tokens,
            duration_seconds=duration,
            topics=conv.get("topics", []),
            timestamp=conv["started_at"],
        )

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for dashboard display.

        Returns:
            Dashboard data including metrics, trends, and recent activity
        """
        now = time.time()

        # Time periods
        last_hour = now - 3600
        last_day = now - 86400
        last_week = now - 604800

        return {
            "summary": {
                "total_conversations": len(self._conversations),
                "total_messages": len(self._messages),
                "total_evaluations": len(self._quality_scores),
                "total_errors": len(self._errors),
            },
            "hourly": self.get_usage_metrics(last_hour, now).__dict__,
            "daily": self.get_usage_metrics(last_day, now).__dict__,
            "weekly": self.get_usage_metrics(last_week, now).__dict__,
            "quality_trend": {
                "hourly": self.get_quality_trend(TimeGranularity.HOUR, 24).__dict__,
                "daily": self.get_quality_trend(TimeGranularity.DAY, 7).__dict__,
            },
            "recent_errors": self._errors[-10:],
            "timestamp": now,
        }

    # Persistence

    def save(self) -> bool:
        """Save analytics data to disk."""
        if not self.storage_path:
            return False

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "conversations": self._conversations,
                "messages": self._messages[-10000:],  # Keep last 10k
                "quality_scores": self._quality_scores[-10000:],
                "errors": self._errors[-1000:],
                "saved_at": time.time(),
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            return True
        except IOError:
            return False

    def _load(self) -> bool:
        """Load analytics data from disk."""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._conversations = data.get("conversations", {})
            self._messages = data.get("messages", [])
            self._quality_scores = data.get("quality_scores", [])
            self._errors = data.get("errors", [])
            return True
        except (IOError, json.JSONDecodeError):
            return False

    def clear(self) -> None:
        """Clear all analytics data."""
        self._conversations.clear()
        self._messages.clear()
        self._quality_scores.clear()
        self._errors.clear()

    # Helper methods

    def _get_time_bucket(self, timestamp: float, granularity: TimeGranularity) -> int:
        """Get time bucket for a timestamp."""
        dt = datetime.fromtimestamp(timestamp)

        if granularity == TimeGranularity.HOUR:
            return int(dt.replace(minute=0, second=0, microsecond=0).timestamp())
        elif granularity == TimeGranularity.DAY:
            return int(dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        elif granularity == TimeGranularity.WEEK:
            start_of_week = dt - timedelta(days=dt.weekday())
            return int(start_of_week.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        elif granularity == TimeGranularity.MONTH:
            return int(dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())

        return int(timestamp)
