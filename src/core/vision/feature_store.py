"""Feature Store Module for Vision System.

This module provides feature management capabilities including:
- Feature definition and registration
- Feature versioning and lineage tracking
- Feature computation and caching
- Feature serving for training and inference
- Feature validation and quality monitoring
- Feature transformation pipelines

Phase 18: Advanced ML Pipeline & AutoML
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class FeatureType(str, Enum):
    """Types of features."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    EMBEDDING = "embedding"
    TIMESTAMP = "timestamp"
    ARRAY = "array"
    STRUCT = "struct"


class FeatureStatus(str, Enum):
    """Feature lifecycle status."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ComputationMode(str, Enum):
    """Feature computation modes."""

    BATCH = "batch"
    STREAMING = "streaming"
    ON_DEMAND = "on_demand"
    PRECOMPUTED = "precomputed"


class TransformationType(str, Enum):
    """Feature transformation types."""

    IDENTITY = "identity"
    FILTER = "filter"
    MAP = "map"
    AGGREGATE = "aggregate"
    JOIN = "join"
    SORT = "sort"
    DEDUPLICATE = "deduplicate"
    ENRICH = "enrich"
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    LOG = "log"
    BUCKETIZE = "bucketize"
    ONE_HOT = "one_hot"
    EMBEDDING = "embedding"
    HASH = "hash"
    CUSTOM = "custom"


class AggregationWindow(str, Enum):
    """Time-based aggregation windows."""

    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOUR = "1h"
    SIX_HOURS = "6h"
    DAY = "1d"
    WEEK = "7d"
    MONTH = "30d"


class DataQualityCheck(str, Enum):
    """Data quality check types."""

    NULL_CHECK = "null_check"
    RANGE_CHECK = "range_check"
    UNIQUE_CHECK = "unique_check"
    TYPE_CHECK = "type_check"
    SCHEMA_CHECK = "schema_check"
    FRESHNESS_CHECK = "freshness_check"
    DRIFT_CHECK = "drift_check"


# ========================
# Dataclasses
# ========================


@dataclass
class FeatureDefinition:
    """Definition of a feature."""

    feature_id: str
    name: str
    feature_type: FeatureType
    description: str = ""
    version: str = "1.0.0"
    status: FeatureStatus = FeatureStatus.DRAFT
    entity_type: str = "default"
    computation_mode: ComputationMode = ComputationMode.BATCH
    default_value: Optional[Any] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class FeatureValue:
    """A computed feature value."""

    feature_id: str
    entity_id: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureGroup:
    """A group of related features."""

    group_id: str
    name: str
    features: List[str]
    description: str = ""
    entity_type: str = "default"
    online_enabled: bool = True
    offline_enabled: bool = True
    ttl_seconds: int = 86400
    tags: List[str] = field(default_factory=list)


@dataclass
class FeatureTransformation:
    """A feature transformation specification."""

    transform_id: str
    name: str
    transform_type: TransformationType
    input_features: List[str]
    output_feature: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    custom_fn: Optional[Callable] = None


@dataclass
class FeatureLineage:
    """Feature lineage information."""

    feature_id: str
    source_features: List[str]
    transformations: List[str]
    data_sources: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class FeatureStatistics:
    """Statistics for a feature."""

    feature_id: str
    count: int
    null_count: int
    unique_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    percentiles: Dict[str, float] = field(default_factory=dict)
    distribution: Dict[str, int] = field(default_factory=dict)
    computed_at: datetime = field(default_factory=datetime.now)


@dataclass
class QualityReport:
    """Feature quality report."""

    feature_id: str
    checks: List[DataQualityCheck]
    passed: bool
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)


# ========================
# Core Classes
# ========================


class FeatureRegistry:
    """Central registry for feature definitions."""

    def __init__(self):
        self._features: Dict[str, FeatureDefinition] = {}
        self._groups: Dict[str, FeatureGroup] = {}
        self._lineage: Dict[str, FeatureLineage] = {}
        self._versions: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()

    def register_feature(self, feature: FeatureDefinition) -> None:
        """Register a new feature."""
        with self._lock:
            key = f"{feature.feature_id}:{feature.version}"
            self._features[key] = feature
            self._versions[feature.feature_id].append(feature.version)

    def get_feature(
        self,
        feature_id: str,
        version: Optional[str] = None,
    ) -> Optional[FeatureDefinition]:
        """Get a feature definition."""
        with self._lock:
            if version is None:
                # Get latest version
                versions = self._versions.get(feature_id, [])
                if not versions:
                    return None
                version = versions[-1]

            key = f"{feature_id}:{version}"
            return self._features.get(key)

    def update_feature(self, feature: FeatureDefinition) -> None:
        """Update a feature definition."""
        with self._lock:
            feature.updated_at = datetime.now()
            key = f"{feature.feature_id}:{feature.version}"
            self._features[key] = feature

    def deprecate_feature(self, feature_id: str, version: str) -> None:
        """Deprecate a feature version."""
        feature = self.get_feature(feature_id, version)
        if feature:
            feature.status = FeatureStatus.DEPRECATED
            self.update_feature(feature)

    def list_features(
        self,
        status: Optional[FeatureStatus] = None,
        entity_type: Optional[str] = None,
    ) -> List[FeatureDefinition]:
        """List registered features."""
        with self._lock:
            features = list(self._features.values())

            if status is not None:
                features = [f for f in features if f.status == status]
            if entity_type is not None:
                features = [f for f in features if f.entity_type == entity_type]

            return features

    def register_group(self, group: FeatureGroup) -> None:
        """Register a feature group."""
        with self._lock:
            self._groups[group.group_id] = group

    def get_group(self, group_id: str) -> Optional[FeatureGroup]:
        """Get a feature group."""
        return self._groups.get(group_id)

    def set_lineage(self, lineage: FeatureLineage) -> None:
        """Set feature lineage."""
        with self._lock:
            self._lineage[lineage.feature_id] = lineage

    def get_lineage(self, feature_id: str) -> Optional[FeatureLineage]:
        """Get feature lineage."""
        return self._lineage.get(feature_id)


class FeatureStore:
    """Feature store for serving features."""

    def __init__(self, registry: Optional[FeatureRegistry] = None):
        self._registry = registry or FeatureRegistry()
        self._online_store: Dict[str, Dict[str, FeatureValue]] = defaultdict(dict)
        self._offline_store: List[FeatureValue] = []
        self._cache: Dict[str, FeatureValue] = {}
        self._cache_ttl = 300  # 5 minutes
        self._lock = threading.RLock()

    def ingest_feature(self, value: FeatureValue) -> None:
        """Ingest a feature value."""
        with self._lock:
            # Store in online store
            entity_key = f"{value.feature_id}:{value.entity_id}"
            self._online_store[value.feature_id][value.entity_id] = value

            # Store in offline store
            self._offline_store.append(value)

            # Update cache
            cache_key = f"{entity_key}:{value.version}"
            self._cache[cache_key] = value

    def get_online_features(
        self,
        feature_ids: List[str],
        entity_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Get features for online serving."""
        result: Dict[str, Dict[str, Any]] = {}

        with self._lock:
            for entity_id in entity_ids:
                result[entity_id] = {}
                for feature_id in feature_ids:
                    value = self._online_store.get(feature_id, {}).get(entity_id)
                    if value:
                        result[entity_id][feature_id] = value.value
                    else:
                        # Get default value from registry
                        feature = self._registry.get_feature(feature_id)
                        if feature:
                            result[entity_id][feature_id] = feature.default_value

        return result

    def get_historical_features(
        self,
        feature_ids: List[str],
        entity_ids: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get historical features for training."""
        result = []

        with self._lock:
            for value in self._offline_store:
                if value.feature_id not in feature_ids:
                    continue
                if value.entity_id not in entity_ids:
                    continue
                if start_time and value.timestamp < start_time:
                    continue
                if end_time and value.timestamp > end_time:
                    continue

                result.append(
                    {
                        "feature_id": value.feature_id,
                        "entity_id": value.entity_id,
                        "value": value.value,
                        "timestamp": value.timestamp,
                    }
                )

        return result

    def materialize(
        self,
        feature_ids: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Materialize features to online store."""
        count = 0
        with self._lock:
            for value in self._offline_store:
                if value.feature_id not in feature_ids:
                    continue
                if value.timestamp < start_time or value.timestamp > end_time:
                    continue

                # Update online store with latest value
                existing = self._online_store.get(value.feature_id, {}).get(value.entity_id)
                if existing is None or value.timestamp > existing.timestamp:
                    self._online_store[value.feature_id][value.entity_id] = value
                    count += 1

        return count


class FeatureComputer:
    """Compute features from raw data."""

    def __init__(self, registry: FeatureRegistry):
        self._registry = registry
        self._transformations: Dict[str, FeatureTransformation] = {}
        self._computation_cache: Dict[str, Any] = {}

    def register_transformation(self, transform: FeatureTransformation) -> None:
        """Register a feature transformation."""
        self._transformations[transform.transform_id] = transform

    def compute_feature(
        self,
        feature_id: str,
        entity_id: str,
        raw_data: Dict[str, Any],
    ) -> FeatureValue:
        """Compute a feature value from raw data."""
        feature = self._registry.get_feature(feature_id)
        if feature is None:
            raise ValueError(f"Feature {feature_id} not found")

        # Find transformation for this feature
        transform = None
        for t in self._transformations.values():
            if t.output_feature == feature_id:
                transform = t
                break

        if transform is None:
            # No transformation, use raw value
            value = raw_data.get(feature_id, feature.default_value)
        else:
            # Apply transformation
            value = self._apply_transformation(transform, raw_data)

        return FeatureValue(
            feature_id=feature_id,
            entity_id=entity_id,
            value=value,
            version=feature.version,
        )

    def _apply_transformation(
        self,
        transform: FeatureTransformation,
        raw_data: Dict[str, Any],
    ) -> Any:
        """Apply a transformation to raw data."""
        input_values = [raw_data.get(f) for f in transform.input_features]

        if transform.transform_type == TransformationType.IDENTITY:
            return input_values[0] if input_values else None

        elif transform.transform_type == TransformationType.NORMALIZE:
            value = input_values[0]
            min_val = transform.parameters.get("min", 0)
            max_val = transform.parameters.get("max", 1)
            if value is not None and max_val != min_val:
                return (value - min_val) / (max_val - min_val)
            return 0.0

        elif transform.transform_type == TransformationType.STANDARDIZE:
            value = input_values[0]
            mean = transform.parameters.get("mean", 0)
            std = transform.parameters.get("std", 1)
            if value is not None and std != 0:
                return (value - mean) / std
            return 0.0

        elif transform.transform_type == TransformationType.LOG:
            import math

            value = input_values[0]
            if value is not None and value > 0:
                return math.log(value)
            return None

        elif transform.transform_type == TransformationType.BUCKETIZE:
            value = input_values[0]
            boundaries = transform.parameters.get("boundaries", [])
            if value is not None:
                for i, boundary in enumerate(boundaries):
                    if value < boundary:
                        return i
                return len(boundaries)
            return None

        elif transform.transform_type == TransformationType.ONE_HOT:
            value = input_values[0]
            vocabulary = transform.parameters.get("vocabulary", [])
            result = [0] * len(vocabulary)
            if value in vocabulary:
                result[vocabulary.index(value)] = 1
            return result

        elif transform.transform_type == TransformationType.HASH:
            value = str(input_values[0]) if input_values[0] is not None else ""
            num_buckets = transform.parameters.get("num_buckets", 1000)
            return int(hashlib.sha256(value.encode()).hexdigest(), 16) % num_buckets

        elif transform.transform_type == TransformationType.CUSTOM:
            if transform.custom_fn is not None:
                return transform.custom_fn(*input_values)
            return None

        return input_values[0] if input_values else None

    def compute_batch(
        self,
        feature_ids: List[str],
        entity_ids: List[str],
        raw_data: Dict[str, Dict[str, Any]],
    ) -> List[FeatureValue]:
        """Compute features in batch."""
        results = []
        for entity_id in entity_ids:
            entity_data = raw_data.get(entity_id, {})
            for feature_id in feature_ids:
                try:
                    value = self.compute_feature(feature_id, entity_id, entity_data)
                    results.append(value)
                except Exception:
                    pass
        return results


class FeatureValidator:
    """Validate feature quality."""

    def __init__(self):
        self._check_functions: Dict[DataQualityCheck, Callable] = {
            DataQualityCheck.NULL_CHECK: self._check_nulls,
            DataQualityCheck.RANGE_CHECK: self._check_range,
            DataQualityCheck.UNIQUE_CHECK: self._check_unique,
            DataQualityCheck.TYPE_CHECK: self._check_type,
        }

    def validate(
        self,
        feature: FeatureDefinition,
        values: List[FeatureValue],
        checks: List[DataQualityCheck],
    ) -> QualityReport:
        """Validate feature values."""
        issues = []
        metrics = {}
        all_passed = True

        for check in checks:
            check_fn = self._check_functions.get(check)
            if check_fn:
                passed, issue, metric = check_fn(feature, values)
                if not passed:
                    all_passed = False
                    if issue:
                        issues.append(issue)
                if metric:
                    metrics.update(metric)

        return QualityReport(
            feature_id=feature.feature_id,
            checks=checks,
            passed=all_passed,
            issues=issues,
            metrics=metrics,
        )

    def _check_nulls(
        self,
        feature: FeatureDefinition,
        values: List[FeatureValue],
    ) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """Check for null values."""
        null_count = sum(1 for v in values if v.value is None)
        total = len(values)
        null_rate = null_count / total if total > 0 else 0

        threshold = 0.1  # 10% null threshold
        passed = null_rate <= threshold

        issue = None
        if not passed:
            issue = f"Null rate {null_rate:.2%} exceeds threshold {threshold:.2%}"

        return passed, issue, {"null_rate": null_rate, "null_count": float(null_count)}

    def _check_range(
        self,
        feature: FeatureDefinition,
        values: List[FeatureValue],
    ) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """Check value ranges."""
        if feature.feature_type not in [FeatureType.NUMERICAL]:
            return True, None, {}

        numeric_values = [
            v.value for v in values if v.value is not None and isinstance(v.value, (int, float))
        ]

        if not numeric_values:
            return True, None, {}

        min_val = min(numeric_values)
        max_val = max(numeric_values)

        expected_min = feature.metadata.get("min_value")
        expected_max = feature.metadata.get("max_value")

        issues = []
        if expected_min is not None and min_val < expected_min:
            issues.append(f"Min value {min_val} below expected {expected_min}")
        if expected_max is not None and max_val > expected_max:
            issues.append(f"Max value {max_val} above expected {expected_max}")

        passed = len(issues) == 0

        return (
            passed,
            "; ".join(issues) if issues else None,
            {
                "min_value": float(min_val),
                "max_value": float(max_val),
            },
        )

    def _check_unique(
        self,
        feature: FeatureDefinition,
        values: List[FeatureValue],
    ) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """Check uniqueness."""
        total = len(values)
        unique_count = len(set(v.value for v in values if v.value is not None))
        unique_rate = unique_count / total if total > 0 else 0

        return (
            True,
            None,
            {
                "unique_count": float(unique_count),
                "unique_rate": unique_rate,
            },
        )

    def _check_type(
        self,
        feature: FeatureDefinition,
        values: List[FeatureValue],
    ) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """Check value types."""
        type_map = {
            FeatureType.NUMERICAL: (int, float),
            FeatureType.BOOLEAN: bool,
            FeatureType.TEXT: str,
            FeatureType.ARRAY: (list, tuple),
        }

        expected_type = type_map.get(feature.feature_type)
        if expected_type is None:
            return True, None, {}

        invalid_count = 0
        for v in values:
            if v.value is not None and not isinstance(v.value, expected_type):
                invalid_count += 1

        total = len(values)
        invalid_rate = invalid_count / total if total > 0 else 0

        passed = invalid_rate < 0.01  # Less than 1% type errors
        issue = None
        if not passed:
            issue = f"Type error rate {invalid_rate:.2%} is too high"

        return passed, issue, {"type_error_rate": invalid_rate}


class FeatureStatisticsComputer:
    """Compute feature statistics."""

    def compute(
        self,
        feature: FeatureDefinition,
        values: List[FeatureValue],
    ) -> FeatureStatistics:
        """Compute statistics for a feature."""
        total = len(values)
        null_count = sum(1 for v in values if v.value is None)
        non_null_values = [v.value for v in values if v.value is not None]
        unique_count = len(set(non_null_values))

        stats = FeatureStatistics(
            feature_id=feature.feature_id,
            count=total,
            null_count=null_count,
            unique_count=unique_count,
        )

        # Compute numeric statistics
        if feature.feature_type == FeatureType.NUMERICAL:
            numeric_values = [v for v in non_null_values if isinstance(v, (int, float))]
            if numeric_values:
                import statistics as stats_lib

                stats.mean = stats_lib.mean(numeric_values)
                stats.std = stats_lib.stdev(numeric_values) if len(numeric_values) > 1 else 0.0
                stats.min_value = min(numeric_values)
                stats.max_value = max(numeric_values)

                # Compute percentiles
                sorted_values = sorted(numeric_values)
                n = len(sorted_values)
                stats.percentiles = {
                    "p25": sorted_values[int(n * 0.25)] if n > 0 else 0,
                    "p50": sorted_values[int(n * 0.50)] if n > 0 else 0,
                    "p75": sorted_values[int(n * 0.75)] if n > 0 else 0,
                    "p95": sorted_values[int(n * 0.95)] if n > 0 else 0,
                    "p99": sorted_values[int(n * 0.99)] if n > 0 else 0,
                }

        # Compute distribution for categorical
        if feature.feature_type == FeatureType.CATEGORICAL:
            from collections import Counter

            stats.distribution = dict(Counter(non_null_values))

        return stats


# ========================
# Vision Provider
# ========================


class FeatureStoreVisionProvider(VisionProvider):
    """Vision provider for feature store capabilities."""

    def __init__(self):
        self._registry: Optional[FeatureRegistry] = None
        self._store: Optional[FeatureStore] = None
        self._computer: Optional[FeatureComputer] = None

    def get_description(self) -> VisionDescription:
        """Get provider description."""
        return VisionDescription(
            name="Feature Store Vision Provider",
            version="1.0.0",
            description="Feature management, versioning, and serving",
            capabilities=[
                "feature_registration",
                "feature_versioning",
                "online_serving",
                "offline_serving",
                "feature_computation",
                "feature_validation",
            ],
        )

    def initialize(self) -> None:
        """Initialize the provider."""
        self._registry = FeatureRegistry()
        self._store = FeatureStore(self._registry)
        self._computer = FeatureComputer(self._registry)

    def shutdown(self) -> None:
        """Shutdown the provider."""
        self._registry = None
        self._store = None
        self._computer = None

    def get_registry(self) -> FeatureRegistry:
        """Get the feature registry."""
        if self._registry is None:
            self.initialize()
        return self._registry

    def get_store(self) -> FeatureStore:
        """Get the feature store."""
        if self._store is None:
            self.initialize()
        return self._store


# ========================
# Factory Functions
# ========================


def create_feature_registry() -> FeatureRegistry:
    """Create a feature registry."""
    return FeatureRegistry()


def create_feature_store(registry: Optional[FeatureRegistry] = None) -> FeatureStore:
    """Create a feature store."""
    return FeatureStore(registry=registry)


def create_feature_definition(
    feature_id: str,
    name: str,
    feature_type: FeatureType,
    description: str = "",
    version: str = "1.0.0",
) -> FeatureDefinition:
    """Create a feature definition."""
    return FeatureDefinition(
        feature_id=feature_id,
        name=name,
        feature_type=feature_type,
        description=description,
        version=version,
    )


def create_feature_group(
    group_id: str,
    name: str,
    features: List[str],
    description: str = "",
) -> FeatureGroup:
    """Create a feature group."""
    return FeatureGroup(
        group_id=group_id,
        name=name,
        features=features,
        description=description,
    )


def create_feature_transformation(
    transform_id: str,
    name: str,
    transform_type: TransformationType,
    input_features: List[str],
    output_feature: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> FeatureTransformation:
    """Create a feature transformation."""
    return FeatureTransformation(
        transform_id=transform_id,
        name=name,
        transform_type=transform_type,
        input_features=input_features,
        output_feature=output_feature,
        parameters=parameters or {},
    )


def create_feature_computer(registry: FeatureRegistry) -> FeatureComputer:
    """Create a feature computer."""
    return FeatureComputer(registry=registry)


def create_feature_validator() -> FeatureValidator:
    """Create a feature validator."""
    return FeatureValidator()


def create_feature_store_provider() -> FeatureStoreVisionProvider:
    """Create a feature store vision provider."""
    return FeatureStoreVisionProvider()
