"""Advanced Data Management & Lifecycle Module for Vision System.

This module provides comprehensive data management capabilities including:
- Data versioning and history tracking
- Data lineage and provenance tracking
- Data quality management and validation
- Data retention policies and archival
- Data catalog and discovery
- Data transformation tracking
- Lifecycle state management

Phase 24: Advanced Data Management & Lifecycle
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider
from .feature_store import TransformationType

# ========================
# Enums
# ========================


class DataState(str, Enum):
    """Data lifecycle states."""

    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"


class VersionType(str, Enum):
    """Types of version changes."""

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    SNAPSHOT = "snapshot"


class LineageRelation(str, Enum):
    """Types of lineage relationships."""

    DERIVED_FROM = "derived_from"
    TRANSFORMED_FROM = "transformed_from"
    COPIED_FROM = "copied_from"
    AGGREGATED_FROM = "aggregated_from"
    JOINED_WITH = "joined_with"
    FILTERED_FROM = "filtered_from"


class QualityDimension(str, Enum):
    """Data quality dimensions."""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


class QualityLevel(str, Enum):
    """Data quality levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class RetentionAction(str, Enum):
    """Actions for data retention."""

    KEEP = "keep"
    ARCHIVE = "archive"
    DELETE = "delete"
    ANONYMIZE = "anonymize"
    COMPRESS = "compress"


class CatalogEntryType(str, Enum):
    """Types of catalog entries."""

    DATASET = "dataset"
    TABLE = "table"
    COLUMN = "column"
    FILE = "file"
    API = "api"
    MODEL = "model"


class AccessLevel(str, Enum):
    """Data access levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


# ========================
# Dataclasses
# ========================


@dataclass
class DataVersion:
    """Represents a version of data."""

    version_id: str
    version_number: str  # semantic versioning: major.minor.patch
    version_type: VersionType
    data_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    is_current: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "version_number": self.version_number,
            "version_type": self.version_type.value,
            "data_hash": self.data_hash,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "message": self.message,
            "parent_version": self.parent_version,
            "is_current": self.is_current,
        }


@dataclass
class LineageNode:
    """Represents a node in the data lineage graph."""

    node_id: str
    name: str
    node_type: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEdge:
    """Represents an edge in the data lineage graph."""

    edge_id: str
    source_id: str
    target_id: str
    relation: LineageRelation
    transformation: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityRule:
    """Defines a data quality rule."""

    rule_id: str
    name: str
    dimension: QualityDimension
    check_fn: Callable[[Any], bool]
    description: str = ""
    severity: str = "warning"
    enabled: bool = True


@dataclass
class QualityResult:
    """Result of a quality check."""

    result_id: str
    rule_id: str
    dimension: QualityDimension
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive quality report for data."""

    report_id: str
    data_id: str
    overall_score: float
    overall_level: QualityLevel
    results: List[QualityResult]
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RetentionPolicy:
    """Defines a data retention policy."""

    policy_id: str
    name: str
    retention_days: int
    action: RetentionAction
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    enabled: bool = True
    description: str = ""


@dataclass
class RetentionResult:
    """Result of retention policy execution."""

    result_id: str
    policy_id: str
    data_id: str
    action_taken: RetentionAction
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""


@dataclass
class CatalogEntry:
    """Entry in the data catalog."""

    entry_id: str
    name: str
    entry_type: CatalogEntryType
    description: str = ""
    owner: str = ""
    access_level: AccessLevel = AccessLevel.INTERNAL
    tags: List[str] = field(default_factory=list)
    schema: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformationRecord:
    """Records a data transformation."""

    record_id: str
    transformation_type: TransformationType
    input_ids: List[str]
    output_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    executed_at: datetime = field(default_factory=datetime.now)
    executed_by: str = "system"
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class DataAsset:
    """Represents a managed data asset."""

    asset_id: str
    name: str
    state: DataState
    current_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    owner: str = ""
    access_level: AccessLevel = AccessLevel.INTERNAL
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecycleConfig:
    """Configuration for data lifecycle management."""

    default_retention_days: int = 365
    archive_after_days: int = 180
    enable_versioning: bool = True
    max_versions: int = 100
    enable_lineage: bool = True
    enable_quality_checks: bool = True
    auto_archive: bool = True
    auto_delete_expired: bool = False


# ========================
# Core Classes
# ========================


class VersionManager:
    """Manages data versioning."""

    def __init__(self, config: Optional[LifecycleConfig] = None) -> None:
        """Initialize version manager."""
        self._config = config or LifecycleConfig()
        self._versions: Dict[str, List[DataVersion]] = defaultdict(list)
        self._current: Dict[str, str] = {}  # asset_id -> version_id
        self._lock = threading.RLock()

    def create_version(
        self,
        asset_id: str,
        data: Any,
        version_type: VersionType = VersionType.MINOR,
        message: str = "",
        created_by: str = "system",
    ) -> DataVersion:
        """Create a new version."""
        with self._lock:
            # Calculate data hash
            data_str = json.dumps(data, sort_keys=True, default=str)
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()

            # Get current version number
            versions = self._versions[asset_id]
            if versions:
                current = versions[-1]
                major, minor, patch = map(int, current.version_number.split("."))
                if version_type == VersionType.MAJOR:
                    major += 1
                    minor = 0
                    patch = 0
                elif version_type == VersionType.MINOR:
                    minor += 1
                    patch = 0
                else:
                    patch += 1
                version_number = f"{major}.{minor}.{patch}"
                parent_version = current.version_id
                # Mark previous as not current
                current.is_current = False
            else:
                version_number = "1.0.0"
                parent_version = None

            version = DataVersion(
                version_id=str(uuid.uuid4()),
                version_number=version_number,
                version_type=version_type,
                data_hash=data_hash,
                message=message,
                created_by=created_by,
                parent_version=parent_version,
                is_current=True,
            )

            versions.append(version)
            self._current[asset_id] = version.version_id

            # Enforce max versions
            if len(versions) > self._config.max_versions:
                self._versions[asset_id] = versions[-self._config.max_versions :]

            return version

    def get_version(self, asset_id: str, version_id: str) -> Optional[DataVersion]:
        """Get a specific version."""
        with self._lock:
            for v in self._versions.get(asset_id, []):
                if v.version_id == version_id:
                    return v
            return None

    def get_current_version(self, asset_id: str) -> Optional[DataVersion]:
        """Get the current version."""
        with self._lock:
            version_id = self._current.get(asset_id)
            if version_id:
                return self.get_version(asset_id, version_id)
            return None

    def get_version_history(self, asset_id: str, limit: int = 100) -> List[DataVersion]:
        """Get version history."""
        with self._lock:
            versions = self._versions.get(asset_id, [])
            return versions[-limit:]

    def rollback_to_version(self, asset_id: str, version_id: str) -> Optional[DataVersion]:
        """Rollback to a specific version."""
        with self._lock:
            version = self.get_version(asset_id, version_id)
            if version:
                # Create new version based on old one
                new_version = DataVersion(
                    version_id=str(uuid.uuid4()),
                    version_number=self._increment_version(asset_id),
                    version_type=VersionType.PATCH,
                    data_hash=version.data_hash,
                    message=f"Rollback to {version.version_number}",
                    parent_version=self._current.get(asset_id),
                    is_current=True,
                )

                # Mark current as not current
                current = self.get_current_version(asset_id)
                if current:
                    current.is_current = False

                self._versions[asset_id].append(new_version)
                self._current[asset_id] = new_version.version_id
                return new_version
            return None

    def _increment_version(self, asset_id: str) -> str:
        """Increment version number."""
        versions = self._versions.get(asset_id, [])
        if versions:
            current = versions[-1]
            major, minor, patch = map(int, current.version_number.split("."))
            return f"{major}.{minor}.{patch + 1}"
        return "1.0.0"

    def compare_versions(
        self, asset_id: str, version_id_1: str, version_id_2: str
    ) -> Dict[str, Any]:
        """Compare two versions."""
        with self._lock:
            v1 = self.get_version(asset_id, version_id_1)
            v2 = self.get_version(asset_id, version_id_2)

            if not v1 or not v2:
                return {"error": "Version not found"}

            return {
                "version_1": v1.to_dict(),
                "version_2": v2.to_dict(),
                "hash_match": v1.data_hash == v2.data_hash,
                "time_delta": abs((v2.created_at - v1.created_at).total_seconds()),
            }


class LineageTracker:
    """Tracks data lineage and provenance."""

    def __init__(self, config: Optional[LifecycleConfig] = None) -> None:
        """Initialize lineage tracker."""
        self._config = config or LifecycleConfig()
        self._nodes: Dict[str, LineageNode] = {}
        self._edges: List[LineageEdge] = []
        self._lock = threading.RLock()

    def register_node(
        self,
        name: str,
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        """Register a lineage node."""
        with self._lock:
            node = LineageNode(
                node_id=str(uuid.uuid4()),
                name=name,
                node_type=node_type,
                metadata=metadata or {},
            )
            self._nodes[node.node_id] = node
            return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: LineageRelation,
        transformation: Optional[str] = None,
    ) -> Optional[LineageEdge]:
        """Add a lineage edge."""
        with self._lock:
            if source_id not in self._nodes or target_id not in self._nodes:
                return None

            edge = LineageEdge(
                edge_id=str(uuid.uuid4()),
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                transformation=transformation,
            )
            self._edges.append(edge)
            return edge

    def get_upstream(self, node_id: str, depth: int = 10) -> List[LineageNode]:
        """Get upstream nodes (ancestors)."""
        with self._lock:
            visited = set()
            result = []
            self._traverse_upstream(node_id, depth, visited, result)
            return result

    def _traverse_upstream(
        self,
        node_id: str,
        depth: int,
        visited: Set[str],
        result: List[LineageNode],
    ) -> None:
        """Traverse upstream nodes."""
        if depth <= 0 or node_id in visited:
            return
        visited.add(node_id)

        for edge in self._edges:
            if edge.target_id == node_id:
                node = self._nodes.get(edge.source_id)
                if node and node.node_id not in visited:
                    result.append(node)
                    self._traverse_upstream(node.node_id, depth - 1, visited, result)

    def get_downstream(self, node_id: str, depth: int = 10) -> List[LineageNode]:
        """Get downstream nodes (descendants)."""
        with self._lock:
            visited = set()
            result = []
            self._traverse_downstream(node_id, depth, visited, result)
            return result

    def _traverse_downstream(
        self,
        node_id: str,
        depth: int,
        visited: Set[str],
        result: List[LineageNode],
    ) -> None:
        """Traverse downstream nodes."""
        if depth <= 0 or node_id in visited:
            return
        visited.add(node_id)

        for edge in self._edges:
            if edge.source_id == node_id:
                node = self._nodes.get(edge.target_id)
                if node and node.node_id not in visited:
                    result.append(node)
                    self._traverse_downstream(node.node_id, depth - 1, visited, result)

    def get_lineage_graph(self, node_id: str) -> Dict[str, Any]:
        """Get complete lineage graph for a node."""
        with self._lock:
            upstream = self.get_upstream(node_id)
            downstream = self.get_downstream(node_id)
            center = self._nodes.get(node_id)

            relevant_edges = [
                e
                for e in self._edges
                if e.source_id == node_id
                or e.target_id == node_id
                or any(n.node_id in (e.source_id, e.target_id) for n in upstream + downstream)
            ]

            return {
                "center": center,
                "upstream": upstream,
                "downstream": downstream,
                "edges": relevant_edges,
            }

    def get_node(self, node_id: str) -> Optional[LineageNode]:
        """Get a node by ID."""
        with self._lock:
            return self._nodes.get(node_id)

    def list_nodes(self) -> List[LineageNode]:
        """List all nodes."""
        with self._lock:
            return list(self._nodes.values())


class QualityManager:
    """Manages data quality rules and checks."""

    def __init__(self, config: Optional[LifecycleConfig] = None) -> None:
        """Initialize quality manager."""
        self._config = config or LifecycleConfig()
        self._rules: Dict[str, QualityRule] = {}
        self._reports: List[QualityReport] = []
        self._lock = threading.RLock()

    def add_rule(self, rule: QualityRule) -> None:
        """Add a quality rule."""
        with self._lock:
            self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a quality rule."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                return True
            return False

    def get_rule(self, rule_id: str) -> Optional[QualityRule]:
        """Get a rule by ID."""
        with self._lock:
            return self._rules.get(rule_id)

    def list_rules(self, dimension: Optional[QualityDimension] = None) -> List[QualityRule]:
        """List rules, optionally filtered by dimension."""
        with self._lock:
            rules = list(self._rules.values())
            if dimension:
                rules = [r for r in rules if r.dimension == dimension]
            return rules

    def check_quality(self, data_id: str, data: Any) -> QualityReport:
        """Run all quality checks on data."""
        with self._lock:
            results = []
            scores = []

            for rule in self._rules.values():
                if not rule.enabled:
                    continue

                try:
                    passed = rule.check_fn(data)
                    score = 1.0 if passed else 0.0
                except Exception as e:
                    passed = False
                    score = 0.0

                result = QualityResult(
                    result_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    dimension=rule.dimension,
                    passed=passed,
                    score=score,
                    message=f"{'Passed' if passed else 'Failed'}: {rule.name}",
                )
                results.append(result)
                scores.append(score)

            # Calculate overall score
            overall_score = sum(scores) / len(scores) if scores else 1.0

            # Determine level
            if overall_score >= 0.9:
                level = QualityLevel.EXCELLENT
            elif overall_score >= 0.75:
                level = QualityLevel.GOOD
            elif overall_score >= 0.5:
                level = QualityLevel.ACCEPTABLE
            elif overall_score >= 0.25:
                level = QualityLevel.POOR
            else:
                level = QualityLevel.CRITICAL

            # Generate recommendations
            recommendations = []
            for result in results:
                if not result.passed:
                    rule = self._rules.get(result.rule_id)
                    if rule:
                        recommendations.append(f"Fix {rule.dimension.value}: {rule.description}")

            report = QualityReport(
                report_id=str(uuid.uuid4()),
                data_id=data_id,
                overall_score=overall_score,
                overall_level=level,
                results=results,
                recommendations=recommendations,
            )
            self._reports.append(report)
            return report

    def get_reports(self, data_id: Optional[str] = None, limit: int = 100) -> List[QualityReport]:
        """Get quality reports."""
        with self._lock:
            reports = self._reports
            if data_id:
                reports = [r for r in reports if r.data_id == data_id]
            return reports[-limit:]


class RetentionManager:
    """Manages data retention policies."""

    def __init__(self, config: Optional[LifecycleConfig] = None) -> None:
        """Initialize retention manager."""
        self._config = config or LifecycleConfig()
        self._policies: Dict[str, RetentionPolicy] = {}
        self._results: List[RetentionResult] = []
        self._lock = threading.RLock()

    def add_policy(self, policy: RetentionPolicy) -> None:
        """Add a retention policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a retention policy."""
        with self._lock:
            if policy_id in self._policies:
                del self._policies[policy_id]
                return True
            return False

    def get_policy(self, policy_id: str) -> Optional[RetentionPolicy]:
        """Get a policy by ID."""
        with self._lock:
            return self._policies.get(policy_id)

    def list_policies(self) -> List[RetentionPolicy]:
        """List all policies."""
        with self._lock:
            return list(self._policies.values())

    def evaluate_asset(self, asset: DataAsset) -> Tuple[RetentionAction, Optional[RetentionPolicy]]:
        """Evaluate which retention action applies to an asset."""
        with self._lock:
            # Sort by priority (higher first)
            sorted_policies = sorted(
                [p for p in self._policies.values() if p.enabled],
                key=lambda p: p.priority,
                reverse=True,
            )

            for policy in sorted_policies:
                # Check if conditions match
                conditions_met = True
                for key, value in policy.conditions.items():
                    asset_value = getattr(asset, key, None)
                    if asset_value is None:
                        asset_value = asset.metadata.get(key)
                    if asset_value != value:
                        conditions_met = False
                        break

                if not conditions_met:
                    continue

                # Check retention period
                age_days = (datetime.now() - asset.created_at).days
                if age_days >= policy.retention_days:
                    return (policy.action, policy)

            return (RetentionAction.KEEP, None)

    def apply_policy(
        self, asset: DataAsset, action_handler: Callable[[DataAsset, RetentionAction], bool]
    ) -> RetentionResult:
        """Apply retention policy to an asset."""
        with self._lock:
            action, policy = self.evaluate_asset(asset)

            success = False
            message = ""

            if action != RetentionAction.KEEP:
                try:
                    success = action_handler(asset, action)
                    message = f"Action {action.value} applied successfully"
                except Exception as e:
                    message = f"Error: {str(e)}"
            else:
                success = True
                message = "No action required"

            result = RetentionResult(
                result_id=str(uuid.uuid4()),
                policy_id=policy.policy_id if policy else "",
                data_id=asset.asset_id,
                action_taken=action,
                success=success,
                message=message,
            )
            self._results.append(result)
            return result

    def get_results(self, limit: int = 100) -> List[RetentionResult]:
        """Get retention results."""
        with self._lock:
            return self._results[-limit:]


class DataCatalog:
    """Data catalog for discovery and metadata management."""

    def __init__(self, config: Optional[LifecycleConfig] = None) -> None:
        """Initialize data catalog."""
        self._config = config or LifecycleConfig()
        self._entries: Dict[str, CatalogEntry] = {}
        self._lock = threading.RLock()

    def register_entry(
        self,
        name: str,
        entry_type: CatalogEntryType,
        description: str = "",
        owner: str = "",
        tags: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None,
        access_level: AccessLevel = AccessLevel.INTERNAL,
    ) -> CatalogEntry:
        """Register a catalog entry."""
        with self._lock:
            entry = CatalogEntry(
                entry_id=str(uuid.uuid4()),
                name=name,
                entry_type=entry_type,
                description=description,
                owner=owner,
                access_level=access_level,
                tags=tags or [],
                schema=schema or {},
            )
            self._entries[entry.entry_id] = entry
            return entry

    def get_entry(self, entry_id: str) -> Optional[CatalogEntry]:
        """Get an entry by ID."""
        with self._lock:
            return self._entries.get(entry_id)

    def search(
        self,
        query: Optional[str] = None,
        entry_type: Optional[CatalogEntryType] = None,
        tags: Optional[List[str]] = None,
        owner: Optional[str] = None,
        access_level: Optional[AccessLevel] = None,
    ) -> List[CatalogEntry]:
        """Search catalog entries."""
        with self._lock:
            results = list(self._entries.values())

            if query:
                query_lower = query.lower()
                results = [
                    e
                    for e in results
                    if query_lower in e.name.lower() or query_lower in e.description.lower()
                ]

            if entry_type:
                results = [e for e in results if e.entry_type == entry_type]

            if tags:
                results = [e for e in results if any(t in e.tags for t in tags)]

            if owner:
                results = [e for e in results if e.owner == owner]

            if access_level:
                results = [e for e in results if e.access_level == access_level]

            return results

    def update_statistics(
        self, entry_id: str, statistics: Dict[str, Any]
    ) -> Optional[CatalogEntry]:
        """Update entry statistics."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry:
                entry.statistics.update(statistics)
                entry.updated_at = datetime.now()
                return entry
            return None

    def add_tags(self, entry_id: str, tags: List[str]) -> Optional[CatalogEntry]:
        """Add tags to an entry."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry:
                for tag in tags:
                    if tag not in entry.tags:
                        entry.tags.append(tag)
                entry.updated_at = datetime.now()
                return entry
            return None

    def remove_entry(self, entry_id: str) -> bool:
        """Remove an entry."""
        with self._lock:
            if entry_id in self._entries:
                del self._entries[entry_id]
                return True
            return False

    def list_entries(self, limit: int = 100) -> List[CatalogEntry]:
        """List all entries."""
        with self._lock:
            return list(self._entries.values())[:limit]


class TransformationTracker:
    """Tracks data transformations."""

    def __init__(self, config: Optional[LifecycleConfig] = None) -> None:
        """Initialize transformation tracker."""
        self._config = config or LifecycleConfig()
        self._records: List[TransformationRecord] = []
        self._lock = threading.RLock()

    def record_transformation(
        self,
        transformation_type: TransformationType,
        input_ids: List[str],
        output_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        executed_by: str = "system",
        duration_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> TransformationRecord:
        """Record a transformation."""
        with self._lock:
            record = TransformationRecord(
                record_id=str(uuid.uuid4()),
                transformation_type=transformation_type,
                input_ids=input_ids,
                output_id=output_id,
                parameters=parameters or {},
                executed_by=executed_by,
                duration_ms=duration_ms,
                success=success,
                error=error,
            )
            self._records.append(record)
            return record

    def get_transformations_for_output(self, output_id: str) -> List[TransformationRecord]:
        """Get transformations that produced an output."""
        with self._lock:
            return [r for r in self._records if r.output_id == output_id]

    def get_transformations_from_input(self, input_id: str) -> List[TransformationRecord]:
        """Get transformations that used an input."""
        with self._lock:
            return [r for r in self._records if input_id in r.input_ids]

    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get transformation statistics."""
        with self._lock:
            if not self._records:
                return {
                    "total": 0,
                    "by_type": {},
                    "success_rate": None,
                    "avg_duration_ms": None,
                }

            by_type: Dict[str, int] = defaultdict(int)
            durations = []
            successes = 0

            for record in self._records:
                by_type[record.transformation_type.value] += 1
                durations.append(record.duration_ms)
                if record.success:
                    successes += 1

            return {
                "total": len(self._records),
                "by_type": dict(by_type),
                "success_rate": successes / len(self._records),
                "avg_duration_ms": sum(durations) / len(durations),
            }

    def list_records(self, limit: int = 100) -> List[TransformationRecord]:
        """List transformation records."""
        with self._lock:
            return self._records[-limit:]


class DataLifecycleHub:
    """Central hub for data lifecycle management."""

    def __init__(self, config: Optional[LifecycleConfig] = None) -> None:
        """Initialize lifecycle hub."""
        self._config = config or LifecycleConfig()
        self._version_manager = VersionManager(self._config)
        self._lineage_tracker = LineageTracker(self._config)
        self._quality_manager = QualityManager(self._config)
        self._retention_manager = RetentionManager(self._config)
        self._catalog = DataCatalog(self._config)
        self._transformation_tracker = TransformationTracker(self._config)
        self._assets: Dict[str, DataAsset] = {}
        self._lock = threading.RLock()

    @property
    def versions(self) -> VersionManager:
        """Get version manager."""
        return self._version_manager

    @property
    def lineage(self) -> LineageTracker:
        """Get lineage tracker."""
        return self._lineage_tracker

    @property
    def quality(self) -> QualityManager:
        """Get quality manager."""
        return self._quality_manager

    @property
    def retention(self) -> RetentionManager:
        """Get retention manager."""
        return self._retention_manager

    @property
    def catalog(self) -> DataCatalog:
        """Get data catalog."""
        return self._catalog

    @property
    def transformations(self) -> TransformationTracker:
        """Get transformation tracker."""
        return self._transformation_tracker

    def create_asset(
        self,
        name: str,
        owner: str = "",
        tags: Optional[List[str]] = None,
        access_level: AccessLevel = AccessLevel.INTERNAL,
        expires_in_days: Optional[int] = None,
    ) -> DataAsset:
        """Create a new data asset."""
        with self._lock:
            asset = DataAsset(
                asset_id=str(uuid.uuid4()),
                name=name,
                state=DataState.DRAFT,
                owner=owner,
                tags=tags or [],
                access_level=access_level,
                expires_at=(
                    datetime.now() + timedelta(days=expires_in_days) if expires_in_days else None
                ),
            )
            self._assets[asset.asset_id] = asset
            return asset

    def get_asset(self, asset_id: str) -> Optional[DataAsset]:
        """Get an asset by ID."""
        with self._lock:
            return self._assets.get(asset_id)

    def update_asset_state(self, asset_id: str, new_state: DataState) -> Optional[DataAsset]:
        """Update asset state."""
        with self._lock:
            asset = self._assets.get(asset_id)
            if asset:
                asset.state = new_state
                asset.updated_at = datetime.now()
                return asset
            return None

    def list_assets(
        self,
        state: Optional[DataState] = None,
        owner: Optional[str] = None,
    ) -> List[DataAsset]:
        """List assets."""
        with self._lock:
            assets = list(self._assets.values())
            if state:
                assets = [a for a in assets if a.state == state]
            if owner:
                assets = [a for a in assets if a.owner == owner]
            return assets

    def get_lifecycle_summary(self) -> Dict[str, Any]:
        """Get lifecycle management summary."""
        with self._lock:
            assets_by_state: Dict[str, int] = defaultdict(int)
            for asset in self._assets.values():
                assets_by_state[asset.state.value] += 1

            return {
                "total_assets": len(self._assets),
                "assets_by_state": dict(assets_by_state),
                "total_versions": sum(len(v) for v in self._version_manager._versions.values()),
                "lineage_nodes": len(self._lineage_tracker._nodes),
                "quality_rules": len(self._quality_manager._rules),
                "retention_policies": len(self._retention_manager._policies),
                "catalog_entries": len(self._catalog._entries),
                "transformations": len(self._transformation_tracker._records),
            }


class ManagedVisionProvider(VisionProvider):
    """Vision provider with data lifecycle management."""

    def __init__(
        self,
        base_provider: VisionProvider,
        hub: Optional[DataLifecycleHub] = None,
    ) -> None:
        """Initialize managed provider."""
        self._base_provider = base_provider
        self._hub = hub or DataLifecycleHub()
        self._request_count = 0
        self._lock = threading.RLock()

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"managed_{self._base_provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with lifecycle tracking."""
        start_time = time.time()

        # Create asset for this request
        asset = self._hub.create_asset(
            name=f"vision_request_{self._request_count}",
        )
        self._hub.update_asset_state(asset.asset_id, DataState.ACTIVE)

        # Register in lineage
        input_node = self._hub.lineage.register_node(
            name="input_image",
            node_type="raw_data",
            metadata={"size": len(image_data)},
        )

        try:
            result = await self._base_provider.analyze_image(image_data, include_description)

            # Create version for result
            if self._hub._config.enable_versioning:
                self._hub.versions.create_version(
                    asset_id=asset.asset_id,
                    data={"summary": result.summary, "confidence": result.confidence},
                    message="Analysis result",
                )

            # Register output in lineage
            output_node = self._hub.lineage.register_node(
                name="analysis_result",
                node_type="processed_data",
                metadata={"confidence": result.confidence},
            )
            self._hub.lineage.add_edge(
                input_node.node_id,
                output_node.node_id,
                LineageRelation.TRANSFORMED_FROM,
                transformation="vision_analysis",
            )

            # Track transformation
            duration_ms = (time.time() - start_time) * 1000
            self._hub.transformations.record_transformation(
                transformation_type=TransformationType.MAP,
                input_ids=[input_node.node_id],
                output_id=output_node.node_id,
                parameters={"include_description": include_description},
                duration_ms=duration_ms,
                success=True,
            )

            self._request_count += 1
            return result

        except Exception as e:
            self._hub.update_asset_state(asset.asset_id, DataState.QUARANTINED)
            raise


# ========================
# Factory Functions
# ========================


def create_lifecycle_config(
    default_retention_days: int = 365,
    archive_after_days: int = 180,
    enable_versioning: bool = True,
    max_versions: int = 100,
    enable_lineage: bool = True,
    enable_quality_checks: bool = True,
    **kwargs: Any,
) -> LifecycleConfig:
    """Create a lifecycle configuration."""
    return LifecycleConfig(
        default_retention_days=default_retention_days,
        archive_after_days=archive_after_days,
        enable_versioning=enable_versioning,
        max_versions=max_versions,
        enable_lineage=enable_lineage,
        enable_quality_checks=enable_quality_checks,
        **kwargs,
    )


def create_data_lifecycle_hub(
    default_retention_days: int = 365,
    enable_versioning: bool = True,
    enable_lineage: bool = True,
    **kwargs: Any,
) -> DataLifecycleHub:
    """Create a data lifecycle hub."""
    config = create_lifecycle_config(
        default_retention_days=default_retention_days,
        enable_versioning=enable_versioning,
        enable_lineage=enable_lineage,
        **kwargs,
    )
    return DataLifecycleHub(config)


def create_quality_rule(
    name: str,
    dimension: QualityDimension,
    check_fn: Callable[[Any], bool],
    description: str = "",
    severity: str = "warning",
) -> QualityRule:
    """Create a quality rule."""
    return QualityRule(
        rule_id=str(uuid.uuid4()),
        name=name,
        dimension=dimension,
        check_fn=check_fn,
        description=description,
        severity=severity,
    )


def create_retention_policy(
    name: str,
    retention_days: int,
    action: RetentionAction,
    conditions: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    description: str = "",
) -> RetentionPolicy:
    """Create a retention policy."""
    return RetentionPolicy(
        policy_id=str(uuid.uuid4()),
        name=name,
        retention_days=retention_days,
        action=action,
        conditions=conditions or {},
        priority=priority,
        description=description,
    )


def create_managed_provider(
    base_provider: VisionProvider,
    hub: Optional[DataLifecycleHub] = None,
) -> ManagedVisionProvider:
    """Create a managed vision provider."""
    return ManagedVisionProvider(base_provider, hub)
