"""Tests for Vision Phase 24: Advanced Data Management & Lifecycle.

This module tests the data lifecycle management capabilities including:
- Data versioning and history tracking
- Data lineage and provenance tracking
- Data quality management and validation
- Data retention policies and archival
- Data catalog and discovery
- Data transformation tracking
- Lifecycle state management
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from src.core.vision import (
    # Enums
    DataState,
    VersionType,
    LineageRelation,
    QualityDimension,
    QualityLevel,
    RetentionAction,
    CatalogEntryType,
    TransformationType,
    AccessLevel,
    # Dataclasses
    DataVersion,
    LineageNode,
    LineageEdge,
    QualityRule,
    QualityResult,
    QualityReport,
    RetentionPolicy,
    RetentionResult,
    CatalogEntry,
    TransformationRecord,
    DataAsset,
    LifecycleConfig,
    # Core classes
    VersionManager,
    LineageTracker,
    QualityManager,
    RetentionManager,
    DataCatalog,
    TransformationTracker,
    DataLifecycleHub,
    ManagedVisionProvider,
    # Factory functions
    create_lifecycle_config,
    create_data_lifecycle_hub,
    create_quality_rule,
    create_retention_policy,
    create_managed_provider,
    # Base
    VisionDescription,
    VisionProvider,
)


# ========================
# Enum Tests
# ========================


class TestDataLifecycleEnums:
    """Tests for Phase 24 enums."""

    def test_data_state_values(self) -> None:
        """Test DataState enum values."""
        assert DataState.DRAFT.value == "draft"
        assert DataState.ACTIVE.value == "active"
        assert DataState.ARCHIVED.value == "archived"
        assert DataState.DELETED.value == "deleted"
        assert DataState.EXPIRED.value == "expired"
        assert DataState.QUARANTINED.value == "quarantined"

    def test_version_type_values(self) -> None:
        """Test VersionType enum values."""
        assert VersionType.MAJOR.value == "major"
        assert VersionType.MINOR.value == "minor"
        assert VersionType.PATCH.value == "patch"
        assert VersionType.SNAPSHOT.value == "snapshot"

    def test_lineage_relation_values(self) -> None:
        """Test LineageRelation enum values."""
        assert LineageRelation.DERIVED_FROM.value == "derived_from"
        assert LineageRelation.TRANSFORMED_FROM.value == "transformed_from"
        assert LineageRelation.COPIED_FROM.value == "copied_from"
        assert LineageRelation.AGGREGATED_FROM.value == "aggregated_from"
        assert LineageRelation.JOINED_WITH.value == "joined_with"
        assert LineageRelation.FILTERED_FROM.value == "filtered_from"

    def test_quality_dimension_values(self) -> None:
        """Test QualityDimension enum values."""
        assert QualityDimension.COMPLETENESS.value == "completeness"
        assert QualityDimension.ACCURACY.value == "accuracy"
        assert QualityDimension.CONSISTENCY.value == "consistency"
        assert QualityDimension.TIMELINESS.value == "timeliness"
        assert QualityDimension.VALIDITY.value == "validity"
        assert QualityDimension.UNIQUENESS.value == "uniqueness"

    def test_quality_level_values(self) -> None:
        """Test QualityLevel enum values."""
        assert QualityLevel.EXCELLENT.value == "excellent"
        assert QualityLevel.GOOD.value == "good"
        assert QualityLevel.ACCEPTABLE.value == "acceptable"
        assert QualityLevel.POOR.value == "poor"
        assert QualityLevel.CRITICAL.value == "critical"

    def test_retention_action_values(self) -> None:
        """Test RetentionAction enum values."""
        assert RetentionAction.KEEP.value == "keep"
        assert RetentionAction.ARCHIVE.value == "archive"
        assert RetentionAction.DELETE.value == "delete"
        assert RetentionAction.ANONYMIZE.value == "anonymize"
        assert RetentionAction.COMPRESS.value == "compress"

    def test_catalog_entry_type_values(self) -> None:
        """Test CatalogEntryType enum values."""
        assert CatalogEntryType.DATASET.value == "dataset"
        assert CatalogEntryType.TABLE.value == "table"
        assert CatalogEntryType.COLUMN.value == "column"
        assert CatalogEntryType.FILE.value == "file"
        assert CatalogEntryType.API.value == "api"
        assert CatalogEntryType.MODEL.value == "model"

    def test_transformation_type_values(self) -> None:
        """Test TransformationType enum values."""
        assert TransformationType.FILTER.value == "filter"
        assert TransformationType.MAP.value == "map"
        assert TransformationType.AGGREGATE.value == "aggregate"
        assert TransformationType.JOIN.value == "join"
        assert TransformationType.SORT.value == "sort"
        assert TransformationType.DEDUPLICATE.value == "deduplicate"
        assert TransformationType.ENRICH.value == "enrich"
        assert TransformationType.NORMALIZE.value == "normalize"

    def test_access_level_values(self) -> None:
        """Test AccessLevel enum values."""
        assert AccessLevel.PUBLIC.value == "public"
        assert AccessLevel.INTERNAL.value == "internal"
        assert AccessLevel.CONFIDENTIAL.value == "confidential"
        assert AccessLevel.RESTRICTED.value == "restricted"


# ========================
# Dataclass Tests
# ========================


class TestDataLifecycleDataclasses:
    """Tests for Phase 24 dataclasses."""

    def test_data_version_creation(self) -> None:
        """Test DataVersion creation."""
        version = DataVersion(
            version_id="v1",
            version_number="1.0.0",
            version_type=VersionType.MAJOR,
            data_hash="abc123",
            message="Initial version",
            created_by="user1",
        )
        assert version.version_id == "v1"
        assert version.version_number == "1.0.0"
        assert version.version_type == VersionType.MAJOR
        assert version.data_hash == "abc123"
        assert version.message == "Initial version"
        assert version.created_by == "user1"
        assert version.is_current is False

    def test_data_version_to_dict(self) -> None:
        """Test DataVersion to_dict method."""
        version = DataVersion(
            version_id="v1",
            version_number="1.0.0",
            version_type=VersionType.MAJOR,
            data_hash="abc123",
            is_current=True,
        )
        d = version.to_dict()
        assert d["version_id"] == "v1"
        assert d["version_number"] == "1.0.0"
        assert d["version_type"] == "major"
        assert d["is_current"] is True

    def test_lineage_node_creation(self) -> None:
        """Test LineageNode creation."""
        node = LineageNode(
            node_id="n1",
            name="source_data",
            node_type="raw",
            metadata={"size": 1000},
        )
        assert node.node_id == "n1"
        assert node.name == "source_data"
        assert node.node_type == "raw"
        assert node.metadata["size"] == 1000

    def test_lineage_edge_creation(self) -> None:
        """Test LineageEdge creation."""
        edge = LineageEdge(
            edge_id="e1",
            source_id="n1",
            target_id="n2",
            relation=LineageRelation.DERIVED_FROM,
            transformation="filter",
        )
        assert edge.edge_id == "e1"
        assert edge.source_id == "n1"
        assert edge.target_id == "n2"
        assert edge.relation == LineageRelation.DERIVED_FROM
        assert edge.transformation == "filter"

    def test_quality_rule_creation(self) -> None:
        """Test QualityRule creation."""
        rule = QualityRule(
            rule_id="r1",
            name="not_null",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda x: x is not None,
            description="Value must not be null",
            severity="error",
        )
        assert rule.rule_id == "r1"
        assert rule.name == "not_null"
        assert rule.dimension == QualityDimension.COMPLETENESS
        assert rule.check_fn("test") is True
        assert rule.check_fn(None) is False

    def test_quality_result_creation(self) -> None:
        """Test QualityResult creation."""
        result = QualityResult(
            result_id="res1",
            rule_id="r1",
            dimension=QualityDimension.COMPLETENESS,
            passed=True,
            score=1.0,
            message="Passed",
        )
        assert result.result_id == "res1"
        assert result.passed is True
        assert result.score == 1.0

    def test_quality_report_creation(self) -> None:
        """Test QualityReport creation."""
        report = QualityReport(
            report_id="rep1",
            data_id="d1",
            overall_score=0.85,
            overall_level=QualityLevel.GOOD,
            results=[],
            recommendations=["Fix nulls"],
        )
        assert report.report_id == "rep1"
        assert report.overall_score == 0.85
        assert report.overall_level == QualityLevel.GOOD

    def test_retention_policy_creation(self) -> None:
        """Test RetentionPolicy creation."""
        policy = RetentionPolicy(
            policy_id="p1",
            name="30_day_archive",
            retention_days=30,
            action=RetentionAction.ARCHIVE,
            conditions={"state": "draft"},
            priority=10,
        )
        assert policy.policy_id == "p1"
        assert policy.retention_days == 30
        assert policy.action == RetentionAction.ARCHIVE
        assert policy.priority == 10

    def test_retention_result_creation(self) -> None:
        """Test RetentionResult creation."""
        result = RetentionResult(
            result_id="rr1",
            policy_id="p1",
            data_id="d1",
            action_taken=RetentionAction.ARCHIVE,
            success=True,
            message="Archived successfully",
        )
        assert result.result_id == "rr1"
        assert result.action_taken == RetentionAction.ARCHIVE
        assert result.success is True

    def test_catalog_entry_creation(self) -> None:
        """Test CatalogEntry creation."""
        entry = CatalogEntry(
            entry_id="c1",
            name="sales_data",
            entry_type=CatalogEntryType.DATASET,
            description="Sales dataset",
            owner="analytics",
            access_level=AccessLevel.INTERNAL,
            tags=["sales", "daily"],
        )
        assert entry.entry_id == "c1"
        assert entry.name == "sales_data"
        assert entry.entry_type == CatalogEntryType.DATASET
        assert entry.access_level == AccessLevel.INTERNAL
        assert "sales" in entry.tags

    def test_transformation_record_creation(self) -> None:
        """Test TransformationRecord creation."""
        record = TransformationRecord(
            record_id="t1",
            transformation_type=TransformationType.FILTER,
            input_ids=["i1", "i2"],
            output_id="o1",
            parameters={"condition": "value > 10"},
            duration_ms=150.5,
            success=True,
        )
        assert record.record_id == "t1"
        assert record.transformation_type == TransformationType.FILTER
        assert len(record.input_ids) == 2
        assert record.duration_ms == 150.5

    def test_data_asset_creation(self) -> None:
        """Test DataAsset creation."""
        asset = DataAsset(
            asset_id="a1",
            name="my_data",
            state=DataState.ACTIVE,
            owner="user1",
            access_level=AccessLevel.CONFIDENTIAL,
            tags=["pii", "encrypted"],
        )
        assert asset.asset_id == "a1"
        assert asset.state == DataState.ACTIVE
        assert asset.access_level == AccessLevel.CONFIDENTIAL

    def test_lifecycle_config_defaults(self) -> None:
        """Test LifecycleConfig default values."""
        config = LifecycleConfig()
        assert config.default_retention_days == 365
        assert config.archive_after_days == 180
        assert config.enable_versioning is True
        assert config.max_versions == 100
        assert config.enable_lineage is True
        assert config.enable_quality_checks is True


# ========================
# VersionManager Tests
# ========================


class TestVersionManager:
    """Tests for VersionManager class."""

    def test_create_first_version(self) -> None:
        """Test creating first version."""
        manager = VersionManager()
        version = manager.create_version(
            asset_id="asset1",
            data={"value": 42},
            version_type=VersionType.MAJOR,
            message="Initial",
        )
        assert version.version_number == "1.0.0"
        assert version.is_current is True
        assert version.parent_version is None

    def test_create_minor_version(self) -> None:
        """Test creating minor version."""
        manager = VersionManager()
        manager.create_version("asset1", {"v": 1})
        v2 = manager.create_version(
            "asset1", {"v": 2}, version_type=VersionType.MINOR
        )
        assert v2.version_number == "1.1.0"

    def test_create_patch_version(self) -> None:
        """Test creating patch version."""
        manager = VersionManager()
        manager.create_version("asset1", {"v": 1})
        v2 = manager.create_version(
            "asset1", {"v": 2}, version_type=VersionType.PATCH
        )
        assert v2.version_number == "1.0.1"

    def test_create_major_version_increment(self) -> None:
        """Test major version increment resets minor and patch."""
        manager = VersionManager()
        manager.create_version("asset1", {"v": 1})
        manager.create_version("asset1", {"v": 2}, version_type=VersionType.MINOR)
        v3 = manager.create_version(
            "asset1", {"v": 3}, version_type=VersionType.MAJOR
        )
        assert v3.version_number == "2.0.0"

    def test_get_current_version(self) -> None:
        """Test getting current version."""
        manager = VersionManager()
        v1 = manager.create_version("asset1", {"v": 1})
        manager.create_version("asset1", {"v": 2})
        current = manager.get_current_version("asset1")
        assert current is not None
        assert current.version_number == "1.1.0"
        assert current.is_current is True

    def test_get_version_history(self) -> None:
        """Test getting version history."""
        manager = VersionManager()
        for i in range(5):
            manager.create_version("asset1", {"v": i})
        history = manager.get_version_history("asset1")
        assert len(history) == 5

    def test_rollback_to_version(self) -> None:
        """Test rollback to previous version."""
        manager = VersionManager()
        v1 = manager.create_version("asset1", {"v": 1})
        manager.create_version("asset1", {"v": 2})
        manager.create_version("asset1", {"v": 3})

        rollback = manager.rollback_to_version("asset1", v1.version_id)
        assert rollback is not None
        assert "Rollback to 1.0.0" in rollback.message
        assert rollback.data_hash == v1.data_hash

    def test_compare_versions(self) -> None:
        """Test comparing two versions."""
        manager = VersionManager()
        v1 = manager.create_version("asset1", {"v": 1})
        v2 = manager.create_version("asset1", {"v": 1})  # Same data

        comparison = manager.compare_versions("asset1", v1.version_id, v2.version_id)
        assert comparison["hash_match"] is True
        assert "time_delta" in comparison

    def test_max_versions_enforcement(self) -> None:
        """Test max versions limit enforcement."""
        config = LifecycleConfig(max_versions=5)
        manager = VersionManager(config)

        for i in range(10):
            manager.create_version("asset1", {"v": i})

        history = manager.get_version_history("asset1")
        assert len(history) == 5


# ========================
# LineageTracker Tests
# ========================


class TestLineageTracker:
    """Tests for LineageTracker class."""

    def test_register_node(self) -> None:
        """Test registering a lineage node."""
        tracker = LineageTracker()
        node = tracker.register_node(
            name="source_data",
            node_type="raw",
            metadata={"format": "csv"},
        )
        assert node.name == "source_data"
        assert node.node_type == "raw"

    def test_add_edge(self) -> None:
        """Test adding a lineage edge."""
        tracker = LineageTracker()
        n1 = tracker.register_node("source", "raw")
        n2 = tracker.register_node("target", "processed")

        edge = tracker.add_edge(
            n1.node_id,
            n2.node_id,
            LineageRelation.TRANSFORMED_FROM,
            transformation="filter",
        )
        assert edge is not None
        assert edge.relation == LineageRelation.TRANSFORMED_FROM

    def test_add_edge_invalid_nodes(self) -> None:
        """Test adding edge with invalid nodes returns None."""
        tracker = LineageTracker()
        edge = tracker.add_edge("invalid1", "invalid2", LineageRelation.DERIVED_FROM)
        assert edge is None

    def test_get_upstream(self) -> None:
        """Test getting upstream nodes."""
        tracker = LineageTracker()
        n1 = tracker.register_node("source1", "raw")
        n2 = tracker.register_node("source2", "raw")
        n3 = tracker.register_node("intermediate", "processed")
        n4 = tracker.register_node("final", "output")

        tracker.add_edge(n1.node_id, n3.node_id, LineageRelation.DERIVED_FROM)
        tracker.add_edge(n2.node_id, n3.node_id, LineageRelation.DERIVED_FROM)
        tracker.add_edge(n3.node_id, n4.node_id, LineageRelation.TRANSFORMED_FROM)

        upstream = tracker.get_upstream(n4.node_id)
        assert len(upstream) == 3  # n3, n1, n2

    def test_get_downstream(self) -> None:
        """Test getting downstream nodes."""
        tracker = LineageTracker()
        n1 = tracker.register_node("source", "raw")
        n2 = tracker.register_node("processed1", "processed")
        n3 = tracker.register_node("processed2", "processed")

        tracker.add_edge(n1.node_id, n2.node_id, LineageRelation.DERIVED_FROM)
        tracker.add_edge(n1.node_id, n3.node_id, LineageRelation.DERIVED_FROM)

        downstream = tracker.get_downstream(n1.node_id)
        assert len(downstream) == 2

    def test_get_lineage_graph(self) -> None:
        """Test getting complete lineage graph."""
        tracker = LineageTracker()
        n1 = tracker.register_node("source", "raw")
        n2 = tracker.register_node("middle", "processed")
        n3 = tracker.register_node("output", "final")

        tracker.add_edge(n1.node_id, n2.node_id, LineageRelation.DERIVED_FROM)
        tracker.add_edge(n2.node_id, n3.node_id, LineageRelation.TRANSFORMED_FROM)

        graph = tracker.get_lineage_graph(n2.node_id)
        assert graph["center"] == n2
        assert len(graph["upstream"]) == 1
        assert len(graph["downstream"]) == 1

    def test_list_nodes(self) -> None:
        """Test listing all nodes."""
        tracker = LineageTracker()
        tracker.register_node("n1", "type1")
        tracker.register_node("n2", "type2")

        nodes = tracker.list_nodes()
        assert len(nodes) == 2


# ========================
# QualityManager Tests
# ========================


class TestQualityManager:
    """Tests for QualityManager class."""

    def test_add_rule(self) -> None:
        """Test adding a quality rule."""
        manager = QualityManager()
        rule = QualityRule(
            rule_id="r1",
            name="not_empty",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda x: len(x) > 0 if x else False,
        )
        manager.add_rule(rule)
        assert manager.get_rule("r1") is not None

    def test_remove_rule(self) -> None:
        """Test removing a quality rule."""
        manager = QualityManager()
        rule = QualityRule(
            rule_id="r1",
            name="test",
            dimension=QualityDimension.ACCURACY,
            check_fn=lambda x: True,
        )
        manager.add_rule(rule)
        assert manager.remove_rule("r1") is True
        assert manager.get_rule("r1") is None

    def test_list_rules_by_dimension(self) -> None:
        """Test listing rules by dimension."""
        manager = QualityManager()
        manager.add_rule(QualityRule(
            rule_id="r1", name="test1",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda x: True,
        ))
        manager.add_rule(QualityRule(
            rule_id="r2", name="test2",
            dimension=QualityDimension.ACCURACY,
            check_fn=lambda x: True,
        ))

        completeness_rules = manager.list_rules(QualityDimension.COMPLETENESS)
        assert len(completeness_rules) == 1

    def test_check_quality_all_pass(self) -> None:
        """Test quality check when all rules pass."""
        manager = QualityManager()
        manager.add_rule(QualityRule(
            rule_id="r1", name="always_pass",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda x: True,
        ))

        report = manager.check_quality("data1", {"test": "value"})
        assert report.overall_score == 1.0
        assert report.overall_level == QualityLevel.EXCELLENT

    def test_check_quality_some_fail(self) -> None:
        """Test quality check when some rules fail."""
        manager = QualityManager()
        manager.add_rule(QualityRule(
            rule_id="r1", name="pass",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda x: True,
        ))
        manager.add_rule(QualityRule(
            rule_id="r2", name="fail",
            dimension=QualityDimension.ACCURACY,
            check_fn=lambda x: False,
            description="Accuracy check",
        ))

        report = manager.check_quality("data1", {"test": "value"})
        assert report.overall_score == 0.5
        assert report.overall_level == QualityLevel.ACCEPTABLE
        assert len(report.recommendations) > 0

    def test_check_quality_levels(self) -> None:
        """Test different quality levels based on scores."""
        manager = QualityManager()

        # Test EXCELLENT (>= 0.9)
        manager.add_rule(QualityRule(
            rule_id="r1", name="pass",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda x: True,
        ))
        report = manager.check_quality("d1", {})
        assert report.overall_level == QualityLevel.EXCELLENT

    def test_get_reports(self) -> None:
        """Test getting quality reports."""
        manager = QualityManager()
        manager.add_rule(QualityRule(
            rule_id="r1", name="test",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda x: True,
        ))

        manager.check_quality("data1", {})
        manager.check_quality("data2", {})

        reports = manager.get_reports()
        assert len(reports) == 2

        data1_reports = manager.get_reports(data_id="data1")
        assert len(data1_reports) == 1


# ========================
# RetentionManager Tests
# ========================


class TestRetentionManager:
    """Tests for RetentionManager class."""

    def test_add_policy(self) -> None:
        """Test adding a retention policy."""
        manager = RetentionManager()
        policy = RetentionPolicy(
            policy_id="p1",
            name="30_day",
            retention_days=30,
            action=RetentionAction.ARCHIVE,
        )
        manager.add_policy(policy)
        assert manager.get_policy("p1") is not None

    def test_remove_policy(self) -> None:
        """Test removing a retention policy."""
        manager = RetentionManager()
        policy = RetentionPolicy(
            policy_id="p1",
            name="test",
            retention_days=30,
            action=RetentionAction.DELETE,
        )
        manager.add_policy(policy)
        assert manager.remove_policy("p1") is True
        assert manager.get_policy("p1") is None

    def test_list_policies(self) -> None:
        """Test listing all policies."""
        manager = RetentionManager()
        manager.add_policy(RetentionPolicy(
            policy_id="p1", name="test1",
            retention_days=30, action=RetentionAction.ARCHIVE,
        ))
        manager.add_policy(RetentionPolicy(
            policy_id="p2", name="test2",
            retention_days=60, action=RetentionAction.DELETE,
        ))

        policies = manager.list_policies()
        assert len(policies) == 2

    def test_evaluate_asset_keep(self) -> None:
        """Test evaluating asset that should be kept."""
        manager = RetentionManager()
        manager.add_policy(RetentionPolicy(
            policy_id="p1", name="30_day",
            retention_days=30, action=RetentionAction.DELETE,
        ))

        # Asset created now should be kept
        asset = DataAsset(
            asset_id="a1", name="test",
            state=DataState.ACTIVE,
        )
        action, policy = manager.evaluate_asset(asset)
        assert action == RetentionAction.KEEP
        assert policy is None

    def test_evaluate_asset_action_needed(self) -> None:
        """Test evaluating asset that needs action."""
        manager = RetentionManager()
        manager.add_policy(RetentionPolicy(
            policy_id="p1", name="30_day",
            retention_days=30, action=RetentionAction.ARCHIVE,
            enabled=True,
        ))

        # Asset created 60 days ago
        asset = DataAsset(
            asset_id="a1", name="test",
            state=DataState.ACTIVE,
            created_at=datetime.now() - timedelta(days=60),
        )
        action, policy = manager.evaluate_asset(asset)
        assert action == RetentionAction.ARCHIVE
        assert policy is not None

    def test_evaluate_asset_with_conditions(self) -> None:
        """Test evaluating asset with policy conditions."""
        manager = RetentionManager()
        manager.add_policy(RetentionPolicy(
            policy_id="p1", name="draft_cleanup",
            retention_days=7, action=RetentionAction.DELETE,
            conditions={"state": DataState.DRAFT},
        ))

        # Draft asset older than 7 days
        draft_asset = DataAsset(
            asset_id="a1", name="test",
            state=DataState.DRAFT,
            created_at=datetime.now() - timedelta(days=10),
        )
        action, _ = manager.evaluate_asset(draft_asset)
        assert action == RetentionAction.DELETE

        # Active asset should not match
        active_asset = DataAsset(
            asset_id="a2", name="test",
            state=DataState.ACTIVE,
            created_at=datetime.now() - timedelta(days=10),
        )
        action, _ = manager.evaluate_asset(active_asset)
        assert action == RetentionAction.KEEP

    def test_apply_policy(self) -> None:
        """Test applying retention policy."""
        manager = RetentionManager()
        manager.add_policy(RetentionPolicy(
            policy_id="p1", name="archive",
            retention_days=30, action=RetentionAction.ARCHIVE,
        ))

        asset = DataAsset(
            asset_id="a1", name="test",
            state=DataState.ACTIVE,
            created_at=datetime.now() - timedelta(days=60),
        )

        handler = MagicMock(return_value=True)
        result = manager.apply_policy(asset, handler)

        assert result.success is True
        assert result.action_taken == RetentionAction.ARCHIVE


# ========================
# DataCatalog Tests
# ========================


class TestDataCatalog:
    """Tests for DataCatalog class."""

    def test_register_entry(self) -> None:
        """Test registering a catalog entry."""
        catalog = DataCatalog()
        entry = catalog.register_entry(
            name="sales_data",
            entry_type=CatalogEntryType.DATASET,
            description="Sales dataset",
            owner="analytics",
            tags=["sales", "daily"],
        )
        assert entry.name == "sales_data"
        assert entry.entry_type == CatalogEntryType.DATASET

    def test_get_entry(self) -> None:
        """Test getting an entry by ID."""
        catalog = DataCatalog()
        entry = catalog.register_entry("test", CatalogEntryType.TABLE)
        retrieved = catalog.get_entry(entry.entry_id)
        assert retrieved == entry

    def test_search_by_query(self) -> None:
        """Test searching by text query."""
        catalog = DataCatalog()
        catalog.register_entry("sales_data", CatalogEntryType.DATASET, description="Daily sales")
        catalog.register_entry("user_data", CatalogEntryType.DATASET, description="User profiles")

        results = catalog.search(query="sales")
        assert len(results) == 1
        assert results[0].name == "sales_data"

    def test_search_by_type(self) -> None:
        """Test searching by entry type."""
        catalog = DataCatalog()
        catalog.register_entry("table1", CatalogEntryType.TABLE)
        catalog.register_entry("dataset1", CatalogEntryType.DATASET)

        results = catalog.search(entry_type=CatalogEntryType.TABLE)
        assert len(results) == 1

    def test_search_by_tags(self) -> None:
        """Test searching by tags."""
        catalog = DataCatalog()
        catalog.register_entry("data1", CatalogEntryType.DATASET, tags=["pii", "sensitive"])
        catalog.register_entry("data2", CatalogEntryType.DATASET, tags=["public"])

        results = catalog.search(tags=["pii"])
        assert len(results) == 1

    def test_search_by_owner(self) -> None:
        """Test searching by owner."""
        catalog = DataCatalog()
        catalog.register_entry("data1", CatalogEntryType.DATASET, owner="team_a")
        catalog.register_entry("data2", CatalogEntryType.DATASET, owner="team_b")

        results = catalog.search(owner="team_a")
        assert len(results) == 1

    def test_search_by_access_level(self) -> None:
        """Test searching by access level."""
        catalog = DataCatalog()
        catalog.register_entry("public_data", CatalogEntryType.DATASET, access_level=AccessLevel.PUBLIC)
        catalog.register_entry("private_data", CatalogEntryType.DATASET, access_level=AccessLevel.RESTRICTED)

        results = catalog.search(access_level=AccessLevel.PUBLIC)
        assert len(results) == 1

    def test_update_statistics(self) -> None:
        """Test updating entry statistics."""
        catalog = DataCatalog()
        entry = catalog.register_entry("data1", CatalogEntryType.DATASET)

        updated = catalog.update_statistics(entry.entry_id, {"row_count": 1000})
        assert updated is not None
        assert updated.statistics["row_count"] == 1000

    def test_add_tags(self) -> None:
        """Test adding tags to an entry."""
        catalog = DataCatalog()
        entry = catalog.register_entry("data1", CatalogEntryType.DATASET, tags=["existing"])

        updated = catalog.add_tags(entry.entry_id, ["new_tag"])
        assert updated is not None
        assert "new_tag" in updated.tags
        assert "existing" in updated.tags

    def test_remove_entry(self) -> None:
        """Test removing an entry."""
        catalog = DataCatalog()
        entry = catalog.register_entry("data1", CatalogEntryType.DATASET)

        assert catalog.remove_entry(entry.entry_id) is True
        assert catalog.get_entry(entry.entry_id) is None


# ========================
# TransformationTracker Tests
# ========================


class TestTransformationTracker:
    """Tests for TransformationTracker class."""

    def test_record_transformation(self) -> None:
        """Test recording a transformation."""
        tracker = TransformationTracker()
        record = tracker.record_transformation(
            transformation_type=TransformationType.FILTER,
            input_ids=["i1", "i2"],
            output_id="o1",
            parameters={"condition": "x > 10"},
            duration_ms=100.5,
        )
        assert record.transformation_type == TransformationType.FILTER
        assert len(record.input_ids) == 2
        assert record.success is True

    def test_get_transformations_for_output(self) -> None:
        """Test getting transformations that produced an output."""
        tracker = TransformationTracker()
        tracker.record_transformation(TransformationType.FILTER, ["i1"], "o1")
        tracker.record_transformation(TransformationType.MAP, ["i2"], "o2")

        results = tracker.get_transformations_for_output("o1")
        assert len(results) == 1

    def test_get_transformations_from_input(self) -> None:
        """Test getting transformations that used an input."""
        tracker = TransformationTracker()
        tracker.record_transformation(TransformationType.FILTER, ["i1"], "o1")
        tracker.record_transformation(TransformationType.MAP, ["i1", "i2"], "o2")

        results = tracker.get_transformations_from_input("i1")
        assert len(results) == 2

    def test_get_transformation_stats(self) -> None:
        """Test getting transformation statistics."""
        tracker = TransformationTracker()
        tracker.record_transformation(
            TransformationType.FILTER, ["i1"], "o1",
            duration_ms=100, success=True
        )
        tracker.record_transformation(
            TransformationType.FILTER, ["i2"], "o2",
            duration_ms=200, success=True
        )
        tracker.record_transformation(
            TransformationType.MAP, ["i3"], "o3",
            duration_ms=150, success=False
        )

        stats = tracker.get_transformation_stats()
        assert stats["total"] == 3
        assert stats["by_type"]["filter"] == 2
        assert stats["by_type"]["map"] == 1
        assert stats["success_rate"] == 2/3
        assert stats["avg_duration_ms"] == 150

    def test_get_transformation_stats_empty(self) -> None:
        """Test getting stats with no transformations."""
        tracker = TransformationTracker()
        stats = tracker.get_transformation_stats()
        assert stats["total"] == 0
        assert stats["success_rate"] is None


# ========================
# DataLifecycleHub Tests
# ========================


class TestDataLifecycleHub:
    """Tests for DataLifecycleHub class."""

    def test_initialization(self) -> None:
        """Test hub initialization."""
        hub = DataLifecycleHub()
        assert hub.versions is not None
        assert hub.lineage is not None
        assert hub.quality is not None
        assert hub.retention is not None
        assert hub.catalog is not None
        assert hub.transformations is not None

    def test_create_asset(self) -> None:
        """Test creating a data asset."""
        hub = DataLifecycleHub()
        asset = hub.create_asset(
            name="test_data",
            owner="team_a",
            tags=["test"],
            access_level=AccessLevel.INTERNAL,
        )
        assert asset.name == "test_data"
        assert asset.state == DataState.DRAFT

    def test_create_asset_with_expiry(self) -> None:
        """Test creating an asset with expiration."""
        hub = DataLifecycleHub()
        asset = hub.create_asset("temp_data", expires_in_days=30)
        assert asset.expires_at is not None

    def test_get_asset(self) -> None:
        """Test getting an asset by ID."""
        hub = DataLifecycleHub()
        created = hub.create_asset("test")
        retrieved = hub.get_asset(created.asset_id)
        assert retrieved == created

    def test_update_asset_state(self) -> None:
        """Test updating asset state."""
        hub = DataLifecycleHub()
        asset = hub.create_asset("test")

        updated = hub.update_asset_state(asset.asset_id, DataState.ACTIVE)
        assert updated is not None
        assert updated.state == DataState.ACTIVE

    def test_list_assets(self) -> None:
        """Test listing assets."""
        hub = DataLifecycleHub()
        hub.create_asset("asset1", owner="user1")
        asset2 = hub.create_asset("asset2", owner="user2")
        hub.update_asset_state(asset2.asset_id, DataState.ACTIVE)

        all_assets = hub.list_assets()
        assert len(all_assets) == 2

        active_only = hub.list_assets(state=DataState.ACTIVE)
        assert len(active_only) == 1

        user1_assets = hub.list_assets(owner="user1")
        assert len(user1_assets) == 1

    def test_get_lifecycle_summary(self) -> None:
        """Test getting lifecycle summary."""
        hub = DataLifecycleHub()

        # Create some assets
        hub.create_asset("asset1")
        asset2 = hub.create_asset("asset2")
        hub.update_asset_state(asset2.asset_id, DataState.ACTIVE)

        # Add versions
        hub.versions.create_version("asset1", {"v": 1})

        # Add lineage nodes
        hub.lineage.register_node("node1", "raw")

        # Add quality rules
        hub.quality.add_rule(QualityRule(
            rule_id="r1", name="test",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda x: True,
        ))

        summary = hub.get_lifecycle_summary()
        assert summary["total_assets"] == 2
        assert summary["assets_by_state"]["draft"] == 1
        assert summary["assets_by_state"]["active"] == 1
        assert summary["total_versions"] == 1
        assert summary["lineage_nodes"] == 1
        assert summary["quality_rules"] == 1

    def test_hub_with_custom_config(self) -> None:
        """Test hub with custom configuration."""
        config = LifecycleConfig(
            max_versions=50,
            enable_versioning=True,
            enable_lineage=False,
        )
        hub = DataLifecycleHub(config)
        assert hub._config.max_versions == 50


# ========================
# ManagedVisionProvider Tests
# ========================


class TestManagedVisionProvider:
    """Tests for ManagedVisionProvider class."""

    def test_provider_name(self) -> None:
        """Test provider name includes 'managed' prefix."""
        mock_provider = MagicMock(spec=VisionProvider)
        mock_provider.provider_name = "test_provider"

        managed = ManagedVisionProvider(mock_provider)
        assert managed.provider_name == "managed_test_provider"

    @pytest.mark.asyncio
    async def test_analyze_image_creates_asset(self) -> None:
        """Test that analyze_image creates a data asset."""
        mock_provider = MagicMock(spec=VisionProvider)
        mock_provider.provider_name = "test"
        mock_provider.analyze_image = AsyncMock(
            return_value=VisionDescription(
                summary="Test analysis",
                confidence=0.95,
            )
        )

        hub = DataLifecycleHub()
        managed = ManagedVisionProvider(mock_provider, hub)

        result = await managed.analyze_image(b"test_image_data")

        assert result.summary == "Test analysis"
        assert result.confidence == 0.95

        # Verify asset was created
        assets = hub.list_assets()
        assert len(assets) == 1
        assert assets[0].state == DataState.ACTIVE

    @pytest.mark.asyncio
    async def test_analyze_image_creates_version(self) -> None:
        """Test that analyze_image creates a version."""
        mock_provider = MagicMock(spec=VisionProvider)
        mock_provider.provider_name = "test"
        mock_provider.analyze_image = AsyncMock(
            return_value=VisionDescription(
                summary="Test",
                confidence=0.9,
            )
        )

        hub = DataLifecycleHub()
        managed = ManagedVisionProvider(mock_provider, hub)

        await managed.analyze_image(b"test")

        # Check versions were created
        summary = hub.get_lifecycle_summary()
        assert summary["total_versions"] == 1

    @pytest.mark.asyncio
    async def test_analyze_image_tracks_lineage(self) -> None:
        """Test that analyze_image tracks lineage."""
        mock_provider = MagicMock(spec=VisionProvider)
        mock_provider.provider_name = "test"
        mock_provider.analyze_image = AsyncMock(
            return_value=VisionDescription(
                summary="Test",
                confidence=0.9,
            )
        )

        hub = DataLifecycleHub()
        managed = ManagedVisionProvider(mock_provider, hub)

        await managed.analyze_image(b"test")

        # Check lineage nodes were created
        nodes = hub.lineage.list_nodes()
        assert len(nodes) == 2  # input and output nodes

    @pytest.mark.asyncio
    async def test_analyze_image_records_transformation(self) -> None:
        """Test that analyze_image records transformation."""
        mock_provider = MagicMock(spec=VisionProvider)
        mock_provider.provider_name = "test"
        mock_provider.analyze_image = AsyncMock(
            return_value=VisionDescription(
                summary="Test",
                confidence=0.9,
            )
        )

        hub = DataLifecycleHub()
        managed = ManagedVisionProvider(mock_provider, hub)

        await managed.analyze_image(b"test")

        # Check transformation was recorded
        records = hub.transformations.list_records()
        assert len(records) == 1
        assert records[0].transformation_type == TransformationType.MAP

    @pytest.mark.asyncio
    async def test_analyze_image_quarantines_on_error(self) -> None:
        """Test that analyze_image quarantines asset on error."""
        mock_provider = MagicMock(spec=VisionProvider)
        mock_provider.provider_name = "test"
        mock_provider.analyze_image = AsyncMock(side_effect=Exception("API Error"))

        hub = DataLifecycleHub()
        managed = ManagedVisionProvider(mock_provider, hub)

        with pytest.raises(Exception, match="API Error"):
            await managed.analyze_image(b"test")

        # Verify asset was quarantined
        assets = hub.list_assets(state=DataState.QUARANTINED)
        assert len(assets) == 1


# ========================
# Factory Function Tests
# ========================


class TestFactoryFunctions:
    """Tests for Phase 24 factory functions."""

    def test_create_lifecycle_config(self) -> None:
        """Test create_lifecycle_config factory."""
        config = create_lifecycle_config(
            default_retention_days=180,
            max_versions=50,
            enable_lineage=False,
        )
        assert config.default_retention_days == 180
        assert config.max_versions == 50
        assert config.enable_lineage is False

    def test_create_data_lifecycle_hub(self) -> None:
        """Test create_data_lifecycle_hub factory."""
        hub = create_data_lifecycle_hub(
            default_retention_days=90,
            enable_versioning=True,
        )
        assert hub._config.default_retention_days == 90
        assert hub._config.enable_versioning is True

    def test_create_quality_rule(self) -> None:
        """Test create_quality_rule factory."""
        rule = create_quality_rule(
            name="not_null",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda x: x is not None,
            description="Value must not be null",
            severity="error",
        )
        assert rule.name == "not_null"
        assert rule.dimension == QualityDimension.COMPLETENESS
        assert rule.rule_id is not None  # Auto-generated

    def test_create_retention_policy(self) -> None:
        """Test create_retention_policy factory."""
        policy = create_retention_policy(
            name="30_day_cleanup",
            retention_days=30,
            action=RetentionAction.DELETE,
            conditions={"state": "draft"},
            priority=10,
        )
        assert policy.name == "30_day_cleanup"
        assert policy.retention_days == 30
        assert policy.action == RetentionAction.DELETE
        assert policy.policy_id is not None  # Auto-generated

    def test_create_managed_provider(self) -> None:
        """Test create_managed_provider factory."""
        mock_provider = MagicMock(spec=VisionProvider)
        mock_provider.provider_name = "test"

        managed = create_managed_provider(mock_provider)
        assert managed.provider_name == "managed_test"


# ========================
# Integration Tests
# ========================


class TestDataLifecycleIntegration:
    """Integration tests for Phase 24 components."""

    def test_complete_data_lifecycle(self) -> None:
        """Test complete data lifecycle from creation to archival."""
        hub = DataLifecycleHub()

        # 1. Create asset
        asset = hub.create_asset(
            name="sales_report",
            owner="analytics",
            tags=["sales", "monthly"],
        )
        assert asset.state == DataState.DRAFT

        # 2. Add to catalog
        catalog_entry = hub.catalog.register_entry(
            name=asset.name,
            entry_type=CatalogEntryType.DATASET,
            owner=asset.owner,
            tags=asset.tags,
        )

        # 3. Create version
        version = hub.versions.create_version(
            asset_id=asset.asset_id,
            data={"period": "2024-01", "total": 1000000},
            message="Initial version",
        )
        asset.current_version = version.version_id

        # 4. Activate asset
        hub.update_asset_state(asset.asset_id, DataState.ACTIVE)

        # 5. Add quality rule and check
        hub.quality.add_rule(QualityRule(
            rule_id="r1",
            name="has_total",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda x: "total" in x if isinstance(x, dict) else False,
        ))
        report = hub.quality.check_quality(
            asset.asset_id,
            {"period": "2024-01", "total": 1000000},
        )
        assert report.overall_level == QualityLevel.EXCELLENT

        # 6. Track lineage
        source = hub.lineage.register_node("raw_sales", "raw")
        processed = hub.lineage.register_node("sales_report", "processed")
        hub.lineage.add_edge(
            source.node_id,
            processed.node_id,
            LineageRelation.AGGREGATED_FROM,
        )

        # 7. Record transformation
        hub.transformations.record_transformation(
            TransformationType.AGGREGATE,
            input_ids=[source.node_id],
            output_id=processed.node_id,
            parameters={"groupby": "month"},
        )

        # Verify lifecycle summary
        summary = hub.get_lifecycle_summary()
        assert summary["total_assets"] == 1
        assert summary["total_versions"] == 1
        assert summary["lineage_nodes"] == 2
        assert summary["transformations"] == 1

    def test_version_lineage_integration(self) -> None:
        """Test integration between versioning and lineage."""
        hub = DataLifecycleHub()

        # Create asset and initial version
        asset = hub.create_asset("data")
        v1 = hub.versions.create_version(asset.asset_id, {"v": 1}, message="v1")

        # Create lineage for v1
        node_v1 = hub.lineage.register_node("data_v1", "version")

        # Create new version
        v2 = hub.versions.create_version(asset.asset_id, {"v": 2}, message="v2")
        node_v2 = hub.lineage.register_node("data_v2", "version")

        # Link versions in lineage
        hub.lineage.add_edge(
            node_v1.node_id,
            node_v2.node_id,
            LineageRelation.DERIVED_FROM,
            transformation="update",
        )

        # Verify lineage
        upstream = hub.lineage.get_upstream(node_v2.node_id)
        assert len(upstream) == 1
        assert upstream[0].name == "data_v1"

    def test_quality_retention_integration(self) -> None:
        """Test integration between quality and retention."""
        hub = DataLifecycleHub()

        # Add retention policy for poor quality data
        hub.retention.add_policy(RetentionPolicy(
            policy_id="poor_quality_cleanup",
            name="Poor Quality Cleanup",
            retention_days=0,
            action=RetentionAction.QUARANTINED if hasattr(RetentionAction, 'QUARANTINED') else RetentionAction.DELETE,
            conditions={},
        ))

        # Add quality rule
        hub.quality.add_rule(QualityRule(
            rule_id="r1",
            name="critical_check",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda x: False,  # Always fails
        ))

        # Check quality
        report = hub.quality.check_quality("data1", {})
        assert report.overall_level == QualityLevel.CRITICAL

    def test_catalog_search_integration(self) -> None:
        """Test catalog search with multiple criteria."""
        hub = DataLifecycleHub()

        # Register multiple entries
        hub.catalog.register_entry(
            "sales_data",
            CatalogEntryType.DATASET,
            owner="analytics",
            tags=["sales", "daily"],
            access_level=AccessLevel.INTERNAL,
        )
        hub.catalog.register_entry(
            "user_data",
            CatalogEntryType.DATASET,
            owner="analytics",
            tags=["pii", "users"],
            access_level=AccessLevel.RESTRICTED,
        )
        hub.catalog.register_entry(
            "public_metrics",
            CatalogEntryType.DATASET,
            owner="marketing",
            tags=["metrics"],
            access_level=AccessLevel.PUBLIC,
        )

        # Search by owner
        analytics_data = hub.catalog.search(owner="analytics")
        assert len(analytics_data) == 2

        # Search by tags
        pii_data = hub.catalog.search(tags=["pii"])
        assert len(pii_data) == 1
        assert pii_data[0].name == "user_data"

        # Search by access level
        public_data = hub.catalog.search(access_level=AccessLevel.PUBLIC)
        assert len(public_data) == 1

    @pytest.mark.asyncio
    async def test_managed_provider_full_workflow(self) -> None:
        """Test managed provider with full data lifecycle workflow."""
        mock_provider = MagicMock(spec=VisionProvider)
        mock_provider.provider_name = "test"
        mock_provider.analyze_image = AsyncMock(
            return_value=VisionDescription(
                summary="CAD drawing analysis",
                confidence=0.92,
            )
        )

        hub = create_data_lifecycle_hub(
            enable_versioning=True,
            enable_lineage=True,
        )

        # Add quality rule
        hub.quality.add_rule(create_quality_rule(
            name="confidence_check",
            dimension=QualityDimension.ACCURACY,
            check_fn=lambda x: x.get("confidence", 0) > 0.5 if isinstance(x, dict) else True,
        ))

        managed = create_managed_provider(mock_provider, hub)

        # Process multiple images
        for i in range(3):
            await managed.analyze_image(f"image_{i}".encode())

        # Verify lifecycle state
        summary = hub.get_lifecycle_summary()
        assert summary["total_assets"] == 3
        assert summary["total_versions"] == 3
        assert summary["lineage_nodes"] == 6  # 2 per request (input + output)
        assert summary["transformations"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
