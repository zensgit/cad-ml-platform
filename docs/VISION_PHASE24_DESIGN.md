# Vision System Phase 24: Advanced Data Management & Lifecycle

## Overview

Phase 24 implements comprehensive data management and lifecycle capabilities for the Vision system, providing enterprise-grade data governance, versioning, lineage tracking, quality management, and retention policies.

## Architecture

### Component Hierarchy

```
DataLifecycleHub
├── VersionManager        # Semantic versioning (major.minor.patch)
├── LineageTracker        # Data provenance graph
├── QualityManager        # Rule-based quality checks
├── RetentionManager      # Configurable retention policies
├── DataCatalog           # Discovery and metadata
├── TransformationTracker # Operation tracking
└── ManagedVisionProvider # Vision integration wrapper
```

### Class Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DataLifecycleHub                              │
├─────────────────────────────────────────────────────────────────────┤
│ - _config: LifecycleConfig                                          │
│ - _version_manager: VersionManager                                  │
│ - _lineage_tracker: LineageTracker                                  │
│ - _quality_manager: QualityManager                                  │
│ - _retention_manager: RetentionManager                              │
│ - _catalog: DataCatalog                                             │
│ - _transformation_tracker: TransformationTracker                    │
│ - _assets: Dict[str, DataAsset]                                     │
├─────────────────────────────────────────────────────────────────────┤
│ + versions: VersionManager                                          │
│ + lineage: LineageTracker                                           │
│ + quality: QualityManager                                           │
│ + retention: RetentionManager                                       │
│ + catalog: DataCatalog                                              │
│ + transformations: TransformationTracker                            │
│ + create_asset(name, owner, tags, ...) -> DataAsset                │
│ + get_asset(asset_id) -> Optional[DataAsset]                        │
│ + update_asset_state(asset_id, new_state) -> Optional[DataAsset]   │
│ + list_assets(state, owner) -> List[DataAsset]                     │
│ + get_lifecycle_summary() -> Dict[str, Any]                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Enums (9 types)

| Enum | Purpose | Values |
|------|---------|--------|
| `DataState` | Asset lifecycle states | DRAFT, ACTIVE, ARCHIVED, DELETED, EXPIRED, QUARANTINED |
| `VersionType` | Version change types | MAJOR, MINOR, PATCH, SNAPSHOT |
| `LineageRelation` | Lineage edge relations | DERIVED_FROM, TRANSFORMED_FROM, COPIED_FROM, AGGREGATED_FROM, JOINED_WITH, FILTERED_FROM |
| `QualityDimension` | Quality check dimensions | COMPLETENESS, ACCURACY, CONSISTENCY, TIMELINESS, VALIDITY, UNIQUENESS |
| `QualityLevel` | Quality assessment levels | EXCELLENT, GOOD, ACCEPTABLE, POOR, CRITICAL |
| `RetentionAction` | Retention policy actions | KEEP, ARCHIVE, DELETE, ANONYMIZE, COMPRESS |
| `CatalogEntryType` | Catalog entry types | DATASET, TABLE, COLUMN, FILE, API, MODEL |
| `TransformationType` | Transformation operations | FILTER, MAP, AGGREGATE, JOIN, SORT, DEDUPLICATE, ENRICH, NORMALIZE |
| `AccessLevel` | Data access levels | PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED |

### 2. Dataclasses (12 types)

| Dataclass | Purpose |
|-----------|---------|
| `DataVersion` | Represents a version with semantic versioning |
| `LineageNode` | Node in the data lineage graph |
| `LineageEdge` | Edge connecting lineage nodes |
| `QualityRule` | Defines a quality check rule |
| `QualityResult` | Result of a single quality check |
| `QualityReport` | Comprehensive quality assessment |
| `RetentionPolicy` | Defines retention rules |
| `RetentionResult` | Result of policy execution |
| `CatalogEntry` | Entry in the data catalog |
| `TransformationRecord` | Records a transformation operation |
| `DataAsset` | Managed data asset |
| `LifecycleConfig` | Configuration for lifecycle management |

### 3. Core Classes (6 managers)

#### VersionManager
Manages data versioning with semantic versioning support.

```python
class VersionManager:
    def create_version(asset_id, data, version_type, message) -> DataVersion
    def get_version(asset_id, version_id) -> Optional[DataVersion]
    def get_current_version(asset_id) -> Optional[DataVersion]
    def get_version_history(asset_id, limit) -> List[DataVersion]
    def rollback_to_version(asset_id, version_id) -> Optional[DataVersion]
    def compare_versions(asset_id, v1_id, v2_id) -> Dict[str, Any]
```

#### LineageTracker
Tracks data lineage and provenance through a graph structure.

```python
class LineageTracker:
    def register_node(name, node_type, metadata) -> LineageNode
    def add_edge(source_id, target_id, relation, transformation) -> Optional[LineageEdge]
    def get_upstream(node_id, depth) -> List[LineageNode]
    def get_downstream(node_id, depth) -> List[LineageNode]
    def get_lineage_graph(node_id) -> Dict[str, Any]
```

#### QualityManager
Manages data quality rules and executes quality checks.

```python
class QualityManager:
    def add_rule(rule: QualityRule) -> None
    def remove_rule(rule_id) -> bool
    def list_rules(dimension) -> List[QualityRule]
    def check_quality(data_id, data) -> QualityReport
    def get_reports(data_id, limit) -> List[QualityReport]
```

#### RetentionManager
Manages data retention policies and execution.

```python
class RetentionManager:
    def add_policy(policy: RetentionPolicy) -> None
    def remove_policy(policy_id) -> bool
    def evaluate_asset(asset) -> Tuple[RetentionAction, Optional[RetentionPolicy]]
    def apply_policy(asset, action_handler) -> RetentionResult
```

#### DataCatalog
Provides data discovery and metadata management.

```python
class DataCatalog:
    def register_entry(name, entry_type, description, ...) -> CatalogEntry
    def get_entry(entry_id) -> Optional[CatalogEntry]
    def search(query, entry_type, tags, owner, access_level) -> List[CatalogEntry]
    def update_statistics(entry_id, statistics) -> Optional[CatalogEntry]
    def add_tags(entry_id, tags) -> Optional[CatalogEntry]
```

#### TransformationTracker
Tracks data transformations and their lineage.

```python
class TransformationTracker:
    def record_transformation(type, input_ids, output_id, params, ...) -> TransformationRecord
    def get_transformations_for_output(output_id) -> List[TransformationRecord]
    def get_transformations_from_input(input_id) -> List[TransformationRecord]
    def get_transformation_stats() -> Dict[str, Any]
```

### 4. Hub Orchestration

```python
class DataLifecycleHub:
    """Central hub coordinating all lifecycle components."""

    @property
    def versions(self) -> VersionManager
    @property
    def lineage(self) -> LineageTracker
    @property
    def quality(self) -> QualityManager
    @property
    def retention(self) -> RetentionManager
    @property
    def catalog(self) -> DataCatalog
    @property
    def transformations(self) -> TransformationTracker

    def create_asset(name, owner, tags, access_level, expires_in_days) -> DataAsset
    def get_asset(asset_id) -> Optional[DataAsset]
    def update_asset_state(asset_id, new_state) -> Optional[DataAsset]
    def list_assets(state, owner) -> List[DataAsset]
    def get_lifecycle_summary() -> Dict[str, Any]
```

### 5. Provider Integration

```python
class ManagedVisionProvider(VisionProvider):
    """Vision provider with automatic lifecycle management."""

    async def analyze_image(image_data, include_description) -> VisionDescription
    # Automatically:
    # - Creates data asset
    # - Tracks versioning
    # - Records lineage
    # - Logs transformations
```

## Factory Functions (5 functions)

| Function | Purpose |
|----------|---------|
| `create_lifecycle_config()` | Create lifecycle configuration |
| `create_data_lifecycle_hub()` | Create hub with custom config |
| `create_quality_rule()` | Create a quality rule |
| `create_retention_policy()` | Create a retention policy |
| `create_managed_provider()` | Wrap provider with lifecycle |

## Data Flow

### Complete Lifecycle Flow

```
1. Asset Creation
   ┌────────────────┐
   │ create_asset() │ → DataAsset (state: DRAFT)
   └───────┬────────┘
           │
2. Version Creation
   ┌───────▼────────┐
   │create_version()│ → DataVersion (1.0.0)
   └───────┬────────┘
           │
3. State Activation
   ┌───────▼─────────────┐
   │update_asset_state() │ → state: ACTIVE
   └───────┬─────────────┘
           │
4. Lineage Tracking
   ┌───────▼──────────┐
   │register_node()   │ → LineageNode
   │add_edge()        │ → LineageEdge
   └───────┬──────────┘
           │
5. Quality Check
   ┌───────▼──────────┐
   │check_quality()   │ → QualityReport
   └───────┬──────────┘
           │
6. Retention Evaluation
   ┌───────▼──────────┐
   │evaluate_asset()  │ → RetentionAction
   │apply_policy()    │ → RetentionResult
   └───────┬──────────┘
           │
7. Archive/Delete
   ┌───────▼─────────────┐
   │update_asset_state() │ → state: ARCHIVED/DELETED
   └─────────────────────┘
```

### Lineage Graph Example

```
     ┌──────────────┐
     │ raw_sales_1  │ (raw data)
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐       ┌──────────────┐
     │ raw_sales_2  │ ──────│ raw_returns  │
     └──────┬───────┘       └──────┬───────┘
            │                      │
            └───────┬──────────────┘
                    │ (AGGREGATED_FROM)
                    ▼
            ┌───────────────┐
            │ sales_report  │ (processed)
            └───────┬───────┘
                    │ (TRANSFORMED_FROM)
                    ▼
            ┌───────────────┐
            │ final_report  │ (output)
            └───────────────┘
```

## Quality Scoring

### Level Thresholds

| Level | Score Range | Description |
|-------|-------------|-------------|
| EXCELLENT | >= 0.90 | All quality rules pass |
| GOOD | >= 0.75 | Most rules pass |
| ACCEPTABLE | >= 0.50 | Half rules pass |
| POOR | >= 0.25 | Few rules pass |
| CRITICAL | < 0.25 | Most rules fail |

### Quality Dimensions

```
COMPLETENESS  ─── Is all required data present?
ACCURACY      ─── Is the data correct?
CONSISTENCY   ─── Is the data internally consistent?
TIMELINESS    ─── Is the data up-to-date?
VALIDITY      ─── Does the data conform to rules?
UNIQUENESS    ─── Are there duplicates?
```

## Usage Examples

### Basic Lifecycle Management

```python
from src.core.vision import (
    create_data_lifecycle_hub,
    DataState,
    AccessLevel,
)

# Create hub
hub = create_data_lifecycle_hub(
    default_retention_days=365,
    enable_versioning=True,
)

# Create and manage asset
asset = hub.create_asset(
    name="sales_data",
    owner="analytics",
    tags=["sales", "monthly"],
    access_level=AccessLevel.INTERNAL,
)

# Create version
version = hub.versions.create_version(
    asset_id=asset.asset_id,
    data={"period": "2024-01", "total": 1000000},
    message="January report",
)

# Activate
hub.update_asset_state(asset.asset_id, DataState.ACTIVE)
```

### Lineage Tracking

```python
# Register nodes
source = hub.lineage.register_node("raw_data", "csv")
processed = hub.lineage.register_node("cleaned_data", "parquet")
output = hub.lineage.register_node("report", "pdf")

# Add edges
hub.lineage.add_edge(source.node_id, processed.node_id, LineageRelation.TRANSFORMED_FROM)
hub.lineage.add_edge(processed.node_id, output.node_id, LineageRelation.DERIVED_FROM)

# Query lineage
upstream = hub.lineage.get_upstream(output.node_id)
graph = hub.lineage.get_lineage_graph(processed.node_id)
```

### Quality Management

```python
from src.core.vision import create_quality_rule, QualityDimension

# Add rules
hub.quality.add_rule(create_quality_rule(
    name="not_null",
    dimension=QualityDimension.COMPLETENESS,
    check_fn=lambda x: x is not None,
    description="Value must not be null",
))

hub.quality.add_rule(create_quality_rule(
    name="valid_range",
    dimension=QualityDimension.VALIDITY,
    check_fn=lambda x: 0 <= x.get("score", 0) <= 100,
    description="Score must be 0-100",
))

# Check quality
report = hub.quality.check_quality("data_1", {"score": 85})
print(f"Quality: {report.overall_level.value}")
```

### Retention Policies

```python
from src.core.vision import create_retention_policy, RetentionAction

# Add policy
hub.retention.add_policy(create_retention_policy(
    name="archive_old_drafts",
    retention_days=30,
    action=RetentionAction.ARCHIVE,
    conditions={"state": DataState.DRAFT},
    priority=10,
))

# Evaluate and apply
action, policy = hub.retention.evaluate_asset(asset)
if action != RetentionAction.KEEP:
    result = hub.retention.apply_policy(asset, handler_fn)
```

### Vision Provider Integration

```python
from src.core.vision import create_managed_provider

# Wrap any vision provider
managed = create_managed_provider(base_provider, hub)

# All operations are automatically tracked
result = await managed.analyze_image(image_data)
# - Creates asset
# - Creates version
# - Tracks lineage
# - Records transformation
```

## Configuration

### LifecycleConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| `default_retention_days` | 365 | Default data retention period |
| `archive_after_days` | 180 | Auto-archive after this period |
| `enable_versioning` | True | Enable version tracking |
| `max_versions` | 100 | Maximum versions per asset |
| `enable_lineage` | True | Enable lineage tracking |
| `enable_quality_checks` | True | Enable quality validation |
| `auto_archive` | True | Auto-archive old data |
| `auto_delete_expired` | False | Auto-delete expired data |

## Thread Safety

All managers use `threading.RLock()` for thread-safe operations:

```python
class VersionManager:
    def __init__(self):
        self._lock = threading.RLock()

    def create_version(self, ...):
        with self._lock:
            # Thread-safe operation
```

## Testing

### Test Coverage: 91 tests

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestDataLifecycleEnums | 9 | All enum values |
| TestDataLifecycleDataclasses | 13 | All dataclass creation |
| TestVersionManager | 9 | Versioning operations |
| TestLineageTracker | 7 | Lineage graph operations |
| TestQualityManager | 7 | Quality rule management |
| TestRetentionManager | 7 | Retention policy management |
| TestDataCatalog | 10 | Catalog operations |
| TestTransformationTracker | 5 | Transformation tracking |
| TestDataLifecycleHub | 8 | Hub orchestration |
| TestManagedVisionProvider | 6 | Provider integration |
| TestFactoryFunctions | 5 | Factory function creation |
| TestDataLifecycleIntegration | 5 | End-to-end workflows |

### Running Tests

```bash
# Run Phase 24 tests only
python -m pytest tests/unit/test_vision_phase24.py -v

# Run all vision phase tests (20-24)
python -m pytest tests/unit/test_vision_phase20.py \
                 tests/unit/test_vision_phase21.py \
                 tests/unit/test_vision_phase22.py \
                 tests/unit/test_vision_phase23.py \
                 tests/unit/test_vision_phase24.py -v
```

## Cumulative Statistics (Phases 20-24)

| Phase | Description | Tests |
|-------|-------------|-------|
| 20 | Event-Driven Architecture | 56 |
| 21 | Advanced Processing Pipeline | 85 |
| 22 | Advanced Security & Governance | 80 |
| 23 | Intelligent Automation & Self-Optimization | 73 |
| 24 | Advanced Data Management & Lifecycle | 91 |
| **Total** | | **385** |

## Integration Points

### With Phase 20 (Event-Driven)
- Lifecycle state changes emit events
- Version creation triggers event notifications
- Quality failures generate alerts

### With Phase 21 (Processing Pipeline)
- Pipeline operations tracked as transformations
- Pipeline outputs versioned automatically
- Pipeline lineage recorded

### With Phase 22 (Security)
- Access level enforcement
- Audit trail integration
- Data classification support

### With Phase 23 (Automation)
- Automated quality monitoring
- Self-tuning retention policies
- Predictive capacity planning

## Future Enhancements

1. **Distributed Lineage**: Cross-system lineage tracking
2. **ML-Based Quality**: Anomaly detection for quality
3. **Time-Travel Queries**: Query data at any version
4. **Data Contracts**: Schema validation and compatibility
5. **Cost Attribution**: Storage and compute cost tracking
