"""API Versioning Module.

Provides comprehensive API versioning:
- Semantic versioning
- Version routing
- Schema migration
- Deprecation management
"""

from src.core.versioning.version import (
    VersionFormat,
    SemanticVersion,
    APIVersion,
    VersionRange,
    VersionRegistry,
)
from src.core.versioning.router import (
    VersioningStrategy,
    VersionedRequest,
    VersionedResponse,
    VersionExtractor,
    URLPathVersionExtractor,
    HeaderVersionExtractor,
    QueryParamVersionExtractor,
    ContentTypeVersionExtractor,
    Route,
    VersionedRouter,
    VersionedAPIGroup,
)
from src.core.versioning.migration import (
    MigrationStep,
    SchemaMigrator,
    FieldChange,
    SchemaTransformBuilder,
    DeprecationNotice,
    DeprecationRegistry,
    MigrationGuide,
    MigrationGuideBuilder,
)

__all__ = [
    # Version
    "VersionFormat",
    "SemanticVersion",
    "APIVersion",
    "VersionRange",
    "VersionRegistry",
    # Router
    "VersioningStrategy",
    "VersionedRequest",
    "VersionedResponse",
    "VersionExtractor",
    "URLPathVersionExtractor",
    "HeaderVersionExtractor",
    "QueryParamVersionExtractor",
    "ContentTypeVersionExtractor",
    "Route",
    "VersionedRouter",
    "VersionedAPIGroup",
    # Migration
    "MigrationStep",
    "SchemaMigrator",
    "FieldChange",
    "SchemaTransformBuilder",
    "DeprecationNotice",
    "DeprecationRegistry",
    "MigrationGuide",
    "MigrationGuideBuilder",
]
