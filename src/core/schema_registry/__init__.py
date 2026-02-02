"""Schema Registry Module.

Provides schema registry capabilities:
- Schema definition and parsing
- Compatibility checking
- Version management
"""

from src.core.schema_registry.schema import (
    SchemaType,
    CompatibilityMode,
    SchemaMetadata,
    Schema,
    SchemaReference,
    SchemaParser,
    JSONSchemaParser,
    AvroSchemaParser,
    ProtobufSchemaParser,
    get_parser,
)
from src.core.schema_registry.compatibility import (
    CompatibilityResult,
    CompatibilityChecker,
    JSONSchemaCompatibilityChecker,
    AvroCompatibilityChecker,
    ProtobufCompatibilityChecker,
    get_compatibility_checker,
)
from src.core.schema_registry.registry import (
    SchemaRegistryError,
    SchemaNotFoundError,
    IncompatibleSchemaError,
    InvalidSchemaError,
    SubjectConfig,
    SchemaRegistry,
    InMemorySchemaRegistry,
    SchemaCache,
    CachingSchemaRegistry,
)

__all__ = [
    # Schema
    "SchemaType",
    "CompatibilityMode",
    "SchemaMetadata",
    "Schema",
    "SchemaReference",
    "SchemaParser",
    "JSONSchemaParser",
    "AvroSchemaParser",
    "ProtobufSchemaParser",
    "get_parser",
    # Compatibility
    "CompatibilityResult",
    "CompatibilityChecker",
    "JSONSchemaCompatibilityChecker",
    "AvroCompatibilityChecker",
    "ProtobufCompatibilityChecker",
    "get_compatibility_checker",
    # Registry
    "SchemaRegistryError",
    "SchemaNotFoundError",
    "IncompatibleSchemaError",
    "InvalidSchemaError",
    "SubjectConfig",
    "SchemaRegistry",
    "InMemorySchemaRegistry",
    "SchemaCache",
    "CachingSchemaRegistry",
]
