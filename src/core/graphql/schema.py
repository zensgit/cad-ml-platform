"""GraphQL Schema Definition.

Defines the main GraphQL schema using Strawberry or similar library patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from src.core.graphql.types import (
    DocumentType,
    ModelType,
    UserType,
    JobType,
    PredictionType,
    PageInfo,
    Connection,
    PaginationType,
    DocumentInput,
    ModelInput,
    PredictionInput,
    MutationResponse,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class FieldDefinition:
    """GraphQL field definition."""
    name: str
    type_name: str
    description: Optional[str] = None
    args: Dict[str, str] = field(default_factory=dict)
    resolver: Optional[Callable] = None
    deprecation_reason: Optional[str] = None


@dataclass
class TypeDefinition:
    """GraphQL type definition."""
    name: str
    fields: List[FieldDefinition]
    description: Optional[str] = None
    interfaces: List[str] = field(default_factory=list)
    directives: List[str] = field(default_factory=list)


class GraphQLSchema:
    """GraphQL schema builder and container."""

    def __init__(self, name: str = "CADMLPlatformSchema"):
        self.name = name
        self._types: Dict[str, TypeDefinition] = {}
        self._queries: Dict[str, FieldDefinition] = {}
        self._mutations: Dict[str, FieldDefinition] = {}
        self._subscriptions: Dict[str, FieldDefinition] = {}
        self._directives: Dict[str, Any] = {}
        self._resolvers: Dict[str, Callable] = {}

    def add_type(self, type_def: TypeDefinition) -> None:
        """Add a type definition."""
        self._types[type_def.name] = type_def

    def add_query(self, field_def: FieldDefinition) -> None:
        """Add a query field."""
        self._queries[field_def.name] = field_def

    def add_mutation(self, field_def: FieldDefinition) -> None:
        """Add a mutation field."""
        self._mutations[field_def.name] = field_def

    def add_subscription(self, field_def: FieldDefinition) -> None:
        """Add a subscription field."""
        self._subscriptions[field_def.name] = field_def

    def register_resolver(self, type_name: str, field_name: str, resolver: Callable) -> None:
        """Register a field resolver."""
        key = f"{type_name}.{field_name}"
        self._resolvers[key] = resolver

    def get_resolver(self, type_name: str, field_name: str) -> Optional[Callable]:
        """Get a field resolver."""
        key = f"{type_name}.{field_name}"
        return self._resolvers.get(key)

    def to_sdl(self) -> str:
        """Generate GraphQL Schema Definition Language."""
        lines = []

        # Types
        for type_def in self._types.values():
            lines.append(self._type_to_sdl(type_def))
            lines.append("")

        # Query type
        if self._queries:
            lines.append("type Query {")
            for field_def in self._queries.values():
                lines.append(f"  {self._field_to_sdl(field_def)}")
            lines.append("}")
            lines.append("")

        # Mutation type
        if self._mutations:
            lines.append("type Mutation {")
            for field_def in self._mutations.values():
                lines.append(f"  {self._field_to_sdl(field_def)}")
            lines.append("}")
            lines.append("")

        # Subscription type
        if self._subscriptions:
            lines.append("type Subscription {")
            for field_def in self._subscriptions.values():
                lines.append(f"  {self._field_to_sdl(field_def)}")
            lines.append("}")

        return "\n".join(lines)

    def _type_to_sdl(self, type_def: TypeDefinition) -> str:
        """Convert type definition to SDL."""
        parts = []

        if type_def.description:
            parts.append(f'"""{type_def.description}"""')

        interfaces = f" implements {' & '.join(type_def.interfaces)}" if type_def.interfaces else ""
        parts.append(f"type {type_def.name}{interfaces} {{")

        for field_def in type_def.fields:
            parts.append(f"  {self._field_to_sdl(field_def)}")

        parts.append("}")
        return "\n".join(parts)

    def _field_to_sdl(self, field_def: FieldDefinition) -> str:
        """Convert field definition to SDL."""
        args = ""
        if field_def.args:
            arg_strs = [f"{name}: {type_name}" for name, type_name in field_def.args.items()]
            args = f"({', '.join(arg_strs)})"

        deprecation = ""
        if field_def.deprecation_reason:
            deprecation = f' @deprecated(reason: "{field_def.deprecation_reason}")'

        return f"{field_def.name}{args}: {field_def.type_name}{deprecation}"


def create_schema() -> GraphQLSchema:
    """Create the default CAD ML Platform GraphQL schema."""
    schema = GraphQLSchema()

    # ========================================================================
    # Document Type
    # ========================================================================
    schema.add_type(TypeDefinition(
        name="Document",
        description="A CAD document in the system",
        fields=[
            FieldDefinition("id", "ID!", "Unique identifier"),
            FieldDefinition("name", "String!", "Document name"),
            FieldDefinition("filePath", "String!", "Path to the file"),
            FieldDefinition("fileType", "String!", "File type (dxf, dwg, etc.)"),
            FieldDefinition("status", "DocumentStatus!", "Processing status"),
            FieldDefinition("fileSize", "Int!", "File size in bytes"),
            FieldDefinition("checksum", "String", "File checksum"),
            FieldDefinition("metadata", "JSON", "Additional metadata"),
            FieldDefinition("tags", "[String!]!", "Document tags"),
            FieldDefinition("owner", "User", "Document owner", resolver=None),
            FieldDefinition("createdAt", "DateTime!", "Creation timestamp"),
            FieldDefinition("updatedAt", "DateTime!", "Last update timestamp"),
            FieldDefinition("processedAt", "DateTime", "Processing completion timestamp"),
        ],
    ))

    # ========================================================================
    # Model Type
    # ========================================================================
    schema.add_type(TypeDefinition(
        name="Model",
        description="An ML model",
        fields=[
            FieldDefinition("id", "ID!", "Unique identifier"),
            FieldDefinition("name", "String!", "Model name"),
            FieldDefinition("modelType", "String!", "Type of model"),
            FieldDefinition("version", "String!", "Model version"),
            FieldDefinition("status", "ModelStatus!", "Model status"),
            FieldDefinition("framework", "String!", "ML framework"),
            FieldDefinition("architecture", "String", "Model architecture"),
            FieldDefinition("parametersCount", "Int!", "Number of parameters"),
            FieldDefinition("metrics", "ModelMetrics", "Performance metrics"),
            FieldDefinition("isDefault", "Boolean!", "Is default model"),
            FieldDefinition("endpointUrl", "String", "Deployment endpoint"),
            FieldDefinition("createdAt", "DateTime!", "Creation timestamp"),
            FieldDefinition("deployedAt", "DateTime", "Deployment timestamp"),
        ],
    ))

    # ========================================================================
    # User Type
    # ========================================================================
    schema.add_type(TypeDefinition(
        name="User",
        description="A user in the system",
        fields=[
            FieldDefinition("id", "ID!", "Unique identifier"),
            FieldDefinition("email", "String!", "Email address"),
            FieldDefinition("username", "String!", "Username"),
            FieldDefinition("displayName", "String", "Display name"),
            FieldDefinition("roles", "[String!]!", "User roles"),
            FieldDefinition("isActive", "Boolean!", "Is active"),
            FieldDefinition("createdAt", "DateTime!", "Creation timestamp"),
            FieldDefinition("documents", "DocumentConnection", "User's documents"),
        ],
    ))

    # ========================================================================
    # Queries
    # ========================================================================
    schema.add_query(FieldDefinition(
        name="document",
        type_name="Document",
        description="Get a document by ID",
        args={"id": "ID!"},
    ))

    schema.add_query(FieldDefinition(
        name="documents",
        type_name="DocumentConnection!",
        description="List documents with pagination",
        args={
            "first": "Int",
            "after": "String",
            "filter": "DocumentFilter",
        },
    ))

    schema.add_query(FieldDefinition(
        name="searchDocuments",
        type_name="DocumentConnection!",
        description="Search documents",
        args={
            "query": "String!",
            "first": "Int",
            "after": "String",
        },
    ))

    schema.add_query(FieldDefinition(
        name="model",
        type_name="Model",
        description="Get a model by ID",
        args={"id": "ID!"},
    ))

    schema.add_query(FieldDefinition(
        name="models",
        type_name="[Model!]!",
        description="List all models",
        args={"status": "ModelStatus"},
    ))

    schema.add_query(FieldDefinition(
        name="me",
        type_name="User",
        description="Get current user",
    ))

    schema.add_query(FieldDefinition(
        name="user",
        type_name="User",
        description="Get a user by ID",
        args={"id": "ID!"},
    ))

    # ========================================================================
    # Mutations
    # ========================================================================
    schema.add_mutation(FieldDefinition(
        name="createDocument",
        type_name="DocumentMutationResponse!",
        description="Create a new document",
        args={"input": "CreateDocumentInput!"},
    ))

    schema.add_mutation(FieldDefinition(
        name="updateDocument",
        type_name="DocumentMutationResponse!",
        description="Update a document",
        args={"id": "ID!", "input": "UpdateDocumentInput!"},
    ))

    schema.add_mutation(FieldDefinition(
        name="deleteDocument",
        type_name="MutationResponse!",
        description="Delete a document",
        args={"id": "ID!"},
    ))

    schema.add_mutation(FieldDefinition(
        name="processDocument",
        type_name="Job!",
        description="Start document processing",
        args={"id": "ID!", "options": "ProcessingOptions"},
    ))

    schema.add_mutation(FieldDefinition(
        name="createModel",
        type_name="ModelMutationResponse!",
        description="Create a new model",
        args={"input": "CreateModelInput!"},
    ))

    schema.add_mutation(FieldDefinition(
        name="deployModel",
        type_name="ModelMutationResponse!",
        description="Deploy a model",
        args={"id": "ID!", "options": "DeploymentOptions"},
    ))

    schema.add_mutation(FieldDefinition(
        name="predict",
        type_name="Prediction!",
        description="Make a prediction",
        args={"input": "PredictionInput!"},
    ))

    # ========================================================================
    # Subscriptions
    # ========================================================================
    schema.add_subscription(FieldDefinition(
        name="documentCreated",
        type_name="Document!",
        description="Subscribe to document creation events",
    ))

    schema.add_subscription(FieldDefinition(
        name="documentProcessed",
        type_name="Document!",
        description="Subscribe to document processing completion",
        args={"id": "ID"},
    ))

    schema.add_subscription(FieldDefinition(
        name="jobProgress",
        type_name="Job!",
        description="Subscribe to job progress updates",
        args={"jobId": "ID!"},
    ))

    schema.add_subscription(FieldDefinition(
        name="modelDeployed",
        type_name="Model!",
        description="Subscribe to model deployment events",
    ))

    return schema


# Global schema instance
_schema: Optional[GraphQLSchema] = None


def get_schema() -> GraphQLSchema:
    """Get global GraphQL schema."""
    global _schema
    if _schema is None:
        _schema = create_schema()
    return _schema


def set_schema(schema: GraphQLSchema) -> None:
    """Set global GraphQL schema."""
    global _schema
    _schema = schema
