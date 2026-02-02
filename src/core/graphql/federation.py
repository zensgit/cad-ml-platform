"""GraphQL Federation Support.

Provides Apollo Federation compatible schema extensions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.graphql.schema import GraphQLSchema, FieldDefinition, TypeDefinition

logger = logging.getLogger(__name__)


@dataclass
class ServiceDefinition:
    """Definition of a federated service."""
    name: str
    url: str
    schema_sdl: str
    types: List[str] = field(default_factory=list)
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityKey:
    """Key fields for an entity."""
    type_name: str
    key_fields: List[str]


class FederatedSchema(GraphQLSchema):
    """GraphQL schema with Federation support.

    Supports Apollo Federation v2 directives:
    - @key: Defines entity identity
    - @external: Field defined in another service
    - @requires: Field requires other fields
    - @provides: Field provides nested fields
    - @shareable: Field can be resolved by multiple services
    - @extends: Type extends from another service
    """

    def __init__(self, name: str = "FederatedSchema"):
        super().__init__(name)
        self._entities: Dict[str, EntityKey] = {}
        self._external_fields: Dict[str, Set[str]] = {}
        self._requires: Dict[str, Dict[str, str]] = {}
        self._provides: Dict[str, Dict[str, str]] = {}
        self._shareable: Dict[str, Set[str]] = {}
        self._services: Dict[str, ServiceDefinition] = {}

    def add_entity(
        self,
        type_name: str,
        key_fields: List[str],
        type_def: Optional[TypeDefinition] = None,
    ) -> None:
        """Register a type as a federation entity.

        Args:
            type_name: Name of the type
            key_fields: Fields that form the entity key
            type_def: Optional type definition to add
        """
        self._entities[type_name] = EntityKey(
            type_name=type_name,
            key_fields=key_fields,
        )

        if type_def:
            # Add @key directive
            key_directive = f'@key(fields: "{" ".join(key_fields)}")'
            if key_directive not in type_def.directives:
                type_def.directives.append(key_directive)
            self.add_type(type_def)

        logger.debug(f"Registered entity: {type_name} with keys: {key_fields}")

    def mark_external(self, type_name: str, field_name: str) -> None:
        """Mark a field as external (defined in another service).

        Args:
            type_name: Type containing the field
            field_name: Field name
        """
        if type_name not in self._external_fields:
            self._external_fields[type_name] = set()
        self._external_fields[type_name].add(field_name)

    def add_requires(self, type_name: str, field_name: str, required_fields: str) -> None:
        """Add @requires directive to a field.

        Args:
            type_name: Type containing the field
            field_name: Field name
            required_fields: Fields required for resolution
        """
        if type_name not in self._requires:
            self._requires[type_name] = {}
        self._requires[type_name][field_name] = required_fields

    def add_provides(self, type_name: str, field_name: str, provided_fields: str) -> None:
        """Add @provides directive to a field.

        Args:
            type_name: Type containing the field
            field_name: Field name
            provided_fields: Nested fields this field provides
        """
        if type_name not in self._provides:
            self._provides[type_name] = {}
        self._provides[type_name][field_name] = provided_fields

    def mark_shareable(self, type_name: str, field_name: str) -> None:
        """Mark a field as shareable.

        Args:
            type_name: Type containing the field
            field_name: Field name
        """
        if type_name not in self._shareable:
            self._shareable[type_name] = set()
        self._shareable[type_name].add(field_name)

    def register_service(self, service: ServiceDefinition) -> None:
        """Register an external service for federation.

        Args:
            service: Service definition
        """
        self._services[service.name] = service
        logger.info(f"Registered federated service: {service.name} at {service.url}")

    def get_entity_resolver(self, type_name: str) -> Optional[Callable]:
        """Get the entity resolver for a type.

        This resolver is used by the gateway to resolve entity references.
        """
        resolver_name = f"resolve_{type_name}_reference"
        return self._resolvers.get(resolver_name)

    def register_entity_resolver(self, type_name: str, resolver: Callable) -> None:
        """Register an entity reference resolver.

        Args:
            type_name: Entity type name
            resolver: Resolver function that takes representation dict
        """
        resolver_name = f"resolve_{type_name}_reference"
        self._resolvers[resolver_name] = resolver

    def to_sdl(self) -> str:
        """Generate Federation-compatible SDL."""
        lines = []

        # Federation directives import
        lines.append("extend schema @link(")
        lines.append('  url: "https://specs.apollo.dev/federation/v2.0"')
        lines.append("  import: [")
        lines.append('    "@key", "@external", "@requires", "@provides", "@shareable"')
        lines.append("  ]")
        lines.append(")")
        lines.append("")

        # Types with federation directives
        for type_name, type_def in self._types.items():
            lines.append(self._type_to_federation_sdl(type_name, type_def))
            lines.append("")

        # Query type
        if self._queries:
            lines.append("type Query {")
            for field_def in self._queries.values():
                lines.append(f"  {self._field_to_sdl(field_def)}")

            # Add _entities query for federation
            if self._entities:
                lines.append("  _entities(representations: [_Any!]!): [_Entity]!")
                lines.append("  _service: _Service!")

            lines.append("}")
            lines.append("")

        # Mutation type
        if self._mutations:
            lines.append("type Mutation {")
            for field_def in self._mutations.values():
                lines.append(f"  {self._field_to_sdl(field_def)}")
            lines.append("}")
            lines.append("")

        # Federation built-in types
        if self._entities:
            entity_types = " | ".join(self._entities.keys())
            lines.append(f"union _Entity = {entity_types}")
            lines.append("")
            lines.append("scalar _Any")
            lines.append("")
            lines.append("type _Service {")
            lines.append("  sdl: String!")
            lines.append("}")

        return "\n".join(lines)

    def _type_to_federation_sdl(self, type_name: str, type_def: TypeDefinition) -> str:
        """Convert type to Federation SDL."""
        parts = []

        if type_def.description:
            parts.append(f'"""{type_def.description}"""')

        # Build type declaration with directives
        interfaces = f" implements {' & '.join(type_def.interfaces)}" if type_def.interfaces else ""

        directives = []
        # Add @key directive for entities
        if type_name in self._entities:
            entity = self._entities[type_name]
            directives.append(f'@key(fields: "{" ".join(entity.key_fields)}")')

        # Add any other directives
        directives.extend(type_def.directives)

        directive_str = " ".join(directives)
        if directive_str:
            directive_str = " " + directive_str

        parts.append(f"type {type_name}{interfaces}{directive_str} {{")

        # Fields with federation directives
        for field_def in type_def.fields:
            field_sdl = self._field_to_federation_sdl(type_name, field_def)
            parts.append(f"  {field_sdl}")

        parts.append("}")
        return "\n".join(parts)

    def _field_to_federation_sdl(self, type_name: str, field_def: FieldDefinition) -> str:
        """Convert field to Federation SDL with directives."""
        args = ""
        if field_def.args:
            arg_strs = [f"{name}: {type_name}" for name, type_name in field_def.args.items()]
            args = f"({', '.join(arg_strs)})"

        directives = []

        # Check for @external
        if type_name in self._external_fields and field_def.name in self._external_fields[type_name]:
            directives.append("@external")

        # Check for @requires
        if type_name in self._requires and field_def.name in self._requires[type_name]:
            required = self._requires[type_name][field_def.name]
            directives.append(f'@requires(fields: "{required}")')

        # Check for @provides
        if type_name in self._provides and field_def.name in self._provides[type_name]:
            provided = self._provides[type_name][field_def.name]
            directives.append(f'@provides(fields: "{provided}")')

        # Check for @shareable
        if type_name in self._shareable and field_def.name in self._shareable[type_name]:
            directives.append("@shareable")

        if field_def.deprecation_reason:
            directives.append(f'@deprecated(reason: "{field_def.deprecation_reason}")')

        directive_str = " ".join(directives)
        if directive_str:
            directive_str = " " + directive_str

        return f"{field_def.name}{args}: {field_def.type_name}{directive_str}"


def create_federated_schema(
    service_name: str = "cad-ml-platform",
) -> FederatedSchema:
    """Create a federation-ready schema for the CAD ML Platform.

    Args:
        service_name: Name of this service in the federation

    Returns:
        FederatedSchema instance
    """
    schema = FederatedSchema(name=service_name)

    # Register Document as an entity
    schema.add_entity(
        "Document",
        key_fields=["id"],
        type_def=TypeDefinition(
            name="Document",
            description="A CAD document (federated entity)",
            fields=[
                FieldDefinition("id", "ID!", "Unique identifier"),
                FieldDefinition("name", "String!", "Document name"),
                FieldDefinition("filePath", "String!", "Path to the file"),
                FieldDefinition("fileType", "String!", "File type"),
                FieldDefinition("status", "DocumentStatus!", "Processing status"),
                FieldDefinition("owner", "User", "Document owner"),
                FieldDefinition("createdAt", "DateTime!", "Creation timestamp"),
            ],
        ),
    )

    # Register Model as an entity
    schema.add_entity(
        "Model",
        key_fields=["id"],
        type_def=TypeDefinition(
            name="Model",
            description="An ML model (federated entity)",
            fields=[
                FieldDefinition("id", "ID!", "Unique identifier"),
                FieldDefinition("name", "String!", "Model name"),
                FieldDefinition("version", "String!", "Model version"),
                FieldDefinition("status", "ModelStatus!", "Model status"),
            ],
        ),
    )

    # Register User as an entity (extended from another service)
    schema.add_entity(
        "User",
        key_fields=["id"],
        type_def=TypeDefinition(
            name="User",
            description="A user (extended from auth service)",
            fields=[
                FieldDefinition("id", "ID!", "Unique identifier"),
                FieldDefinition("documents", "DocumentConnection", "User's documents"),
            ],
            directives=["@extends"],
        ),
    )
    schema.mark_external("User", "id")

    # Add queries
    schema.add_query(FieldDefinition(
        name="document",
        type_name="Document",
        args={"id": "ID!"},
    ))

    schema.add_query(FieldDefinition(
        name="documents",
        type_name="DocumentConnection!",
        args={"first": "Int", "after": "String"},
    ))

    schema.add_query(FieldDefinition(
        name="model",
        type_name="Model",
        args={"id": "ID!"},
    ))

    # Add mutations
    schema.add_mutation(FieldDefinition(
        name="createDocument",
        type_name="DocumentMutationResponse!",
        args={"input": "CreateDocumentInput!"},
    ))

    schema.add_mutation(FieldDefinition(
        name="predict",
        type_name="Prediction!",
        args={"input": "PredictionInput!"},
    ))

    return schema


class FederationGateway:
    """Gateway for coordinating federated GraphQL services."""

    def __init__(self):
        self._services: Dict[str, ServiceDefinition] = {}
        self._entity_map: Dict[str, str] = {}  # type_name -> service_name

    def register_service(self, service: ServiceDefinition) -> None:
        """Register a service with the gateway.

        Args:
            service: Service definition
        """
        self._services[service.name] = service

        # Map entity types to services
        for type_name in service.types:
            self._entity_map[type_name] = service.name

        logger.info(f"Gateway registered service: {service.name}")

    def get_service_for_type(self, type_name: str) -> Optional[ServiceDefinition]:
        """Get the service that owns a type.

        Args:
            type_name: GraphQL type name

        Returns:
            Service definition or None
        """
        service_name = self._entity_map.get(type_name)
        if service_name:
            return self._services.get(service_name)
        return None

    async def resolve_entity(
        self,
        type_name: str,
        representation: Dict[str, Any],
    ) -> Optional[Any]:
        """Resolve an entity reference through federation.

        Args:
            type_name: Entity type name
            representation: Entity representation with key fields

        Returns:
            Resolved entity
        """
        service = self.get_service_for_type(type_name)
        if not service:
            logger.warning(f"No service found for entity type: {type_name}")
            return None

        # In a real implementation, this would make a subgraph query
        # to the appropriate service
        logger.debug(f"Resolving {type_name} from {service.name}")
        return None

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all registered services.

        Returns:
            Dict mapping service name to health status
        """
        results = {}
        for name, service in self._services.items():
            try:
                # In real implementation, would check health endpoint
                results[name] = True
            except Exception:
                results[name] = False
        return results

    def compose_supergraph_sdl(self) -> str:
        """Compose supergraph SDL from all services.

        Returns:
            Combined SDL for the supergraph
        """
        parts = []
        for service in self._services.values():
            parts.append(f"# Service: {service.name}")
            parts.append(service.schema_sdl)
            parts.append("")
        return "\n".join(parts)
