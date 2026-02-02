"""GraphQL Gateway module.

Provides:
- Schema definition and types
- Query and mutation resolvers
- Federation support
- DataLoader for N+1 prevention
"""

from src.core.graphql.schema import (
    GraphQLSchema,
    create_schema,
    get_schema,
)
from src.core.graphql.types import (
    DocumentType,
    ModelType,
    UserType,
    PaginationType,
    PageInfo,
)
from src.core.graphql.resolvers import (
    Resolver,
    QueryResolver,
    MutationResolver,
    SubscriptionResolver,
)
from src.core.graphql.federation import (
    FederatedSchema,
    ServiceDefinition,
    create_federated_schema,
)
from src.core.graphql.dataloader import (
    DataLoader,
    BatchLoader,
    CachedDataLoader,
    create_dataloaders,
)

__all__ = [
    # Schema
    "GraphQLSchema",
    "create_schema",
    "get_schema",
    # Types
    "DocumentType",
    "ModelType",
    "UserType",
    "PaginationType",
    "PageInfo",
    # Resolvers
    "Resolver",
    "QueryResolver",
    "MutationResolver",
    "SubscriptionResolver",
    # Federation
    "FederatedSchema",
    "ServiceDefinition",
    "create_federated_schema",
    # DataLoader
    "DataLoader",
    "BatchLoader",
    "CachedDataLoader",
    "create_dataloaders",
]
