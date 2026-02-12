"""GraphQL Resolvers.

Provides resolver base classes and implementations.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

from src.core.graphql.types import (
    DocumentType,
    ModelType,
    UserType,
    Connection,
    Edge,
    PageInfo,
    MutationResponse,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ResolverContext:
    """Context passed to resolvers."""
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    roles: List[str] = None
    permissions: List[str] = None
    request: Optional[Any] = None
    dataloaders: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.permissions is None:
            self.permissions = []

    def has_permission(self, permission: str) -> bool:
        """Check if context has permission."""
        return permission in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if context has role."""
        return role in self.roles


class Resolver(ABC):
    """Base class for GraphQL resolvers."""

    @abstractmethod
    async def resolve(self, parent: Any, info: Any, **kwargs: Any) -> Any:
        """Resolve the field.

        Args:
            parent: Parent object
            info: GraphQL resolve info
            **kwargs: Field arguments

        Returns:
            Resolved value
        """
        pass


class QueryResolver(Resolver):
    """Base class for query resolvers."""
    pass


class MutationResolver(Resolver):
    """Base class for mutation resolvers."""

    async def validate(self, context: ResolverContext, **kwargs: Any) -> Optional[str]:
        """Validate mutation input.

        Returns:
            Error message if validation fails, None otherwise
        """
        return None


class SubscriptionResolver(Resolver):
    """Base class for subscription resolvers."""

    @abstractmethod
    async def subscribe(self, parent: Any, info: Any, **kwargs: Any):
        """Subscribe to events.

        Yields:
            Events as they occur
        """
        pass

    async def resolve(self, parent: Any, info: Any, **kwargs: Any) -> Any:
        """Default resolve returns the subscription."""
        return parent


# ============================================================================
# Document Resolvers
# ============================================================================

class DocumentQueryResolver(QueryResolver):
    """Resolver for document queries."""

    def __init__(self, repository: Any = None):
        self.repository = repository

    async def resolve(self, parent: Any, info: Any, **kwargs: Any) -> Optional[DocumentType]:
        """Resolve single document query."""
        document_id = kwargs.get("id")
        if not document_id:
            return None

        # Use dataloader if available
        context = getattr(info, "context", None)
        if context and hasattr(context, "dataloaders"):
            loader = context.dataloaders.get("documents")
            if loader:
                return await loader.load(document_id)

        # Fallback to repository
        if self.repository:
            return await self.repository.get(document_id)

        return None


class DocumentsQueryResolver(QueryResolver):
    """Resolver for documents list query."""

    def __init__(self, repository: Any = None):
        self.repository = repository

    async def resolve(
        self,
        parent: Any,
        info: Any,
        first: int = 20,
        after: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Connection[DocumentType]:
        """Resolve documents list query with pagination."""
        # Decode cursor
        offset = 0
        if after:
            try:
                offset = int(base64.b64decode(after).decode()) + 1
            except Exception:
                offset = 0

        # Fetch documents
        documents = []
        total_count = 0

        if self.repository:
            documents = await self.repository.list(
                offset=offset,
                limit=first + 1,  # Fetch one extra to check hasNextPage
                filter=filter,
            )
            total_count = await self.repository.count(filter=filter)

        # Build edges
        has_next_page = len(documents) > first
        documents = documents[:first]

        edges = []
        for i, doc in enumerate(documents):
            cursor = base64.b64encode(str(offset + i).encode()).decode()
            edges.append(Edge(node=doc, cursor=cursor))

        # Build page info
        page_info = PageInfo(
            has_next_page=has_next_page,
            has_previous_page=offset > 0,
            start_cursor=edges[0].cursor if edges else None,
            end_cursor=edges[-1].cursor if edges else None,
            total_count=total_count,
        )

        return Connection(
            edges=edges,
            page_info=page_info,
            total_count=total_count,
        )


class CreateDocumentMutationResolver(MutationResolver):
    """Resolver for createDocument mutation."""

    def __init__(self, repository: Any = None):
        self.repository = repository

    async def validate(self, context: ResolverContext, **kwargs: Any) -> Optional[str]:
        """Validate document creation."""
        input_data = kwargs.get("input", {})

        if not input_data.get("name"):
            return "Document name is required"

        if not input_data.get("file_path"):
            return "File path is required"

        return None

    async def resolve(
        self,
        parent: Any,
        info: Any,
        input: Dict[str, Any],
        **kwargs: Any,
    ) -> MutationResponse:
        """Create a new document."""
        context = getattr(info, "context", ResolverContext())

        # Validate
        error = await self.validate(context, input=input)
        if error:
            return MutationResponse(success=False, message=error)

        try:
            # Create document
            document = None
            if self.repository:
                document = await self.repository.create(
                    name=input.get("name"),
                    file_path=input.get("file_path"),
                    file_type=input.get("file_type"),
                    metadata=input.get("metadata", {}),
                    tags=input.get("tags", []),
                    owner_id=context.user_id,
                    tenant_id=context.tenant_id,
                )

            return MutationResponse(
                success=True,
                message="Document created successfully",
                data=document,
            )

        except Exception as e:
            logger.error(f"Error creating document: {e}")
            return MutationResponse(
                success=False,
                message=str(e),
                errors=[str(e)],
            )


class DeleteDocumentMutationResolver(MutationResolver):
    """Resolver for deleteDocument mutation."""

    def __init__(self, repository: Any = None):
        self.repository = repository

    async def resolve(
        self,
        parent: Any,
        info: Any,
        id: str,
        **kwargs: Any,
    ) -> MutationResponse:
        """Delete a document."""
        try:
            if self.repository:
                await self.repository.delete(id)

            return MutationResponse(
                success=True,
                message="Document deleted successfully",
            )

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return MutationResponse(
                success=False,
                message=str(e),
                errors=[str(e)],
            )


# ============================================================================
# Model Resolvers
# ============================================================================

class ModelQueryResolver(QueryResolver):
    """Resolver for model queries."""

    def __init__(self, repository: Any = None):
        self.repository = repository

    async def resolve(self, parent: Any, info: Any, **kwargs: Any) -> Optional[ModelType]:
        """Resolve single model query."""
        model_id = kwargs.get("id")
        if not model_id:
            return None

        if self.repository:
            return await self.repository.get(model_id)

        return None


class PredictMutationResolver(MutationResolver):
    """Resolver for predict mutation."""

    def __init__(self, model_service: Any = None):
        self.model_service = model_service

    async def resolve(
        self,
        parent: Any,
        info: Any,
        input: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make a prediction."""
        model_id = input.get("model_id")
        document_id = input.get("document_id")
        input_data = input.get("input_data")

        try:
            result = None
            if self.model_service:
                result = await self.model_service.predict(
                    model_id=model_id,
                    document_id=document_id,
                    input_data=input_data,
                )

            return result or {"prediction": None, "confidence": 0.0}

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


# ============================================================================
# Subscription Resolvers
# ============================================================================

class DocumentCreatedSubscriptionResolver(SubscriptionResolver):
    """Resolver for documentCreated subscription."""

    def __init__(self, pubsub: Any = None):
        self.pubsub = pubsub

    async def subscribe(self, parent: Any, info: Any, **kwargs: Any):
        """Subscribe to document creation events."""
        if self.pubsub:
            async for event in self.pubsub.subscribe("document.created"):
                yield event
        else:
            # Fallback: No pubsub, yield nothing
            return

    async def resolve(self, parent: Any, info: Any, **kwargs: Any) -> Any:
        return parent


class JobProgressSubscriptionResolver(SubscriptionResolver):
    """Resolver for jobProgress subscription."""

    def __init__(self, pubsub: Any = None):
        self.pubsub = pubsub

    async def subscribe(self, parent: Any, info: Any, job_id: str, **kwargs: Any):
        """Subscribe to job progress events."""
        if self.pubsub:
            channel = f"job.progress.{job_id}"
            async for event in self.pubsub.subscribe(channel):
                yield event

    async def resolve(self, parent: Any, info: Any, **kwargs: Any) -> Any:
        return parent


# ============================================================================
# Resolver Registry
# ============================================================================

class ResolverRegistry:
    """Registry for GraphQL resolvers."""

    def __init__(self):
        self._resolvers: Dict[str, Dict[str, Resolver]] = {
            "Query": {},
            "Mutation": {},
            "Subscription": {},
        }
        self._field_resolvers: Dict[str, Dict[str, Resolver]] = {}

    def register_query(self, name: str, resolver: Resolver) -> None:
        """Register a query resolver."""
        self._resolvers["Query"][name] = resolver

    def register_mutation(self, name: str, resolver: Resolver) -> None:
        """Register a mutation resolver."""
        self._resolvers["Mutation"][name] = resolver

    def register_subscription(self, name: str, resolver: Resolver) -> None:
        """Register a subscription resolver."""
        self._resolvers["Subscription"][name] = resolver

    def register_field(self, type_name: str, field_name: str, resolver: Resolver) -> None:
        """Register a field resolver."""
        if type_name not in self._field_resolvers:
            self._field_resolvers[type_name] = {}
        self._field_resolvers[type_name][field_name] = resolver

    def get_resolver(self, type_name: str, field_name: str) -> Optional[Resolver]:
        """Get a resolver."""
        if type_name in self._resolvers:
            return self._resolvers[type_name].get(field_name)
        if type_name in self._field_resolvers:
            return self._field_resolvers[type_name].get(field_name)
        return None


# Global resolver registry
_resolver_registry: Optional[ResolverRegistry] = None


def get_resolver_registry() -> ResolverRegistry:
    """Get global resolver registry."""
    global _resolver_registry
    if _resolver_registry is None:
        _resolver_registry = ResolverRegistry()
    return _resolver_registry


# ============================================================================
# Resolver Decorators
# ============================================================================

def require_auth(resolver_func: Callable) -> Callable:
    """Decorator to require authentication."""
    @wraps(resolver_func)
    async def wrapper(self, parent: Any, info: Any, **kwargs: Any) -> Any:
        context = getattr(info, "context", None)
        if not context or not context.user_id:
            raise PermissionError("Authentication required")
        return await resolver_func(self, parent, info, **kwargs)
    return wrapper


def require_permission(permission: str) -> Callable:
    """Decorator to require specific permission."""
    def decorator(resolver_func: Callable) -> Callable:
        @wraps(resolver_func)
        async def wrapper(self, parent: Any, info: Any, **kwargs: Any) -> Any:
            context = getattr(info, "context", None)
            if not context or not context.has_permission(permission):
                raise PermissionError(f"Permission required: {permission}")
            return await resolver_func(self, parent, info, **kwargs)
        return wrapper
    return decorator


def require_role(role: str) -> Callable:
    """Decorator to require specific role."""
    def decorator(resolver_func: Callable) -> Callable:
        @wraps(resolver_func)
        async def wrapper(self, parent: Any, info: Any, **kwargs: Any) -> Any:
            context = getattr(info, "context", None)
            if not context or not context.has_role(role):
                raise PermissionError(f"Role required: {role}")
            return await resolver_func(self, parent, info, **kwargs)
        return wrapper
    return decorator
