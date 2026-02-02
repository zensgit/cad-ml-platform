"""CQRS Queries.

Provides query handling infrastructure:
- Query definitions
- Query handlers
- Query bus
- Read models
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)


@dataclass
class Query:
    """Base class for queries."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def query_type(self) -> str:
        return self.__class__.__name__


@dataclass
class QueryResult(Generic[TypeVar('T')]):
    """Result of query execution."""
    success: bool
    query_id: str
    data: Optional[Any] = None
    error: Optional[str] = None
    total_count: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None


T = TypeVar('T', bound=Query)
R = TypeVar('R')


class QueryHandler(ABC, Generic[T, R]):
    """Base class for query handlers."""

    @abstractmethod
    async def handle(self, query: T) -> QueryResult[R]:
        """Handle a query.

        Args:
            query: The query to handle.

        Returns:
            QueryResult with data.
        """
        pass


class QueryBus:
    """Dispatches queries to handlers."""

    def __init__(self):
        self._handlers: Dict[Type[Query], QueryHandler] = {}
        self._middleware: List[Callable] = []

    def register_handler(
        self,
        query_type: Type[T],
        handler: QueryHandler[T, Any],
    ) -> None:
        """Register a handler for a query type."""
        self._handlers[query_type] = handler

    def add_middleware(
        self,
        middleware: Callable[[Query, Callable], QueryResult],
    ) -> None:
        """Add middleware to the query pipeline."""
        self._middleware.append(middleware)

    async def dispatch(self, query: Query) -> QueryResult:
        """Dispatch a query to its handler.

        Args:
            query: The query to dispatch.

        Returns:
            QueryResult with data.
        """
        query_type = type(query)

        handler = self._handlers.get(query_type)
        if not handler:
            return QueryResult(
                success=False,
                query_id=query.query_id,
                error=f"No handler registered for {query_type.__name__}",
            )

        # Execute with middleware
        async def execute() -> QueryResult:
            return await handler.handle(query)

        return await self._execute_with_middleware(query, execute)

    async def _execute_with_middleware(
        self,
        query: Query,
        handler: Callable,
    ) -> QueryResult:
        """Execute handler with middleware chain."""
        async def chain(index: int) -> QueryResult:
            if index >= len(self._middleware):
                return await handler()

            middleware = self._middleware[index]
            return await middleware(query, lambda: chain(index + 1))

        return await chain(0)


# Example Queries

@dataclass
class GetOrderQuery(Query):
    """Query to get an order by ID."""
    order_id: str = ""


@dataclass
class GetOrdersByCustomerQuery(Query):
    """Query to get orders for a customer."""
    customer_id: str = ""
    status: Optional[str] = None
    page: int = 1
    page_size: int = 10


@dataclass
class GetOrdersByStatusQuery(Query):
    """Query to get orders by status."""
    status: str = ""
    page: int = 1
    page_size: int = 10


@dataclass
class SearchOrdersQuery(Query):
    """Query to search orders."""
    customer_id: Optional[str] = None
    status: Optional[str] = None
    min_total: Optional[float] = None
    max_total: Optional[float] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    page: int = 1
    page_size: int = 10


# Example Read Models

@dataclass
class OrderReadModel:
    """Read model for orders."""
    order_id: str
    customer_id: str
    status: str
    items: List[Dict[str, Any]]
    total: float
    created_at: datetime
    updated_at: Optional[datetime] = None


@dataclass
class OrderListItem:
    """Lightweight order for lists."""
    order_id: str
    customer_id: str
    status: str
    item_count: int
    total: float
    created_at: datetime


# Example Handlers

class GetOrderHandler(QueryHandler[GetOrderQuery, OrderReadModel]):
    """Handler for GetOrderQuery."""

    def __init__(self, projection):
        self._projection = projection

    async def handle(self, query: GetOrderQuery) -> QueryResult[OrderReadModel]:
        order = self._projection.get_order(query.order_id)

        if not order:
            return QueryResult(
                success=False,
                query_id=query.query_id,
                error=f"Order not found: {query.order_id}",
            )

        return QueryResult(
            success=True,
            query_id=query.query_id,
            data=order,
        )


class GetOrdersByCustomerHandler(QueryHandler[GetOrdersByCustomerQuery, List[OrderListItem]]):
    """Handler for GetOrdersByCustomerQuery."""

    def __init__(self, projection):
        self._projection = projection

    async def handle(self, query: GetOrdersByCustomerQuery) -> QueryResult[List[OrderListItem]]:
        orders = self._projection.get_orders_by_customer(query.customer_id)

        if query.status:
            orders = [o for o in orders if o["status"] == query.status]

        # Pagination
        total_count = len(orders)
        start = (query.page - 1) * query.page_size
        end = start + query.page_size
        page_orders = orders[start:end]

        return QueryResult(
            success=True,
            query_id=query.query_id,
            data=page_orders,
            total_count=total_count,
            page=query.page,
            page_size=query.page_size,
        )


class GetOrdersByStatusHandler(QueryHandler[GetOrdersByStatusQuery, List[OrderListItem]]):
    """Handler for GetOrdersByStatusQuery."""

    def __init__(self, projection):
        self._projection = projection

    async def handle(self, query: GetOrdersByStatusQuery) -> QueryResult[List[OrderListItem]]:
        orders = self._projection.get_orders_by_status(query.status)

        # Pagination
        total_count = len(orders)
        start = (query.page - 1) * query.page_size
        end = start + query.page_size
        page_orders = orders[start:end]

        return QueryResult(
            success=True,
            query_id=query.query_id,
            data=page_orders,
            total_count=total_count,
            page=query.page,
            page_size=query.page_size,
        )


# Middleware

async def caching_middleware(
    query: Query,
    next_handler: Callable,
    cache: Optional[Dict[str, Any]] = None,
) -> QueryResult:
    """Middleware that caches query results."""
    if cache is None:
        return await next_handler()

    cache_key = f"{query.query_type}:{hash(str(query.__dict__))}"

    if cache_key in cache:
        logger.debug(f"Cache hit for {query.query_type}")
        return cache[cache_key]

    result = await next_handler()

    if result.success:
        cache[cache_key] = result

    return result


async def logging_middleware(
    query: Query,
    next_handler: Callable,
) -> QueryResult:
    """Middleware that logs query execution."""
    logger.info(f"Executing query: {query.query_type}")
    result = await next_handler()
    if result.success:
        logger.info(f"Query {query.query_type} succeeded")
    else:
        logger.warning(f"Query {query.query_type} failed: {result.error}")
    return result


# Global query bus
_query_bus: Optional[QueryBus] = None


def get_query_bus() -> QueryBus:
    """Get the global query bus."""
    global _query_bus
    if _query_bus is None:
        _query_bus = QueryBus()
    return _query_bus
