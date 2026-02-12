"""CQRS Module.

Provides Command Query Responsibility Segregation:
- Command handling and validation
- Query handling and caching
- Read model separation
- Middleware pipeline
"""

from src.core.cqrs.commands import (
    Command,
    CommandResult,
    CommandHandler,
    CommandValidator,
    CommandBus,
    CreateOrderCommand,
    AddItemCommand,
    RemoveItemCommand,
    SubmitOrderCommand,
    CancelOrderCommand,
    CreateOrderHandler,
    AddItemHandler,
    CreateOrderValidator,
    AddItemValidator,
    logging_middleware,
    timing_middleware,
    get_command_bus,
)
from src.core.cqrs.queries import (
    Query,
    QueryResult,
    QueryHandler,
    QueryBus,
    GetOrderQuery,
    GetOrdersByCustomerQuery,
    GetOrdersByStatusQuery,
    SearchOrdersQuery,
    OrderReadModel,
    OrderListItem,
    GetOrderHandler,
    GetOrdersByCustomerHandler,
    GetOrdersByStatusHandler,
    caching_middleware,
    logging_middleware as query_logging_middleware,
    get_query_bus,
)

__all__ = [
    # Commands
    "Command",
    "CommandResult",
    "CommandHandler",
    "CommandValidator",
    "CommandBus",
    "CreateOrderCommand",
    "AddItemCommand",
    "RemoveItemCommand",
    "SubmitOrderCommand",
    "CancelOrderCommand",
    "CreateOrderHandler",
    "AddItemHandler",
    "CreateOrderValidator",
    "AddItemValidator",
    "logging_middleware",
    "timing_middleware",
    "get_command_bus",
    # Queries
    "Query",
    "QueryResult",
    "QueryHandler",
    "QueryBus",
    "GetOrderQuery",
    "GetOrdersByCustomerQuery",
    "GetOrdersByStatusQuery",
    "SearchOrdersQuery",
    "OrderReadModel",
    "OrderListItem",
    "GetOrderHandler",
    "GetOrdersByCustomerHandler",
    "GetOrdersByStatusHandler",
    "caching_middleware",
    "query_logging_middleware",
    "get_query_bus",
]
