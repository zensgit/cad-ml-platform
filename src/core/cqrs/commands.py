"""CQRS Commands.

Provides command handling infrastructure:
- Command definitions
- Command handlers
- Command bus
- Validation
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
class Command:
    """Base class for commands."""
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def command_type(self) -> str:
        return self.__class__.__name__


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    command_id: str
    aggregate_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


T = TypeVar('T', bound=Command)


class CommandHandler(ABC, Generic[T]):
    """Base class for command handlers."""

    @abstractmethod
    async def handle(self, command: T) -> CommandResult:
        """Handle a command.

        Args:
            command: The command to handle.

        Returns:
            CommandResult with outcome.
        """
        pass


class CommandValidator(ABC, Generic[T]):
    """Base class for command validators."""

    @abstractmethod
    def validate(self, command: T) -> List[str]:
        """Validate a command.

        Args:
            command: The command to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        pass


class CommandBus:
    """Dispatches commands to handlers."""

    def __init__(self):
        self._handlers: Dict[Type[Command], CommandHandler] = {}
        self._validators: Dict[Type[Command], List[CommandValidator]] = {}
        self._middleware: List[Callable] = []

    def register_handler(
        self,
        command_type: Type[T],
        handler: CommandHandler[T],
    ) -> None:
        """Register a handler for a command type."""
        self._handlers[command_type] = handler

    def register_validator(
        self,
        command_type: Type[T],
        validator: CommandValidator[T],
    ) -> None:
        """Register a validator for a command type."""
        if command_type not in self._validators:
            self._validators[command_type] = []
        self._validators[command_type].append(validator)

    def add_middleware(
        self,
        middleware: Callable[[Command, Callable], CommandResult],
    ) -> None:
        """Add middleware to the command pipeline."""
        self._middleware.append(middleware)

    async def dispatch(self, command: Command) -> CommandResult:
        """Dispatch a command to its handler.

        Args:
            command: The command to dispatch.

        Returns:
            CommandResult with outcome.
        """
        command_type = type(command)

        # Validate
        validators = self._validators.get(command_type, [])
        for validator in validators:
            errors = validator.validate(command)
            if errors:
                return CommandResult(
                    success=False,
                    command_id=command.command_id,
                    error="; ".join(errors),
                    error_code="VALIDATION_ERROR",
                )

        # Find handler
        handler = self._handlers.get(command_type)
        if not handler:
            return CommandResult(
                success=False,
                command_id=command.command_id,
                error=f"No handler registered for {command_type.__name__}",
                error_code="HANDLER_NOT_FOUND",
            )

        # Execute with middleware
        async def execute() -> CommandResult:
            return await handler.handle(command)

        result = await self._execute_with_middleware(command, execute)
        return result

    async def _execute_with_middleware(
        self,
        command: Command,
        handler: Callable,
    ) -> CommandResult:
        """Execute handler with middleware chain."""
        async def chain(index: int) -> CommandResult:
            if index >= len(self._middleware):
                return await handler()

            middleware = self._middleware[index]
            return await middleware(command, lambda: chain(index + 1))

        return await chain(0)


# Example Commands

@dataclass
class CreateOrderCommand(Command):
    """Command to create a new order."""
    customer_id: str = ""


@dataclass
class AddItemCommand(Command):
    """Command to add an item to an order."""
    order_id: str = ""
    product_id: str = ""
    quantity: int = 1
    price: float = 0.0


@dataclass
class RemoveItemCommand(Command):
    """Command to remove an item from an order."""
    order_id: str = ""
    product_id: str = ""


@dataclass
class SubmitOrderCommand(Command):
    """Command to submit an order."""
    order_id: str = ""


@dataclass
class CancelOrderCommand(Command):
    """Command to cancel an order."""
    order_id: str = ""
    reason: str = ""


# Example Handlers

class CreateOrderHandler(CommandHandler[CreateOrderCommand]):
    """Handler for CreateOrderCommand."""

    def __init__(self, repository):
        self._repository = repository

    async def handle(self, command: CreateOrderCommand) -> CommandResult:
        from src.core.eventsourcing.aggregate import Order

        try:
            order = Order()
            order.create(command.customer_id)
            await self._repository.save(order)

            return CommandResult(
                success=True,
                command_id=command.command_id,
                aggregate_id=order.id,
                data={"order_id": order.id},
            )
        except Exception as e:
            return CommandResult(
                success=False,
                command_id=command.command_id,
                error=str(e),
                error_code="CREATE_ORDER_FAILED",
            )


class AddItemHandler(CommandHandler[AddItemCommand]):
    """Handler for AddItemCommand."""

    def __init__(self, repository):
        self._repository = repository

    async def handle(self, command: AddItemCommand) -> CommandResult:
        try:
            order = await self._repository.get(command.order_id)
            if not order:
                return CommandResult(
                    success=False,
                    command_id=command.command_id,
                    error=f"Order not found: {command.order_id}",
                    error_code="ORDER_NOT_FOUND",
                )

            order.add_item(command.product_id, command.quantity, command.price)
            await self._repository.save(order)

            return CommandResult(
                success=True,
                command_id=command.command_id,
                aggregate_id=command.order_id,
            )
        except Exception as e:
            return CommandResult(
                success=False,
                command_id=command.command_id,
                error=str(e),
                error_code="ADD_ITEM_FAILED",
            )


# Example Validators

class CreateOrderValidator(CommandValidator[CreateOrderCommand]):
    """Validator for CreateOrderCommand."""

    def validate(self, command: CreateOrderCommand) -> List[str]:
        errors = []
        if not command.customer_id:
            errors.append("customer_id is required")
        return errors


class AddItemValidator(CommandValidator[AddItemCommand]):
    """Validator for AddItemCommand."""

    def validate(self, command: AddItemCommand) -> List[str]:
        errors = []
        if not command.order_id:
            errors.append("order_id is required")
        if not command.product_id:
            errors.append("product_id is required")
        if command.quantity <= 0:
            errors.append("quantity must be positive")
        if command.price < 0:
            errors.append("price cannot be negative")
        return errors


# Middleware examples

async def logging_middleware(
    command: Command,
    next_handler: Callable,
) -> CommandResult:
    """Middleware that logs command execution."""
    logger.info(f"Executing command: {command.command_type}")
    result = await next_handler()
    if result.success:
        logger.info(f"Command {command.command_type} succeeded")
    else:
        logger.warning(f"Command {command.command_type} failed: {result.error}")
    return result


async def timing_middleware(
    command: Command,
    next_handler: Callable,
) -> CommandResult:
    """Middleware that times command execution."""
    import time
    start = time.time()
    result = await next_handler()
    duration = time.time() - start
    logger.debug(f"Command {command.command_type} took {duration:.3f}s")
    return result


# Global command bus
_command_bus: Optional[CommandBus] = None


def get_command_bus() -> CommandBus:
    """Get the global command bus."""
    global _command_bus
    if _command_bus is None:
        _command_bus = CommandBus()
    return _command_bus
