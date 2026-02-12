"""Workflow State Machine.

Provides state machine primitives for workflow management:
- State definitions
- Transitions with guards
- Event-driven state changes
- State persistence
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)


@dataclass
class State:
    """A state in the state machine."""
    name: str
    is_initial: bool = False
    is_final: bool = False
    on_enter: Optional[Callable[["StateContext"], None]] = None
    on_exit: Optional[Callable[["StateContext"], None]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.name == other.name
        return self.name == other


@dataclass
class Transition:
    """A transition between states."""
    name: str
    source: str  # State name
    target: str  # State name
    guard: Optional[Callable[["StateContext"], bool]] = None
    action: Optional[Callable[["StateContext"], Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateContext:
    """Context for state machine execution."""
    machine_id: str
    current_state: str
    data: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def set(self, key: str, value: Any) -> None:
        """Set a value in context data."""
        self.data[key] = value
        self.updated_at = datetime.utcnow()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context data."""
        return self.data.get(key, default)


class StateMachine:
    """Finite state machine implementation."""

    def __init__(
        self,
        machine_id: Optional[str] = None,
        name: str = "StateMachine",
    ):
        self.machine_id = machine_id or str(uuid.uuid4())
        self.name = name
        self._states: Dict[str, State] = {}
        self._transitions: Dict[str, List[Transition]] = {}  # source -> transitions
        self._initial_state: Optional[str] = None
        self._context: Optional[StateContext] = None

    def add_state(
        self,
        name: str,
        is_initial: bool = False,
        is_final: bool = False,
        on_enter: Optional[Callable[[StateContext], None]] = None,
        on_exit: Optional[Callable[[StateContext], None]] = None,
        **metadata: Any,
    ) -> "StateMachine":
        """Add a state to the machine.

        Args:
            name: State name.
            is_initial: Whether this is the initial state.
            is_final: Whether this is a final state.
            on_enter: Callback when entering state.
            on_exit: Callback when exiting state.
            **metadata: Additional metadata.

        Returns:
            Self for chaining.
        """
        state = State(
            name=name,
            is_initial=is_initial,
            is_final=is_final,
            on_enter=on_enter,
            on_exit=on_exit,
            metadata=metadata,
        )
        self._states[name] = state

        if is_initial:
            if self._initial_state:
                raise ValueError(f"Initial state already set: {self._initial_state}")
            self._initial_state = name

        return self

    def add_transition(
        self,
        name: str,
        source: str,
        target: str,
        guard: Optional[Callable[[StateContext], bool]] = None,
        action: Optional[Callable[[StateContext], Any]] = None,
        **metadata: Any,
    ) -> "StateMachine":
        """Add a transition between states.

        Args:
            name: Transition name (event trigger).
            source: Source state name.
            target: Target state name.
            guard: Optional guard condition.
            action: Optional action to execute.
            **metadata: Additional metadata.

        Returns:
            Self for chaining.
        """
        if source not in self._states:
            raise ValueError(f"Source state not found: {source}")
        if target not in self._states:
            raise ValueError(f"Target state not found: {target}")

        transition = Transition(
            name=name,
            source=source,
            target=target,
            guard=guard,
            action=action,
            metadata=metadata,
        )

        if source not in self._transitions:
            self._transitions[source] = []
        self._transitions[source].append(transition)

        return self

    def initialize(self, data: Optional[Dict[str, Any]] = None) -> StateContext:
        """Initialize the state machine.

        Args:
            data: Initial context data.

        Returns:
            StateContext for the machine.
        """
        if not self._initial_state:
            raise ValueError("No initial state defined")

        self._context = StateContext(
            machine_id=self.machine_id,
            current_state=self._initial_state,
            data=data or {},
            history=[self._initial_state],
        )

        # Call on_enter for initial state
        initial = self._states[self._initial_state]
        if initial.on_enter:
            initial.on_enter(self._context)

        return self._context

    @property
    def context(self) -> Optional[StateContext]:
        """Get the current context."""
        return self._context

    @property
    def current_state(self) -> Optional[str]:
        """Get the current state name."""
        return self._context.current_state if self._context else None

    def get_state(self, name: str) -> Optional[State]:
        """Get a state by name."""
        return self._states.get(name)

    def get_available_transitions(self) -> List[Transition]:
        """Get transitions available from current state."""
        if not self._context:
            return []

        transitions = self._transitions.get(self._context.current_state, [])
        return [
            t for t in transitions
            if not t.guard or t.guard(self._context)
        ]

    def can_trigger(self, event: str) -> bool:
        """Check if an event can be triggered from current state."""
        for t in self.get_available_transitions():
            if t.name == event:
                return True
        return False

    def trigger(self, event: str) -> bool:
        """Trigger a transition by event name.

        Args:
            event: Event/transition name.

        Returns:
            True if transition was successful.
        """
        if not self._context:
            raise ValueError("State machine not initialized")

        # Find matching transition
        transitions = self._transitions.get(self._context.current_state, [])
        transition = None

        for t in transitions:
            if t.name == event:
                # Check guard
                if t.guard and not t.guard(self._context):
                    continue
                transition = t
                break

        if not transition:
            logger.warning(
                f"No valid transition for event '{event}' from state '{self._context.current_state}'"
            )
            return False

        # Execute transition
        source_state = self._states[transition.source]
        target_state = self._states[transition.target]

        # Exit current state
        if source_state.on_exit:
            source_state.on_exit(self._context)

        # Execute transition action
        if transition.action:
            transition.action(self._context)

        # Update state
        self._context.current_state = transition.target
        self._context.history.append(transition.target)
        self._context.updated_at = datetime.utcnow()

        # Enter new state
        if target_state.on_enter:
            target_state.on_enter(self._context)

        logger.info(
            f"State machine {self.machine_id}: {transition.source} -> {transition.target} (event: {event})"
        )

        return True

    def is_final(self) -> bool:
        """Check if current state is final."""
        if not self._context:
            return False
        state = self._states.get(self._context.current_state)
        return state.is_final if state else False

    def reset(self, data: Optional[Dict[str, Any]] = None) -> StateContext:
        """Reset the state machine to initial state."""
        return self.initialize(data)

    def get_reachable_states(self, from_state: Optional[str] = None) -> Set[str]:
        """Get all states reachable from a given state."""
        start = from_state or self._context.current_state if self._context else self._initial_state
        if not start:
            return set()

        reachable = set()
        queue = [start]

        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)

            for transition in self._transitions.get(current, []):
                if transition.target not in reachable:
                    queue.append(transition.target)

        return reachable


class StateMachineBuilder:
    """Fluent builder for state machines."""

    def __init__(self, name: str = "StateMachine"):
        self._machine = StateMachine(name=name)
        self._last_state: Optional[str] = None

    def state(
        self,
        name: str,
        is_initial: bool = False,
        is_final: bool = False,
        on_enter: Optional[Callable[[StateContext], None]] = None,
        on_exit: Optional[Callable[[StateContext], None]] = None,
    ) -> "StateMachineBuilder":
        """Add a state."""
        self._machine.add_state(
            name, is_initial, is_final, on_enter, on_exit
        )
        self._last_state = name
        return self

    def initial_state(
        self,
        name: str,
        on_enter: Optional[Callable[[StateContext], None]] = None,
    ) -> "StateMachineBuilder":
        """Add initial state."""
        return self.state(name, is_initial=True, on_enter=on_enter)

    def final_state(
        self,
        name: str,
        on_enter: Optional[Callable[[StateContext], None]] = None,
    ) -> "StateMachineBuilder":
        """Add final state."""
        return self.state(name, is_final=True, on_enter=on_enter)

    def transition(
        self,
        event: str,
        source: str,
        target: str,
        guard: Optional[Callable[[StateContext], bool]] = None,
        action: Optional[Callable[[StateContext], Any]] = None,
    ) -> "StateMachineBuilder":
        """Add a transition."""
        self._machine.add_transition(event, source, target, guard, action)
        return self

    def on(
        self,
        event: str,
        target: str,
        guard: Optional[Callable[[StateContext], bool]] = None,
        action: Optional[Callable[[StateContext], Any]] = None,
    ) -> "StateMachineBuilder":
        """Add transition from last state."""
        if not self._last_state:
            raise ValueError("No state defined yet")
        return self.transition(event, self._last_state, target, guard, action)

    def build(self) -> StateMachine:
        """Build and return the state machine."""
        return self._machine


# Predefined workflow states
class WorkflowStates(str, Enum):
    """Common workflow states."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def create_approval_workflow() -> StateMachine:
    """Create a standard approval workflow state machine."""
    return (
        StateMachineBuilder("ApprovalWorkflow")
        .initial_state(WorkflowStates.DRAFT)
        .state(WorkflowStates.PENDING_APPROVAL)
        .state(WorkflowStates.APPROVED)
        .state(WorkflowStates.REJECTED)
        .state(WorkflowStates.IN_PROGRESS)
        .state(WorkflowStates.PAUSED)
        .final_state(WorkflowStates.COMPLETED)
        .final_state(WorkflowStates.FAILED)
        .final_state(WorkflowStates.CANCELLED)
        # Draft transitions
        .transition("submit", WorkflowStates.DRAFT, WorkflowStates.PENDING_APPROVAL)
        .transition("cancel", WorkflowStates.DRAFT, WorkflowStates.CANCELLED)
        # Approval transitions
        .transition("approve", WorkflowStates.PENDING_APPROVAL, WorkflowStates.APPROVED)
        .transition("reject", WorkflowStates.PENDING_APPROVAL, WorkflowStates.REJECTED)
        .transition("cancel", WorkflowStates.PENDING_APPROVAL, WorkflowStates.CANCELLED)
        # Approved transitions
        .transition("start", WorkflowStates.APPROVED, WorkflowStates.IN_PROGRESS)
        .transition("cancel", WorkflowStates.APPROVED, WorkflowStates.CANCELLED)
        # Rejected transitions
        .transition("revise", WorkflowStates.REJECTED, WorkflowStates.DRAFT)
        # In progress transitions
        .transition("pause", WorkflowStates.IN_PROGRESS, WorkflowStates.PAUSED)
        .transition("complete", WorkflowStates.IN_PROGRESS, WorkflowStates.COMPLETED)
        .transition("fail", WorkflowStates.IN_PROGRESS, WorkflowStates.FAILED)
        .transition("cancel", WorkflowStates.IN_PROGRESS, WorkflowStates.CANCELLED)
        # Paused transitions
        .transition("resume", WorkflowStates.PAUSED, WorkflowStates.IN_PROGRESS)
        .transition("cancel", WorkflowStates.PAUSED, WorkflowStates.CANCELLED)
        .build()
    )


def create_document_workflow() -> StateMachine:
    """Create a document processing workflow state machine."""
    return (
        StateMachineBuilder("DocumentWorkflow")
        .initial_state("uploaded")
        .state("validating")
        .state("validated")
        .state("processing")
        .state("processed")
        .state("reviewing")
        .state("reviewed")
        .final_state("published")
        .final_state("archived")
        .final_state("error")
        # Upload -> Validate
        .transition("validate", "uploaded", "validating")
        .transition("error", "uploaded", "error")
        # Validating
        .transition("pass", "validating", "validated")
        .transition("fail", "validating", "error")
        # Validated -> Process
        .transition("process", "validated", "processing")
        # Processing
        .transition("complete", "processing", "processed")
        .transition("fail", "processing", "error")
        # Processed -> Review
        .transition("review", "processed", "reviewing")
        .transition("skip_review", "processed", "reviewed")
        # Reviewing
        .transition("approve", "reviewing", "reviewed")
        .transition("reject", "reviewing", "processed")
        # Reviewed -> Publish
        .transition("publish", "reviewed", "published")
        .transition("archive", "reviewed", "archived")
        # Error recovery
        .transition("retry", "error", "uploaded")
        .build()
    )
