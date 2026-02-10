"""Tests for workflow state_machine module to improve coverage."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.core.workflow.state_machine import (
    State,
    Transition,
    StateContext,
    StateMachine,
    StateMachineBuilder,
    WorkflowStates,
    create_approval_workflow,
    create_document_workflow,
)


class TestState:
    """Tests for State class."""

    def test_state_hash(self):
        """Test State.__hash__ (line 35)."""
        state = State(name="test_state")
        assert hash(state) == hash("test_state")

    def test_state_eq_with_state(self):
        """Test State.__eq__ with another State (lines 38-39)."""
        state1 = State(name="test")
        state2 = State(name="test")
        state3 = State(name="other")

        assert state1 == state2
        assert not (state1 == state3)

    def test_state_eq_with_string(self):
        """Test State.__eq__ with string (line 40)."""
        state = State(name="test")

        assert state == "test"
        assert not (state == "other")


class TestStateContext:
    """Tests for StateContext class."""

    def test_context_set_updates_timestamp(self):
        """Test StateContext.set updates timestamp."""
        context = StateContext(
            machine_id="m1",
            current_state="initial",
        )
        original_time = context.updated_at

        import time
        time.sleep(0.01)
        context.set("key", "value")

        assert context.data["key"] == "value"
        assert context.updated_at >= original_time

    def test_context_get_with_default(self):
        """Test StateContext.get with default value."""
        context = StateContext(
            machine_id="m1",
            current_state="initial",
        )

        result = context.get("nonexistent", "default_value")
        assert result == "default_value"


class TestStateMachine:
    """Tests for StateMachine class."""

    @pytest.fixture
    def simple_machine(self):
        """Create a simple state machine."""
        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)
        machine.add_state("middle")
        machine.add_state("final", is_final=True)
        machine.add_transition("go", "initial", "middle")
        machine.add_transition("finish", "middle", "final")
        return machine

    def test_add_state_duplicate_initial(self):
        """Test add_state raises error for duplicate initial state (line 123)."""
        machine = StateMachine(name="test")
        machine.add_state("state1", is_initial=True)

        with pytest.raises(ValueError, match="Initial state already set"):
            machine.add_state("state2", is_initial=True)

    def test_add_transition_source_not_found(self):
        """Test add_transition raises error for invalid source (line 151)."""
        machine = StateMachine(name="test")
        machine.add_state("target")

        with pytest.raises(ValueError, match="Source state not found"):
            machine.add_transition("event", "nonexistent", "target")

    def test_add_transition_target_not_found(self):
        """Test add_transition raises error for invalid target (line 153)."""
        machine = StateMachine(name="test")
        machine.add_state("source")

        with pytest.raises(ValueError, match="Target state not found"):
            machine.add_transition("event", "source", "nonexistent")

    def test_initialize_without_initial_state(self):
        """Test initialize raises error when no initial state (line 180)."""
        machine = StateMachine(name="test")
        machine.add_state("state1")  # Not initial

        with pytest.raises(ValueError, match="No initial state defined"):
            machine.initialize()

    def test_initialize_calls_on_enter(self):
        """Test initialize calls on_enter callback (line 192)."""
        on_enter_mock = MagicMock()

        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True, on_enter=on_enter_mock)

        context = machine.initialize()

        on_enter_mock.assert_called_once_with(context)

    def test_context_property(self, simple_machine):
        """Test context property (line 199)."""
        assert simple_machine.context is None

        simple_machine.initialize()
        assert simple_machine.context is not None
        assert simple_machine.context.current_state == "initial"

    def test_get_state(self, simple_machine):
        """Test get_state method (line 208)."""
        state = simple_machine.get_state("initial")
        assert state is not None
        assert state.name == "initial"

        none_state = simple_machine.get_state("nonexistent")
        assert none_state is None

    def test_get_available_transitions_not_initialized(self):
        """Test get_available_transitions without initialization (lines 212-213)."""
        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)

        transitions = machine.get_available_transitions()
        assert transitions == []

    def test_get_available_transitions_with_guard(self):
        """Test get_available_transitions respects guards (lines 216-219)."""
        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)
        machine.add_state("target1")
        machine.add_state("target2")

        # Add transition with guard that returns False
        machine.add_transition(
            "blocked", "initial", "target1",
            guard=lambda ctx: False
        )
        # Add transition with guard that returns True
        machine.add_transition(
            "allowed", "initial", "target2",
            guard=lambda ctx: True
        )

        machine.initialize()
        available = machine.get_available_transitions()

        assert len(available) == 1
        assert available[0].name == "allowed"

    def test_can_trigger(self, simple_machine):
        """Test can_trigger method (lines 223-226)."""
        simple_machine.initialize()

        assert simple_machine.can_trigger("go") is True
        assert simple_machine.can_trigger("nonexistent") is False

    def test_trigger_without_initialization(self):
        """Test trigger raises error when not initialized (line 238)."""
        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)

        with pytest.raises(ValueError, match="not initialized"):
            machine.trigger("event")

    def test_trigger_calls_on_exit(self):
        """Test trigger calls on_exit callback (line 264)."""
        on_exit_mock = MagicMock()

        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True, on_exit=on_exit_mock)
        machine.add_state("target")
        machine.add_transition("go", "initial", "target")

        machine.initialize()
        machine.trigger("go")

        on_exit_mock.assert_called_once()

    def test_trigger_calls_action(self):
        """Test trigger calls transition action (line 268)."""
        action_mock = MagicMock()

        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)
        machine.add_state("target")
        machine.add_transition("go", "initial", "target", action=action_mock)

        machine.initialize()
        machine.trigger("go")

        action_mock.assert_called_once()

    def test_trigger_calls_on_enter(self):
        """Test trigger calls on_enter callback for target state (line 277)."""
        on_enter_mock = MagicMock()

        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)
        machine.add_state("target", on_enter=on_enter_mock)
        machine.add_transition("go", "initial", "target")

        machine.initialize()
        on_enter_mock.reset_mock()  # Clear call from initialization
        machine.trigger("go")

        on_enter_mock.assert_called_once()

    def test_is_final_not_initialized(self):
        """Test is_final returns False when not initialized (line 288)."""
        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)

        assert machine.is_final() is False

    def test_is_final_with_initialized(self):
        """Test is_final returns correct value when initialized (lines 289-290)."""
        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)
        machine.add_state("final", is_final=True)
        machine.add_transition("finish", "initial", "final")

        machine.initialize()
        assert machine.is_final() is False

        machine.trigger("finish")
        assert machine.is_final() is True

    def test_reset_reinitializes(self, simple_machine):
        """Test reset method (line 294)."""
        simple_machine.initialize()
        simple_machine.trigger("go")

        assert simple_machine.current_state == "middle"

        simple_machine.reset()
        assert simple_machine.current_state == "initial"

    def test_get_reachable_states_no_context(self):
        """Test get_reachable_states without context uses initial (lines 298-300)."""
        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)
        machine.add_state("target")
        machine.add_transition("go", "initial", "target")

        reachable = machine.get_reachable_states()

        assert "initial" in reachable
        assert "target" in reachable

    def test_get_reachable_states_no_initial(self):
        """Test get_reachable_states without initial state returns empty set."""
        machine = StateMachine(name="test")
        machine.add_state("state1")  # Not initial

        reachable = machine.get_reachable_states()
        assert reachable == set()

    def test_get_reachable_states_from_specific_state(self):
        """Test get_reachable_states from specific state (lines 302-315)."""
        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)
        machine.add_state("middle")
        machine.add_state("final", is_final=True)
        machine.add_state("isolated")  # No transitions to/from

        machine.add_transition("go1", "initial", "middle")
        machine.add_transition("go2", "middle", "final")

        reachable = machine.get_reachable_states("middle")

        assert "middle" in reachable
        assert "final" in reachable
        assert "isolated" not in reachable

    def test_get_reachable_states_with_cycle(self):
        """Test get_reachable_states handles cycles (line 308 - already visited)."""
        machine = StateMachine(name="test")
        machine.add_state("a", is_initial=True)
        machine.add_state("b")
        machine.add_state("c")

        machine.add_transition("go_ab", "a", "b")
        machine.add_transition("go_bc", "b", "c")
        machine.add_transition("go_ca", "c", "a")  # Creates cycle

        reachable = machine.get_reachable_states("a")

        assert "a" in reachable
        assert "b" in reachable
        assert "c" in reachable


class TestStateMachineBuilder:
    """Tests for StateMachineBuilder class."""

    def test_on_without_state(self):
        """Test on raises error when no state defined (lines 376-377)."""
        builder = StateMachineBuilder(name="test")

        with pytest.raises(ValueError, match="No state defined yet"):
            builder.on("event", "target")

    def test_on_with_state(self):
        """Test on adds transition from last state (line 378)."""
        builder = (
            StateMachineBuilder(name="test")
            .initial_state("initial")
            .state("target")
        )
        # Go back to initial and add transition
        builder._last_state = "initial"
        builder.on("go", "target")

        machine = builder.build()
        machine.initialize()

        assert machine.can_trigger("go")


class TestCreateDocumentWorkflow:
    """Tests for create_document_workflow function."""

    def test_create_document_workflow(self):
        """Test create_document_workflow creates valid workflow (line 438)."""
        workflow = create_document_workflow()

        assert workflow is not None
        assert workflow.name == "DocumentWorkflow"

        # Initialize and check initial state
        workflow.initialize()
        assert workflow.current_state == "uploaded"


class TestCreateApprovalWorkflow:
    """Tests for create_approval_workflow function."""

    def test_create_approval_workflow(self):
        """Test create_approval_workflow creates valid workflow."""
        workflow = create_approval_workflow()

        assert workflow is not None
        assert workflow.name == "ApprovalWorkflow"

        workflow.initialize()
        assert workflow.current_state == WorkflowStates.DRAFT


class TestTransitionGuards:
    """Tests for transition guard functionality."""

    def test_trigger_with_guard_false_no_transition(self):
        """Test trigger returns False when guard fails."""
        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)
        machine.add_state("target")
        machine.add_transition(
            "blocked", "initial", "target",
            guard=lambda ctx: False
        )

        machine.initialize()
        result = machine.trigger("blocked")

        assert result is False
        assert machine.current_state == "initial"

    def test_trigger_with_guard_true_transitions(self):
        """Test trigger succeeds when guard passes."""
        machine = StateMachine(name="test")
        machine.add_state("initial", is_initial=True)
        machine.add_state("target")
        machine.add_transition(
            "allowed", "initial", "target",
            guard=lambda ctx: True
        )

        machine.initialize()
        result = machine.trigger("allowed")

        assert result is True
        assert machine.current_state == "target"
