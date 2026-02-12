"""
Search Space Definition for Hyperparameter Tuning.

Provides declarative search space definition for Optuna optimization.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ParamType(str, Enum):
    """Parameter types for hyperparameter search."""
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    LOG_INT = "log_int"
    LOG_FLOAT = "log_float"


@dataclass
class HyperParameter(ABC):
    """Base class for hyperparameter definition."""
    name: str
    description: str = ""

    @abstractmethod
    def suggest(self, trial: Any) -> Any:
        """Suggest a value using Optuna trial."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        pass


@dataclass
class IntParam(HyperParameter):
    """Integer hyperparameter."""
    low: int = 1
    high: int = 100
    step: int = 1
    log: bool = False

    def suggest(self, trial: Any) -> int:
        """Suggest integer value."""
        if self.log:
            return trial.suggest_int(self.name, self.low, self.high, log=True)
        return trial.suggest_int(self.name, self.low, self.high, step=self.step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": ParamType.LOG_INT.value if self.log else ParamType.INT.value,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
            "description": self.description,
        }


@dataclass
class FloatParam(HyperParameter):
    """Float hyperparameter."""
    low: float = 0.0
    high: float = 1.0
    step: Optional[float] = None
    log: bool = False

    def suggest(self, trial: Any) -> float:
        """Suggest float value."""
        if self.log:
            return trial.suggest_float(self.name, self.low, self.high, log=True)
        if self.step:
            return trial.suggest_float(self.name, self.low, self.high, step=self.step)
        return trial.suggest_float(self.name, self.low, self.high)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": ParamType.LOG_FLOAT.value if self.log else ParamType.FLOAT.value,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
            "description": self.description,
        }


@dataclass
class CategoricalParam(HyperParameter):
    """Categorical hyperparameter."""
    choices: List[Any] = field(default_factory=list)

    def suggest(self, trial: Any) -> Any:
        """Suggest categorical value."""
        return trial.suggest_categorical(self.name, self.choices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": ParamType.CATEGORICAL.value,
            "choices": self.choices,
            "description": self.description,
        }


@dataclass
class ConditionalParam(HyperParameter):
    """Conditional hyperparameter that depends on another parameter."""
    parent_name: str = ""
    parent_values: List[Any] = field(default_factory=list)
    child_param: HyperParameter = field(default_factory=lambda: IntParam(name="default"))

    def suggest(self, trial: Any) -> Optional[Any]:
        """Suggest value if condition is met."""
        parent_value = trial.params.get(self.parent_name)
        if parent_value in self.parent_values:
            return self.child_param.suggest(trial)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "conditional",
            "parent_name": self.parent_name,
            "parent_values": self.parent_values,
            "child_param": self.child_param.to_dict(),
            "description": self.description,
        }


class SearchSpace:
    """
    Search space for hyperparameter optimization.

    Provides declarative definition of hyperparameter search spaces
    with support for conditional parameters and constraints.
    """

    def __init__(self, name: str = "default"):
        """
        Initialize search space.

        Args:
            name: Name of the search space
        """
        self._name = name
        self._params: Dict[str, HyperParameter] = {}
        self._constraints: List[Callable[[Dict[str, Any]], bool]] = []
        self._defaults: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> Dict[str, HyperParameter]:
        return self._params.copy()

    @property
    def param_names(self) -> List[str]:
        return list(self._params.keys())

    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        step: int = 1,
        log: bool = False,
        description: str = "",
        default: Optional[int] = None,
    ) -> "SearchSpace":
        """Add integer parameter."""
        self._params[name] = IntParam(
            name=name,
            low=low,
            high=high,
            step=step,
            log=log,
            description=description,
        )
        if default is not None:
            self._defaults[name] = default
        return self

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        step: Optional[float] = None,
        log: bool = False,
        description: str = "",
        default: Optional[float] = None,
    ) -> "SearchSpace":
        """Add float parameter."""
        self._params[name] = FloatParam(
            name=name,
            low=low,
            high=high,
            step=step,
            log=log,
            description=description,
        )
        if default is not None:
            self._defaults[name] = default
        return self

    def add_categorical(
        self,
        name: str,
        choices: List[Any],
        description: str = "",
        default: Optional[Any] = None,
    ) -> "SearchSpace":
        """Add categorical parameter."""
        self._params[name] = CategoricalParam(
            name=name,
            choices=choices,
            description=description,
        )
        if default is not None:
            self._defaults[name] = default
        return self

    def add_conditional(
        self,
        name: str,
        parent_name: str,
        parent_values: List[Any],
        child_param: HyperParameter,
        description: str = "",
    ) -> "SearchSpace":
        """Add conditional parameter."""
        self._params[name] = ConditionalParam(
            name=name,
            parent_name=parent_name,
            parent_values=parent_values,
            child_param=child_param,
            description=description,
        )
        return self

    def add_constraint(
        self,
        constraint_fn: Callable[[Dict[str, Any]], bool],
    ) -> "SearchSpace":
        """
        Add constraint function.

        Args:
            constraint_fn: Function that takes params dict and returns True if valid
        """
        self._constraints.append(constraint_fn)
        return self

    def suggest(self, trial: Any) -> Dict[str, Any]:
        """
        Suggest hyperparameters using Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        # First suggest non-conditional params
        for name, param in self._params.items():
            if not isinstance(param, ConditionalParam):
                params[name] = param.suggest(trial)

        # Then suggest conditional params
        for name, param in self._params.items():
            if isinstance(param, ConditionalParam):
                value = param.suggest(trial)
                if value is not None:
                    params[name] = value

        return params

    def validate(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameters against constraints.

        Args:
            params: Dictionary of hyperparameters

        Returns:
            True if all constraints are satisfied
        """
        for constraint in self._constraints:
            try:
                if not constraint(params):
                    return False
            except Exception:
                return False
        return True

    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for all parameters."""
        return self._defaults.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert search space to dictionary."""
        return {
            "name": self._name,
            "params": {name: param.to_dict() for name, param in self._params.items()},
            "defaults": self._defaults,
            "num_constraints": len(self._constraints),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchSpace":
        """Create search space from dictionary."""
        space = cls(name=data.get("name", "default"))

        for name, param_data in data.get("params", {}).items():
            param_type = param_data.get("type")

            if param_type in (ParamType.INT.value, ParamType.LOG_INT.value):
                space.add_int(
                    name=name,
                    low=param_data["low"],
                    high=param_data["high"],
                    step=param_data.get("step", 1),
                    log=param_data.get("log", False),
                    description=param_data.get("description", ""),
                )
            elif param_type in (ParamType.FLOAT.value, ParamType.LOG_FLOAT.value):
                space.add_float(
                    name=name,
                    low=param_data["low"],
                    high=param_data["high"],
                    step=param_data.get("step"),
                    log=param_data.get("log", False),
                    description=param_data.get("description", ""),
                )
            elif param_type == ParamType.CATEGORICAL.value:
                space.add_categorical(
                    name=name,
                    choices=param_data["choices"],
                    description=param_data.get("description", ""),
                )

        space._defaults = data.get("defaults", {})
        return space


# Predefined search spaces for common use cases

def create_graph_classifier_space() -> SearchSpace:
    """Create search space for graph classifier."""
    return (
        SearchSpace("graph_classifier")
        .add_float("lr", 1e-5, 1e-2, log=True, default=1e-3, description="Learning rate")
        .add_int("hidden_dim", 32, 256, step=32, default=64, description="Hidden dimension")
        .add_int("batch_size", 2, 16, step=2, default=4, description="Batch size")
        .add_int("epochs", 10, 100, step=10, default=30, description="Number of epochs")
        .add_categorical("model", ["gcn", "edge_sage"], default="gcn", description="Model architecture")
        .add_categorical("loss", ["cross_entropy", "focal", "logit_adjusted"], default="cross_entropy", description="Loss function")
        .add_categorical("class_weighting", ["none", "inverse", "sqrt"], default="none", description="Class weighting")
        .add_categorical("scheduler", ["none", "cosine", "warmup_cosine"], default="none", description="LR scheduler")
        .add_float("focal_gamma", 1.0, 3.0, step=0.5, default=2.0, description="Focal loss gamma")
        .add_int("early_stop_patience", 5, 20, step=5, default=10, description="Early stopping patience")
    )


def create_neural_network_space() -> SearchSpace:
    """Create search space for general neural networks."""
    return (
        SearchSpace("neural_network")
        .add_float("lr", 1e-6, 1e-1, log=True, default=1e-3, description="Learning rate")
        .add_int("hidden_layers", 1, 5, default=2, description="Number of hidden layers")
        .add_int("hidden_units", 32, 512, step=32, default=128, description="Hidden units per layer")
        .add_float("dropout", 0.0, 0.5, step=0.1, default=0.1, description="Dropout rate")
        .add_categorical("activation", ["relu", "gelu", "swish"], default="relu", description="Activation function")
        .add_categorical("optimizer", ["adam", "adamw", "sgd"], default="adam", description="Optimizer")
        .add_float("weight_decay", 1e-6, 1e-2, log=True, default=1e-4, description="Weight decay")
    )
