"""
Hyperparameter Tuning Module (M2).

Provides Optuna-based hyperparameter optimization with:
- Search space definition
- Multiple optimization strategies
- Integration with experiment tracking (M1)
- Integration with model evaluation (M3)
"""

from src.ml.tuning.search_space import (
    SearchSpace,
    HyperParameter,
    IntParam,
    FloatParam,
    CategoricalParam,
    ConditionalParam,
)
from src.ml.tuning.optimizer import (
    HyperOptimizer,
    OptimizationConfig,
    OptimizationResult,
    TrialResult,
)
from src.ml.tuning.strategies import (
    SamplerType,
    PrunerType,
    get_sampler,
    get_pruner,
)
from src.ml.tuning.callbacks import (
    TuningCallback,
    EarlyStoppingCallback,
    ExperimentTrackerCallback,
    ProgressCallback,
)
from src.ml.tuning.integration import (
    create_tuning_objective,
    tune_model,
    TuningContext,
)

__all__ = [
    # Search space
    "SearchSpace",
    "HyperParameter",
    "IntParam",
    "FloatParam",
    "CategoricalParam",
    "ConditionalParam",
    # Optimizer
    "HyperOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "TrialResult",
    # Strategies
    "SamplerType",
    "PrunerType",
    "get_sampler",
    "get_pruner",
    # Callbacks
    "TuningCallback",
    "EarlyStoppingCallback",
    "ExperimentTrackerCallback",
    "ProgressCallback",
    # Integration
    "create_tuning_objective",
    "tune_model",
    "TuningContext",
]
