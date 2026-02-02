"""
Optimization Strategies for Hyperparameter Tuning.

Provides Optuna sampler and pruner configurations.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SamplerType(str, Enum):
    """Optuna sampler types."""
    TPE = "tpe"  # Tree-structured Parzen Estimator (default)
    RANDOM = "random"  # Random search
    GRID = "grid"  # Grid search
    CMAES = "cmaes"  # CMA-ES for continuous params
    NSGAII = "nsgaii"  # Multi-objective
    QMCSAMPLER = "qmc"  # Quasi-Monte Carlo


class PrunerType(str, Enum):
    """Optuna pruner types."""
    MEDIAN = "median"  # Median pruning (default)
    PERCENTILE = "percentile"  # Percentile-based pruning
    SUCCESSIVE_HALVING = "successive_halving"  # Async successive halving
    HYPERBAND = "hyperband"  # Hyperband algorithm
    THRESHOLD = "threshold"  # Threshold-based pruning
    PATIENT = "patient"  # Patient pruning
    NONE = "none"  # No pruning


def get_sampler(
    sampler_type: SamplerType = SamplerType.TPE,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Get Optuna sampler instance.

    Args:
        sampler_type: Type of sampler to use
        seed: Random seed for reproducibility
        **kwargs: Additional sampler-specific arguments

    Returns:
        Optuna sampler instance
    """
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna not installed, returning None sampler")
        return None

    if sampler_type == SamplerType.TPE:
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=kwargs.get("n_startup_trials", 10),
            multivariate=kwargs.get("multivariate", True),
        )

    elif sampler_type == SamplerType.RANDOM:
        return optuna.samplers.RandomSampler(seed=seed)

    elif sampler_type == SamplerType.GRID:
        search_space = kwargs.get("search_space", {})
        return optuna.samplers.GridSampler(search_space)

    elif sampler_type == SamplerType.CMAES:
        return optuna.samplers.CmaEsSampler(
            seed=seed,
            restart_strategy=kwargs.get("restart_strategy", "ipop"),
        )

    elif sampler_type == SamplerType.NSGAII:
        return optuna.samplers.NSGAIISampler(
            seed=seed,
            population_size=kwargs.get("population_size", 50),
        )

    elif sampler_type == SamplerType.QMCSAMPLER:
        return optuna.samplers.QMCSampler(
            seed=seed,
            qmc_type=kwargs.get("qmc_type", "sobol"),
        )

    else:
        logger.warning(f"Unknown sampler type: {sampler_type}, using TPE")
        return optuna.samplers.TPESampler(seed=seed)


def get_pruner(
    pruner_type: PrunerType = PrunerType.MEDIAN,
    **kwargs: Any,
) -> Any:
    """
    Get Optuna pruner instance.

    Args:
        pruner_type: Type of pruner to use
        **kwargs: Additional pruner-specific arguments

    Returns:
        Optuna pruner instance or None
    """
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna not installed, returning None pruner")
        return None

    if pruner_type == PrunerType.NONE:
        return optuna.pruners.NopPruner()

    elif pruner_type == PrunerType.MEDIAN:
        return optuna.pruners.MedianPruner(
            n_startup_trials=kwargs.get("n_startup_trials", 5),
            n_warmup_steps=kwargs.get("n_warmup_steps", 5),
            interval_steps=kwargs.get("interval_steps", 1),
        )

    elif pruner_type == PrunerType.PERCENTILE:
        return optuna.pruners.PercentilePruner(
            percentile=kwargs.get("percentile", 25.0),
            n_startup_trials=kwargs.get("n_startup_trials", 5),
            n_warmup_steps=kwargs.get("n_warmup_steps", 5),
        )

    elif pruner_type == PrunerType.SUCCESSIVE_HALVING:
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=kwargs.get("min_resource", 1),
            reduction_factor=kwargs.get("reduction_factor", 3),
        )

    elif pruner_type == PrunerType.HYPERBAND:
        return optuna.pruners.HyperbandPruner(
            min_resource=kwargs.get("min_resource", 1),
            max_resource=kwargs.get("max_resource", 100),
            reduction_factor=kwargs.get("reduction_factor", 3),
        )

    elif pruner_type == PrunerType.THRESHOLD:
        lower = kwargs.get("lower", None)
        upper = kwargs.get("upper", None)
        return optuna.pruners.ThresholdPruner(lower=lower, upper=upper)

    elif pruner_type == PrunerType.PATIENT:
        return optuna.pruners.PatientPruner(
            wrapped_pruner=optuna.pruners.MedianPruner(),
            patience=kwargs.get("patience", 3),
        )

    else:
        logger.warning(f"Unknown pruner type: {pruner_type}, using Median")
        return optuna.pruners.MedianPruner()


def get_strategy_config(
    strategy: str = "default",
) -> Dict[str, Any]:
    """
    Get predefined strategy configuration.

    Args:
        strategy: Strategy name

    Returns:
        Configuration dictionary
    """
    strategies = {
        "default": {
            "sampler": SamplerType.TPE,
            "pruner": PrunerType.MEDIAN,
            "sampler_kwargs": {"multivariate": True},
            "pruner_kwargs": {"n_warmup_steps": 5},
        },
        "fast": {
            "sampler": SamplerType.TPE,
            "pruner": PrunerType.HYPERBAND,
            "sampler_kwargs": {"n_startup_trials": 5},
            "pruner_kwargs": {"min_resource": 1, "reduction_factor": 4},
        },
        "thorough": {
            "sampler": SamplerType.TPE,
            "pruner": PrunerType.PATIENT,
            "sampler_kwargs": {"n_startup_trials": 20, "multivariate": True},
            "pruner_kwargs": {"patience": 5},
        },
        "grid": {
            "sampler": SamplerType.GRID,
            "pruner": PrunerType.NONE,
            "sampler_kwargs": {},
            "pruner_kwargs": {},
        },
        "random": {
            "sampler": SamplerType.RANDOM,
            "pruner": PrunerType.MEDIAN,
            "sampler_kwargs": {},
            "pruner_kwargs": {"n_warmup_steps": 3},
        },
        "bayesian": {
            "sampler": SamplerType.TPE,
            "pruner": PrunerType.SUCCESSIVE_HALVING,
            "sampler_kwargs": {"multivariate": True, "n_startup_trials": 10},
            "pruner_kwargs": {"reduction_factor": 3},
        },
        "evolutionary": {
            "sampler": SamplerType.CMAES,
            "pruner": PrunerType.NONE,
            "sampler_kwargs": {"restart_strategy": "ipop"},
            "pruner_kwargs": {},
        },
        "multi_objective": {
            "sampler": SamplerType.NSGAII,
            "pruner": PrunerType.NONE,
            "sampler_kwargs": {"population_size": 50},
            "pruner_kwargs": {},
        },
    }

    return strategies.get(strategy, strategies["default"])
