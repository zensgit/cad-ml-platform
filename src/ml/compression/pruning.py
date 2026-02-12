"""
Model pruning for network compression.

Provides structured and unstructured pruning methods.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class PruningMethod(str, Enum):
    """Pruning method."""
    MAGNITUDE = "magnitude"  # L1 magnitude-based
    RANDOM = "random"  # Random pruning
    GRADIENT = "gradient"  # Gradient-based
    TAYLOR = "taylor"  # Taylor expansion
    LOTTERY = "lottery"  # Lottery ticket hypothesis


class PruningScope(str, Enum):
    """Pruning scope."""
    LOCAL = "local"  # Per-layer sparsity
    GLOBAL = "global"  # Global sparsity across model


class PruningSchedule(str, Enum):
    """Pruning schedule."""
    ONE_SHOT = "one_shot"  # Prune once
    ITERATIVE = "iterative"  # Gradual pruning
    CUBIC = "cubic"  # Cubic schedule


@dataclass
class PruningConfig:
    """Pruning configuration."""
    method: PruningMethod = PruningMethod.MAGNITUDE
    scope: PruningScope = PruningScope.LOCAL
    schedule: PruningSchedule = PruningSchedule.ONE_SHOT
    sparsity: float = 0.5  # Target sparsity (0-1)
    structured: bool = False  # Structured vs unstructured
    structure_type: str = "filter"  # filter, channel, block
    block_size: Tuple[int, int] = (4, 4)  # For block pruning
    iterations: int = 5  # For iterative pruning
    initial_sparsity: float = 0.0  # Starting sparsity
    final_sparsity: float = 0.5  # Final sparsity
    modules_to_prune: Optional[List[str]] = None
    modules_to_exclude: Optional[List[str]] = None


@dataclass
class PruningResult:
    """Result of pruning."""
    original_params: int
    pruned_params: int
    sparsity: float
    original_size_mb: float
    pruned_size_mb: float
    compression_ratio: float
    original_latency_ms: float
    pruned_latency_ms: float
    speedup: float
    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None
    accuracy_drop: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_params": self.original_params,
            "pruned_params": self.pruned_params,
            "sparsity": round(self.sparsity, 4),
            "original_size_mb": round(self.original_size_mb, 2),
            "pruned_size_mb": round(self.pruned_size_mb, 2),
            "compression_ratio": round(self.compression_ratio, 2),
            "original_latency_ms": round(self.original_latency_ms, 2),
            "pruned_latency_ms": round(self.pruned_latency_ms, 2),
            "speedup": round(self.speedup, 2),
            "accuracy_before": self.accuracy_before,
            "accuracy_after": self.accuracy_after,
            "accuracy_drop": round(self.accuracy_drop, 4) if self.accuracy_drop else None,
        }


class Pruner:
    """
    Model pruner for network compression.

    Supports various pruning methods and schedules.
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Initialize pruner.

        Args:
            config: Pruning configuration
        """
        self._config = config or PruningConfig()
        self._masks: Dict[str, Any] = {}

    @property
    def config(self) -> PruningConfig:
        return self._config

    @property
    def masks(self) -> Dict[str, Any]:
        return self._masks.copy()

    def prune(
        self,
        model: Any,
        eval_fn: Optional[Callable] = None,
        fine_tune_fn: Optional[Callable] = None,
    ) -> Tuple[Any, PruningResult]:
        """
        Prune a model.

        Args:
            model: PyTorch model to prune
            eval_fn: Function to evaluate model accuracy
            fine_tune_fn: Function to fine-tune after pruning

        Returns:
            (pruned_model, result)
        """
        try:
            import torch
            import torch.nn.utils.prune as prune
        except ImportError:
            raise ImportError("PyTorch required for pruning")

        # Measure original model
        original_params = self._count_parameters(model)
        original_size = self._get_model_size(model)
        original_latency = self._measure_latency(model)
        original_accuracy = eval_fn(model) if eval_fn else None

        # Clone model
        model = copy.deepcopy(model)

        # Select pruning method
        if self._config.schedule == PruningSchedule.ONE_SHOT:
            pruned_model = self._prune_one_shot(model)
        elif self._config.schedule == PruningSchedule.ITERATIVE:
            pruned_model = self._prune_iterative(model, fine_tune_fn)
        elif self._config.schedule == PruningSchedule.CUBIC:
            pruned_model = self._prune_cubic(model, fine_tune_fn)
        else:
            raise ValueError(f"Unknown schedule: {self._config.schedule}")

        # Make pruning permanent
        pruned_model = self._make_permanent(pruned_model)

        # Fine-tune if provided
        if fine_tune_fn and self._config.schedule == PruningSchedule.ONE_SHOT:
            fine_tune_fn(pruned_model)

        # Measure pruned model
        pruned_params = self._count_nonzero_parameters(pruned_model)
        pruned_size = self._get_model_size(pruned_model)
        pruned_latency = self._measure_latency(pruned_model)
        pruned_accuracy = eval_fn(pruned_model) if eval_fn else None

        # Calculate metrics
        sparsity = 1 - (pruned_params / original_params) if original_params > 0 else 0
        compression_ratio = original_size / pruned_size if pruned_size > 0 else 1.0
        speedup = original_latency / pruned_latency if pruned_latency > 0 else 1.0
        accuracy_drop = None
        if original_accuracy is not None and pruned_accuracy is not None:
            accuracy_drop = original_accuracy - pruned_accuracy

        result = PruningResult(
            original_params=original_params,
            pruned_params=pruned_params,
            sparsity=sparsity,
            original_size_mb=original_size,
            pruned_size_mb=pruned_size,
            compression_ratio=compression_ratio,
            original_latency_ms=original_latency,
            pruned_latency_ms=pruned_latency,
            speedup=speedup,
            accuracy_before=original_accuracy,
            accuracy_after=pruned_accuracy,
            accuracy_drop=accuracy_drop,
        )

        logger.info(f"Pruning complete: {sparsity:.2%} sparsity, {speedup:.2f}x speedup")
        return pruned_model, result

    def _prune_one_shot(self, model: Any) -> Any:
        """Apply one-shot pruning."""
        import torch
        import torch.nn.utils.prune as prune

        # Get prunable modules
        modules = self._get_prunable_modules(model)

        if self._config.structured:
            # Structured pruning
            for name, module in modules:
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    if self._config.structure_type == "filter":
                        prune.ln_structured(
                            module,
                            name="weight",
                            amount=self._config.sparsity,
                            n=2,
                            dim=0,
                        )
                    elif self._config.structure_type == "channel":
                        prune.ln_structured(
                            module,
                            name="weight",
                            amount=self._config.sparsity,
                            n=2,
                            dim=1,
                        )
        else:
            # Unstructured pruning
            if self._config.scope == PruningScope.GLOBAL:
                parameters_to_prune = [
                    (module, "weight") for name, module in modules
                    if hasattr(module, "weight")
                ]
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=self._get_pruning_method(),
                    amount=self._config.sparsity,
                )
            else:
                for name, module in modules:
                    if hasattr(module, "weight"):
                        self._apply_local_pruning(module)

        return model

    def _prune_iterative(
        self,
        model: Any,
        fine_tune_fn: Optional[Callable] = None,
    ) -> Any:
        """Apply iterative pruning."""
        current_sparsity = self._config.initial_sparsity
        sparsity_step = (self._config.final_sparsity - current_sparsity) / self._config.iterations

        for i in range(self._config.iterations):
            current_sparsity += sparsity_step
            logger.info(f"Iteration {i+1}/{self._config.iterations}: target sparsity {current_sparsity:.2%}")

            # Update config and prune
            temp_config = PruningConfig(
                method=self._config.method,
                scope=self._config.scope,
                schedule=PruningSchedule.ONE_SHOT,
                sparsity=current_sparsity,
                structured=self._config.structured,
            )
            temp_pruner = Pruner(temp_config)
            model = temp_pruner._prune_one_shot(model)

            # Fine-tune
            if fine_tune_fn:
                fine_tune_fn(model)

        return model

    def _prune_cubic(
        self,
        model: Any,
        fine_tune_fn: Optional[Callable] = None,
    ) -> Any:
        """Apply cubic schedule pruning."""
        for i in range(self._config.iterations):
            # Cubic sparsity schedule
            progress = i / self._config.iterations
            current_sparsity = self._config.final_sparsity * (1 - (1 - progress) ** 3)
            current_sparsity = max(current_sparsity, self._config.initial_sparsity)

            logger.info(f"Iteration {i+1}/{self._config.iterations}: target sparsity {current_sparsity:.2%}")

            temp_config = PruningConfig(
                method=self._config.method,
                scope=self._config.scope,
                schedule=PruningSchedule.ONE_SHOT,
                sparsity=current_sparsity,
                structured=self._config.structured,
            )
            temp_pruner = Pruner(temp_config)
            model = temp_pruner._prune_one_shot(model)

            if fine_tune_fn:
                fine_tune_fn(model)

        return model

    def _get_prunable_modules(self, model: Any) -> List[Tuple[str, Any]]:
        """Get modules that can be pruned."""
        import torch

        modules = []
        exclude = set(self._config.modules_to_exclude or [])
        include = set(self._config.modules_to_prune) if self._config.modules_to_prune else None

        for name, module in model.named_modules():
            if name in exclude:
                continue
            if include and name not in include:
                continue
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d)):
                modules.append((name, module))

        return modules

    def _get_pruning_method(self):
        """Get PyTorch pruning method."""
        import torch.nn.utils.prune as prune

        method_map = {
            PruningMethod.MAGNITUDE: prune.L1Unstructured,
            PruningMethod.RANDOM: prune.RandomUnstructured,
        }
        return method_map.get(self._config.method, prune.L1Unstructured)

    def _apply_local_pruning(self, module: Any) -> None:
        """Apply local pruning to a module."""
        import torch.nn.utils.prune as prune

        if self._config.method == PruningMethod.MAGNITUDE:
            prune.l1_unstructured(module, name="weight", amount=self._config.sparsity)
        elif self._config.method == PruningMethod.RANDOM:
            prune.random_unstructured(module, name="weight", amount=self._config.sparsity)
        else:
            prune.l1_unstructured(module, name="weight", amount=self._config.sparsity)

    def _make_permanent(self, model: Any) -> Any:
        """Make pruning masks permanent."""
        import torch.nn.utils.prune as prune

        for name, module in model.named_modules():
            if hasattr(module, "weight_orig"):
                try:
                    prune.remove(module, "weight")
                except Exception:
                    pass

        return model

    def _count_parameters(self, model: Any) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in model.parameters())

    def _count_nonzero_parameters(self, model: Any) -> int:
        """Count non-zero parameters."""
        import torch
        return sum((p != 0).sum().item() for p in model.parameters())

    def _get_model_size(self, model: Any) -> float:
        """Get model size in MB."""
        import torch
        import io

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.tell() / (1024 * 1024)

    def _measure_latency(self, model: Any, num_runs: int = 100) -> float:
        """Measure inference latency in ms."""
        import torch

        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 64).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                try:
                    model(dummy_input)
                except Exception:
                    break

        # Measure
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                try:
                    model(dummy_input)
                except Exception:
                    break

        total_time = (time.time() - start) * 1000
        return total_time / num_runs

    def get_sparsity(self, model: Any) -> Dict[str, float]:
        """Get sparsity per layer."""
        import torch

        sparsity = {}
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                weight = module.weight
                total = weight.numel()
                zeros = (weight == 0).sum().item()
                sparsity[name] = zeros / total if total > 0 else 0

        return sparsity


def prune_model(
    model: Any,
    sparsity: float = 0.5,
    method: PruningMethod = PruningMethod.MAGNITUDE,
    structured: bool = False,
) -> Tuple[Any, PruningResult]:
    """
    Convenience function to prune a model.

    Args:
        model: Model to prune
        sparsity: Target sparsity
        method: Pruning method
        structured: Use structured pruning

    Returns:
        (pruned_model, result)
    """
    config = PruningConfig(method=method, sparsity=sparsity, structured=structured)
    pruner = Pruner(config)
    return pruner.prune(model)


def prune_unstructured(model: Any, sparsity: float = 0.5) -> Any:
    """Apply unstructured pruning."""
    config = PruningConfig(sparsity=sparsity, structured=False)
    pruner = Pruner(config)
    pruned, _ = pruner.prune(model)
    return pruned


def prune_structured(model: Any, sparsity: float = 0.5) -> Any:
    """Apply structured pruning."""
    config = PruningConfig(sparsity=sparsity, structured=True)
    pruner = Pruner(config)
    pruned, _ = pruner.prune(model)
    return pruned
