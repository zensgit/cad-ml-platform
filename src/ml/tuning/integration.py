"""
Integration module for hyperparameter tuning.

Provides high-level integration with existing training scripts,
experiment tracking (M1), and model evaluation (M3).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from src.ml.tuning.search_space import SearchSpace, create_graph_classifier_space
from src.ml.tuning.optimizer import (
    HyperOptimizer,
    OptimizationConfig,
    OptimizationResult,
)
from src.ml.tuning.strategies import SamplerType, PrunerType
from src.ml.tuning.callbacks import (
    TuningCallback,
    EarlyStoppingCallback,
    ExperimentTrackerCallback,
    ProgressCallback,
    CompositeCallback,
)

if TYPE_CHECKING:
    from src.ml.experiment import ExperimentTracker
    from src.ml.evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TuningContext:
    """
    Context for hyperparameter tuning session.

    Provides integration with experiment tracking and model evaluation.
    """
    search_space: SearchSpace
    optimizer: HyperOptimizer
    experiment_tracker: Optional["ExperimentTracker"] = None
    model_evaluator: Optional["ModelEvaluator"] = None
    output_dir: Path = field(default_factory=lambda: Path("tuning_results"))
    best_params: Optional[Dict[str, Any]] = None
    best_value: Optional[float] = None
    result: Optional[OptimizationResult] = None

    def save_result(self, filename: str = "tuning_result.json") -> Path:
        """Save tuning result to file."""
        if self.result is None:
            raise ValueError("No result to save")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        self.result.save(path)
        return path

    def get_best_config(self) -> Dict[str, Any]:
        """Get best configuration as a dictionary."""
        if self.result is not None:
            return {
                "params": self.result.best_params,
                "value": self.result.best_value,
                "trial_number": self.result.best_trial_number,
            }

        # Fallback to direct values
        return {
            "params": self.best_params or {},
            "value": self.best_value,
            "trial_number": -1,
        }


def create_tuning_objective(
    train_fn: Callable[..., float],
    search_space: SearchSpace,
    fixed_params: Optional[Dict[str, Any]] = None,
    report_intermediate: bool = True,
    intermediate_key: str = "val_acc",
) -> Callable[[Any], float]:
    """
    Create an Optuna objective function from a training function.

    Args:
        train_fn: Training function that takes hyperparameters and returns metric
        search_space: Search space for hyperparameters
        fixed_params: Fixed parameters to always include
        report_intermediate: Whether to report intermediate values for pruning
        intermediate_key: Key for intermediate value in train_fn results

    Returns:
        Optuna-compatible objective function
    """
    fixed_params = fixed_params or {}

    def objective(trial: Any) -> float:
        # Suggest hyperparameters from search space
        params = search_space.suggest(trial)

        # Merge with fixed params
        all_params = {**fixed_params, **params}

        # Create callback for intermediate reporting
        def report_callback(step: int, metrics: Dict[str, float]) -> bool:
            """Report intermediate value and check for pruning."""
            if report_intermediate and intermediate_key in metrics:
                trial.report(metrics[intermediate_key], step)
                if trial.should_prune():
                    raise TrialPruned()
            return True

        # Add report callback to params
        all_params["_trial_callback"] = report_callback

        try:
            result = train_fn(**all_params)

            # Handle different return types
            if isinstance(result, dict):
                return result.get(intermediate_key, result.get("metric", 0.0))
            return float(result)

        except TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            raise

    return objective


class TrialPruned(Exception):
    """Exception raised when a trial should be pruned."""
    pass


def tune_model(
    train_fn: Callable[..., float],
    search_space: Optional[SearchSpace] = None,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    direction: str = "maximize",
    strategy: str = "default",
    experiment_tracker: Optional["ExperimentTracker"] = None,
    model_evaluator: Optional["ModelEvaluator"] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List[TuningCallback]] = None,
    early_stopping_patience: int = 10,
    seed: Optional[int] = None,
    output_dir: Union[str, Path] = "tuning_results",
    study_name: str = "model_tuning",
    verbose: bool = True,
) -> TuningContext:
    """
    High-level function to tune a model.

    Args:
        train_fn: Training function that takes hyperparameters and returns metric
        search_space: Search space (uses default graph classifier space if None)
        n_trials: Number of optimization trials
        timeout: Optional timeout in seconds
        direction: "maximize" or "minimize"
        strategy: Optimization strategy name
        experiment_tracker: Optional experiment tracker for logging
        model_evaluator: Optional model evaluator for final evaluation
        fixed_params: Fixed parameters to always include
        callbacks: Additional callbacks
        early_stopping_patience: Patience for early stopping (0 to disable)
        seed: Random seed
        output_dir: Output directory for results
        study_name: Name for the Optuna study
        verbose: Whether to show progress

    Returns:
        TuningContext with optimization results
    """
    # Default search space
    if search_space is None:
        search_space = create_graph_classifier_space()

    # Create config from strategy
    config = OptimizationConfig.from_strategy(
        strategy,
        n_trials=n_trials,
        timeout=timeout,
        direction=direction,
        seed=seed,
        study_name=study_name,
        show_progress=verbose,
    )

    # Create optimizer
    optimizer = HyperOptimizer(search_space, config)

    # Add callbacks
    all_callbacks: List[TuningCallback] = []

    # Progress callback
    if verbose:
        all_callbacks.append(ProgressCallback(
            n_trials=n_trials,
            direction=direction,
            verbose=True,
        ))

    # Early stopping callback
    if early_stopping_patience > 0:
        all_callbacks.append(EarlyStoppingCallback(
            patience=early_stopping_patience,
            direction=direction,
        ))

    # Experiment tracker callback
    if experiment_tracker is not None:
        all_callbacks.append(ExperimentTrackerCallback(
            tracker=experiment_tracker,
            experiment_name=study_name,
        ))

    # User callbacks
    if callbacks:
        all_callbacks.extend(callbacks)

    for cb in all_callbacks:
        optimizer.add_callback(cb)

    # Create objective
    objective = create_tuning_objective(
        train_fn=train_fn,
        search_space=search_space,
        fixed_params=fixed_params,
    )

    # Run optimization
    logger.info(f"Starting hyperparameter tuning: {n_trials} trials, strategy={strategy}")
    result = optimizer.optimize(objective)

    # Create context
    output_dir = Path(output_dir)
    context = TuningContext(
        search_space=search_space,
        optimizer=optimizer,
        experiment_tracker=experiment_tracker,
        model_evaluator=model_evaluator,
        output_dir=output_dir,
        best_params=result.best_params,
        best_value=result.best_value,
        result=result,
    )

    # Save result
    context.save_result()

    # Log summary
    logger.info(f"Tuning complete: best_value={result.best_value:.4f if result.best_value else 'N/A'}")
    logger.info(f"Best params: {result.best_params}")
    logger.info(f"Completed: {result.n_completed}, Pruned: {result.n_pruned}, Failed: {result.n_failed}")

    return context


@contextmanager
def tuning_session(
    search_space: Optional[SearchSpace] = None,
    experiment_name: str = "tuning_session",
    output_dir: Union[str, Path] = "tuning_results",
):
    """
    Context manager for a tuning session.

    Usage:
        with tuning_session() as ctx:
            result = ctx.optimizer.optimize(objective)
    """
    if search_space is None:
        search_space = create_graph_classifier_space()

    config = OptimizationConfig(study_name=experiment_name)
    optimizer = HyperOptimizer(search_space, config)

    context = TuningContext(
        search_space=search_space,
        optimizer=optimizer,
        output_dir=Path(output_dir),
    )

    try:
        yield context
    finally:
        # Save any results
        if context.result is not None:
            context.save_result()


def create_train_2d_graph_objective(
    manifest_path: str,
    dxf_dir: str,
    device: str = "cpu",
    max_samples: int = 0,
) -> Tuple[Callable[[Any], float], SearchSpace]:
    """
    Create objective function for training 2D graph classifier.

    This integrates with the existing train_2d_graph.py script.

    Args:
        manifest_path: Path to manifest CSV
        dxf_dir: Path to DXF directory
        device: Device to use
        max_samples: Max samples (0 for all)

    Returns:
        (objective_function, search_space)
    """
    search_space = create_graph_classifier_space()

    def objective(trial: Any) -> float:
        # Get hyperparameters
        params = search_space.suggest(trial)

        # Import training components
        try:
            import torch
            from torch.utils.data import DataLoader, Subset

            from src.ml.train.dataset_2d import DXFManifestDataset
            from src.ml.train.model_2d import EdgeGraphSageClassifier, SimpleGraphClassifier
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            raise

        # Create dataset
        use_edge_attr = params.get("model", "gcn") == "edge_sage"
        dataset = DXFManifestDataset(
            manifest_path,
            dxf_dir,
            node_dim=params.get("node_dim", 10),
            return_edge_attr=use_edge_attr,
        )

        if max_samples > 0:
            dataset.samples = dataset.samples[:max_samples]

        if len(dataset) == 0:
            raise ValueError("Empty dataset")

        # Split data
        import random
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        split = max(1, int(len(indices) * 0.8))
        train_idx = indices[:split]
        val_idx = indices[split:] or indices[:1]

        # Simple collate function
        def collate(batch):
            xs, edges, edge_attrs, labels, filenames = [], [], [], [], []
            for graph, label in batch:
                xs.append(graph["x"])
                edges.append(graph["edge_index"])
                edge_attrs.append(graph.get("edge_attr"))
                labels.append(label)
                filenames.append(graph.get("file_name", ""))
            return xs, edges, edge_attrs, labels, filenames

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=params.get("batch_size", 4),
            shuffle=True,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=params.get("batch_size", 4),
            shuffle=False,
            collate_fn=collate,
        )

        # Create model
        label_map = dataset.get_label_map()
        num_classes = len(label_map)
        node_dim = params.get("node_dim", 10)
        hidden_dim = params.get("hidden_dim", 64)

        if params.get("model", "gcn") == "edge_sage":
            model = EdgeGraphSageClassifier(
                node_dim, params.get("edge_dim", 4), hidden_dim, num_classes
            )
        else:
            model = SimpleGraphClassifier(node_dim, hidden_dim, num_classes)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 1e-3))
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        epochs = params.get("epochs", 30)
        best_acc = 0.0
        patience = params.get("early_stop_patience", 10)
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for xs, edges, edge_attrs, labels, _ in train_loader:
                optimizer.zero_grad()
                batch_loss = 0.0
                count = 0

                for x, edge_index, edge_attr, label in zip(xs, edges, edge_attrs, labels):
                    if x.numel() == 0:
                        continue

                    x = x.to(device)
                    edge_index = edge_index.to(device)
                    if edge_attr is not None:
                        edge_attr = edge_attr.to(device)
                    label = label.to(device)

                    if use_edge_attr:
                        logits = model(x, edge_index, edge_attr)
                    else:
                        logits = model(x, edge_index)

                    loss = criterion(logits, label.view(1))
                    batch_loss += loss
                    count += 1

                if count > 0:
                    (batch_loss / count).backward()
                    optimizer.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xs, edges, edge_attrs, labels, _ in val_loader:
                    for x, edge_index, edge_attr, label in zip(xs, edges, edge_attrs, labels):
                        if x.numel() == 0:
                            continue

                        x = x.to(device)
                        edge_index = edge_index.to(device)
                        if edge_attr is not None:
                            edge_attr = edge_attr.to(device)

                        if use_edge_attr:
                            logits = model(x, edge_index, edge_attr)
                        else:
                            logits = model(x, edge_index)

                        pred = int(torch.argmax(logits, dim=1)[0])
                        if pred == int(label):
                            correct += 1
                        total += 1

            acc = correct / max(1, total)

            # Report intermediate value for pruning
            trial.report(acc, epoch)
            if trial.should_prune():
                raise TrialPruned()

            # Early stopping
            if acc > best_acc:
                best_acc = acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return best_acc

    return objective, search_space


# Convenience functions for common use cases

def quick_tune(
    train_fn: Callable[..., float],
    n_trials: int = 20,
    direction: str = "maximize",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Quick hyperparameter tuning with minimal configuration.

    Returns:
        Best parameters dictionary
    """
    context = tune_model(
        train_fn=train_fn,
        n_trials=n_trials,
        direction=direction,
        strategy="fast",
        verbose=True,
        **kwargs,
    )
    return context.best_params or {}


def thorough_tune(
    train_fn: Callable[..., float],
    n_trials: int = 100,
    direction: str = "maximize",
    **kwargs: Any,
) -> TuningContext:
    """
    Thorough hyperparameter tuning with comprehensive search.

    Returns:
        TuningContext with full results
    """
    return tune_model(
        train_fn=train_fn,
        n_trials=n_trials,
        direction=direction,
        strategy="thorough",
        verbose=True,
        **kwargs,
    )
