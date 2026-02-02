"""
Pipeline stages for E2E ML workflow.

Provides modular, composable pipeline stages.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


class StageStatus(str, Enum):
    """Stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of stage execution."""
    stage_name: str
    status: StageStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == StageStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "execution_time": round(self.execution_time, 3),
            "error": self.error,
            "metrics": self.metrics,
        }


class PipelineStage(ABC, Generic[T, U]):
    """
    Base class for pipeline stages.

    Each stage transforms input of type T to output of type U.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: str = "",
        skip_on_error: bool = False,
    ):
        """
        Initialize pipeline stage.

        Args:
            name: Stage name (uses class name if None)
            description: Stage description
            skip_on_error: Skip stage instead of failing pipeline
        """
        self._name = name or self.__class__.__name__
        self._description = description
        self._skip_on_error = skip_on_error
        self._status = StageStatus.PENDING
        self._last_result: Optional[StageResult] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def status(self) -> StageStatus:
        return self._status

    @property
    def last_result(self) -> Optional[StageResult]:
        return self._last_result

    def execute(self, input_data: T, context: Dict[str, Any]) -> StageResult:
        """
        Execute the stage.

        Args:
            input_data: Input data
            context: Pipeline context

        Returns:
            StageResult
        """
        self._status = StageStatus.RUNNING
        start_time = time.time()

        try:
            # Pre-execution hook
            self.on_start(input_data, context)

            # Run stage
            output = self.run(input_data, context)

            # Calculate metrics
            metrics = self.compute_metrics(input_data, output, context)

            execution_time = time.time() - start_time
            self._status = StageStatus.COMPLETED

            result = StageResult(
                stage_name=self._name,
                status=StageStatus.COMPLETED,
                output=output,
                execution_time=execution_time,
                metrics=metrics,
            )

            # Post-execution hook
            self.on_complete(result, context)

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Stage {self._name} failed: {e}")

            if self._skip_on_error:
                self._status = StageStatus.SKIPPED
                result = StageResult(
                    stage_name=self._name,
                    status=StageStatus.SKIPPED,
                    error=str(e),
                    execution_time=execution_time,
                )
            else:
                self._status = StageStatus.FAILED
                result = StageResult(
                    stage_name=self._name,
                    status=StageStatus.FAILED,
                    error=str(e),
                    execution_time=execution_time,
                )

            self.on_error(e, context)

        self._last_result = result
        return result

    @abstractmethod
    def run(self, input_data: T, context: Dict[str, Any]) -> U:
        """
        Run the stage logic.

        Args:
            input_data: Input data
            context: Pipeline context

        Returns:
            Output data
        """
        pass

    def on_start(self, input_data: T, context: Dict[str, Any]) -> None:
        """Hook called before stage execution."""
        logger.info(f"Starting stage: {self._name}")

    def on_complete(self, result: StageResult, context: Dict[str, Any]) -> None:
        """Hook called after successful execution."""
        logger.info(f"Completed stage: {self._name} ({result.execution_time:.2f}s)")

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Hook called on error."""
        logger.error(f"Stage {self._name} error: {error}")

    def compute_metrics(
        self,
        input_data: T,
        output: U,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute stage metrics."""
        return {}

    def validate_input(self, input_data: T) -> bool:
        """Validate input data."""
        return True

    def validate_output(self, output: U) -> bool:
        """Validate output data."""
        return True


class DataLoadingStage(PipelineStage[Dict[str, Any], Dict[str, Any]]):
    """Stage for loading data from various sources."""

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        loader_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize data loading stage.

        Args:
            data_path: Path to data
            loader_fn: Custom loader function
        """
        super().__init__(name="DataLoading", **kwargs)
        self._data_path = Path(data_path) if data_path else None
        self._loader_fn = loader_fn

    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Load data."""
        # Use custom loader if provided
        if self._loader_fn:
            data = self._loader_fn(self._data_path, context)
            return {"data": data, **input_data}

        # Default loading logic
        data_path = self._data_path or input_data.get("data_path")
        if not data_path:
            raise ValueError("No data path provided")

        data_path = Path(data_path)

        if data_path.is_dir():
            # Load directory of files
            files = list(data_path.rglob("*.dxf")) + list(data_path.rglob("*.dwg"))
            data = {"files": files, "count": len(files)}
        elif data_path.suffix == ".json":
            import json
            with open(data_path) as f:
                data = json.load(f)
        elif data_path.suffix in (".pt", ".pth"):
            try:
                import torch
                data = torch.load(data_path)
            except ImportError:
                raise ImportError("PyTorch required for .pt/.pth files")
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")

        return {"data": data, "source_path": str(data_path), **input_data}

    def compute_metrics(
        self,
        input_data: Dict[str, Any],
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        data = output.get("data", {})
        if isinstance(data, dict) and "count" in data:
            return {"loaded_files": data["count"]}
        elif isinstance(data, list):
            return {"loaded_samples": len(data)}
        return {}


class PreprocessingStage(PipelineStage[Dict[str, Any], Dict[str, Any]]):
    """Stage for data preprocessing."""

    def __init__(
        self,
        transforms: Optional[List[Callable]] = None,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Initialize preprocessing stage.

        Args:
            transforms: List of transform functions
            normalize: Whether to normalize data
        """
        super().__init__(name="Preprocessing", **kwargs)
        self._transforms = transforms or []
        self._normalize = normalize

    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        data = input_data.get("data")
        if data is None:
            raise ValueError("No data to preprocess")

        # Apply transforms
        for transform in self._transforms:
            data = transform(data)

        # Normalize if enabled
        if self._normalize and isinstance(data, dict):
            data = self._normalize_data(data)

        return {**input_data, "data": data, "preprocessed": True}

    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data values."""
        # Basic normalization logic
        return data

    def compute_metrics(
        self,
        input_data: Dict[str, Any],
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"transforms_applied": len(self._transforms)}


class FeatureExtractionStage(PipelineStage[Dict[str, Any], Dict[str, Any]]):
    """Stage for feature extraction."""

    def __init__(
        self,
        feature_extractors: Optional[List[Callable]] = None,
        feature_dim: int = 128,
        **kwargs,
    ):
        """
        Initialize feature extraction stage.

        Args:
            feature_extractors: List of feature extractor functions
            feature_dim: Feature dimension
        """
        super().__init__(name="FeatureExtraction", **kwargs)
        self._extractors = feature_extractors or []
        self._feature_dim = feature_dim

    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features."""
        data = input_data.get("data")
        if data is None:
            raise ValueError("No data for feature extraction")

        features = []
        for extractor in self._extractors:
            feature = extractor(data, context)
            features.append(feature)

        # Combine features if multiple extractors
        if len(features) > 1:
            try:
                import torch
                combined = torch.cat(features, dim=-1)
            except ImportError:
                combined = features
        elif len(features) == 1:
            combined = features[0]
        else:
            combined = data  # No extractors, pass through

        return {**input_data, "features": combined}

    def compute_metrics(
        self,
        input_data: Dict[str, Any],
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        features = output.get("features")
        metrics = {"extractors_used": len(self._extractors)}

        if hasattr(features, "shape"):
            metrics["feature_shape"] = list(features.shape)

        return metrics


class TrainingStage(PipelineStage[Dict[str, Any], Dict[str, Any]]):
    """Stage for model training."""

    def __init__(
        self,
        model: Optional[Any] = None,
        optimizer_class: Optional[type] = None,
        criterion: Optional[Any] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_split: float = 0.1,
        callbacks: Optional[List[Callable]] = None,
        **kwargs,
    ):
        """
        Initialize training stage.

        Args:
            model: Model to train
            optimizer_class: Optimizer class
            criterion: Loss function
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Validation split ratio
            callbacks: Training callbacks
        """
        super().__init__(name="Training", **kwargs)
        self._model = model
        self._optimizer_class = optimizer_class
        self._criterion = criterion
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._val_split = validation_split
        self._callbacks = callbacks or []

    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Train model."""
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset, random_split
        except ImportError:
            raise ImportError("PyTorch required for training")

        # Get model
        model = self._model or context.get("model")
        if model is None:
            raise ValueError("No model provided for training")

        # Get data
        features = input_data.get("features")
        labels = input_data.get("labels")

        if features is None:
            raise ValueError("No features for training")

        # Create dataset
        if labels is not None:
            dataset = TensorDataset(features, labels)
        else:
            dataset = TensorDataset(features)

        # Split into train/val
        val_size = int(len(dataset) * self._val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size) if val_size > 0 else None

        # Setup optimizer
        optimizer_class = self._optimizer_class or torch.optim.Adam
        optimizer = optimizer_class(model.parameters(), lr=self._lr)

        # Setup criterion
        criterion = self._criterion or torch.nn.CrossEntropyLoss()

        # Training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self._epochs):
            # Training
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                else:
                    x = batch[0].to(device)
                    y = None

                optimizer.zero_grad()
                output = model(x)

                if y is not None:
                    loss = criterion(output, y)
                else:
                    loss = output.mean()  # Unsupervised

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        if len(batch) == 2:
                            x, y = batch
                            x, y = x.to(device), y.to(device)
                        else:
                            x = batch[0].to(device)
                            y = None

                        output = model(x)
                        if y is not None:
                            loss = criterion(output, y)
                        else:
                            loss = output.mean()
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                history["val_loss"].append(val_loss)

            # Callbacks
            for callback in self._callbacks:
                callback(epoch, model, history, context)

            logger.debug(f"Epoch {epoch+1}/{self._epochs}: train_loss={train_loss:.4f}")

        # Store model in context
        context["trained_model"] = model

        return {
            **input_data,
            "model": model,
            "training_history": history,
        }

    def compute_metrics(
        self,
        input_data: Dict[str, Any],
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        history = output.get("training_history", {})
        metrics = {
            "epochs": self._epochs,
            "batch_size": self._batch_size,
        }

        if history.get("train_loss"):
            metrics["final_train_loss"] = history["train_loss"][-1]
        if history.get("val_loss"):
            metrics["final_val_loss"] = history["val_loss"][-1]

        return metrics


class EvaluationStage(PipelineStage[Dict[str, Any], Dict[str, Any]]):
    """Stage for model evaluation."""

    def __init__(
        self,
        metrics_fn: Optional[List[Callable]] = None,
        threshold: float = 0.5,
        **kwargs,
    ):
        """
        Initialize evaluation stage.

        Args:
            metrics_fn: List of metric functions
            threshold: Classification threshold
        """
        super().__init__(name="Evaluation", **kwargs)
        self._metrics_fn = metrics_fn or []
        self._threshold = threshold

    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for evaluation")

        model = input_data.get("model") or context.get("trained_model")
        if model is None:
            raise ValueError("No model to evaluate")

        features = input_data.get("test_features") or input_data.get("features")
        labels = input_data.get("test_labels") or input_data.get("labels")

        if features is None:
            raise ValueError("No features for evaluation")

        # Evaluate
        device = next(model.parameters()).device
        model.eval()

        with torch.no_grad():
            features = features.to(device)
            predictions = model(features)

        # Calculate metrics
        eval_metrics = {}

        if labels is not None:
            labels = labels.to(device)

            # Accuracy for classification
            if predictions.dim() > 1 and predictions.size(1) > 1:
                pred_classes = predictions.argmax(dim=1)
                accuracy = (pred_classes == labels).float().mean().item()
                eval_metrics["accuracy"] = accuracy

            # Custom metrics
            for metric_fn in self._metrics_fn:
                metric_name = metric_fn.__name__
                metric_value = metric_fn(predictions, labels)
                eval_metrics[metric_name] = metric_value

        return {
            **input_data,
            "predictions": predictions,
            "evaluation_metrics": eval_metrics,
        }

    def compute_metrics(
        self,
        input_data: Dict[str, Any],
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return output.get("evaluation_metrics", {})


class InferenceStage(PipelineStage[Dict[str, Any], Dict[str, Any]]):
    """Stage for model inference."""

    def __init__(
        self,
        model: Optional[Any] = None,
        batch_size: int = 32,
        post_process_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize inference stage.

        Args:
            model: Model for inference
            batch_size: Batch size
            post_process_fn: Post-processing function
        """
        super().__init__(name="Inference", **kwargs)
        self._model = model
        self._batch_size = batch_size
        self._post_process_fn = post_process_fn

    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for inference")

        model = self._model or input_data.get("model") or context.get("trained_model")
        if model is None:
            raise ValueError("No model for inference")

        features = input_data.get("features")
        if features is None:
            raise ValueError("No features for inference")

        # Inference
        device = next(model.parameters()).device
        model.eval()

        # Batch inference
        all_predictions = []
        num_samples = features.size(0)

        with torch.no_grad():
            for i in range(0, num_samples, self._batch_size):
                batch = features[i:i + self._batch_size].to(device)
                pred = model(batch)
                all_predictions.append(pred.cpu())

        predictions = torch.cat(all_predictions, dim=0)

        # Post-process
        if self._post_process_fn:
            predictions = self._post_process_fn(predictions)

        return {
            **input_data,
            "predictions": predictions,
        }

    def compute_metrics(
        self,
        input_data: Dict[str, Any],
        output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        predictions = output.get("predictions")
        if hasattr(predictions, "shape"):
            return {"prediction_shape": list(predictions.shape)}
        return {}
