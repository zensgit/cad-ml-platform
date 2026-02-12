"""
Model worker for inference serving.

Provides:
- Model loading and management
- Inference execution
- Performance tracking
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.ml.serving.request import Prediction, InferenceRequest, InferenceResponse, RequestStatus

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for model worker."""
    device: str = "cpu"
    max_batch_size: int = 32
    warmup_iterations: int = 3
    enable_profiling: bool = False


@dataclass
class WorkerStats:
    """Statistics for a model worker."""
    model_name: str
    model_version: str = ""
    load_time: float = 0.0
    warmup_time: float = 0.0
    total_requests: int = 0
    total_samples: int = 0
    total_inference_time: float = 0.0
    errors: int = 0
    last_inference_time: Optional[float] = None

    @property
    def avg_latency_ms(self) -> float:
        """Average inference latency in ms."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_inference_time / self.total_requests) * 1000

    @property
    def avg_sample_latency_ms(self) -> float:
        """Average latency per sample in ms."""
        if self.total_samples == 0:
            return 0.0
        return (self.total_inference_time / self.total_samples) * 1000

    @property
    def throughput(self) -> float:
        """Samples per second."""
        if self.total_inference_time == 0:
            return 0.0
        return self.total_samples / self.total_inference_time

    @property
    def error_rate(self) -> float:
        """Error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.errors / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "load_time_ms": round(self.load_time * 1000, 2),
            "warmup_time_ms": round(self.warmup_time * 1000, 2),
            "total_requests": self.total_requests,
            "total_samples": self.total_samples,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_sample_latency_ms": round(self.avg_sample_latency_ms, 2),
            "throughput": round(self.throughput, 2),
            "errors": self.errors,
            "error_rate": round(self.error_rate, 4),
        }


class ModelWorker:
    """
    Worker for model inference.

    Handles:
    - Model loading and unloading
    - Batch inference execution
    - Performance tracking
    """

    def __init__(
        self,
        model_path: str,
        model_name: str,
        config: Optional[WorkerConfig] = None,
        predict_fn: Optional[Callable] = None,
    ):
        """
        Initialize model worker.

        Args:
            model_path: Path to model file
            model_name: Name of the model
            config: Worker configuration
            predict_fn: Custom prediction function
        """
        self._model_path = Path(model_path)
        self._model_name = model_name
        self._config = config or WorkerConfig()
        self._predict_fn = predict_fn

        self._model: Any = None
        self._model_version: str = ""
        self._label_map: Optional[Dict[int, str]] = None
        self._is_loaded = False

        self._stats = WorkerStats(model_name=model_name)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def stats(self) -> WorkerStats:
        """Get worker statistics."""
        return self._stats

    def load(self) -> float:
        """
        Load model into memory.

        Returns:
            Load time in seconds
        """
        if self._is_loaded:
            return 0.0

        start_time = time.time()

        try:
            import torch

            checkpoint = torch.load(
                self._model_path,
                map_location=self._config.device,
                weights_only=False,
            )

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    self._label_map = checkpoint.get("label_map")
                    self._model_version = checkpoint.get("version", "")

                    # Try to infer model architecture
                    model = self._create_model_from_config(checkpoint)
                    if model:
                        model.load_state_dict(state_dict)
                        self._model = model
                    else:
                        # Store state dict for later
                        self._model = checkpoint
                else:
                    self._model = checkpoint
            else:
                self._model = checkpoint

            # Move to device
            if hasattr(self._model, "to"):
                self._model = self._model.to(self._config.device)

            # Set eval mode
            if hasattr(self._model, "eval"):
                self._model.eval()

            self._is_loaded = True
            load_time = time.time() - start_time
            self._stats.load_time = load_time
            self._stats.model_version = self._model_version

            logger.info(
                f"Loaded model {self._model_name} in {load_time*1000:.2f}ms "
                f"on {self._config.device}"
            )

            return load_time

        except Exception as e:
            logger.error(f"Failed to load model {self._model_name}: {e}")
            raise

    def unload(self) -> None:
        """Unload model from memory."""
        if not self._is_loaded:
            return

        try:
            import torch

            if hasattr(self._model, "cpu"):
                self._model.cpu()

            del self._model
            self._model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._is_loaded = False
            logger.info(f"Unloaded model {self._model_name}")

        except Exception as e:
            logger.error(f"Error unloading model {self._model_name}: {e}")

    def warmup(self, sample_input: Optional[Any] = None) -> float:
        """
        Warm up model with sample inference.

        Args:
            sample_input: Sample input for warmup

        Returns:
            Warmup time in seconds
        """
        if not self._is_loaded:
            self.load()

        start_time = time.time()

        try:
            import torch

            # Generate sample input if not provided
            if sample_input is None:
                # Create dummy input based on model
                sample_input = self._create_dummy_input()

            # Run warmup iterations
            with torch.no_grad():
                for _ in range(self._config.warmup_iterations):
                    self._run_inference([sample_input])

            warmup_time = time.time() - start_time
            self._stats.warmup_time = warmup_time

            logger.info(f"Warmed up model {self._model_name} in {warmup_time*1000:.2f}ms")
            return warmup_time

        except Exception as e:
            logger.warning(f"Warmup failed for {self._model_name}: {e}")
            return 0.0

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """
        Execute inference for a request.

        Args:
            request: Inference request

        Returns:
            InferenceResponse
        """
        if not self._is_loaded:
            self.load()

        request.start()
        start_time = time.time()

        try:
            # Run inference
            results = self._run_inference(request.inputs)

            # Build predictions
            predictions = []
            for result in results:
                if isinstance(result, dict):
                    predictions.append(Prediction(
                        label=result.get("label", 0),
                        confidence=result.get("confidence", 0.0),
                        probabilities=result.get("probabilities"),
                        embeddings=result.get("embeddings"),
                    ))
                elif isinstance(result, tuple):
                    predictions.append(Prediction(
                        label=result[0],
                        confidence=result[1] if len(result) > 1 else 1.0,
                        probabilities=result[2] if len(result) > 2 else None,
                    ))
                else:
                    predictions.append(Prediction(
                        label=result,
                        confidence=1.0,
                    ))

            request.complete()
            latency = time.time() - start_time

            # Update stats
            self._stats.total_requests += 1
            self._stats.total_samples += len(request.inputs)
            self._stats.total_inference_time += latency
            self._stats.last_inference_time = latency

            return InferenceResponse(
                request_id=request.request_id,
                predictions=predictions,
                model_name=self._model_name,
                model_version=self._model_version,
                latency_ms=latency * 1000,
            )

        except Exception as e:
            request.fail()
            self._stats.errors += 1
            logger.error(f"Inference failed for {self._model_name}: {e}")

            return InferenceResponse.error_response(
                request_id=request.request_id,
                model_name=self._model_name,
                error=str(e),
            )

    def predict_batch(self, inputs: List[Any]) -> List[Prediction]:
        """
        Execute batch inference.

        Args:
            inputs: List of inputs

        Returns:
            List of Predictions
        """
        request = InferenceRequest(
            model_name=self._model_name,
            inputs=inputs,
        )
        response = self.predict(request)
        return response.predictions

    def _run_inference(self, inputs: List[Any]) -> List[Any]:
        """Run model inference on inputs."""
        if self._predict_fn:
            return self._predict_fn(self._model, inputs)

        try:
            import torch

            # Default inference logic
            with torch.no_grad():
                if hasattr(self._model, "predict"):
                    return self._model.predict(inputs)
                elif hasattr(self._model, "forward"):
                    # Convert inputs to tensor if needed
                    if isinstance(inputs[0], torch.Tensor):
                        batch = torch.stack(inputs)
                    else:
                        batch = torch.tensor(inputs)

                    batch = batch.to(self._config.device)
                    outputs = self._model(batch)

                    # Convert outputs to predictions
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    probs = torch.softmax(outputs, dim=-1)
                    confidences, labels = torch.max(probs, dim=-1)

                    results = []
                    for i in range(len(inputs)):
                        label = int(labels[i].item())
                        if self._label_map:
                            label = self._label_map.get(label, label)
                        results.append({
                            "label": label,
                            "confidence": float(confidences[i].item()),
                            "probabilities": probs[i].tolist(),
                        })
                    return results
                else:
                    raise ValueError(f"Model {self._model_name} has no predict or forward method")

        except ImportError:
            # Non-PyTorch model
            if hasattr(self._model, "predict"):
                return self._model.predict(inputs)
            raise ValueError(f"Cannot run inference for model {self._model_name}")

    def _create_model_from_config(self, checkpoint: Dict) -> Optional[Any]:
        """Try to create model from checkpoint config."""
        config = checkpoint.get("config", {})

        if not config:
            return None

        # Try to infer model type
        model_type = config.get("model_type", checkpoint.get("model_type", ""))
        node_dim = config.get("node_dim", config.get("node_input_dim", 64))
        hidden_dim = config.get("hidden_dim", 64)
        num_classes = config.get("num_classes", len(checkpoint.get("label_map", {})))

        if not num_classes:
            return None

        try:
            if model_type in ("gcn", "edge_sage"):
                from src.ml.train.model_2d import SimpleGraphClassifier, EdgeGraphSageClassifier

                if model_type == "edge_sage":
                    edge_dim = config.get("edge_dim", 4)
                    return EdgeGraphSageClassifier(node_dim, edge_dim, hidden_dim, num_classes)
                else:
                    return SimpleGraphClassifier(node_dim, hidden_dim, num_classes)
        except ImportError:
            pass

        return None

    def _create_dummy_input(self) -> Any:
        """Create dummy input for warmup."""
        try:
            import torch
            # Default: 64-dim node features
            return torch.randn(10, 64)
        except ImportError:
            return [[0.0] * 64 for _ in range(10)]

    def get_info(self) -> Dict[str, Any]:
        """Get worker info."""
        return {
            "model_name": self._model_name,
            "model_path": str(self._model_path),
            "model_version": self._model_version,
            "device": self._config.device,
            "is_loaded": self._is_loaded,
            "stats": self._stats.to_dict(),
        }
