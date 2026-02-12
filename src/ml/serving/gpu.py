"""
GPU-optimized inference module.

Provides:
- GPU memory management
- Multi-GPU support
- CUDA stream parallelism
- Mixed precision inference
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    device_id: int
    name: str
    total_memory_mb: float
    free_memory_mb: float
    utilization_percent: float = 0.0
    temperature_c: Optional[float] = None

    @property
    def used_memory_mb(self) -> float:
        return self.total_memory_mb - self.free_memory_mb

    @property
    def memory_utilization_percent(self) -> float:
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "total_memory_mb": round(self.total_memory_mb, 2),
            "free_memory_mb": round(self.free_memory_mb, 2),
            "used_memory_mb": round(self.used_memory_mb, 2),
            "memory_utilization_percent": round(self.memory_utilization_percent, 1),
            "utilization_percent": round(self.utilization_percent, 1),
            "temperature_c": self.temperature_c,
        }


@dataclass
class GPUConfig:
    """Configuration for GPU inference."""
    device_ids: Optional[List[int]] = None  # None = auto-detect
    memory_fraction: float = 0.9  # Max fraction of GPU memory to use
    enable_mixed_precision: bool = True
    enable_cuda_graphs: bool = False  # Experimental
    enable_tensor_cores: bool = True
    pin_memory: bool = True
    non_blocking: bool = True
    num_streams: int = 2  # CUDA streams for parallel execution


class GPUManager:
    """
    Manages GPU resources for inference.

    Provides:
    - GPU memory tracking
    - Multi-GPU load balancing
    - Memory-aware model placement
    """

    _instance: Optional["GPUManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "GPUManager":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[GPUConfig] = None):
        if self._initialized:
            return

        self._config = config or GPUConfig()
        self._available_gpus: List[int] = []
        self._gpu_info: Dict[int, GPUInfo] = {}
        self._model_placements: Dict[str, int] = {}  # model_name -> device_id
        self._device_locks: Dict[int, threading.Lock] = {}
        self._streams: Dict[int, List[Any]] = {}  # device_id -> CUDA streams

        self._detect_gpus()
        self._initialized = True

    def _detect_gpus(self) -> None:
        """Detect available GPUs."""
        try:
            import torch

            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                device_ids = self._config.device_ids or list(range(num_gpus))

                for device_id in device_ids:
                    if device_id >= num_gpus:
                        continue

                    self._available_gpus.append(device_id)
                    self._device_locks[device_id] = threading.Lock()

                    # Create CUDA streams
                    self._streams[device_id] = [
                        torch.cuda.Stream(device=device_id)
                        for _ in range(self._config.num_streams)
                    ]

                    # Get initial GPU info
                    self._gpu_info[device_id] = self._get_gpu_info(device_id)

                logger.info(f"Detected {len(self._available_gpus)} GPUs: {self._available_gpus}")

            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._available_gpus = [-1]  # Use -1 for MPS
                logger.info("Detected Apple MPS device")

        except ImportError:
            logger.warning("PyTorch not available, GPU support disabled")

    def _get_gpu_info(self, device_id: int) -> GPUInfo:
        """Get GPU information."""
        try:
            import torch

            props = torch.cuda.get_device_properties(device_id)
            total_memory = props.total_memory / (1024 ** 2)

            # Get current memory usage
            torch.cuda.set_device(device_id)
            allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
            free_memory = total_memory - reserved

            return GPUInfo(
                device_id=device_id,
                name=props.name,
                total_memory_mb=total_memory,
                free_memory_mb=free_memory,
            )

        except Exception as e:
            logger.warning(f"Failed to get GPU info for device {device_id}: {e}")
            return GPUInfo(
                device_id=device_id,
                name="unknown",
                total_memory_mb=0,
                free_memory_mb=0,
            )

    @property
    def available_gpus(self) -> List[int]:
        """Get list of available GPU device IDs."""
        return self._available_gpus.copy()

    @property
    def num_gpus(self) -> int:
        """Get number of available GPUs."""
        return len(self._available_gpus)

    @property
    def has_gpu(self) -> bool:
        """Check if any GPU is available."""
        return len(self._available_gpus) > 0

    def get_device(self, model_name: Optional[str] = None) -> str:
        """
        Get best device for inference.

        Args:
            model_name: Optional model name for placement lookup

        Returns:
            Device string (e.g., "cuda:0", "cpu")
        """
        if not self.has_gpu:
            return "cpu"

        # Check if model has assigned placement
        if model_name and model_name in self._model_placements:
            device_id = self._model_placements[model_name]
            return f"cuda:{device_id}" if device_id >= 0 else "mps"

        # Find GPU with most free memory
        best_device = self._available_gpus[0]
        best_free_memory = 0

        for device_id in self._available_gpus:
            if device_id < 0:  # MPS
                return "mps"

            info = self._get_gpu_info(device_id)
            self._gpu_info[device_id] = info

            if info.free_memory_mb > best_free_memory:
                best_free_memory = info.free_memory_mb
                best_device = device_id

        return f"cuda:{best_device}"

    def assign_model(self, model_name: str, device_id: Optional[int] = None) -> str:
        """
        Assign a model to a specific GPU.

        Args:
            model_name: Model name
            device_id: GPU device ID (None for auto-assign)

        Returns:
            Assigned device string
        """
        if device_id is None:
            # Auto-assign to GPU with most free memory
            device = self.get_device()
            if device.startswith("cuda:"):
                device_id = int(device.split(":")[1])
            elif device == "mps":
                device_id = -1
            else:
                device_id = -2  # CPU

        self._model_placements[model_name] = device_id
        logger.info(f"Assigned model {model_name} to device {device_id}")

        if device_id >= 0:
            return f"cuda:{device_id}"
        elif device_id == -1:
            return "mps"
        return "cpu"

    def release_model(self, model_name: str) -> None:
        """Release model placement."""
        self._model_placements.pop(model_name, None)

    @contextmanager
    def device_context(self, device_id: int):
        """Context manager for GPU operations."""
        try:
            import torch

            if device_id >= 0:
                with self._device_locks.get(device_id, threading.Lock()):
                    with torch.cuda.device(device_id):
                        yield
            else:
                yield

        except ImportError:
            yield

    def get_stream(self, device_id: int, stream_idx: int = 0) -> Optional[Any]:
        """Get CUDA stream for device."""
        streams = self._streams.get(device_id, [])
        if streams and stream_idx < len(streams):
            return streams[stream_idx]
        return None

    def synchronize(self, device_id: Optional[int] = None) -> None:
        """Synchronize GPU operations."""
        try:
            import torch

            if device_id is not None:
                torch.cuda.synchronize(device_id)
            else:
                torch.cuda.synchronize()

        except (ImportError, RuntimeError):
            pass

    def clear_cache(self, device_id: Optional[int] = None) -> None:
        """Clear GPU memory cache."""
        try:
            import torch

            if device_id is not None:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()

        except (ImportError, RuntimeError):
            pass

    def get_memory_info(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """Get memory information for GPU(s)."""
        if device_id is not None:
            info = self._get_gpu_info(device_id)
            return info.to_dict()

        return {
            device_id: self._get_gpu_info(device_id).to_dict()
            for device_id in self._available_gpus
            if device_id >= 0
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get GPU manager statistics."""
        return {
            "available_gpus": self._available_gpus,
            "num_gpus": self.num_gpus,
            "model_placements": dict(self._model_placements),
            "gpu_info": {
                device_id: self._get_gpu_info(device_id).to_dict()
                for device_id in self._available_gpus
                if device_id >= 0
            },
        }


class MixedPrecisionInference:
    """
    Mixed precision (FP16/BF16) inference support.

    Provides:
    - Automatic mixed precision
    - Memory savings
    - Performance improvements on supported hardware
    """

    def __init__(
        self,
        dtype: str = "float16",
        enabled: bool = True,
    ):
        """
        Initialize mixed precision inference.

        Args:
            dtype: Data type ("float16", "bfloat16")
            enabled: Whether mixed precision is enabled
        """
        self._dtype_str = dtype
        self._enabled = enabled
        self._dtype = None
        self._autocast_context = None

        self._setup()

    def _setup(self) -> None:
        """Setup mixed precision."""
        if not self._enabled:
            return

        try:
            import torch

            if self._dtype_str == "bfloat16":
                self._dtype = torch.bfloat16
            else:
                self._dtype = torch.float16

            logger.info(f"Mixed precision enabled with dtype={self._dtype_str}")

        except ImportError:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def autocast(self, device_type: str = "cuda"):
        """Context manager for automatic mixed precision."""
        if not self._enabled:
            yield
            return

        try:
            import torch

            if device_type == "cuda" and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=self._dtype):
                    yield
            elif device_type == "cpu":
                with torch.cpu.amp.autocast(dtype=self._dtype):
                    yield
            else:
                yield

        except (ImportError, AttributeError):
            yield

    def convert_model(self, model: Any) -> Any:
        """Convert model to mixed precision."""
        if not self._enabled:
            return model

        try:
            import torch

            if hasattr(model, "half") and self._dtype == torch.float16:
                return model.half()
            elif hasattr(model, "bfloat16") and self._dtype == torch.bfloat16:
                return model.bfloat16()

        except (ImportError, RuntimeError) as e:
            logger.warning(f"Failed to convert model to mixed precision: {e}")

        return model


class CUDAGraphExecutor:
    """
    CUDA Graph execution for reduced kernel launch overhead.

    Note: Experimental feature, requires static input shapes.
    """

    def __init__(self, model: Any, sample_input: Any, device: str = "cuda:0"):
        """
        Initialize CUDA Graph executor.

        Args:
            model: PyTorch model
            sample_input: Sample input for graph capture
            device: CUDA device
        """
        self._model = model
        self._sample_input = sample_input
        self._device = device
        self._graph: Optional[Any] = None
        self._static_input: Optional[Any] = None
        self._static_output: Optional[Any] = None
        self._captured = False

    def capture(self) -> bool:
        """Capture CUDA graph."""
        try:
            import torch

            if not torch.cuda.is_available():
                return False

            self._model.eval()

            # Prepare static tensors
            if isinstance(self._sample_input, torch.Tensor):
                self._static_input = self._sample_input.clone().to(self._device)
            else:
                self._static_input = torch.tensor(self._sample_input).to(self._device)

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = self._model(self._static_input)

            torch.cuda.synchronize()

            # Capture graph
            self._graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(self._graph):
                self._static_output = self._model(self._static_input)

            self._captured = True
            logger.info("CUDA Graph captured successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to capture CUDA Graph: {e}")
            return False

    def execute(self, inputs: Any) -> Any:
        """Execute captured CUDA graph."""
        if not self._captured:
            raise RuntimeError("CUDA Graph not captured")

        try:
            import torch

            # Copy input to static buffer
            if isinstance(inputs, torch.Tensor):
                self._static_input.copy_(inputs)
            else:
                self._static_input.copy_(torch.tensor(inputs))

            # Replay graph
            self._graph.replay()

            # Return copy of output
            return self._static_output.clone()

        except Exception as e:
            logger.error(f"CUDA Graph execution failed: {e}")
            raise

    @property
    def is_captured(self) -> bool:
        return self._captured


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def get_best_device() -> str:
    """Get best available device for inference."""
    return get_gpu_manager().get_device()
