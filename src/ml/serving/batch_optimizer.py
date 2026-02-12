"""
Batch inference optimizer for improved throughput.

Provides:
- Adaptive batch sizing
- Memory-aware batching
- Batch padding and unpadding
- Batch inference strategies
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BatchStrategy(str, Enum):
    """Batch inference strategies."""
    FIXED = "fixed"  # Fixed batch size
    ADAPTIVE = "adaptive"  # Adjust based on latency
    MEMORY_AWARE = "memory_aware"  # Adjust based on GPU memory
    THROUGHPUT = "throughput"  # Maximize throughput


@dataclass
class BatchOptimizerConfig:
    """Configuration for batch optimizer."""
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    initial_batch_size: int = 16
    min_batch_size: int = 1
    max_batch_size: int = 64
    target_latency_ms: float = 100.0
    max_memory_percent: float = 80.0  # Max GPU memory to use
    adjustment_factor: float = 0.2  # Batch size adjustment rate
    warmup_batches: int = 5  # Batches before optimization kicks in
    enable_padding: bool = True
    pad_to_multiple: int = 8  # Pad batch size to multiple for efficiency


@dataclass
class BatchMetrics:
    """Metrics for batch inference."""
    batch_size: int
    latency_ms: float
    throughput: float  # samples/second
    memory_used_mb: float = 0.0
    padding_overhead: float = 0.0  # fraction of padded samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "latency_ms": round(self.latency_ms, 2),
            "throughput": round(self.throughput, 2),
            "memory_used_mb": round(self.memory_used_mb, 2),
            "padding_overhead": round(self.padding_overhead, 4),
        }


class BatchOptimizer:
    """
    Optimizer for batch inference.

    Dynamically adjusts batch size based on:
    - Latency targets
    - GPU memory availability
    - Throughput optimization
    """

    def __init__(self, config: Optional[BatchOptimizerConfig] = None):
        """
        Initialize batch optimizer.

        Args:
            config: Optimizer configuration
        """
        self._config = config or BatchOptimizerConfig()
        self._current_batch_size = self._config.initial_batch_size
        self._metrics_history: List[BatchMetrics] = []
        self._batch_count = 0
        self._warmup_complete = False

    @property
    def current_batch_size(self) -> int:
        return self._current_batch_size

    @property
    def config(self) -> BatchOptimizerConfig:
        return self._config

    def get_optimal_batch_size(self, pending_count: int) -> int:
        """
        Get optimal batch size for current conditions.

        Args:
            pending_count: Number of pending requests

        Returns:
            Optimal batch size
        """
        if self._config.strategy == BatchStrategy.FIXED:
            batch_size = min(self._config.initial_batch_size, pending_count)
        elif not self._warmup_complete:
            batch_size = min(self._current_batch_size, pending_count)
        else:
            batch_size = min(self._current_batch_size, pending_count)

        # Apply padding if enabled
        if self._config.enable_padding and batch_size > 1:
            batch_size = self._pad_batch_size(batch_size)

        return max(self._config.min_batch_size, min(batch_size, self._config.max_batch_size))

    def _pad_batch_size(self, size: int) -> int:
        """Pad batch size to multiple for efficiency."""
        multiple = self._config.pad_to_multiple
        if size % multiple == 0:
            return size
        return ((size // multiple) + 1) * multiple

    def record_batch(self, metrics: BatchMetrics) -> None:
        """
        Record batch metrics and adjust batch size.

        Args:
            metrics: Batch metrics
        """
        self._metrics_history.append(metrics)
        self._batch_count += 1

        # Keep limited history
        if len(self._metrics_history) > 100:
            self._metrics_history.pop(0)

        # Check warmup
        if self._batch_count >= self._config.warmup_batches:
            self._warmup_complete = True

        # Adjust batch size based on strategy
        if self._warmup_complete and self._config.strategy != BatchStrategy.FIXED:
            self._adjust_batch_size(metrics)

    def _adjust_batch_size(self, latest: BatchMetrics) -> None:
        """Adjust batch size based on strategy and metrics."""
        if self._config.strategy == BatchStrategy.ADAPTIVE:
            self._adjust_for_latency(latest)
        elif self._config.strategy == BatchStrategy.MEMORY_AWARE:
            self._adjust_for_memory(latest)
        elif self._config.strategy == BatchStrategy.THROUGHPUT:
            self._adjust_for_throughput(latest)

    def _adjust_for_latency(self, latest: BatchMetrics) -> None:
        """Adjust batch size to meet latency target."""
        target = self._config.target_latency_ms
        adjustment = self._config.adjustment_factor

        if latest.latency_ms > target * 1.2:
            # Latency too high, reduce batch size
            new_size = int(self._current_batch_size * (1 - adjustment))
            self._current_batch_size = max(self._config.min_batch_size, new_size)
            logger.debug(f"Reduced batch size to {self._current_batch_size} (latency={latest.latency_ms:.1f}ms)")

        elif latest.latency_ms < target * 0.8:
            # Latency low, can increase batch size
            new_size = int(self._current_batch_size * (1 + adjustment))
            self._current_batch_size = min(self._config.max_batch_size, new_size)
            logger.debug(f"Increased batch size to {self._current_batch_size} (latency={latest.latency_ms:.1f}ms)")

    def _adjust_for_memory(self, latest: BatchMetrics) -> None:
        """Adjust batch size based on memory usage."""
        max_memory = self._config.max_memory_percent
        adjustment = self._config.adjustment_factor

        # Estimate memory per sample
        if latest.batch_size > 0 and latest.memory_used_mb > 0:
            memory_per_sample = latest.memory_used_mb / latest.batch_size

            # Get available memory
            try:
                from src.ml.serving.gpu import get_gpu_manager
                gpu_info = get_gpu_manager().get_memory_info()
                if gpu_info:
                    first_gpu = list(gpu_info.values())[0] if isinstance(gpu_info, dict) else gpu_info
                    if isinstance(first_gpu, dict):
                        free_memory = first_gpu.get("free_memory_mb", 0)
                        total_memory = first_gpu.get("total_memory_mb", 1)
                        memory_usage = (total_memory - free_memory) / total_memory * 100

                        if memory_usage > max_memory:
                            new_size = int(self._current_batch_size * (1 - adjustment))
                            self._current_batch_size = max(self._config.min_batch_size, new_size)
                        elif memory_usage < max_memory * 0.7:
                            new_size = int(self._current_batch_size * (1 + adjustment))
                            self._current_batch_size = min(self._config.max_batch_size, new_size)
            except (ImportError, Exception):
                # Fall back to latency-based adjustment
                self._adjust_for_latency(latest)

    def _adjust_for_throughput(self, latest: BatchMetrics) -> None:
        """Adjust batch size to maximize throughput."""
        if len(self._metrics_history) < 3:
            return

        # Get recent throughputs
        recent = self._metrics_history[-3:]
        avg_throughput = sum(m.throughput for m in recent) / len(recent)

        # Try increasing batch size if throughput is improving
        if latest.throughput > avg_throughput * 1.05:
            new_size = int(self._current_batch_size * 1.1)
            self._current_batch_size = min(self._config.max_batch_size, new_size)
        elif latest.throughput < avg_throughput * 0.9:
            new_size = int(self._current_batch_size * 0.9)
            self._current_batch_size = max(self._config.min_batch_size, new_size)

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        if not self._metrics_history:
            return {
                "current_batch_size": self._current_batch_size,
                "warmup_complete": self._warmup_complete,
                "batch_count": self._batch_count,
            }

        recent = self._metrics_history[-10:]
        return {
            "current_batch_size": self._current_batch_size,
            "warmup_complete": self._warmup_complete,
            "batch_count": self._batch_count,
            "strategy": self._config.strategy.value,
            "avg_latency_ms": round(sum(m.latency_ms for m in recent) / len(recent), 2),
            "avg_throughput": round(sum(m.throughput for m in recent) / len(recent), 2),
            "avg_batch_size": round(sum(m.batch_size for m in recent) / len(recent), 1),
        }

    def reset(self) -> None:
        """Reset optimizer state."""
        self._current_batch_size = self._config.initial_batch_size
        self._metrics_history.clear()
        self._batch_count = 0
        self._warmup_complete = False


class BatchPadder:
    """
    Handles batch padding for variable-length inputs.

    Supports:
    - Sequence padding
    - Graph padding
    - Attention mask generation
    """

    def __init__(
        self,
        pad_token: Any = 0,
        max_length: Optional[int] = None,
        padding_side: str = "right",
    ):
        """
        Initialize batch padder.

        Args:
            pad_token: Token to use for padding
            max_length: Maximum sequence length (None for dynamic)
            padding_side: "left" or "right"
        """
        self._pad_token = pad_token
        self._max_length = max_length
        self._padding_side = padding_side

    def pad_batch(
        self,
        sequences: List[Any],
        return_attention_mask: bool = True,
    ) -> Tuple[Any, Optional[Any]]:
        """
        Pad a batch of sequences.

        Args:
            sequences: List of sequences
            return_attention_mask: Whether to return attention mask

        Returns:
            (padded_batch, attention_mask)
        """
        try:
            import torch

            # Get max length
            lengths = [len(seq) if hasattr(seq, "__len__") else 1 for seq in sequences]
            max_len = self._max_length or max(lengths)

            # Pad sequences
            padded = []
            masks = [] if return_attention_mask else None

            for seq in sequences:
                if isinstance(seq, torch.Tensor):
                    seq_len = seq.shape[0]
                    if seq_len < max_len:
                        pad_len = max_len - seq_len
                        if self._padding_side == "right":
                            pad_tensor = torch.full((pad_len,) + seq.shape[1:], self._pad_token, dtype=seq.dtype)
                            padded_seq = torch.cat([seq, pad_tensor], dim=0)
                        else:
                            pad_tensor = torch.full((pad_len,) + seq.shape[1:], self._pad_token, dtype=seq.dtype)
                            padded_seq = torch.cat([pad_tensor, seq], dim=0)
                    else:
                        padded_seq = seq[:max_len]
                    padded.append(padded_seq)

                    if return_attention_mask:
                        mask = torch.zeros(max_len)
                        if self._padding_side == "right":
                            mask[:min(seq_len, max_len)] = 1
                        else:
                            mask[max(0, max_len - seq_len):] = 1
                        masks.append(mask)
                else:
                    # Handle list inputs
                    seq_len = len(seq) if hasattr(seq, "__len__") else 1
                    if seq_len < max_len:
                        pad_len = max_len - seq_len
                        if self._padding_side == "right":
                            padded_seq = list(seq) + [self._pad_token] * pad_len
                        else:
                            padded_seq = [self._pad_token] * pad_len + list(seq)
                    else:
                        padded_seq = list(seq)[:max_len]
                    padded.append(padded_seq)

                    if return_attention_mask:
                        mask = [0] * max_len
                        if self._padding_side == "right":
                            for i in range(min(seq_len, max_len)):
                                mask[i] = 1
                        else:
                            for i in range(max(0, max_len - seq_len), max_len):
                                mask[i] = 1
                        masks.append(mask)

            # Stack into batch
            if isinstance(padded[0], torch.Tensor):
                batch = torch.stack(padded)
                mask_batch = torch.stack(masks) if masks else None
            else:
                batch = padded
                mask_batch = masks

            return batch, mask_batch

        except ImportError:
            # Non-PyTorch fallback
            max_len = self._max_length or max(len(s) if hasattr(s, "__len__") else 1 for s in sequences)
            padded = []
            masks = []

            for seq in sequences:
                seq_list = list(seq) if hasattr(seq, "__iter__") else [seq]
                seq_len = len(seq_list)

                if seq_len < max_len:
                    pad_len = max_len - seq_len
                    if self._padding_side == "right":
                        padded.append(seq_list + [self._pad_token] * pad_len)
                    else:
                        padded.append([self._pad_token] * pad_len + seq_list)
                else:
                    padded.append(seq_list[:max_len])

                if return_attention_mask:
                    mask = [0] * max_len
                    if self._padding_side == "right":
                        mask[:min(seq_len, max_len)] = [1] * min(seq_len, max_len)
                    else:
                        mask[max(0, max_len - seq_len):] = [1] * min(seq_len, max_len)
                    masks.append(mask)

            return padded, masks if return_attention_mask else None

    def unpad_batch(
        self,
        batch: Any,
        attention_mask: Optional[Any] = None,
        original_lengths: Optional[List[int]] = None,
    ) -> List[Any]:
        """
        Remove padding from batch.

        Args:
            batch: Padded batch
            attention_mask: Attention mask (alternative to lengths)
            original_lengths: Original sequence lengths

        Returns:
            List of unpadded sequences
        """
        try:
            import torch

            unpadded = []
            batch_size = len(batch)

            for i in range(batch_size):
                if original_lengths:
                    length = original_lengths[i]
                elif attention_mask is not None:
                    if isinstance(attention_mask, torch.Tensor):
                        length = int(attention_mask[i].sum().item())
                    else:
                        length = sum(attention_mask[i])
                else:
                    # No length info, return as-is
                    unpadded.append(batch[i])
                    continue

                if self._padding_side == "right":
                    unpadded.append(batch[i][:length])
                else:
                    unpadded.append(batch[i][-length:])

            return unpadded

        except ImportError:
            # Non-PyTorch fallback
            return list(batch)


class BatchInferenceRunner:
    """
    Runs batch inference with optimization.

    Combines:
    - Batch size optimization
    - Padding/unpadding
    - Memory management
    """

    def __init__(
        self,
        model: Any,
        optimizer: Optional[BatchOptimizer] = None,
        padder: Optional[BatchPadder] = None,
        device: str = "cpu",
    ):
        """
        Initialize batch inference runner.

        Args:
            model: Model for inference
            optimizer: Batch optimizer
            padder: Batch padder
            device: Device for inference
        """
        self._model = model
        self._optimizer = optimizer or BatchOptimizer()
        self._padder = padder or BatchPadder()
        self._device = device

    def run_batch(
        self,
        inputs: List[Any],
        return_metrics: bool = False,
    ) -> Tuple[List[Any], Optional[BatchMetrics]]:
        """
        Run batch inference.

        Args:
            inputs: Input samples
            return_metrics: Whether to return metrics

        Returns:
            (outputs, metrics)
        """
        try:
            import torch

            start_time = time.time()
            batch_size = len(inputs)

            # Pad batch if needed
            needs_padding = any(
                hasattr(inp, "__len__") and len(inp) != len(inputs[0])
                if hasattr(inputs[0], "__len__") else False
                for inp in inputs
            )

            if needs_padding:
                padded, mask = self._padder.pad_batch(inputs)
            else:
                padded = inputs
                mask = None

            # Move to device
            if isinstance(padded, torch.Tensor):
                padded = padded.to(self._device)
            elif isinstance(padded, list) and padded and isinstance(padded[0], torch.Tensor):
                padded = torch.stack(padded).to(self._device)

            # Run inference
            with torch.no_grad():
                if hasattr(self._model, "forward"):
                    outputs = self._model(padded)
                elif hasattr(self._model, "predict"):
                    outputs = self._model.predict(padded)
                else:
                    raise ValueError("Model has no forward or predict method")

            # Unpad if needed
            if needs_padding and mask is not None:
                outputs = self._padder.unpad_batch(outputs, mask)

            latency_ms = (time.time() - start_time) * 1000

            # Record metrics
            metrics = None
            if return_metrics:
                # Get memory usage
                memory_mb = 0.0
                if torch.cuda.is_available() and "cuda" in self._device:
                    memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)

                metrics = BatchMetrics(
                    batch_size=batch_size,
                    latency_ms=latency_ms,
                    throughput=batch_size / (latency_ms / 1000) if latency_ms > 0 else 0,
                    memory_used_mb=memory_mb,
                )
                self._optimizer.record_batch(metrics)

            return outputs, metrics

        except ImportError:
            # Non-PyTorch fallback
            start_time = time.time()

            if hasattr(self._model, "predict"):
                outputs = self._model.predict(inputs)
            else:
                outputs = [self._model(inp) for inp in inputs]

            latency_ms = (time.time() - start_time) * 1000

            metrics = None
            if return_metrics:
                metrics = BatchMetrics(
                    batch_size=len(inputs),
                    latency_ms=latency_ms,
                    throughput=len(inputs) / (latency_ms / 1000) if latency_ms > 0 else 0,
                )
                self._optimizer.record_batch(metrics)

            return outputs, metrics

    def get_stats(self) -> Dict[str, Any]:
        """Get runner statistics."""
        return {
            "device": self._device,
            "optimizer": self._optimizer.get_stats(),
        }
