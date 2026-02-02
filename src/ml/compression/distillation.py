"""
Knowledge distillation for model compression.

Trains a smaller student model to mimic a larger teacher model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DistillationLoss(str, Enum):
    """Distillation loss type."""
    KL_DIVERGENCE = "kl_divergence"
    MSE = "mse"
    COSINE = "cosine"
    ATTENTION = "attention"
    HIDDEN_STATES = "hidden_states"


@dataclass
class DistillationConfig:
    """Distillation configuration."""
    temperature: float = 4.0  # Softmax temperature
    alpha: float = 0.5  # Balance between soft and hard targets
    loss_type: DistillationLoss = DistillationLoss.KL_DIVERGENCE
    use_attention_transfer: bool = False
    use_hidden_states: bool = False
    hidden_layers_to_match: Optional[List[int]] = None
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01


@dataclass
class DistillationResult:
    """Result of distillation."""
    teacher_params: int
    student_params: int
    compression_ratio: float
    teacher_accuracy: float
    student_accuracy: float
    accuracy_retention: float
    training_time_seconds: float
    final_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher_params": self.teacher_params,
            "student_params": self.student_params,
            "compression_ratio": round(self.compression_ratio, 2),
            "teacher_accuracy": round(self.teacher_accuracy, 4),
            "student_accuracy": round(self.student_accuracy, 4),
            "accuracy_retention": round(self.accuracy_retention, 4),
            "training_time_seconds": round(self.training_time_seconds, 2),
            "final_loss": round(self.final_loss, 4),
        }


class DistillationTrainer:
    """
    Knowledge distillation trainer.

    Trains a student model to mimic a teacher model.
    """

    def __init__(self, config: Optional[DistillationConfig] = None):
        """
        Initialize distillation trainer.

        Args:
            config: Distillation configuration
        """
        self._config = config or DistillationConfig()
        self._history: Dict[str, List[float]] = {
            "train_loss": [],
            "distill_loss": [],
            "hard_loss": [],
            "val_accuracy": [],
        }

    @property
    def config(self) -> DistillationConfig:
        return self._config

    @property
    def history(self) -> Dict[str, List[float]]:
        return self._history.copy()

    def distill(
        self,
        teacher: Any,
        student: Any,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        eval_fn: Optional[Callable] = None,
    ) -> Tuple[Any, DistillationResult]:
        """
        Perform knowledge distillation.

        Args:
            teacher: Teacher model (larger)
            student: Student model (smaller)
            train_loader: Training data loader
            val_loader: Validation data loader
            eval_fn: Evaluation function

        Returns:
            (trained_student, result)
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch required for distillation")

        start_time = time.time()

        # Setup
        device = next(teacher.parameters()).device
        student = student.to(device)
        teacher.eval()

        # Count parameters
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())

        # Evaluate teacher
        teacher_accuracy = eval_fn(teacher) if eval_fn else 0.0

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        # Setup loss
        hard_loss_fn = nn.CrossEntropyLoss()
        soft_loss_fn = self._get_distillation_loss()

        # Training loop
        final_loss = 0.0
        for epoch in range(self._config.epochs):
            student.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                # Get inputs and targets
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    inputs = batch.to(device)
                    targets = None

                optimizer.zero_grad()

                # Forward pass
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)

                student_outputs = student(inputs)

                # Calculate distillation loss
                soft_loss = soft_loss_fn(
                    student_outputs / self._config.temperature,
                    teacher_outputs / self._config.temperature,
                )
                soft_loss = soft_loss * (self._config.temperature ** 2)

                # Calculate hard loss
                if targets is not None:
                    hard_loss = hard_loss_fn(student_outputs, targets)
                    loss = self._config.alpha * soft_loss + (1 - self._config.alpha) * hard_loss
                else:
                    loss = soft_loss
                    hard_loss = torch.tensor(0.0)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            # Record history
            avg_loss = epoch_loss / max(num_batches, 1)
            self._history["train_loss"].append(avg_loss)
            final_loss = avg_loss

            # Validation
            if val_loader:
                val_acc = self._evaluate(student, val_loader, device)
                self._history["val_accuracy"].append(val_acc)
                logger.info(f"Epoch {epoch+1}/{self._config.epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{self._config.epochs}: loss={avg_loss:.4f}")

        training_time = time.time() - start_time

        # Final evaluation
        student_accuracy = eval_fn(student) if eval_fn else 0.0
        accuracy_retention = student_accuracy / teacher_accuracy if teacher_accuracy > 0 else 0.0
        compression_ratio = teacher_params / student_params if student_params > 0 else 1.0

        result = DistillationResult(
            teacher_params=teacher_params,
            student_params=student_params,
            compression_ratio=compression_ratio,
            teacher_accuracy=teacher_accuracy,
            student_accuracy=student_accuracy,
            accuracy_retention=accuracy_retention,
            training_time_seconds=training_time,
            final_loss=final_loss,
        )

        logger.info(
            f"Distillation complete: {compression_ratio:.2f}x compression, "
            f"{accuracy_retention:.2%} accuracy retention"
        )
        return student, result

    def _get_distillation_loss(self) -> Callable:
        """Get distillation loss function."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        def kl_loss(student_logits, teacher_logits):
            return F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )

        def mse_loss(student_logits, teacher_logits):
            return F.mse_loss(student_logits, teacher_logits)

        def cosine_loss(student_logits, teacher_logits):
            return 1 - F.cosine_similarity(student_logits, teacher_logits, dim=-1).mean()

        loss_map = {
            DistillationLoss.KL_DIVERGENCE: kl_loss,
            DistillationLoss.MSE: mse_loss,
            DistillationLoss.COSINE: cosine_loss,
        }

        return loss_map.get(self._config.loss_type, kl_loss)

    def _evaluate(self, model: Any, loader: Any, device: Any) -> float:
        """Evaluate model accuracy."""
        import torch

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    continue

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total if total > 0 else 0.0

    def save_student(self, model: Any, path: Union[str, Path]) -> None:
        """Save trained student model."""
        import torch

        path = Path(path)
        torch.save(model.state_dict(), path)
        logger.info(f"Saved student model to {path}")


def distill_model(
    teacher: Any,
    student: Any,
    train_loader: Any,
    val_loader: Optional[Any] = None,
    temperature: float = 4.0,
    alpha: float = 0.5,
    epochs: int = 10,
) -> Tuple[Any, DistillationResult]:
    """
    Convenience function for knowledge distillation.

    Args:
        teacher: Teacher model
        student: Student model
        train_loader: Training data
        val_loader: Validation data
        temperature: Softmax temperature
        alpha: Balance factor
        epochs: Training epochs

    Returns:
        (trained_student, result)
    """
    config = DistillationConfig(
        temperature=temperature,
        alpha=alpha,
        epochs=epochs,
    )
    trainer = DistillationTrainer(config)
    return trainer.distill(teacher, student, train_loader, val_loader)
