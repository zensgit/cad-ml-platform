"""
知识蒸馏模块

使用文件名/文本分类器作为"教师"信号，蒸馏到 GNN "学生"模型。
目标：在无文件名场景下提升 GNN 模型的准确率。

Feature Flags:
    DISTILLATION_ENABLED: 是否启用蒸馏训练 (default: false)
    DISTILLATION_ALPHA: CE/KL 混合系数 (default: 0.3)
    DISTILLATION_TEMPERATURE: 软标签温度 (default: 3.0)
    DISTILLATION_TEACHER_TYPE: 教师类型 filename|titleblock|hybrid (default: hybrid)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TeacherType(str, Enum):
    """教师模型类型"""

    FILENAME = "filename"
    TITLEBLOCK = "titleblock"
    HYBRID = "hybrid"


@dataclass
class DistillationConfig:
    """蒸馏配置"""

    alpha: float = 0.3  # CE 损失权重 (1-alpha 为 KL 权重)
    temperature: float = 3.0  # 软标签温度
    teacher_type: str = "hybrid"
    hard_label_weight: float = 0.7
    soft_label_weight: float = 0.3


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数

    L = α * CE(student, hard_labels) + (1-α) * T² * KL(student_soft, teacher_soft)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        temperature: float = 3.0,
        hard_loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.alpha = float(os.getenv("DISTILLATION_ALPHA", str(alpha)))
        self.temperature = float(
            os.getenv("DISTILLATION_TEMPERATURE", str(temperature))
        )
        self.hard_loss_fn = hard_loss_fn

        logger.info(
            "DistillationLoss initialized",
            extra={
                "alpha": self.alpha,
                "temperature": self.temperature,
            },
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算蒸馏损失

        Args:
            student_logits: 学生模型 logits (N, C)
            teacher_logits: 教师模型 logits (N, C)
            hard_labels: 真实标签 (N,)

        Returns:
            (total_loss, loss_components)
        """
        # Hard-label loss (defaults to plain CE). When supplied, `hard_loss_fn`
        # should have the same signature as CrossEntropyLoss: (logits, targets).
        if self.hard_loss_fn is not None:
            ce_loss = self.hard_loss_fn(student_logits, hard_labels)
        else:
            ce_loss = F.cross_entropy(student_logits, hard_labels)

        # 软标签 KL 散度损失
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")

        # 温度缩放
        kl_loss = kl_loss * (self.temperature**2)

        # 总损失
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        components = {
            "ce_loss": ce_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, components


class TeacherModel:
    """
    教师模型封装

    将 FilenameClassifier/HybridClassifier 封装为教师信号生成器
    """

    def __init__(
        self,
        teacher_type: str = "hybrid",
        label_to_idx: Optional[Dict[str, int]] = None,
        num_classes: int = 144,
    ):
        self.teacher_type = TeacherType(
            os.getenv("DISTILLATION_TEACHER_TYPE", teacher_type)
        )
        self.label_to_idx = label_to_idx or {}
        self.num_classes = num_classes

        # 懒加载分类器
        self._classifier = None

        logger.info(
            "TeacherModel initialized",
            extra={
                "teacher_type": self.teacher_type.value,
                "num_classes": self.num_classes,
            },
        )

    @property
    def classifier(self):
        """懒加载分类器"""
        if self._classifier is None:
            if self.teacher_type == TeacherType.HYBRID:
                from src.ml.hybrid_classifier import get_hybrid_classifier

                self._classifier = get_hybrid_classifier()
            elif self.teacher_type == TeacherType.FILENAME:
                from src.ml.filename_classifier import get_filename_classifier

                self._classifier = get_filename_classifier()
            else:
                from src.ml.titleblock_extractor import TitleBlockClassifier

                self._classifier = TitleBlockClassifier()
        return self._classifier

    def generate_soft_labels(
        self,
        filenames: List[str],
        file_bytes_list: Optional[List[Optional[bytes]]] = None,
        default_confidence: float = 0.1,
    ) -> torch.Tensor:
        """
        生成软标签

        Args:
            filenames: 文件名列表
            file_bytes_list: 每个样本的文件字节（可选；titleblock/hybrid 教师需要）
            default_confidence: 未匹配时的默认置信度

        Returns:
            软标签 logits (N, C)
        """
        batch_size = len(filenames)
        soft_labels = torch.zeros(batch_size, self.num_classes)
        if file_bytes_list is None:
            file_bytes_list = [None] * batch_size
        if len(file_bytes_list) != batch_size:
            raise ValueError(
                "file_bytes_list length must match filenames length "
                f"({len(file_bytes_list)} != {batch_size})"
            )

        for i, (filename, file_bytes) in enumerate(zip(filenames, file_bytes_list)):
            try:
                if self.teacher_type == TeacherType.HYBRID:
                    # Hybrid teacher can use titleblock/process signals if file_bytes are provided.
                    result = self.classifier.classify(filename, file_bytes=file_bytes)
                    label = result.label
                    confidence = result.confidence
                elif self.teacher_type == TeacherType.FILENAME:
                    result = self.classifier.predict(filename)
                    label = result.get("label")
                    confidence = result.get("confidence", 0.0)
                else:
                    # Titleblock teacher requires DXF entities.
                    label = None
                    confidence = 0.0
                    if file_bytes:
                        try:
                            from src.utils.dxf_io import read_dxf_entities_from_bytes

                            dxf_entities = read_dxf_entities_from_bytes(file_bytes)
                            result = self.classifier.predict(dxf_entities)
                            label = result.get("label")
                            confidence = result.get("confidence", 0.0)
                        except Exception as exc:
                            logger.debug(
                                "Titleblock teacher parse failed for %s: %s",
                                filename,
                                exc,
                            )

                if label:
                    label = str(label).strip()

                if label and label not in self.label_to_idx:
                    # Some training flows normalize fine labels into coarse buckets
                    # (e.g. "对接法兰" -> "法兰"). Prefer that mapping before falling
                    # back to "other"/uniform labels.
                    from src.ml.label_normalization import normalize_dxf_label

                    normalized = normalize_dxf_label(label, default=None)
                    if normalized != label and normalized in self.label_to_idx:
                        label = normalized

                if (
                    label
                    and label not in self.label_to_idx
                    and "other" in self.label_to_idx
                ):
                    # When the distillation teacher predicts a label outside the
                    # student's label map (e.g., after manifest cleaning),
                    # fall back to "other" with a low confidence.
                    label = "other"
                    confidence = min(float(confidence or 0.0), 0.25)

                if label and label in self.label_to_idx:
                    idx = self.label_to_idx[label]
                    # 将置信度转换为 logit
                    soft_labels[i, idx] = confidence * 5.0  # 放大以产生更明确的分布
                else:
                    # 均匀分布
                    soft_labels[i, :] = default_confidence

            except Exception as e:
                logger.debug(f"Teacher prediction failed for {filename}: {e}")
                soft_labels[i, :] = default_confidence

        return soft_labels

    def get_teacher_predictions(
        self,
        filenames: List[str],
        file_bytes_list: Optional[List[Optional[bytes]]] = None,
    ) -> List[Dict[str, Any]]:
        """获取教师预测"""
        if file_bytes_list is None:
            file_bytes_list = [None] * len(filenames)
        if len(file_bytes_list) != len(filenames):
            raise ValueError(
                "file_bytes_list length must match filenames length "
                f"({len(file_bytes_list)} != {len(filenames)})"
            )
        predictions = []
        for filename, file_bytes in zip(filenames, file_bytes_list):
            try:
                if self.teacher_type == TeacherType.HYBRID:
                    result = self.classifier.classify(filename, file_bytes=file_bytes)
                    predictions.append(
                        {
                            "label": result.label,
                            "confidence": result.confidence,
                            "source": result.source.value,
                        }
                    )
                elif self.teacher_type == TeacherType.FILENAME:
                    result = self.classifier.predict(filename)
                    predictions.append(result)
                else:
                    if not file_bytes:
                        predictions.append(
                            {"label": None, "confidence": 0.0, "source": "titleblock"}
                        )
                        continue
                    from src.utils.dxf_io import read_dxf_entities_from_bytes

                    entities = read_dxf_entities_from_bytes(file_bytes)
                    result = self.classifier.predict(entities)
                    predictions.append(result)
            except Exception:
                predictions.append({"label": None, "confidence": 0.0})
        return predictions


class DistillationTrainer:
    """蒸馏训练器"""

    def __init__(
        self,
        student_model: nn.Module,
        teacher: TeacherModel,
        config: Optional[DistillationConfig] = None,
    ):
        self.student = student_model
        self.teacher = teacher
        self.config = config or DistillationConfig()
        self.loss_fn = DistillationLoss(
            alpha=self.config.alpha,
            temperature=self.config.temperature,
        )

        logger.info(
            "DistillationTrainer initialized",
            extra={
                "alpha": self.config.alpha,
                "temperature": self.config.temperature,
            },
        )

    def train_step(
        self,
        batch_data: Dict[str, Any],
        filenames: List[str],
        hard_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        单步训练

        Args:
            batch_data: 批次数据 (图数据)
            filenames: 文件名列表
            hard_labels: 真实标签
            optimizer: 优化器

        Returns:
            损失组件字典
        """
        self.student.train()
        optimizer.zero_grad()

        # 学生前向传播
        student_logits = self.student(batch_data)

        # 教师软标签
        teacher_logits = self.teacher.generate_soft_labels(filenames)
        teacher_logits = teacher_logits.to(student_logits.device)

        # 计算损失
        loss, components = self.loss_fn(student_logits, teacher_logits, hard_labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        return components

    def evaluate_without_teacher(
        self,
        dataloader: Any,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        评估学生模型 (不使用教师信号)

        用于验证蒸馏后模型在无文件名场景的表现
        """
        self.student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data, labels in dataloader:
                if isinstance(batch_data, dict):
                    batch_data = {k: v.to(device) for k, v in batch_data.items()}
                labels = labels.to(device)

                logits = self.student(batch_data)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }


def is_distillation_enabled() -> bool:
    """检查蒸馏训练是否启用"""
    return os.getenv("DISTILLATION_ENABLED", "false").lower() == "true"
