"""
类别不平衡治理模块

提供多种类别不平衡处理策略：
- Class Weights (类别权重)
- Focal Loss (焦点损失)
- Logit Adjustment (Logit 调整)

Feature Flags:
    CLASS_BALANCE_STRATEGY: 策略 none|weights|focal|logit_adj (default: focal)
    CLASS_WEIGHT_MODE: 权重模式 inverse|sqrt|log (default: sqrt)
    FOCAL_ALPHA: Focal Loss alpha (default: 0.25)
    FOCAL_GAMMA: Focal Loss gamma (default: 2.0)
    LOGIT_ADJ_TAU: Logit Adjustment tau (default: 1.0)
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BalanceStrategy(str, Enum):
    """类别平衡策略"""
    NONE = "none"
    WEIGHTS = "weights"
    FOCAL = "focal"
    LOGIT_ADJ = "logit_adj"


class WeightMode(str, Enum):
    """权重计算模式"""
    INVERSE = "inverse"      # 1 / count
    SQRT = "sqrt"            # sqrt(max_count / count)
    LOG = "log"              # log(total / count)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: Optional[float] = None,
        reduction: str = "mean",
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if alpha is None:
            alpha = float(os.getenv("FOCAL_ALPHA", "0.25"))
        if gamma is None:
            gamma = float(os.getenv("FOCAL_GAMMA", "2.0"))
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LogitAdjustedLoss(nn.Module):
    """
    Logit Adjusted Cross Entropy Loss

    对 logits 进行类别频率调整，缓解长尾分布问题
    """

    def __init__(
        self,
        class_counts: List[int],
        tau: Optional[float] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if tau is None:
            tau = float(os.getenv("LOGIT_ADJ_TAU", "1.0"))
        self.tau = float(tau)
        self.reduction = reduction

        # 计算类别先验
        total = sum(class_counts)
        priors = [c / total for c in class_counts]
        self.register_buffer(
            "log_priors",
            torch.tensor([math.log(p + 1e-9) for p in priors])
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        # 调整 logits
        adjusted_logits = inputs + self.tau * self.log_priors.to(inputs.device)
        return F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)


class ClassBalancer:
    """类别平衡器"""

    def __init__(
        self,
        strategy: str = "focal",
        weight_mode: str = "sqrt",
        focal_alpha: Optional[float] = None,
        focal_gamma: Optional[float] = None,
        logit_adj_tau: Optional[float] = None,
    ):
        self.strategy = BalanceStrategy(os.getenv("CLASS_BALANCE_STRATEGY", strategy))
        self.weight_mode = WeightMode(os.getenv("CLASS_WEIGHT_MODE", weight_mode))
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.logit_adj_tau = logit_adj_tau

        logger.info(
            "ClassBalancer initialized",
            extra={
                "strategy": self.strategy.value,
                "weight_mode": self.weight_mode.value,
                "focal_alpha": self.focal_alpha,
                "focal_gamma": self.focal_gamma,
                "logit_adj_tau": self.logit_adj_tau,
            },
        )

    def compute_class_weights(
        self,
        labels: List[int],
        num_classes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        计算类别权重

        Args:
            labels: 标签列表
            num_classes: 类别数

        Returns:
            类别权重张量
        """
        counts = Counter(labels)
        if num_classes is None:
            num_classes = max(counts.keys()) + 1

        max_count = max(counts.values())
        total = len(labels)

        weights = []
        for i in range(num_classes):
            count = counts.get(i, 1)  # 避免除零

            if self.weight_mode == WeightMode.INVERSE:
                w = 1.0 / count
            elif self.weight_mode == WeightMode.SQRT:
                w = math.sqrt(max_count / count)
            elif self.weight_mode == WeightMode.LOG:
                w = math.log(total / count + 1)
            else:
                w = 1.0

            weights.append(w)

        # 归一化
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        weights_tensor = weights_tensor / weights_tensor.sum() * num_classes

        return weights_tensor

    def get_loss_function(
        self,
        labels: Optional[List[int]] = None,
        num_classes: Optional[int] = None,
        class_counts: Optional[List[int]] = None,
    ) -> nn.Module:
        """
        获取损失函数

        Args:
            labels: 训练标签（用于计算权重）
            num_classes: 类别数
            class_counts: 各类别样本数

        Returns:
            损失函数模块
        """
        if self.strategy == BalanceStrategy.NONE:
            return nn.CrossEntropyLoss()

        elif self.strategy == BalanceStrategy.WEIGHTS:
            if labels is None:
                return nn.CrossEntropyLoss()
            weights = self.compute_class_weights(labels, num_classes)
            return nn.CrossEntropyLoss(weight=weights)

        elif self.strategy == BalanceStrategy.FOCAL:
            weights = None
            if labels is not None:
                weights = self.compute_class_weights(labels, num_classes)
            return FocalLoss(
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                class_weights=weights,
            )

        elif self.strategy == BalanceStrategy.LOGIT_ADJ:
            if class_counts is None and labels is not None:
                counts = Counter(labels)
                num_classes = num_classes or (max(counts.keys()) + 1)
                class_counts = [counts.get(i, 1) for i in range(num_classes)]
            if class_counts is None:
                return nn.CrossEntropyLoss()
            return LogitAdjustedLoss(class_counts, tau=self.logit_adj_tau)

        return nn.CrossEntropyLoss()

    def get_class_distribution(self, labels: List[int]) -> Dict[str, Any]:
        """获取类别分布统计"""
        counts = Counter(labels)
        total = len(labels)

        stats = {
            "total": total,
            "num_classes": len(counts),
            "class_counts": dict(counts),
            "min_count": min(counts.values()),
            "max_count": max(counts.values()),
            "imbalance_ratio": max(counts.values()) / max(min(counts.values()), 1),
        }

        return stats


# 全局单例
_BALANCER: Optional[ClassBalancer] = None


def get_class_balancer() -> ClassBalancer:
    """获取全局 ClassBalancer 实例"""
    global _BALANCER
    if _BALANCER is None:
        _BALANCER = ClassBalancer()
    return _BALANCER


def reset_class_balancer() -> None:
    """重置全局实例"""
    global _BALANCER
    _BALANCER = None
