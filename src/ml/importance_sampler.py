"""
节点重要性采样模块

解决 DXF 图形中节点截断问题，通过重要性采样保留关键实体信息。

Feature Flags:
    DXF_MAX_NODES: 最大节点数 (default: 200)
    DXF_SAMPLING_STRATEGY: 采样策略 importance|random|hybrid (default: importance)
    DXF_SAMPLING_SEED: 随机种子，确保可重复性 (default: 42)
    DXF_TEXT_PRIORITY_RATIO: 文本实体优先占比上限 (default: 0.3)
    DXF_FRAME_PRIORITY_RATIO: 边框/标题栏等“框架实体”占比上限 (default: 1.0 = 不限制)
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SamplingStrategy(str, Enum):
    """采样策略"""
    IMPORTANCE = "importance"
    RANDOM = "random"
    HYBRID = "hybrid"


class EntityPriority(int, Enum):
    """实体优先级 (越高越重要)"""
    TEXT = 100           # 文本实体 (语义信息最丰富)
    DIMENSION = 90       # 尺寸标注
    TITLE_BLOCK = 85     # 标题栏区域实体
    BORDER = 80          # 边界实体
    CIRCLE = 70          # 圆 (孔、圆角特征)
    ARC = 65             # 圆弧
    LONG_LINE = 60       # 长线段 (轮廓线)
    INSERT = 55          # 块引用
    POLYLINE = 50        # 多段线
    SHORT_LINE = 40      # 短线段
    OTHER = 10           # 其他


@dataclass
class EntityInfo:
    """实体信息"""
    index: int
    entity: Any
    dtype: str
    priority: int
    length: float = 0.0
    center: Tuple[float, float] = (0.0, 0.0)
    is_border: bool = False
    is_title_block: bool = False
    is_text: bool = False
    text_content: str = ""


@dataclass
class SamplingResult:
    """采样结果"""
    sampled_entities: List[Any]
    original_count: int
    sampled_count: int
    strategy: str
    stats: Dict[str, int] = field(default_factory=dict)


class ImportanceSampler:
    """重要性采样器"""

    def __init__(
        self,
        max_nodes: int = 200,
        strategy: str = "importance",
        seed: int = 42,
        text_priority_ratio: float = 0.3,
        frame_priority_ratio: float = 1.0,
    ):
        """
        初始化采样器

        Args:
            max_nodes: 最大节点数
            strategy: 采样策略 (importance/random/hybrid)
            seed: 随机种子
            text_priority_ratio: 文本实体占比上限
        """
        self.max_nodes = int(os.getenv("DXF_MAX_NODES", str(max_nodes)))
        self.strategy = SamplingStrategy(os.getenv("DXF_SAMPLING_STRATEGY", strategy))
        self.seed = int(os.getenv("DXF_SAMPLING_SEED", str(seed)))
        self.text_priority_ratio = float(os.getenv("DXF_TEXT_PRIORITY_RATIO", str(text_priority_ratio)))
        self.text_priority_ratio = max(0.0, min(1.0, self.text_priority_ratio))
        self.frame_priority_ratio = float(
            os.getenv("DXF_FRAME_PRIORITY_RATIO", str(frame_priority_ratio))
        )
        # Keep ratios in [0,1] to avoid surprising sampling behavior.
        self.frame_priority_ratio = max(0.0, min(1.0, self.frame_priority_ratio))

        # 设置随机种子确保可重复性
        random.seed(self.seed)

        logger.info(
            "ImportanceSampler initialized",
            extra={
                "max_nodes": self.max_nodes,
                "strategy": self.strategy.value,
                "seed": self.seed,
                "text_ratio": self.text_priority_ratio,
                "frame_ratio": self.frame_priority_ratio,
            },
        )

    def _calculate_priority(
        self,
        entity: Any,
        dtype: str,
        length: float,
        max_dim: float,
        center: Tuple[float, float],
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[int, bool, bool]:
        """
        计算实体优先级

        Returns:
            (优先级, 是否边界实体, 是否标题栏实体)
        """
        min_x, min_y, max_x, max_y = bbox
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)
        tol = max_dim * 0.02

        is_border = False
        is_title_block = False

        # 检查边界实体
        if dtype in {"LINE", "LWPOLYLINE"} and length >= 0.8 * max_dim:
            cx, cy = center
            if (abs(cx - min_x) <= tol or abs(cx - max_x) <= tol or
                abs(cy - min_y) <= tol or abs(cy - max_y) <= tol):
                is_border = True

        # 检查标题栏区域 (右下角)
        cx, cy = center
        if cx >= min_x + 0.6 * width and cy <= min_y + 0.4 * height:
            is_title_block = True

        # 计算优先级
        if dtype in {"TEXT", "MTEXT"}:
            priority = EntityPriority.TEXT.value
        elif dtype == "DIMENSION":
            priority = EntityPriority.DIMENSION.value
        elif is_title_block:
            priority = EntityPriority.TITLE_BLOCK.value
        elif is_border:
            priority = EntityPriority.BORDER.value
        elif dtype == "CIRCLE":
            priority = EntityPriority.CIRCLE.value
        elif dtype == "ARC":
            priority = EntityPriority.ARC.value
        elif dtype == "INSERT":
            priority = EntityPriority.INSERT.value
        elif dtype == "LWPOLYLINE":
            priority = EntityPriority.POLYLINE.value
        elif dtype == "LINE":
            if length >= 0.5 * max_dim:
                priority = EntityPriority.LONG_LINE.value
            else:
                priority = EntityPriority.SHORT_LINE.value
        else:
            priority = EntityPriority.OTHER.value

        return priority, is_border, is_title_block

    def sample(self, entities: List[Any], bbox: Optional[Tuple[float, float, float, float]] = None) -> SamplingResult:
        """
        对实体列表进行重要性采样

        Args:
            entities: 实体列表
            bbox: 边界框 (min_x, min_y, max_x, max_y)

        Returns:
            SamplingResult
        """
        original_count = len(entities)

        if original_count <= self.max_nodes:
            # 无需采样
            return SamplingResult(
                sampled_entities=entities,
                original_count=original_count,
                sampled_count=original_count,
                strategy="no_sampling",
                stats={"reason": "under_limit"},
            )

        # 计算边界框
        if bbox is None:
            all_points = []
            for e in entities:
                try:
                    dtype = e.dxftype()
                    if dtype == "LINE":
                        all_points.append((float(e.dxf.start.x), float(e.dxf.start.y)))
                        all_points.append((float(e.dxf.end.x), float(e.dxf.end.y)))
                    elif dtype in {"CIRCLE", "ARC"}:
                        all_points.append((float(e.dxf.center.x), float(e.dxf.center.y)))
                except Exception:
                    pass

            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                bbox = (0, 0, 100, 100)

        min_x, min_y, max_x, max_y = bbox
        max_dim = max(max_x - min_x, max_y - min_y, 1.0)

        # 计算每个实体的优先级
        entity_infos: List[EntityInfo] = []

        for idx, e in enumerate(entities):
            try:
                dtype = e.dxftype()
                length = 0.0
                center = (0.0, 0.0)

                if dtype == "LINE":
                    start = e.dxf.start
                    end = e.dxf.end
                    length = float(start.distance(end))
                    mid = (start + end) / 2
                    center = (float(mid.x), float(mid.y))
                elif dtype in {"CIRCLE", "ARC"}:
                    c = e.dxf.center
                    center = (float(c.x), float(c.y))
                    length = float(e.dxf.radius) * 2

                priority, is_border, is_title_block = self._calculate_priority(
                    e, dtype, length, max_dim, center, bbox
                )

                entity_infos.append(EntityInfo(
                    index=idx,
                    entity=e,
                    dtype=dtype,
                    priority=priority,
                    length=length,
                    center=center,
                    is_border=is_border,
                    is_title_block=is_title_block,
                    is_text=dtype in {"TEXT", "MTEXT", "DIMENSION"},
                ))
            except Exception as e:
                logger.debug(f"Error processing entity {idx}: {e}")

        if not entity_infos:
            return SamplingResult(
                sampled_entities=[],
                original_count=original_count,
                sampled_count=0,
                strategy=self.strategy.value,
                stats={"error": "no_valid_entities"},
            )

        # 根据策略采样
        if self.strategy == SamplingStrategy.RANDOM:
            sampled = self._random_sample(entity_infos)
        elif self.strategy == SamplingStrategy.IMPORTANCE:
            sampled = self._importance_sample(entity_infos)
        else:  # HYBRID
            sampled = self._hybrid_sample(entity_infos)

        # 统计
        stats = {
            "text_count": sum(1 for e in sampled if e.is_text),
            "border_count": sum(1 for e in sampled if e.is_border),
            "title_block_count": sum(1 for e in sampled if e.is_title_block),
            "frame_count": sum(1 for e in sampled if (e.is_border or e.is_title_block)),
            "original_count": original_count,
        }

        return SamplingResult(
            sampled_entities=[e.entity for e in sampled],
            original_count=original_count,
            sampled_count=len(sampled),
            strategy=self.strategy.value,
            stats=stats,
        )

    def _random_sample(self, entity_infos: List[EntityInfo]) -> List[EntityInfo]:
        """随机采样"""
        random.seed(self.seed)  # 确保可重复
        return random.sample(entity_infos, min(self.max_nodes, len(entity_infos)))

    def _importance_sample(self, entity_infos: List[EntityInfo]) -> List[EntityInfo]:
        """重要性采样"""
        # 按优先级排序 (稳定排序)
        sorted_entities = sorted(entity_infos, key=lambda e: (-e.priority, e.index))

        # 限制文本实体占比
        max_text = int(self.max_nodes * self.text_priority_ratio)
        text_entities = [e for e in sorted_entities if e.is_text][:max_text]
        selected = {e.index for e in text_entities}

        # 限制“框架实体”占比：边框/标题栏的线段在多数图纸中高度相似，几何-only
        # 场景下容易导致采样结果过于一致，从而造成模型输出塌缩。
        max_frame = int(self.max_nodes * self.frame_priority_ratio)
        frame_entities = [
            e
            for e in sorted_entities
            if (not e.is_text)
            and (e.is_border or e.is_title_block)
            and e.index not in selected
        ][:max_frame]
        selected.update({e.index for e in frame_entities})

        # Prefer non-frame geometry entities to avoid collapsing on title block/border.
        other_entities = [
            e
            for e in sorted_entities
            if (not e.is_text)
            and (not (e.is_border or e.is_title_block))
            and e.index not in selected
        ]
        # If we still need to fill slots (rare edge cases), allow remaining frame
        # entities as a fallback so the graph size stays stable.
        frame_overflow_entities = [
            e
            for e in sorted_entities
            if (not e.is_text)
            and (e.is_border or e.is_title_block)
            and e.index not in selected
        ]

        # 组合
        sampled = list(text_entities) + list(frame_entities)
        remaining_slots = self.max_nodes - len(sampled)
        if remaining_slots > 0:
            sampled.extend(other_entities[:remaining_slots])
            remaining_slots = self.max_nodes - len(sampled)
        if remaining_slots > 0:
            sampled.extend(frame_overflow_entities[:remaining_slots])

        return sampled[:self.max_nodes]

    def _hybrid_sample(self, entity_infos: List[EntityInfo]) -> List[EntityInfo]:
        """混合采样 (50% 重要性 + 50% 随机)"""
        importance_count = self.max_nodes // 2
        random_count = self.max_nodes - importance_count

        # 重要性采样部分
        sorted_entities = sorted(entity_infos, key=lambda e: (-e.priority, e.index))
        importance_sampled = sorted_entities[:importance_count]

        # 随机采样部分 (从剩余实体中)
        remaining = [e for e in entity_infos if e not in importance_sampled]
        random.seed(self.seed)
        random_sampled = random.sample(remaining, min(random_count, len(remaining)))

        return importance_sampled + random_sampled


# 全局单例
_SAMPLER: Optional[ImportanceSampler] = None


def get_importance_sampler() -> ImportanceSampler:
    """获取全局 ImportanceSampler 实例"""
    global _SAMPLER
    if _SAMPLER is None:
        _SAMPLER = ImportanceSampler()
    return _SAMPLER


def reset_importance_sampler() -> None:
    """重置全局实例"""
    global _SAMPLER
    _SAMPLER = None
