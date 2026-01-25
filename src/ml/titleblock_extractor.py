"""
标题栏文本特征提取模块

从 DXF 图纸中提取标题栏区域的文本信息，作为分类的辅助特征。

Feature Flags:
    TITLEBLOCK_ENABLED: 是否启用标题栏特征 (default: false)
    TITLEBLOCK_REGION_X_RATIO: 标题栏 X 区域比例 (default: 0.6)
    TITLEBLOCK_REGION_Y_RATIO: 标题栏 Y 区域比例 (default: 0.4)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 常见标题栏关键词模式
DRAWING_NUMBER_PATTERNS = [
    r"图[号纸][:：]?\s*([A-Za-z0-9\-]+)",
    r"[Dd]rawing\s*[Nn]o\.?\s*[:：]?\s*([A-Za-z0-9\-]+)",
    r"([A-Z]{1,3}\d{6,12})",  # 如 J2925001, BTJ01239901522
]

PART_NAME_PATTERNS = [
    r"名称[:：]?\s*([\u4e00-\u9fa5]+)",
    r"零件[:：]?\s*([\u4e00-\u9fa5]+)",
    r"[Nn]ame[:：]?\s*([\u4e00-\u9fa5A-Za-z]+)",
]

MATERIAL_PATTERNS = [
    r"材[料质][:：]?\s*([\u4e00-\u9fa5A-Za-z0-9\-]+)",
    r"[Mm]aterial[:：]?\s*([\u4e00-\u9fa5A-Za-z0-9\-]+)",
]


@dataclass
class TitleBlockInfo:
    """标题栏信息"""
    drawing_number: Optional[str] = None
    part_name: Optional[str] = None
    material: Optional[str] = None
    scale: Optional[str] = None
    revision: Optional[str] = None
    raw_texts: List[str] = field(default_factory=list)
    region_entities_count: int = 0
    confidence: float = 0.0


class TitleBlockExtractor:
    """标题栏文本提取器"""

    def __init__(
        self,
        region_x_ratio: float = 0.6,
        region_y_ratio: float = 0.4,
    ):
        """
        初始化提取器

        Args:
            region_x_ratio: 标题栏 X 区域起始比例 (从右侧)
            region_y_ratio: 标题栏 Y 区域结束比例 (从底部)
        """
        self.region_x_ratio = float(os.getenv("TITLEBLOCK_REGION_X_RATIO", str(region_x_ratio)))
        self.region_y_ratio = float(os.getenv("TITLEBLOCK_REGION_Y_RATIO", str(region_y_ratio)))

        logger.info(
            "TitleBlockExtractor initialized",
            extra={
                "region_x_ratio": self.region_x_ratio,
                "region_y_ratio": self.region_y_ratio,
            },
        )

    def _is_in_title_block(
        self,
        x: float,
        y: float,
        bbox: Tuple[float, float, float, float],
    ) -> bool:
        """检查坐标是否在标题栏区域"""
        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y

        # 标题栏通常在右下角
        title_x_start = min_x + width * self.region_x_ratio
        title_y_end = min_y + height * self.region_y_ratio

        return x >= title_x_start and y <= title_y_end

    def extract_from_entities(
        self,
        entities: List[Any],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> TitleBlockInfo:
        """
        从实体列表提取标题栏信息

        Args:
            entities: DXF 实体列表
            bbox: 边界框

        Returns:
            TitleBlockInfo
        """
        result = TitleBlockInfo()

        if not entities:
            return result

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
                    elif dtype in {"TEXT", "MTEXT"}:
                        insert = getattr(e.dxf, "insert", None) or getattr(e.dxf, "location", None)
                        if insert:
                            all_points.append((float(insert.x), float(insert.y)))
                except Exception:
                    pass

            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                bbox = (0, 0, 100, 100)

        # 提取标题栏区域的文本
        title_texts: List[str] = []

        for e in entities:
            try:
                dtype = e.dxftype()
                if dtype not in {"TEXT", "MTEXT", "DIMENSION"}:
                    continue

                # 获取位置
                insert = getattr(e.dxf, "insert", None) or getattr(e.dxf, "location", None)
                if insert is None:
                    continue

                x, y = float(insert.x), float(insert.y)

                # 检查是否在标题栏区域
                if not self._is_in_title_block(x, y, bbox):
                    continue

                result.region_entities_count += 1

                # 提取文本内容
                if dtype == "TEXT":
                    text = str(getattr(e.dxf, "text", "") or "")
                elif dtype == "MTEXT":
                    text = str(getattr(e, "plain_text", lambda: "")() or "")
                    if not text:
                        text = str(getattr(e, "text", "") or "")
                elif dtype == "DIMENSION":
                    text = str(getattr(e.dxf, "text", "") or "")
                else:
                    text = ""

                text = text.strip()
                if text:
                    title_texts.append(text)

            except Exception as e:
                logger.debug(f"Error extracting text: {e}")

        result.raw_texts = title_texts

        # 解析文本内容
        combined_text = " ".join(title_texts)

        # 提取图号
        for pattern in DRAWING_NUMBER_PATTERNS:
            match = re.search(pattern, combined_text)
            if match:
                result.drawing_number = match.group(1)
                break

        # 提取零件名称
        for pattern in PART_NAME_PATTERNS:
            match = re.search(pattern, combined_text)
            if match:
                result.part_name = match.group(1)
                break

        # 提取材料
        for pattern in MATERIAL_PATTERNS:
            match = re.search(pattern, combined_text)
            if match:
                result.material = match.group(1)
                break

        # 计算置信度
        confidence = 0.0
        if result.drawing_number:
            confidence += 0.3
        if result.part_name:
            confidence += 0.4
        if result.material:
            confidence += 0.2
        if result.raw_texts:
            confidence += 0.1

        result.confidence = min(confidence, 1.0)

        return result

    def extract_from_msp(self, msp: Any) -> TitleBlockInfo:
        """从 modelspace 提取标题栏信息"""
        entities = list(msp)
        return self.extract_from_entities(entities)


class TitleBlockClassifier:
    """基于标题栏信息的分类器"""

    def __init__(self, synonyms: Optional[Dict[str, List[str]]] = None):
        self.extractor = TitleBlockExtractor()
        self.synonyms = synonyms or {}
        self._matcher: Dict[str, str] = {}

        if self.synonyms:
            self._build_matcher()

    def _build_matcher(self) -> None:
        """构建标签匹配器"""
        for label, aliases in self.synonyms.items():
            self._matcher[label.lower()] = label
            for alias in aliases:
                if alias:
                    self._matcher[alias.lower()] = label

    def predict(
        self,
        entities: List[Any],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
        """
        基于标题栏信息预测

        Returns:
            预测结果字典
        """
        info = self.extractor.extract_from_entities(entities, bbox)

        result = {
            "source": "titleblock",
            "label": None,
            "confidence": 0.0,
            "title_block_info": {
                "drawing_number": info.drawing_number,
                "part_name": info.part_name,
                "material": info.material,
                "raw_texts_count": len(info.raw_texts),
                "region_entities_count": info.region_entities_count,
            },
            "status": "no_match",
        }

        # 尝试从零件名称匹配标签
        if info.part_name and self._matcher:
            key = info.part_name.lower()
            if key in self._matcher:
                result["label"] = self._matcher[key]
                result["confidence"] = 0.85
                result["status"] = "matched"
            else:
                # 部分匹配
                for label_key, label in self._matcher.items():
                    if label_key in key or key in label_key:
                        result["label"] = label
                        result["confidence"] = 0.6
                        result["status"] = "partial_match"
                        break

        return result


# 全局单例
_EXTRACTOR: Optional[TitleBlockExtractor] = None


def get_titleblock_extractor() -> TitleBlockExtractor:
    """获取全局 TitleBlockExtractor 实例"""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = TitleBlockExtractor()
    return _EXTRACTOR


def is_titleblock_enabled() -> bool:
    """检查标题栏特征是否启用"""
    return os.getenv("TITLEBLOCK_ENABLED", "false").lower() == "true"
