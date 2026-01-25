"""
FilenameClassifier - 基于文件名的零件分类器

从文件名中提取零件名称，与同义词表匹配获得标准标签。
这是一个高效的弱监督信号，在文件名规范的场景下准确率可达 95%+。

Feature Flags:
    FILENAME_CLASSIFIER_ENABLED: 是否启用 (default: true)
    FILENAME_MIN_CONF: 最低置信度阈值 (default: 0.5)
    FILENAME_EXACT_MATCH_CONF: 精确匹配置信度 (default: 0.95)
    FILENAME_PARTIAL_MATCH_CONF: 部分匹配置信度 (default: 0.7)
    FILENAME_FUZZY_MATCH_CONF: 模糊匹配置信度 (default: 0.5)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 默认同义词表路径
DEFAULT_SYNONYMS_PATH = Path(__file__).resolve().parents[2] / "data/knowledge/label_synonyms_template.json"

# 版本后缀正则 (匹配 v1, v2, _v1, -v2, V1 等)
_VERSION_SUFFIX_RE = re.compile(r"(?:[_\-\s]?[vV]\d+)$")

# 比较文件前缀
_COMPARE_PREFIX_RE = re.compile(r"^比较[_\-\s]*")


class FilenameClassifier:
    """基于文件名的零件分类器"""

    def __init__(
        self,
        synonyms_path: Optional[str] = None,
        exact_match_conf: float = 0.95,
        partial_match_conf: float = 0.7,
        fuzzy_match_conf: float = 0.5,
    ):
        """
        初始化分类器

        Args:
            synonyms_path: 同义词表 JSON 路径
            exact_match_conf: 精确匹配置信度
            partial_match_conf: 部分匹配置信度
            fuzzy_match_conf: 模糊匹配置信度
        """
        self.exact_match_conf = float(os.getenv("FILENAME_EXACT_MATCH_CONF", str(exact_match_conf)))
        self.partial_match_conf = float(os.getenv("FILENAME_PARTIAL_MATCH_CONF", str(partial_match_conf)))
        self.fuzzy_match_conf = float(os.getenv("FILENAME_FUZZY_MATCH_CONF", str(fuzzy_match_conf)))

        # 加载同义词表
        path = Path(synonyms_path) if synonyms_path else DEFAULT_SYNONYMS_PATH
        self.synonyms: Dict[str, List[str]] = self._load_synonyms(path)
        self.matcher: Dict[str, str] = self._build_matcher()

        logger.info(
            "FilenameClassifier initialized",
            extra={
                "synonyms_path": str(path),
                "label_count": len(self.synonyms),
                "exact_conf": self.exact_match_conf,
                "partial_conf": self.partial_match_conf,
                "fuzzy_conf": self.fuzzy_match_conf,
            },
        )

    def _load_synonyms(self, path: Path) -> Dict[str, List[str]]:
        """加载同义词表"""
        if not path.exists():
            logger.warning(f"Synonyms file not found: {path}")
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {k: v for k, v in data.items() if isinstance(v, list)}
        except Exception as e:
            logger.error(f"Failed to load synonyms: {e}")
            return {}

    def _build_matcher(self) -> Dict[str, str]:
        """构建标签匹配器：零件名 -> 标准标签"""
        matcher: Dict[str, str] = {}
        for label, aliases in self.synonyms.items():
            # 标准标签本身
            matcher[label.lower()] = label
            # 所有别名
            for alias in aliases:
                if alias:
                    matcher[alias.lower()] = label
        return matcher

    def extract_part_name(self, filename: str) -> Optional[str]:
        """
        从文件名提取零件名称

        支持格式:
        - J2925001-01人孔v2.dxf -> 人孔
        - BTJ01239901522-00拖轮组件v1.dxf -> 拖轮组件
        - 比较_LTJ012306102-0084调节螺栓v1 vs xxx.dxf -> 调节螺栓

        Args:
            filename: 文件名

        Returns:
            提取的零件名称，或 None
        """
        if not filename:
            return None

        # 获取文件主名（不含扩展名）
        basename = Path(filename).stem

        # 处理比较文件格式
        if _COMPARE_PREFIX_RE.match(basename):
            basename = _COMPARE_PREFIX_RE.sub("", basename)
            # 取 "vs" 之前的部分
            if " vs " in basename.lower():
                basename = basename.lower().split(" vs ")[0].strip()
                # 重新提取
                basename = Path(basename).stem

        # 移除版本后缀
        basename = _VERSION_SUFFIX_RE.sub("", basename).strip()

        # 尝试多种模式提取中文部分
        patterns = [
            # 模式1: 编号-编号-零件名 (如 J0224025-06-01-03出料凸缘)
            r"[-\d]+([^\d\-][^v]*?)$",
            # 模式2: 编号零件名 (如 BTJ01239901522-00拖轮组件)
            r"\d+[-\d]*[A-Za-z]*[-\d]*([\u4e00-\u9fa5()（）]+.*?)$",
            # 模式3: 纯中文部分
            r"([\u4e00-\u9fa5()（）]+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, basename, re.IGNORECASE)
            if match:
                part_name = match.group(1).strip()
                # 清理可能的数字前缀
                part_name = re.sub(r"^[-\d]+", "", part_name).strip()
                if part_name and len(part_name) >= 2:
                    return part_name

        return None

    def match_label(self, part_name: str) -> Tuple[Optional[str], float, str]:
        """
        匹配零件名到标准标签

        Args:
            part_name: 零件名称

        Returns:
            (标签, 置信度, 匹配类型)
        """
        if not part_name:
            return None, 0.0, "no_input"

        key = part_name.lower().strip()

        # 精确匹配
        if key in self.matcher:
            return self.matcher[key], self.exact_match_conf, "exact"

        # 部分匹配（零件名包含标签或标签包含零件名）
        best_match: Optional[str] = None
        best_overlap = 0

        for label_key, label in self.matcher.items():
            if label_key in key:
                overlap = len(label_key)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = label
            elif key in label_key:
                overlap = len(key)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = label

        if best_match and best_overlap >= 2:
            # 根据重叠比例调整置信度
            ratio = best_overlap / max(len(key), 1)
            if ratio >= 0.8:
                return best_match, self.partial_match_conf, "partial_high"
            else:
                return best_match, self.fuzzy_match_conf, "partial_low"

        return None, 0.0, "no_match"

    def predict(self, filename: str) -> Dict[str, Any]:
        """
        预测文件的零件类型

        Args:
            filename: 文件名

        Returns:
            预测结果字典
        """
        result: Dict[str, Any] = {
            "source": "filename",
            "filename": filename,
            "extracted_name": None,
            "label": None,
            "confidence": 0.0,
            "match_type": "none",
            "status": "no_match",
        }

        # 提取零件名
        part_name = self.extract_part_name(filename)
        result["extracted_name"] = part_name

        if not part_name:
            result["status"] = "extraction_failed"
            return result

        # 匹配标签
        label, confidence, match_type = self.match_label(part_name)
        result["label"] = label
        result["confidence"] = confidence
        result["match_type"] = match_type

        if label:
            result["status"] = "matched"
        else:
            result["status"] = "no_match"

        return result

    def predict_batch(self, filenames: List[str]) -> List[Dict[str, Any]]:
        """批量预测"""
        return [self.predict(f) for f in filenames]


# 全局单例
_CLASSIFIER: Optional[FilenameClassifier] = None


def get_filename_classifier() -> FilenameClassifier:
    """获取全局 FilenameClassifier 实例"""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = FilenameClassifier()
    return _CLASSIFIER


def reset_filename_classifier() -> None:
    """重置全局实例（用于测试）"""
    global _CLASSIFIER
    _CLASSIFIER = None
