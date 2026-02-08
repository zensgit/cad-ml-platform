"""
CAD部件分类推理服务

提供DXF/DWG图纸的部件类型识别功能
支持V2 (28维/7类), V6 (48维/5类), V16 (超级集成/5类, 99.88%准确率) 模型
"""

import io
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.dxf_features import extract_features_v6

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """分类结果"""
    category: str  # 预测类别
    confidence: float  # 置信度
    probabilities: Dict[str, float]  # 各类别概率
    features: Optional[Dict[str, float]] = None  # 提取的特征
    model_version: str = "v2"  # 模型版本
    needs_review: bool = False  # 是否需要人工审核
    review_reason: Optional[str] = None  # 审核原因
    top2_category: Optional[str] = None  # 第二预测类别
    top2_confidence: Optional[float] = None  # 第二预测置信度


class PartClassifier:
    """部件分类器 - 支持多版本模型"""

    def __init__(self, model_path: str = "models/cad_classifier_v6.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.id_to_label = None
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.version = "v2"  # 默认版本
        self._load_model()

    def _load_model(self):
        """加载模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        input_dim = checkpoint["input_dim"]
        hidden_dim = checkpoint["hidden_dim"]
        num_classes = checkpoint["num_classes"]
        self.version = checkpoint.get("version")
        if not self.version:
            self.version = self._infer_version(input_dim, num_classes)

        # 根据版本选择模型架构
        if self.version == "v8":
            self.model = self._build_v8_model(input_dim, hidden_dim, num_classes)
        elif self.version in ("v6", "v7"):
            self.model = self._build_v6_model(input_dim, hidden_dim, num_classes)
        else:
            self.model = self._build_v2_model(input_dim, hidden_dim, num_classes)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.id_to_label = {int(k): v for k, v in checkpoint["id_to_label"].items()}
        self.num_classes = num_classes
        self.input_dim = input_dim

        logger.info(f"模型加载成功 (版本: {self.version})，类别: {list(self.id_to_label.values())}")

    def _infer_version(self, input_dim: int, num_classes: int) -> str:
        """基于维度推断模型版本，避免缺少version字段时加载错误架构。"""
        if input_dim >= 48 and num_classes <= 10:
            return "v6"
        return "v2"

    def _build_v2_model(self, input_dim, hidden_dim, num_classes):
        """V2模型架构 (28维特征)"""
        class ImprovedClassifier(nn.Module):
            def __init__(self, in_dim, hid_dim, n_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(hid_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(hid_dim, hid_dim // 2),
                    nn.BatchNorm1d(hid_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hid_dim // 2, n_classes)
                )

            def forward(self, x):
                return self.net(x)

        return ImprovedClassifier(input_dim, hidden_dim, num_classes)

    def _build_v6_model(self, input_dim, hidden_dim, num_classes):
        """V6模型架构 (48维特征)"""
        class ImprovedClassifierV6(nn.Module):
            def __init__(self, in_dim, hid_dim, n_classes, dropout=0.5):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),

                    nn.Linear(hid_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),

                    nn.Linear(hid_dim, hid_dim // 2),
                    nn.BatchNorm1d(hid_dim // 2),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout * 0.6),

                    nn.Linear(hid_dim // 2, hid_dim // 4),
                    nn.BatchNorm1d(hid_dim // 4),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout * 0.4),

                    nn.Linear(hid_dim // 4, n_classes)
                )

            def forward(self, x):
                return self.net(x)

        return ImprovedClassifierV6(input_dim, hidden_dim, num_classes)

    def _build_v8_model(self, input_dim, hidden_dim, num_classes):
        """V8模型架构 (48维特征, 与V6/V7相同)"""
        # V8使用与V6/V7相同的架构
        return self._build_v6_model(input_dim, hidden_dim, num_classes)

    def extract_features(self, dxf_path: str) -> Optional[np.ndarray]:
        """从DXF文件提取特征 - 根据模型版本选择特征维度"""
        version = getattr(self, "version", "v2")
        if version in ("v6", "v7", "v8"):
            return self._extract_features_v6(dxf_path)
        else:
            return self._extract_features_v2(dxf_path)

    def _extract_features_v2(self, dxf_path: str) -> Optional[np.ndarray]:
        """V2特征提取 (28维)"""
        try:
            import ezdxf

            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            entity_types = []
            layer_names = []
            all_points = []
            circle_radii = []
            line_lengths = []

            for entity in msp:
                etype = entity.dxftype()
                entity_types.append(etype)

                if hasattr(entity.dxf, 'layer'):
                    layer_names.append(entity.dxf.layer)

                try:
                    if etype == "LINE":
                        start = (entity.dxf.start.x, entity.dxf.start.y)
                        end = (entity.dxf.end.x, entity.dxf.end.y)
                        all_points.extend([start, end])
                        length = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                        line_lengths.append(length)
                    elif etype == "CIRCLE":
                        center = (entity.dxf.center.x, entity.dxf.center.y)
                        all_points.append(center)
                        circle_radii.append(entity.dxf.radius)
                    elif etype == "ARC":
                        center = (entity.dxf.center.x, entity.dxf.center.y)
                        all_points.append(center)
                        circle_radii.append(entity.dxf.radius)
                    elif etype in ["TEXT", "MTEXT"]:
                        if hasattr(entity.dxf, 'insert'):
                            all_points.append((entity.dxf.insert.x, entity.dxf.insert.y))
                    elif etype in ["LWPOLYLINE", "POLYLINE"]:
                        if hasattr(entity, 'get_points'):
                            for pt in entity.get_points():
                                all_points.append((pt[0], pt[1]))
                    elif etype == "INSERT":
                        if hasattr(entity.dxf, 'insert'):
                            all_points.append((entity.dxf.insert.x, entity.dxf.insert.y))
                except Exception:
                    pass

            type_counts = Counter(entity_types)
            total_entities = len(entity_types)

            features = []

            # 实体类型比例 (12)
            for etype in ["LINE", "CIRCLE", "ARC", "LWPOLYLINE", "POLYLINE",
                          "SPLINE", "ELLIPSE", "TEXT", "MTEXT", "DIMENSION",
                          "HATCH", "INSERT"]:
                features.append(type_counts.get(etype, 0) / max(total_entities, 1))

            # 几何统计 (4)
            features.append(np.log1p(total_entities) / 10)
            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                features.extend([np.log1p(width) / 10, np.log1p(height) / 10,
                               np.clip(width / max(height, 0.001), 0, 10) / 10])
            else:
                features.extend([0, 0, 0.5])

            # 圆/弧 (2)
            if circle_radii:
                features.extend([np.log1p(np.mean(circle_radii)) / 5,
                               np.log1p(np.std(circle_radii)) / 5 if len(circle_radii) > 1 else 0])
            else:
                features.extend([0, 0])

            # 线段 (2)
            if line_lengths:
                features.extend([np.log1p(np.mean(line_lengths)) / 5,
                               np.log1p(np.std(line_lengths)) / 5 if len(line_lengths) > 1 else 0])
            else:
                features.extend([0, 0])

            # 图层 (4)
            unique_layers = len(set(layer_names))
            features.append(np.log1p(unique_layers) / 3)
            layer_lower = [l.lower() for l in layer_names]
            features.append(1.0 if any('dim' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('text' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('center' in l for l in layer_lower) else 0.0)

            # 复杂度 (4)
            features.append((type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0)) / max(total_entities, 1))
            features.append((type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0)) / max(total_entities, 1))
            features.append(type_counts.get("INSERT", 0) / max(total_entities, 1))
            features.append(type_counts.get("DIMENSION", 0) / max(total_entities, 1))

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None

    def _extract_features_v6(self, dxf_path: str) -> Optional[np.ndarray]:
        """V6特征提取 (48维)"""
        return extract_features_v6(dxf_path, log=logger)

    def predict(self, dxf_path: str, confidence_threshold: float = 0.6) -> Optional[ClassificationResult]:
        """预测DXF文件的部件类别

        Args:
            dxf_path: DXF文件路径
            confidence_threshold: 置信度阈值，低于此值标记为需要人工审核
        """
        features = self.extract_features(dxf_path)
        if features is None:
            return None

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)[0]

            # 获取top2预测
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            pred_id = sorted_indices[0].item()
            confidence = sorted_probs[0].item()
            top2_id = sorted_indices[1].item()
            top2_confidence = sorted_probs[1].item()

        probabilities = {
            self.id_to_label[i]: probs[i].item()
            for i in range(self.num_classes)
        }

        # 检查是否需要人工审核
        needs_review = False
        review_reason = None
        margin = confidence - top2_confidence

        if confidence < confidence_threshold:
            needs_review = True
            review_reason = f"置信度({confidence:.1%})低于阈值({confidence_threshold:.0%})"
        elif margin < 0.1:  # top1和top2差距小于10%
            needs_review = True
            review_reason = f"预测不确定(差距仅{margin:.1%})"

        return ClassificationResult(
            category=self.id_to_label[pred_id],
            confidence=confidence,
            probabilities=probabilities,
            model_version=self.version,
            needs_review=needs_review,
            review_reason=review_reason,
            top2_category=self.id_to_label[top2_id],
            top2_confidence=top2_confidence,
        )

    def predict_batch(self, dxf_paths: List[str]) -> List[Optional[ClassificationResult]]:
        """批量预测"""
        return [self.predict(p) for p in dxf_paths]


# 全局实例
_classifier: Optional[PartClassifier] = None


def get_part_classifier() -> PartClassifier:
    """获取分类器单例"""
    global _classifier
    if _classifier is None:
        configured_model = os.getenv("CAD_CLASSIFIER_MODEL")
        if configured_model:
            model_path = configured_model
        elif Path("models/cad_classifier_v2.pt").exists():
            # Backward-compatible convenience default used by unit tests.
            model_path = "models/cad_classifier_v2.pt"
        else:
            model_path = "models/cad_classifier_v6.pt"
        _classifier = PartClassifier(model_path)
    return _classifier


def classify_part(dxf_path: str) -> Optional[ClassificationResult]:
    """便捷函数：分类单个文件"""
    return get_part_classifier().predict(dxf_path)


# ============== V16 超级集成分类器 ==============

class _DeepGeoBranch(nn.Module):
    """V14几何分支"""
    def __init__(self, geo_dim: int = 48, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(geo_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.net(x)


class _MultiScaleVisualBranch(nn.Module):
    """V14视觉分支"""
    def __init__(self):
        super().__init__()
        self.shallow = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.deep = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(32 + 64 + 128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        shallow = self.shallow(x)
        mid = self.mid(x)
        deep = self.deep(x)
        combined = torch.cat([shallow, mid, deep], dim=1)
        return self.fuse(combined)


class _FusionModelV14(nn.Module):
    """V14融合模型"""
    def __init__(self, geo_dim: int = 48, num_classes: int = 5):
        super().__init__()
        self.geo_branch = _DeepGeoBranch(geo_dim, 256)
        self.visual_branch = _MultiScaleVisualBranch()
        self.geo_weight = nn.Parameter(torch.tensor(0.7))
        self.visual_weight = nn.Parameter(torch.tensor(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.geo_classifier = nn.Linear(128, num_classes)

    def forward(self, img, geo):
        geo_feat = self.geo_branch(geo)
        visual_feat = self.visual_branch(img)
        geo_w = torch.sigmoid(self.geo_weight)
        visual_w = torch.sigmoid(self.visual_weight)
        total = geo_w + visual_w
        geo_w, visual_w = geo_w / total, visual_w / total
        fused = torch.cat([geo_feat * geo_w, visual_feat * visual_w], dim=1)
        return self.classifier(fused)


class PartClassifierV16:
    """V16超级集成分类器 (99.65%准确率)

    结合V6纯几何模型和V14视觉+几何融合集成模型

    特性:
    - 置信度阈值机制：低于阈值时标记为"需人工审核"
    - Top-2预测：返回前两个最可能的类别供参考
    - 边界案例识别：特定零件类型可能存在分类歧义
    - 快速模式：可选择性跳过V14视觉分支以提高推理速度
    - 特征缓存：缓存已提取的特征和渲染图像，重复分类时跳过I/O
    """

    # 类别映射 - 必须与训练时的labels.json一致
    # label_to_id: 传动件=0, 其他=1, 壳体类=2, 轴类=3, 连接件=4
    CATEGORIES = ["传动件", "其他", "壳体类", "轴类", "连接件"]
    IMG_SIZE = 128

    # 置信度阈值配置
    CONFIDENCE_THRESHOLD = 0.85  # 低于此值需人工审核
    MARGIN_THRESHOLD = 0.15  # top1和top2差距小于此值时需审核

    # 已知边界案例（这些类别组合容易混淆）
    KNOWN_AMBIGUOUS_PAIRS = [
        ("连接件", "传动件"),  # 如卡簧、挡圈
        ("壳体类", "其他"),  # 如端盖、法兰
    ]

    # 速度模式配置 (优化后延迟，GPU FP16)
    SPEED_MODES = {
        "accurate": {"v14_folds": 5, "use_fast_render": False},  # 完整精度 ~120ms (优化前~230ms)
        "balanced": {"v14_folds": 3, "use_fast_render": True},   # 平衡模式 ~85ms (优化前~150ms)
        "fast": {"v14_folds": 1, "use_fast_render": True},       # 快速模式 ~60ms (优化前~100ms)
        "v6_only": {"v14_folds": 0, "use_fast_render": False},   # 仅V6 ~45ms (优化前~50ms, 99.34%准确率)
    }

    # 缓存配置
    DEFAULT_CACHE_SIZE = 1000  # 默认缓存条目数

    def __init__(self, model_dir: str = "models", confidence_threshold: float = None,
                 speed_mode: str = "accurate", enable_cache: bool = True,
                 cache_size: int = None, use_fp16: bool = None):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        self.v6_model = None
        self.v14_models = []
        self.v6_mean = None
        self.v6_std = None
        self.v6_weight = 0.3  # 更新为优化后的权重
        self.v14_weight = 0.7
        self.loaded = False
        # 允许自定义置信度阈值
        self.confidence_threshold = confidence_threshold or self.CONFIDENCE_THRESHOLD
        # 速度模式
        if speed_mode not in self.SPEED_MODES:
            raise ValueError(f"无效速度模式: {speed_mode}，可选: {list(self.SPEED_MODES.keys())}")
        self.speed_mode = speed_mode
        self._speed_config = self.SPEED_MODES[speed_mode]

        # FP16半精度推理 - GPU上自动启用，CPU不支持
        if use_fp16 is None:
            # 自动检测: CUDA支持FP16，MPS部分支持，CPU不支持
            self._use_fp16 = self.device.type == "cuda"
        else:
            self._use_fp16 = use_fp16 and self.device.type in ("cuda", "mps")
        self._dtype = torch.float16 if self._use_fp16 else torch.float32

        # 特征缓存 - 使用LRU策略
        self._enable_cache = enable_cache
        self._cache_size = cache_size or self.DEFAULT_CACHE_SIZE
        self._feature_cache: Dict[str, np.ndarray] = {}  # file_hash -> features
        self._image_cache: Dict[str, np.ndarray] = {}    # file_hash -> rendered image
        self._tensor_cache: Dict[str, torch.Tensor] = {}  # file_hash -> 预转换的tensor (GPU)
        self._cache_order: List[str] = []  # LRU顺序追踪
        self._cache_stats = {"hits": 0, "misses": 0}

    def _get_file_cache_key(self, file_path: str) -> str:
        """生成文件缓存键（基于路径和修改时间）"""
        import hashlib
        path = Path(file_path)
        try:
            mtime = path.stat().st_mtime
            key_str = f"{path.resolve()}:{mtime}"
            return hashlib.md5(key_str.encode()).hexdigest()[:16]
        except OSError:
            return hashlib.md5(str(path.resolve()).encode()).hexdigest()[:16]

    def _cache_get(self, cache_key: str) -> tuple:
        """从缓存获取特征和图像，返回(features, image)或(None, None)"""
        if not self._enable_cache or cache_key not in self._feature_cache:
            self._cache_stats["misses"] += 1
            return None, None

        self._cache_stats["hits"] += 1
        # 更新LRU顺序
        if cache_key in self._cache_order:
            self._cache_order.remove(cache_key)
        self._cache_order.append(cache_key)

        features = self._feature_cache.get(cache_key)
        image = self._image_cache.get(cache_key)
        return features, image

    def _cache_put(self, cache_key: str, features: np.ndarray, image: np.ndarray = None):
        """存入缓存，自动LRU淘汰"""
        if not self._enable_cache:
            return

        # LRU淘汰
        while len(self._cache_order) >= self._cache_size:
            old_key = self._cache_order.pop(0)
            self._feature_cache.pop(old_key, None)
            self._image_cache.pop(old_key, None)
            self._tensor_cache.pop(old_key, None)

        self._feature_cache[cache_key] = features
        if image is not None:
            self._image_cache[cache_key] = image
        self._cache_order.append(cache_key)

    def clear_cache(self):
        """清空缓存"""
        self._feature_cache.clear()
        self._image_cache.clear()
        self._tensor_cache.clear()
        self._cache_order.clear()
        self._cache_stats = {"hits": 0, "misses": 0}

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total if total > 0 else 0
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_rate": hit_rate,
            "cache_size": len(self._feature_cache),
            "max_size": self._cache_size,
        }

    @property
    def use_fp16(self) -> bool:
        """是否启用FP16半精度推理"""
        return self._use_fp16

    @property
    def dtype_str(self) -> str:
        """当前使用的数据类型"""
        return "fp16" if self._use_fp16 else "fp32"

    def _load_models(self):
        """加载V6和V14模型"""
        if self.loaded:
            return

        # 加载配置
        config_path = self.model_dir / "cad_classifier_v16_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.v6_weight = config['components']['v6']['weight']
            self.v14_weight = config['components']['v14_ensemble']['weight']

        # 加载V6
        v6_path = self.model_dir / "cad_classifier_v6.pt"
        if not v6_path.exists():
            raise FileNotFoundError(f"V6模型不存在: {v6_path}")

        v6_ckpt = torch.load(v6_path, map_location=self.device, weights_only=False)

        class ImprovedClassifierV6(nn.Module):
            def __init__(self, in_dim, hid_dim, n_classes, dropout=0.5):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),
                    nn.Linear(hid_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),
                    nn.Linear(hid_dim, hid_dim // 2),
                    nn.BatchNorm1d(hid_dim // 2),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout * 0.6),
                    nn.Linear(hid_dim // 2, hid_dim // 4),
                    nn.BatchNorm1d(hid_dim // 4),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout * 0.4),
                    nn.Linear(hid_dim // 4, n_classes)
                )
            def forward(self, x):
                return self.net(x)

        self.v6_model = ImprovedClassifierV6(48, 256, 5, 0.5)
        self.v6_model.load_state_dict(v6_ckpt['model_state_dict'])
        self.v6_model.to(self.device)
        self.v6_model.eval()
        # FP16转换
        if self._use_fp16:
            self.v6_model = self.v6_model.half()
        self.v6_mean = v6_ckpt.get('feature_mean', np.zeros(48))
        self.v6_std = v6_ckpt.get('feature_std', np.ones(48))

        # 加载V14集成
        v14_path = self.model_dir / "cad_classifier_v14_ensemble.pt"
        if v14_path.exists():
            v14_ckpt = torch.load(v14_path, map_location=self.device, weights_only=False)
            for fold_state in v14_ckpt['fold_states']:
                model = _FusionModelV14(48, 5)
                model.load_state_dict(fold_state)
                model.to(self.device)
                model.eval()
                # FP16转换
                if self._use_fp16:
                    model = model.half()
                self.v14_models.append(model)
            logger.info(f"V14集成模型加载完成，{len(self.v14_models)}个折叠")
        else:
            logger.warning(f"V14模型不存在: {v14_path}，将仅使用V6")

        self.loaded = True
        fp16_str = ", FP16启用" if self._use_fp16 else ""
        logger.info(f"V16模型加载完成，设备: {self.device}{fp16_str}")

    def _extract_features(self, dxf_path: str, ezdxf_doc=None) -> Optional[np.ndarray]:
        """提取48维几何特征（与V6训练时一致）"""
        return extract_features_v6(dxf_path, log=logger, ezdxf_doc=ezdxf_doc)

    def _check_needs_review(self, top1_cat: str, top1_conf: float,
                            top2_cat: str, top2_conf: float) -> tuple:
        """检查是否需要人工审核

        Returns:
            (needs_review: bool, reason: str or None)
        """
        reasons = []

        # 检查1: 置信度低于阈值
        if top1_conf < self.confidence_threshold:
            reasons.append(f"置信度({top1_conf:.1%})低于阈值({self.confidence_threshold:.0%})")

        # 检查2: top1和top2差距过小
        margin = top1_conf - top2_conf
        if margin < self.MARGIN_THRESHOLD:
            reasons.append(f"预测不确定(差距仅{margin:.1%})")

        # 检查3: 是否属于已知边界案例
        pair = tuple(sorted([top1_cat, top2_cat]))
        for ambiguous_pair in self.KNOWN_AMBIGUOUS_PAIRS:
            if pair == tuple(sorted(ambiguous_pair)):
                reasons.append(f"已知边界案例({top1_cat}/{top2_cat})")
                break

        if reasons:
            return True, "; ".join(reasons)
        return False, None

    def _render_dxf(self, dxf_path: str, ezdxf_doc=None) -> Optional[np.ndarray]:
        """渲染DXF为灰度图 (根据速度配置选择渲染方式)"""
        if self._speed_config["use_fast_render"]:
            return self._render_dxf_fast(dxf_path, ezdxf_doc)
        # accurate模式尝试matplotlib，失败则回退到PIL
        result = self._render_dxf_matplotlib(dxf_path)
        if result is None:
            return self._render_dxf_fast(dxf_path, ezdxf_doc)
        return result

    def _render_dxf_fast(self, dxf_path: str, ezdxf_doc=None) -> Optional[np.ndarray]:
        """快速PIL渲染 (~35ms vs matplotlib的~55ms)

        Args:
            dxf_path: DXF文件路径
            ezdxf_doc: 可选的预加载ezdxf文档对象，避免重复读取
        """
        try:
            import ezdxf
            from PIL import Image, ImageDraw

            if ezdxf_doc is not None:
                doc = ezdxf_doc
            else:
                doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            all_x, all_y = [], []
            entities_data = []

            for entity in msp:
                try:
                    etype = entity.dxftype()
                    if etype == "LINE":
                        x1, y1 = entity.dxf.start.x, entity.dxf.start.y
                        x2, y2 = entity.dxf.end.x, entity.dxf.end.y
                        entities_data.append(("LINE", (x1, y1, x2, y2)))
                        all_x.extend([x1, x2])
                        all_y.extend([y1, y2])
                    elif etype == "CIRCLE":
                        cx, cy = entity.dxf.center.x, entity.dxf.center.y
                        r = entity.dxf.radius
                        entities_data.append(("CIRCLE", (cx, cy, r)))
                        all_x.extend([cx - r, cx + r])
                        all_y.extend([cy - r, cy + r])
                    elif etype == "ARC":
                        cx, cy = entity.dxf.center.x, entity.dxf.center.y
                        r = entity.dxf.radius
                        start_angle = entity.dxf.start_angle
                        end_angle = entity.dxf.end_angle
                        entities_data.append(("ARC", (cx, cy, r, start_angle, end_angle)))
                        all_x.extend([cx - r, cx + r])
                        all_y.extend([cy - r, cy + r])
                    elif etype in ["LWPOLYLINE", "POLYLINE"]:
                        if hasattr(entity, 'get_points'):
                            pts = list(entity.get_points())
                            if len(pts) >= 2:
                                entities_data.append(("POLYLINE", pts))
                                for p in pts:
                                    all_x.append(p[0])
                                    all_y.append(p[1])
                except Exception:
                    pass

            if not all_x or not all_y:
                return None

            # 计算边界和缩放
            margin = 0.1
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_range = max(x_max - x_min, 1e-6)
            y_range = max(y_max - y_min, 1e-6)

            # 扩展边界
            x_min -= margin * x_range
            x_max += margin * x_range
            y_min -= margin * y_range
            y_max += margin * y_range
            x_range = x_max - x_min
            y_range = y_max - y_min

            # 保持宽高比
            if x_range > y_range:
                scale = self.IMG_SIZE / x_range
                offset_y = (self.IMG_SIZE - y_range * scale) / 2
                offset_x = 0
            else:
                scale = self.IMG_SIZE / y_range
                offset_x = (self.IMG_SIZE - x_range * scale) / 2
                offset_y = 0

            def transform(x, y):
                px = (x - x_min) * scale + offset_x
                py = self.IMG_SIZE - ((y - y_min) * scale + offset_y)  # 翻转Y轴
                return px, py

            # 创建图像
            img = Image.new('L', (self.IMG_SIZE, self.IMG_SIZE), color=255)
            draw = ImageDraw.Draw(img)

            for etype, data in entities_data:
                if etype == "LINE":
                    x1, y1, x2, y2 = data
                    p1 = transform(x1, y1)
                    p2 = transform(x2, y2)
                    draw.line([p1, p2], fill=0, width=1)
                elif etype == "CIRCLE":
                    cx, cy, r = data
                    # PIL椭圆需要bbox
                    p1 = transform(cx - r, cy + r)
                    p2 = transform(cx + r, cy - r)
                    draw.ellipse([p1, p2], outline=0, width=1)
                elif etype == "ARC":
                    cx, cy, r, start, end = data
                    p1 = transform(cx - r, cy + r)
                    p2 = transform(cx + r, cy - r)
                    # PIL arc 角度是逆时针从3点钟方向
                    draw.arc([p1, p2], start=-end, end=-start, fill=0, width=1)
                elif etype == "POLYLINE":
                    pts = [transform(p[0], p[1]) for p in data]
                    if len(pts) >= 2:
                        draw.line(pts, fill=0, width=1)

            return np.array(img, dtype=np.float32) / 255.0

        except Exception as e:
            logger.error(f"快速渲染失败: {e}")
            return None

    def _render_dxf_matplotlib(self, dxf_path: str) -> Optional[np.ndarray]:
        """渲染DXF为灰度图"""
        try:
            import ezdxf
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            fig = plt.figure(figsize=(self.IMG_SIZE/100, self.IMG_SIZE/100), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_aspect('equal')

            all_x, all_y = [], []
            for entity in msp:
                try:
                    etype = entity.dxftype()
                    if etype == "LINE":
                        x1, y1 = entity.dxf.start.x, entity.dxf.start.y
                        x2, y2 = entity.dxf.end.x, entity.dxf.end.y
                        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.8)
                        all_x.extend([x1, x2])
                        all_y.extend([y1, y2])
                    elif etype == "CIRCLE":
                        cx, cy = entity.dxf.center.x, entity.dxf.center.y
                        r = entity.dxf.radius
                        circle = plt.Circle((cx, cy), r, fill=False, color='k', linewidth=0.8)
                        ax.add_patch(circle)
                        all_x.extend([cx-r, cx+r])
                        all_y.extend([cy-r, cy+r])
                    elif etype == "ARC":
                        from matplotlib.patches import Arc
                        cx, cy = entity.dxf.center.x, entity.dxf.center.y
                        r = entity.dxf.radius
                        arc = Arc((cx, cy), 2*r, 2*r, angle=0, theta1=entity.dxf.start_angle,
                                 theta2=entity.dxf.end_angle, color='k', linewidth=0.8)
                        ax.add_patch(arc)
                        all_x.extend([cx-r, cx+r])
                        all_y.extend([cy-r, cy+r])
                    elif etype in ["LWPOLYLINE", "POLYLINE"]:
                        if hasattr(entity, 'get_points'):
                            pts = list(entity.get_points())
                            if len(pts) >= 2:
                                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                                ax.plot(xs, ys, 'k-', linewidth=0.8)
                                all_x.extend(xs)
                                all_y.extend(ys)
                except Exception as exc:
                    logger.debug("DXF渲染实体跳过: %s", exc)

            if not all_x or not all_y:
                plt.close(fig)
                return None

            margin = 0.1
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_range, y_range = max(x_max - x_min, 1e-6), max(y_max - y_min, 1e-6)
            ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
            ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
            ax.axis('off')

            with io.BytesIO() as buf:
                plt.savefig(buf, format='png', dpi=100, facecolor='white', bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                buf.seek(0)

                from PIL import Image
                img = Image.open(buf).convert('L').resize((self.IMG_SIZE, self.IMG_SIZE))
                return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            logger.error(f"渲染失败: {e}")
            return None

    def predict(self, file_path: str) -> Optional[ClassificationResult]:
        """预测CAD文件的部件类别（支持DXF和DWG）

        Args:
            file_path: DXF或DWG文件路径

        Returns:
            ClassificationResult 包含:
            - category: 预测类别
            - confidence: 置信度
            - needs_review: 是否需要人工审核
            - review_reason: 需要审核的原因
            - top2_category: 第二可能的类别
            - top2_confidence: 第二类别的置信度
        """
        self._load_models()

        # 处理DWG文件 - 转换为DXF
        dxf_path = file_path
        temp_dxf = None
        if file_path.lower().endswith('.dwg'):
            dxf_path, temp_dxf = self._convert_dwg_to_dxf(file_path)
            if dxf_path is None:
                return None

        try:
            result = self._predict_dxf(dxf_path)
            return result
        finally:
            # 清理临时文件
            if temp_dxf:
                try:
                    import os
                    os.unlink(temp_dxf)
                except Exception:
                    pass

    def _convert_dwg_to_dxf(self, dwg_path: str) -> tuple:
        """将DWG转换为临时DXF文件

        Returns:
            (dxf_path, temp_file_path) 或 (None, None) 如果转换失败
        """
        try:
            from src.core.cad.dwg.converter import DWGConverter
            import tempfile

            converter = DWGConverter()
            if not converter.is_available:
                logger.warning("DWG转换器不可用，无法处理DWG文件")
                return None, None

            # 创建临时文件
            temp_fd, temp_path = tempfile.mkstemp(suffix='.dxf')
            import os
            os.close(temp_fd)

            result = converter.convert(dwg_path, temp_path)
            if result.success:
                logger.debug(f"DWG转换成功: {dwg_path} -> {temp_path}")
                return temp_path, temp_path
            else:
                logger.error(f"DWG转换失败: {result.error_message}")
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                return None, None

        except ImportError:
            logger.warning("DWG转换模块不可用")
            return None, None
        except Exception as e:
            logger.error(f"DWG转换异常: {e}")
            return None, None

    def _predict_dxf(self, dxf_path: str) -> Optional[ClassificationResult]:
        """预测DXF文件的部件类别（内部方法）"""
        # 检查缓存
        cache_key = self._get_file_cache_key(dxf_path)
        cached_features, cached_img = self._cache_get(cache_key)

        if cached_features is not None:
            # 缓存命中 - 跳过I/O
            features = cached_features
            img = cached_img
        else:
            # 缓存未命中 - 读取并提取特征
            try:
                import ezdxf
                ezdxf_doc = ezdxf.readfile(dxf_path)
            except Exception as e:
                logger.error(f"读取DXF文件失败: {e}")
                return None

            features = self._extract_features(dxf_path, ezdxf_doc=ezdxf_doc)
            if features is None:
                return None

            # 获取速度配置
            v14_folds = self._speed_config["v14_folds"]
            use_v14 = v14_folds > 0 and self.v14_models

            # V14需要图像渲染 (复用ezdxf_doc)
            if use_v14:
                img = self._render_dxf(dxf_path, ezdxf_doc=ezdxf_doc)
                if img is None:
                    img = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
            else:
                img = None

            # 存入缓存
            self._cache_put(cache_key, features, img)

        # 获取速度配置（缓存命中时也需要）
        v14_folds = self._speed_config["v14_folds"]
        use_v14 = v14_folds > 0 and self.v14_models

        with torch.inference_mode():
            # 转换为tensor并应用正确的dtype
            x_v6 = torch.tensor(features, dtype=self._dtype, device=self.device).unsqueeze(0)
            v6_probs = torch.softmax(self.v6_model(x_v6), dim=1)

            if use_v14 and img is not None:
                x_geo = torch.tensor(features, dtype=self._dtype, device=self.device).unsqueeze(0)
                x_img = torch.tensor(img, dtype=self._dtype, device=self.device).unsqueeze(0).unsqueeze(0)

                # 根据速度模式选择使用的V14折叠数量
                models_to_use = self.v14_models[:v14_folds]
                n_models = len(models_to_use)

                # 优化: 批量推理 - 将输入扩展为batch，一次性计算所有折叠
                if n_models > 1:
                    # 批量处理: [n_models, 1, H, W] 和 [n_models, 48]
                    x_img_batch = x_img.expand(n_models, -1, -1, -1)
                    x_geo_batch = x_geo.expand(n_models, -1)

                    # 并行计算所有模型 (通过循环但tensor已预分配)
                    v14_logits_list = []
                    for i, model in enumerate(models_to_use):
                        logits = model(x_img_batch[i:i+1], x_geo_batch[i:i+1])
                        v14_logits_list.append(logits)
                    v14_logits = torch.cat(v14_logits_list, dim=0)
                    v14_probs = torch.softmax(v14_logits, dim=1).mean(dim=0, keepdim=True)
                else:
                    # 单模型直接推理
                    v14_probs = torch.softmax(models_to_use[0](x_img, x_geo), dim=1)

                # 融合
                final_probs = self.v6_weight * v6_probs + self.v14_weight * v14_probs
            else:
                # v6_only 模式
                final_probs = v6_probs

        # 获取top-2预测 (转换为float32确保兼容性)
        probs_np = final_probs[0].float().cpu().numpy()
        sorted_indices = np.argsort(probs_np)[::-1]  # 降序

        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]

        top1_cat = self.CATEGORIES[top1_idx]
        top1_conf = probs_np[top1_idx]
        top2_cat = self.CATEGORIES[top2_idx]
        top2_conf = probs_np[top2_idx]

        # 检查是否需要人工审核
        needs_review, review_reason = self._check_needs_review(
            top1_cat, top1_conf, top2_cat, top2_conf
        )

        probabilities = {
            self.CATEGORIES[i]: float(probs_np[i])
            for i in range(len(self.CATEGORIES))
        }

        # 根据速度模式设置版本标识
        version = f"v16_{self.speed_mode}" if self.speed_mode != "accurate" else "v16"

        return ClassificationResult(
            category=top1_cat,
            confidence=float(top1_conf),
            probabilities=probabilities,
            model_version=version,
            needs_review=needs_review,
            review_reason=review_reason,
            top2_category=top2_cat,
            top2_confidence=float(top2_conf)
        )

    def predict_batch(self, dxf_paths: List[str], max_workers: int = None) -> List[Optional[ClassificationResult]]:
        """批量预测（并行处理）

        Args:
            dxf_paths: DXF文件路径列表
            max_workers: 最大并行数，默认为CPU核心数

        Returns:
            分类结果列表，与输入顺序对应
        """
        if not dxf_paths:
            return []

        # 单文件不需要并行
        if len(dxf_paths) == 1:
            return [self.predict(dxf_paths[0])]

        # 确保模型已加载（避免并行时重复加载）
        self._load_models()

        import concurrent.futures
        import os

        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, len(dxf_paths), 8)

        # 并行处理文件I/O和特征提取（主要瓶颈）
        results = [None] * len(dxf_paths)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.predict, path): idx
                for idx, path in enumerate(dxf_paths)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"批量预测失败 [{idx}]: {e}")
                    results[idx] = None

        return results


# V16全局实例
_classifier_v16: Optional[PartClassifierV16] = None


def get_part_classifier_v16() -> PartClassifierV16:
    """获取V16分类器单例"""
    global _classifier_v16
    if _classifier_v16 is None:
        _classifier_v16 = PartClassifierV16()
    return _classifier_v16


def classify_part_v16(dxf_path: str) -> Optional[ClassificationResult]:
    """便捷函数：使用V16分类单个文件"""
    return get_part_classifier_v16().predict(dxf_path)
