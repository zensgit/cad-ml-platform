"""
CAD部件分类推理服务

提供DXF/DWG图纸的部件类型识别功能
支持V2 (28维/7类), V6 (48维/5类), V16 (超级集成/5类, 99.88%准确率) 模型
"""

import io
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        if self.version in ("v6", "v7", "v8"):
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
        try:
            import ezdxf

            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            entity_types = []
            layer_names = []
            all_points = []
            circle_radii = []
            arc_radii = []
            arc_angles = []
            line_lengths = []
            polyline_vertex_counts = []
            dimension_count = 0
            hatch_count = 0
            block_names = []

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
                        arc_radii.append(entity.dxf.radius)
                        angle = abs(entity.dxf.end_angle - entity.dxf.start_angle)
                        if angle > 180:
                            angle = 360 - angle
                        arc_angles.append(angle)
                    elif etype in ["TEXT", "MTEXT"]:
                        if hasattr(entity.dxf, 'insert'):
                            all_points.append((entity.dxf.insert.x, entity.dxf.insert.y))
                    elif etype in ["LWPOLYLINE", "POLYLINE"]:
                        if hasattr(entity, 'get_points'):
                            pts = list(entity.get_points())
                            polyline_vertex_counts.append(len(pts))
                            for pt in pts:
                                all_points.append((pt[0], pt[1]))
                    elif etype == "INSERT":
                        if hasattr(entity.dxf, 'insert'):
                            all_points.append((entity.dxf.insert.x, entity.dxf.insert.y))
                        if hasattr(entity.dxf, 'name'):
                            block_names.append(entity.dxf.name)
                    elif etype == "DIMENSION":
                        dimension_count += 1
                    elif etype == "HATCH":
                        hatch_count += 1
                except Exception:
                    pass

            type_counts = Counter(entity_types)
            total_entities = len(entity_types)

            features = []

            # 1-12: 实体类型比例
            for etype in ["LINE", "CIRCLE", "ARC", "LWPOLYLINE", "POLYLINE",
                          "SPLINE", "ELLIPSE", "TEXT", "MTEXT", "DIMENSION",
                          "HATCH", "INSERT"]:
                features.append(type_counts.get(etype, 0) / max(total_entities, 1))

            # 13-16: 基础几何
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

            # 17-22: 圆/弧
            if circle_radii:
                features.extend([np.log1p(np.mean(circle_radii)) / 5,
                               np.log1p(np.std(circle_radii)) / 5 if len(circle_radii) > 1 else 0,
                               len(circle_radii) / max(total_entities, 1)])
            else:
                features.extend([0, 0, 0])

            if arc_radii:
                features.extend([np.log1p(np.mean(arc_radii)) / 5,
                               np.mean(arc_angles) / 180 if arc_angles else 0,
                               len(arc_radii) / max(total_entities, 1)])
            else:
                features.extend([0, 0, 0])

            # 23-26: 线段
            if line_lengths:
                features.extend([np.log1p(np.mean(line_lengths)) / 5,
                               np.log1p(np.std(line_lengths)) / 5 if len(line_lengths) > 1 else 0,
                               np.log1p(np.max(line_lengths)) / 5,
                               np.log1p(np.min(line_lengths)) / 5])
            else:
                features.extend([0, 0, 0, 0])

            # 27-32: 图层
            unique_layers = len(set(layer_names))
            features.append(np.log1p(unique_layers) / 3)
            layer_lower = [l.lower() for l in layer_names]
            features.append(1.0 if any('dim' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('text' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('center' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('hidden' in l or 'hid' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('section' in l or 'cut' in l for l in layer_lower) else 0.0)

            # 33-36: 复杂度
            features.append((type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0)) / max(total_entities, 1))
            features.append((type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0)) / max(total_entities, 1))
            features.append(type_counts.get("INSERT", 0) / max(total_entities, 1))
            features.append(dimension_count / max(total_entities, 1))

            # 37-40: 空间分布
            if all_points and len(all_points) > 1:
                xs = np.array([p[0] for p in all_points])
                ys = np.array([p[1] for p in all_points])
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                features.append(np.log1p(len(all_points) / max(area, 0.001)) / 10)
                features.append(np.std(xs) / max(max(xs) - min(xs), 0.001))
                features.append(np.std(ys) / max(max(ys) - min(ys), 0.001))
                center_offset = np.sqrt((np.mean(xs) - (max(xs)+min(xs))/2)**2 +
                                       (np.mean(ys) - (max(ys)+min(ys))/2)**2)
                features.append(np.log1p(center_offset) / 5)
            else:
                features.extend([0, 0.5, 0.5, 0])

            # 41-44: 形状复杂度
            if polyline_vertex_counts:
                features.extend([np.log1p(np.mean(polyline_vertex_counts)) / 3,
                               np.log1p(np.max(polyline_vertex_counts)) / 4])
            else:
                features.extend([0, 0])
            features.append(hatch_count / max(total_entities, 1))
            features.append(np.log1p(len(set(block_names))) / 3)

            # 45-48: 类型特征
            curved = type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0) + type_counts.get("ELLIPSE", 0)
            straight = type_counts.get("LINE", 0)
            features.append(np.clip(curved / max(straight, 1), 0, 5) / 5)
            annotation = type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0) + dimension_count
            features.append(annotation / max(total_entities, 1))
            geometry = straight + curved + type_counts.get("LWPOLYLINE", 0) + type_counts.get("POLYLINE", 0)
            features.append(np.clip(geometry / max(annotation, 1), 0, 20) / 20)
            features.append(len(circle_radii) / max(total_entities, 1))

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None

    def predict(self, dxf_path: str) -> Optional[ClassificationResult]:
        """预测DXF文件的部件类别"""
        features = self.extract_features(dxf_path)
        if features is None:
            return None

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_id = probs.argmax().item()
            confidence = probs[pred_id].item()

        probabilities = {
            self.id_to_label[i]: probs[i].item()
            for i in range(self.num_classes)
        }

        return ClassificationResult(
            category=self.id_to_label[pred_id],
            confidence=confidence,
            probabilities=probabilities,
            model_version=self.version
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
        _classifier = PartClassifier()
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
    """V16超级集成分类器 (99.88%准确率)

    结合V6纯几何模型和V14视觉+几何融合集成模型

    特性:
    - 置信度阈值机制：低于阈值时标记为"需人工审核"
    - Top-2预测：返回前两个最可能的类别供参考
    - 边界案例识别：特定零件类型可能存在分类歧义
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

    def __init__(self, model_dir: str = "models", confidence_threshold: float = None):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        self.v6_model = None
        self.v14_models = []
        self.v6_mean = None
        self.v6_std = None
        self.v6_weight = 0.6
        self.v14_weight = 0.4
        self.loaded = False
        # 允许自定义置信度阈值
        self.confidence_threshold = confidence_threshold or self.CONFIDENCE_THRESHOLD

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
                self.v14_models.append(model)
            logger.info(f"V14集成模型加载完成，{len(self.v14_models)}个折叠")
        else:
            logger.warning(f"V14模型不存在: {v14_path}，将仅使用V6")

        self.loaded = True
        logger.info(f"V16模型加载完成，设备: {self.device}")

    def _extract_features(self, dxf_path: str) -> Optional[np.ndarray]:
        """提取48维几何特征（与V6训练时一致）"""
        try:
            import ezdxf
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            entity_types = []
            layer_names = []
            all_points = []
            circle_radii = []
            arc_radii = []
            arc_angles = []
            line_lengths = []
            polyline_vertex_counts = []
            dimension_count = 0
            hatch_count = 0
            block_names = []

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
                        line_lengths.append(np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2))
                    elif etype == "CIRCLE":
                        center = (entity.dxf.center.x, entity.dxf.center.y)
                        all_points.append(center)
                        circle_radii.append(entity.dxf.radius)
                    elif etype == "ARC":
                        center = (entity.dxf.center.x, entity.dxf.center.y)
                        all_points.append(center)
                        arc_radii.append(entity.dxf.radius)
                        angle = abs(entity.dxf.end_angle - entity.dxf.start_angle)
                        if angle > 180:
                            angle = 360 - angle
                        arc_angles.append(angle)
                    elif etype in ["TEXT", "MTEXT"]:
                        if hasattr(entity.dxf, 'insert'):
                            all_points.append((entity.dxf.insert.x, entity.dxf.insert.y))
                    elif etype in ["LWPOLYLINE", "POLYLINE"]:
                        if hasattr(entity, 'get_points'):
                            pts = list(entity.get_points())
                            polyline_vertex_counts.append(len(pts))
                            for pt in pts:
                                all_points.append((pt[0], pt[1]))
                    elif etype == "INSERT":
                        if hasattr(entity.dxf, 'insert'):
                            all_points.append((entity.dxf.insert.x, entity.dxf.insert.y))
                        if hasattr(entity.dxf, 'name'):
                            block_names.append(entity.dxf.name)
                    elif etype == "DIMENSION":
                        dimension_count += 1
                    elif etype == "HATCH":
                        hatch_count += 1
                except:
                    pass

            type_counts = Counter(entity_types)
            total_entities = len(entity_types)
            features = []

            for etype in ["LINE", "CIRCLE", "ARC", "LWPOLYLINE", "POLYLINE",
                          "SPLINE", "ELLIPSE", "TEXT", "MTEXT", "DIMENSION", "HATCH", "INSERT"]:
                features.append(type_counts.get(etype, 0) / max(total_entities, 1))

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

            if circle_radii:
                features.extend([np.log1p(np.mean(circle_radii)) / 5,
                               np.log1p(np.std(circle_radii)) / 5 if len(circle_radii) > 1 else 0,
                               len(circle_radii) / max(total_entities, 1)])
            else:
                features.extend([0, 0, 0])

            if arc_radii:
                features.extend([np.log1p(np.mean(arc_radii)) / 5,
                               np.mean(arc_angles) / 180 if arc_angles else 0,
                               len(arc_radii) / max(total_entities, 1)])
            else:
                features.extend([0, 0, 0])

            if line_lengths:
                features.extend([np.log1p(np.mean(line_lengths)) / 5,
                               np.log1p(np.std(line_lengths)) / 5 if len(line_lengths) > 1 else 0,
                               np.log1p(np.max(line_lengths)) / 5,
                               np.log1p(np.min(line_lengths)) / 5])
            else:
                features.extend([0, 0, 0, 0])

            unique_layers = len(set(layer_names))
            features.append(np.log1p(unique_layers) / 3)
            layer_lower = [l.lower() for l in layer_names]
            features.append(1.0 if any('dim' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('text' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('center' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('hidden' in l or 'hid' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('section' in l or 'cut' in l for l in layer_lower) else 0.0)

            features.append((type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0)) / max(total_entities, 1))
            features.append((type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0)) / max(total_entities, 1))
            features.append(type_counts.get("INSERT", 0) / max(total_entities, 1))
            features.append(dimension_count / max(total_entities, 1))

            if all_points and len(all_points) > 1:
                xs = np.array([p[0] for p in all_points])
                ys = np.array([p[1] for p in all_points])
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                features.append(np.log1p(len(all_points) / max(area, 0.001)) / 10)
                features.append(np.std(xs) / max(max(xs) - min(xs), 0.001))
                features.append(np.std(ys) / max(max(ys) - min(ys), 0.001))
                center_offset = np.sqrt((np.mean(xs) - (max(xs)+min(xs))/2)**2 +
                                       (np.mean(ys) - (max(ys)+min(ys))/2)**2)
                features.append(np.log1p(center_offset) / 5)
            else:
                features.extend([0, 0.5, 0.5, 0])

            if polyline_vertex_counts:
                features.extend([np.log1p(np.mean(polyline_vertex_counts)) / 3,
                               np.log1p(np.max(polyline_vertex_counts)) / 4])
            else:
                features.extend([0, 0])
            features.append(hatch_count / max(total_entities, 1))
            features.append(np.log1p(len(set(block_names))) / 3)

            curved = type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0) + type_counts.get("ELLIPSE", 0)
            straight = type_counts.get("LINE", 0)
            features.append(np.clip(curved / max(straight, 1), 0, 5) / 5)
            annotation = type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0) + dimension_count
            features.append(annotation / max(total_entities, 1))
            geometry = straight + curved + type_counts.get("LWPOLYLINE", 0) + type_counts.get("POLYLINE", 0)
            features.append(np.clip(geometry / max(annotation, 1), 0, 20) / 20)
            features.append(len(circle_radii) / max(total_entities, 1))

            return np.array(features, dtype=np.float32)
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None

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

    def _render_dxf(self, dxf_path: str) -> Optional[np.ndarray]:
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
                except:
                    pass

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

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, facecolor='white', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)

            from PIL import Image
            img = Image.open(buf).convert('L').resize((self.IMG_SIZE, self.IMG_SIZE))
            return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            logger.error(f"渲染失败: {e}")
            return None

    def predict(self, dxf_path: str) -> Optional[ClassificationResult]:
        """预测DXF文件的部件类别

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

        features = self._extract_features(dxf_path)
        if features is None:
            return None

        # V6预测 (使用原始特征，因为训练时未保存标准化参数)
        with torch.no_grad():
            x_v6 = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            v6_probs = torch.softmax(self.v6_model(x_v6), dim=1)

        # V14预测 (使用原始特征，因为训练时未标准化)
        if self.v14_models:
            img = self._render_dxf(dxf_path)
            if img is None:
                img = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)

            with torch.no_grad():
                # V14使用原始特征（不标准化）
                x_geo = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                x_img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(self.device)

                v14_probs_list = []
                for model in self.v14_models:
                    probs = torch.softmax(model(x_img, x_geo), dim=1)
                    v14_probs_list.append(probs)
                v14_probs = torch.mean(torch.stack(v14_probs_list), dim=0)

            # 融合
            final_probs = self.v6_weight * v6_probs + self.v14_weight * v14_probs
        else:
            final_probs = v6_probs

        # 获取top-2预测
        probs_np = final_probs[0].cpu().numpy()
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

        return ClassificationResult(
            category=top1_cat,
            confidence=float(top1_conf),
            probabilities=probabilities,
            model_version="v16",
            needs_review=needs_review,
            review_reason=review_reason,
            top2_category=top2_cat,
            top2_confidence=float(top2_conf)
        )

    def predict_batch(self, dxf_paths: List[str]) -> List[Optional[ClassificationResult]]:
        """批量预测"""
        return [self.predict(p) for p in dxf_paths]


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
