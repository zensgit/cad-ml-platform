"""
CAD部件分类推理服务

提供DXF/DWG图纸的部件类型识别功能
支持V2 (28维/7类), V6 (48维/5类) 和 V7 (48维/5类) 模型
"""

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


class PartClassifier:
    """部件分类器 - 支持多版本模型"""

    def __init__(self, model_path: str = "models/cad_classifier_v7.pt"):
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
        self.version = checkpoint.get("version", "v2")

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
