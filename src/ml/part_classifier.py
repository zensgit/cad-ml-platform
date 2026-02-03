"""
CAD部件分类推理服务

提供DXF/DWG图纸的部件类型识别功能
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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


class PartClassifier:
    """部件分类器"""

    def __init__(self, model_path: str = "models/cad_classifier_v2.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.id_to_label = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """加载模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 重建模型架构
        input_dim = checkpoint["input_dim"]
        hidden_dim = checkpoint["hidden_dim"]
        num_classes = checkpoint["num_classes"]

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

        self.model = ImprovedClassifier(input_dim, hidden_dim, num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.id_to_label = {int(k): v for k, v in checkpoint["id_to_label"].items()}
        self.num_classes = num_classes

        logger.info(f"模型加载成功，类别: {list(self.id_to_label.values())}")

    def extract_features(self, dxf_path: str) -> Optional[np.ndarray]:
        """从DXF文件提取特征"""
        try:
            import ezdxf
            from collections import Counter

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
                except:
                    pass

            type_counts = Counter(entity_types)
            total_entities = len(entity_types)

            features = []

            # 实体类型比例
            for etype in ["LINE", "CIRCLE", "ARC", "LWPOLYLINE", "POLYLINE",
                          "SPLINE", "ELLIPSE", "TEXT", "MTEXT", "DIMENSION",
                          "HATCH", "INSERT"]:
                ratio = type_counts.get(etype, 0) / max(total_entities, 1)
                features.append(ratio)

            # 几何统计
            features.append(np.log1p(total_entities) / 10)

            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                features.append(np.log1p(width) / 10)
                features.append(np.log1p(height) / 10)
                features.append(np.clip(width / max(height, 0.001), 0, 10) / 10)
            else:
                features.extend([0, 0, 0.5])

            if circle_radii:
                features.append(np.log1p(np.mean(circle_radii)) / 5)
                features.append(np.log1p(np.std(circle_radii)) / 5)
            else:
                features.extend([0, 0])

            if line_lengths:
                features.append(np.log1p(np.mean(line_lengths)) / 5)
                features.append(np.log1p(np.std(line_lengths)) / 5)
            else:
                features.extend([0, 0])

            # 图层特征
            unique_layers = len(set(layer_names))
            features.append(np.log1p(unique_layers) / 3)

            layer_lower = [l.lower() for l in layer_names]
            features.append(1.0 if any('dim' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('text' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('center' in l for l in layer_lower) else 0.0)

            # 复杂度特征
            arc_ratio = (type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0)) / max(total_entities, 1)
            features.append(arc_ratio)

            text_ratio = (type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0)) / max(total_entities, 1)
            features.append(text_ratio)

            insert_ratio = type_counts.get("INSERT", 0) / max(total_entities, 1)
            features.append(insert_ratio)

            dim_ratio = type_counts.get("DIMENSION", 0) / max(total_entities, 1)
            features.append(dim_ratio)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None

    def predict(self, dxf_path: str) -> Optional[ClassificationResult]:
        """预测DXF文件的部件类别"""
        features = self.extract_features(dxf_path)
        if features is None:
            return None

        # 转换为tensor
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_id = probs.argmax().item()
            confidence = probs[pred_id].item()

        # 构建结果
        probabilities = {
            self.id_to_label[i]: probs[i].item()
            for i in range(self.num_classes)
        }

        return ClassificationResult(
            category=self.id_to_label[pred_id],
            confidence=confidence,
            probabilities=probabilities
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
