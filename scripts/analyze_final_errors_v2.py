#!/usr/bin/env python3
"""
分析V16超级集成的最终错误样本 - 使用正确的数据集
"""

import json
import logging
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 使用V6训练时的数据集！
REAL_DATA_DIR = Path("data/training_v5")  # V6训练数据集
MODEL_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CATEGORIES = ["轴类", "传动件", "壳体类", "连接件", "其他"]
CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}
IDX_TO_CAT = {i: cat for i, cat in enumerate(CATEGORIES)}

SUSPICIOUS_SAMPLES = [
    "其他/old_0033.dxf",
    "其他/old_0085.dxf",
    "连接件/old_0008.dxf",
    "其他/new_0208.dxf",
]


def extract_geometric_features(dxf_path: str) -> np.ndarray:
    """提取48维几何特征"""
    try:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        entity_counts = Counter()
        all_x, all_y = [], []
        line_lengths = []
        circle_radii = []
        arc_angles = []
        layer_names = set()

        for entity in msp:
            etype = entity.dxftype()
            entity_counts[etype] += 1

            if hasattr(entity.dxf, 'layer'):
                layer_names.add(entity.dxf.layer)

            try:
                if etype == "LINE":
                    x1, y1 = entity.dxf.start.x, entity.dxf.start.y
                    x2, y2 = entity.dxf.end.x, entity.dxf.end.y
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    line_lengths.append(length)
                    all_x.extend([x1, x2])
                    all_y.extend([y1, y2])
                elif etype == "CIRCLE":
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r = entity.dxf.radius
                    circle_radii.append(r)
                    all_x.extend([cx-r, cx+r])
                    all_y.extend([cy-r, cy+r])
                elif etype == "ARC":
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r = entity.dxf.radius
                    angle = abs(entity.dxf.end_angle - entity.dxf.start_angle)
                    arc_angles.append(angle)
                    all_x.extend([cx-r, cx+r])
                    all_y.extend([cy-r, cy+r])
                elif etype in ["LWPOLYLINE", "POLYLINE"]:
                    if hasattr(entity, 'get_points'):
                        pts = list(entity.get_points())
                        for p in pts:
                            all_x.append(p[0])
                            all_y.append(p[1])
            except:
                pass

        total = sum(entity_counts.values())
        if total == 0:
            return np.zeros(48)

        # 基础实体统计 (10维)
        features = [
            total,
            entity_counts.get("LINE", 0),
            entity_counts.get("CIRCLE", 0),
            entity_counts.get("ARC", 0),
            entity_counts.get("LWPOLYLINE", 0) + entity_counts.get("POLYLINE", 0),
            entity_counts.get("SPLINE", 0),
            entity_counts.get("ELLIPSE", 0),
            entity_counts.get("POINT", 0),
            entity_counts.get("TEXT", 0) + entity_counts.get("MTEXT", 0),
            len(layer_names),
        ]

        # 比例特征 (8维)
        features.extend([
            entity_counts.get("LINE", 0) / total,
            entity_counts.get("CIRCLE", 0) / total,
            entity_counts.get("ARC", 0) / total,
            (entity_counts.get("CIRCLE", 0) + entity_counts.get("ARC", 0)) / total,
            (entity_counts.get("LWPOLYLINE", 0) + entity_counts.get("POLYLINE", 0)) / total,
            entity_counts.get("SPLINE", 0) / total,
            (entity_counts.get("TEXT", 0) + entity_counts.get("MTEXT", 0)) / total,
            len(entity_counts) / 20,
        ])

        # 边界框特征 (8维)
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            width = x_max - x_min
            height = y_max - y_min
            aspect_ratio = width / max(height, 1e-6)
            area = width * height
            features.extend([
                width, height, aspect_ratio, area,
                np.log1p(width), np.log1p(height),
                np.log1p(area), 1.0 if aspect_ratio > 2 else 0.0,
            ])
        else:
            features.extend([0]*8)

        # 线段特征 (6维)
        if line_lengths:
            features.extend([
                np.mean(line_lengths), np.std(line_lengths),
                np.min(line_lengths), np.max(line_lengths),
                len(line_lengths), np.log1p(np.sum(line_lengths)),
            ])
        else:
            features.extend([0]*6)

        # 圆特征 (6维)
        if circle_radii:
            features.extend([
                np.mean(circle_radii), np.std(circle_radii),
                np.min(circle_radii), np.max(circle_radii),
                len(circle_radii), np.log1p(np.sum(circle_radii)),
            ])
        else:
            features.extend([0]*6)

        # 弧特征 (6维)
        if arc_angles:
            features.extend([
                np.mean(arc_angles), np.std(arc_angles),
                np.min(arc_angles), np.max(arc_angles),
                len(arc_angles), np.sum(arc_angles) / 360,
            ])
        else:
            features.extend([0]*6)

        # 复杂度特征 (4维)
        features.extend([
            total / max(len(layer_names), 1),
            np.log1p(total),
            len(entity_counts),
            (entity_counts.get("SPLINE", 0) + entity_counts.get("ELLIPSE", 0)) / max(total, 1),
        ])

        return np.array(features[:48], dtype=np.float32)

    except Exception as e:
        return np.zeros(48, dtype=np.float32)


class ImprovedClassifierV6(nn.Module):
    """V6分类器结构"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.6),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.4),

            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    """加载数据集"""
    manifest_path = REAL_DATA_DIR / "manifest.json"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    return manifest


def main():
    logger.info("=" * 60)
    logger.info("分析V16最终错误样本 (使用V6训练数据集)")
    logger.info("=" * 60)

    # 加载V6模型
    logger.info("\n加载V6几何分类器...")
    v6_path = MODEL_DIR / "cad_classifier_v6.pt"
    v6_checkpoint = torch.load(v6_path, map_location=DEVICE, weights_only=False)

    v6_model = ImprovedClassifierV6(input_dim=48, hidden_dim=256, num_classes=5, dropout=0.5)
    v6_model.load_state_dict(v6_checkpoint['model_state_dict'])
    v6_model.to(DEVICE)
    v6_model.eval()

    # 特征标准化参数
    feature_mean = v6_checkpoint.get('feature_mean', np.zeros(48))
    feature_std = v6_checkpoint.get('feature_std', np.ones(48))

    # 加载数据
    manifest = load_data()
    logger.info(f"总样本数: {len(manifest)}")

    # V6预测所有样本
    logger.info("\nV6预测...")
    errors = []
    correct = 0
    total = 0

    for item in manifest:
        file_path = REAL_DATA_DIR / item["file"]
        true_cat = item["category"]

        features = extract_geometric_features(str(file_path))
        features_norm = (features - feature_mean) / (feature_std + 1e-8)

        with torch.no_grad():
            x = torch.FloatTensor(features_norm).unsqueeze(0).to(DEVICE)
            logits = v6_model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_cat = IDX_TO_CAT[pred_idx]
            confidence = probs[0, pred_idx].item()

        total += 1
        if pred_cat == true_cat:
            correct += 1
        else:
            errors.append({
                "file": item["file"],
                "true": true_cat,
                "pred": pred_cat,
                "confidence": confidence,
                "is_suspicious": item["file"] in SUSPICIOUS_SAMPLES,
            })

    logger.info(f"\nV6准确率: {correct}/{total} = {100*correct/total:.2f}%")
    logger.info(f"错误数: {len(errors)}")

    # 分析错误样本
    logger.info("\n" + "=" * 60)
    logger.info("错误样本详细分析")
    logger.info("=" * 60)

    for i, err in enumerate(errors):
        marker = "⚠️ 可疑" if err['is_suspicious'] else ""
        logger.info(f"\n{i+1}. {err['file']} {marker}")
        logger.info(f"   标注: {err['true']} → 预测: {err['pred']} (置信度: {err['confidence']:.2%})")

    # 统计
    suspicious_count = sum(1 for e in errors if e['is_suspicious'])
    non_suspicious = [e for e in errors if not e['is_suspicious']]

    logger.info("\n" + "=" * 60)
    logger.info("汇总")
    logger.info("=" * 60)
    logger.info(f"总错误数: {len(errors)}")
    logger.info(f"已标记可疑样本: {suspicious_count}")
    logger.info(f"新发现的错误样本: {len(non_suspicious)}")

    # 计算修正后准确率
    clean_accuracy = (total - len(non_suspicious)) / total
    logger.info(f"\n若所有可疑样本标注有误，修正后准确率: {100*clean_accuracy:.2f}%")

    if non_suspicious:
        logger.info("\n新发现的错误样本（需要进一步分析）:")
        for e in non_suspicious:
            logger.info(f"  {e['file']}: {e['true']} → {e['pred']} ({e['confidence']:.2%})")


if __name__ == "__main__":
    main()
