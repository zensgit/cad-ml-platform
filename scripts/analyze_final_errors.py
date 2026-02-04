#!/usr/bin/env python3
"""
分析V16超级集成的最终错误样本
确定是否还有提升空间
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

REAL_DATA_DIR = Path("data/training_v7")
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
        logger.error(f"提取特征失败 {dxf_path}: {e}")
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
    logger.info("分析V16最终错误样本")
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

    # 计算各类别的特征统计
    logger.info("\n计算各类别特征统计...")
    class_features = {cat: [] for cat in CATEGORIES}

    for item in manifest:
        file_path = REAL_DATA_DIR / item["file"]
        category = item["category"]
        features = extract_geometric_features(str(file_path))
        class_features[category].append(features)

    # 计算类中心
    class_centers = {}
    for cat in CATEGORIES:
        if class_features[cat]:
            class_centers[cat] = np.mean(class_features[cat], axis=0)

    # V6预测所有样本
    logger.info("\nV6预测结果分析...")
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
            # 计算到各类中心的距离
            distances = {}
            for cat, center in class_centers.items():
                dist = np.linalg.norm(features - center)
                distances[cat] = dist

            errors.append({
                "file": item["file"],
                "true": true_cat,
                "pred": pred_cat,
                "confidence": confidence,
                "is_suspicious": item["file"] in SUSPICIOUS_SAMPLES,
                "distances": distances,
            })

    logger.info(f"\nV6准确率: {correct}/{total} = {100*correct/total:.2f}%")
    logger.info(f"错误数: {len(errors)}")

    # 分析每个错误样本
    logger.info("\n" + "=" * 60)
    logger.info("错误样本详细分析")
    logger.info("=" * 60)

    for i, err in enumerate(errors):
        logger.info(f"\n{i+1}. {err['file']}")
        logger.info(f"   标注: {err['true']} → 预测: {err['pred']} (置信度: {err['confidence']:.2%})")
        logger.info(f"   可疑样本: {'是' if err['is_suspicious'] else '否'}")

        # 距离分析
        sorted_dists = sorted(err['distances'].items(), key=lambda x: x[1])
        logger.info(f"   到各类中心距离:")
        for cat, dist in sorted_dists:
            marker = "← 标注" if cat == err['true'] else ("← 预测" if cat == err['pred'] else "")
            logger.info(f"      {cat}: {dist:.2f} {marker}")

        # 判断更接近哪个类
        nearest_cat = sorted_dists[0][0]
        if nearest_cat == err['pred']:
            logger.info(f"   📌 特征最接近预测类 '{err['pred']}'，标注可能有误")
        elif nearest_cat == err['true']:
            logger.info(f"   ✓ 特征最接近标注类 '{err['true']}'，是困难样本")
        else:
            logger.info(f"   ⚠️ 特征最接近 '{nearest_cat}'，但预测为 '{err['pred']}'")

    # 统计分析
    suspicious_count = sum(1 for e in errors if e['is_suspicious'])
    label_issue_count = sum(1 for e in errors if sorted(e['distances'].items(), key=lambda x: x[1])[0][0] == e['pred'])

    logger.info("\n" + "=" * 60)
    logger.info("错误分类汇总")
    logger.info("=" * 60)
    logger.info(f"总错误数: {len(errors)}")
    logger.info(f"已标记可疑样本: {suspicious_count}")
    logger.info(f"特征接近预测类(可能标注错误): {label_issue_count}")
    logger.info(f"特征接近标注类(真正困难样本): {len(errors) - label_issue_count}")

    # 如果修正所有可能的标注错误
    potential_accuracy = (total - (len(errors) - label_issue_count)) / total
    logger.info(f"\n若修正所有疑似标注错误，理论准确率: {100*potential_accuracy:.2f}%")

    # 检查是否还有未标记的可疑样本
    logger.info("\n" + "=" * 60)
    logger.info("建议添加到可疑样本列表")
    logger.info("=" * 60)

    new_suspicious = []
    for err in errors:
        if not err['is_suspicious']:
            sorted_dists = sorted(err['distances'].items(), key=lambda x: x[1])
            if sorted_dists[0][0] == err['pred']:
                new_suspicious.append(err['file'])
                logger.info(f"  {err['file']}: {err['true']} → {err['pred']}")

    if not new_suspicious:
        logger.info("  无新增可疑样本")

    # 最终结论
    logger.info("\n" + "=" * 60)
    logger.info("结论")
    logger.info("=" * 60)

    clean_errors = len(errors) - label_issue_count
    clean_total = total - label_issue_count
    clean_accuracy = (clean_total - clean_errors) / clean_total if clean_total > 0 else 1.0

    logger.info(f"1. 当前V6准确率: {100*correct/total:.2f}% ({correct}/{total})")
    logger.info(f"2. 排除标注问题后: {100*(total-len(errors))/(total-label_issue_count):.2f}%")
    logger.info(f"3. 真正困难样本数: {clean_errors}")

    if clean_errors == 0:
        logger.info("\n🎉 所有错误均为疑似标注问题，模型已达理论最优!")
    else:
        logger.info(f"\n还有 {clean_errors} 个真正困难样本需要进一步分析")


if __name__ == "__main__":
    main()
