#!/usr/bin/env python3
"""
分析V13错误案例，找出改进方向
"""

import json
import logging
import sys
import io
from pathlib import Path
from typing import Optional
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REAL_DATA_DIR = Path("data/training_v7")
MODEL_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

IMG_SIZE = 128


def render_dxf_to_grayscale(dxf_path: str, size: int = IMG_SIZE) -> Optional[np.ndarray]:
    """将DXF渲染为灰度图像"""
    try:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        fig = plt.figure(figsize=(size/100, size/100), dpi=100)
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
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r = entity.dxf.radius
                    from matplotlib.patches import Arc
                    arc = Arc((cx, cy), 2*r, 2*r, angle=0, theta1=entity.dxf.start_angle, theta2=entity.dxf.end_angle, color='k', linewidth=0.8)
                    ax.add_patch(arc)
                    all_x.extend([cx-r, cx+r])
                    all_y.extend([cy-r, cy+r])
                elif etype in ["LWPOLYLINE", "POLYLINE"]:
                    if hasattr(entity, 'get_points'):
                        pts = list(entity.get_points())
                        if len(pts) >= 2:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
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
        x_range = x_max - x_min or 1
        y_range = y_max - y_min or 1
        ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        ax.axis('off')
        ax.set_facecolor('white')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='white', edgecolor='none', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf).convert('L')
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0

        plt.close(fig)
        buf.close()

        return img_array
    except:
        return None


class GeometricFeatureExtractor:
    """48维几何特征提取器"""

    def extract(self, dxf_path: str) -> Optional[np.ndarray]:
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
                except:
                    pass

            type_counts = Counter(entity_types)
            total_entities = len(entity_types)
            features = []

            for etype in ["LINE", "CIRCLE", "ARC", "LWPOLYLINE", "POLYLINE",
                          "SPLINE", "ELLIPSE", "TEXT", "MTEXT", "DIMENSION",
                          "HATCH", "INSERT"]:
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

            curved = type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0) + type_counts.get("ELLIPSE", 0) + type_counts.get("SPLINE", 0)
            straight = type_counts.get("LINE", 0)
            features.append(np.clip(curved / max(straight, 1), 0, 5) / 5)
            annotation = type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0) + dimension_count
            features.append(annotation / max(total_entities, 1))
            geometry = straight + curved + type_counts.get("LWPOLYLINE", 0) + type_counts.get("POLYLINE", 0)
            features.append(np.clip(geometry / max(annotation, 1), 0, 20) / 20)
            features.append(len(circle_radii) / max(total_entities, 1))

            return np.array(features, dtype=np.float32)
        except:
            return None


class FusionModelV13(nn.Module):
    """V13模型架构"""

    def __init__(self, geo_dim: int, num_classes: int):
        super().__init__()

        self.visual_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.geo_branch = nn.Sequential(
            nn.Linear(geo_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
        )

        fusion_dim = 128 + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, img, geo):
        visual_feat = self.visual_branch(img)
        geo_feat = self.geo_branch(geo)
        fused = torch.cat([visual_feat, geo_feat], dim=1)
        attn = self.attention(fused)
        visual_weighted = visual_feat * attn[:, 0:1]
        geo_weighted = geo_feat * attn[:, 1:2]
        fused_weighted = torch.cat([visual_weighted, geo_weighted], dim=1)
        out = self.fusion(fused_weighted)
        return out, attn


def analyze_errors():
    """分析V13错误案例"""
    logger.info("=" * 60)
    logger.info("分析V13错误案例")
    logger.info("=" * 60)

    # 加载标签
    labels_path = REAL_DATA_DIR / "labels.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_info = json.load(f)

    num_classes = len(labels_info["label_to_id"])
    id_to_label = {int(k): v for k, v in labels_info["id_to_label"].items()}

    # 加载V13模型
    checkpoint = torch.load(MODEL_DIR / "cad_classifier_v13.pt", map_location=DEVICE, weights_only=False)
    model = FusionModelV13(checkpoint["geo_dim"], checkpoint["num_classes"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 加载数据
    manifest_path = REAL_DATA_DIR / "manifest.json"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    geo_extractor = GeometricFeatureExtractor()

    # 分析所有样本
    errors = []
    correct_samples = []
    attention_stats = defaultdict(list)
    confusion = defaultdict(lambda: defaultdict(int))

    logger.info(f"分析 {len(manifest)} 个样本...")

    for i, item in enumerate(manifest):
        file_path = REAL_DATA_DIR / item["file"]
        true_label = item["label_id"]
        category = item["category"]

        img = render_dxf_to_grayscale(str(file_path))
        geo = geo_extractor.extract(str(file_path))

        if img is None or geo is None:
            continue

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        geo_tensor = torch.tensor(geo, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output, attn = model(img_tensor, geo_tensor)
            probs = torch.softmax(output, dim=1)
            pred_label = output.argmax(1).item()
            confidence = probs[0, pred_label].item()
            visual_weight = attn[0, 0].item()
            geo_weight = attn[0, 1].item()

        confusion[true_label][pred_label] += 1
        attention_stats[category].append((visual_weight, geo_weight))

        if pred_label != true_label:
            errors.append({
                "file": item["file"],
                "true": id_to_label[true_label],
                "pred": id_to_label[pred_label],
                "confidence": confidence,
                "visual_weight": visual_weight,
                "geo_weight": geo_weight,
                "true_prob": probs[0, true_label].item()
            })
        else:
            correct_samples.append({
                "file": item["file"],
                "category": category,
                "confidence": confidence,
                "visual_weight": visual_weight,
                "geo_weight": geo_weight
            })

    # 输出分析结果
    total = sum(sum(confusion[t].values()) for t in confusion)
    correct = sum(confusion[t][t] for t in confusion)
    logger.info(f"\n总准确率: {correct/total:.2%} ({correct}/{total})")

    logger.info("\n混淆矩阵:")
    cats = sorted(id_to_label.keys())
    header = "真实\\预测  " + "  ".join([id_to_label[c][:4] for c in cats])
    logger.info(header)
    for t in cats:
        row = f"{id_to_label[t][:6]:8s}"
        for p in cats:
            row += f"  {confusion[t][p]:4d}"
        logger.info(row)

    logger.info("\n注意力权重分析 (视觉 vs 几何):")
    for cat in sorted(attention_stats.keys()):
        weights = attention_stats[cat]
        visual_avg = np.mean([w[0] for w in weights])
        geo_avg = np.mean([w[1] for w in weights])
        logger.info(f"  {cat}: 视觉={visual_avg:.2%}, 几何={geo_avg:.2%}")

    logger.info(f"\n错误案例分析 ({len(errors)} 个):")

    # 按错误类型分组
    error_types = defaultdict(list)
    for e in errors:
        key = f"{e['true']} -> {e['pred']}"
        error_types[key].append(e)

    for error_type, cases in sorted(error_types.items(), key=lambda x: -len(x[1])):
        logger.info(f"\n  {error_type}: {len(cases)} 个")
        # 显示前3个案例
        for case in cases[:3]:
            logger.info(f"    - {case['file']}")
            logger.info(f"      置信度={case['confidence']:.2%}, 正确类概率={case['true_prob']:.2%}")
            logger.info(f"      视觉权重={case['visual_weight']:.2%}, 几何权重={case['geo_weight']:.2%}")

    # 分析低置信度正确预测
    low_conf_correct = [s for s in correct_samples if s['confidence'] < 0.8]
    logger.info(f"\n低置信度正确预测 ({len(low_conf_correct)} 个, <80%):")
    for s in low_conf_correct[:5]:
        logger.info(f"  {s['file']}: {s['category']}, 置信度={s['confidence']:.2%}")

    return errors, confusion, attention_stats


if __name__ == "__main__":
    analyze_errors()
