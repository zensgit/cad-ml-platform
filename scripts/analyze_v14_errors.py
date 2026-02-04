#!/usr/bin/env python3
"""
分析V14集成模型的错误案例，找出最后5个错误的原因
"""

import json
import logging
import sys
import io
from pathlib import Path
from typing import Optional, List
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
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


class DeepGeoBranch(nn.Module):
    def __init__(self, geo_dim: int, hidden_dim: int = 256):
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


class MultiScaleVisualBranch(nn.Module):
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


class FusionModelV14(nn.Module):
    def __init__(self, geo_dim: int, num_classes: int):
        super().__init__()

        self.geo_branch = DeepGeoBranch(geo_dim, hidden_dim=256)
        self.visual_branch = MultiScaleVisualBranch()

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

    def forward(self, img, geo, return_aux=False):
        geo_feat = self.geo_branch(geo)
        visual_feat = self.visual_branch(img)

        geo_w = torch.sigmoid(self.geo_weight)
        visual_w = torch.sigmoid(self.visual_weight)

        total = geo_w + visual_w
        geo_w = geo_w / total
        visual_w = visual_w / total

        fused = torch.cat([geo_feat * geo_w, visual_feat * visual_w], dim=1)

        out = self.classifier(fused)

        if return_aux:
            geo_out = self.geo_classifier(geo_feat)
            return out, geo_out, (geo_w.item(), visual_w.item())

        return out


def analyze_v14_errors():
    """分析V14集成模型的错误"""
    logger.info("=" * 60)
    logger.info("分析V14集成模型错误案例")
    logger.info("=" * 60)

    # 加载标签
    labels_path = REAL_DATA_DIR / "labels.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_info = json.load(f)

    id_to_label = {int(k): v for k, v in labels_info["id_to_label"].items()}

    # 加载集成模型
    ensemble_data = torch.load(MODEL_DIR / "cad_classifier_v14_ensemble.pt", map_location=DEVICE, weights_only=False)
    fold_states = ensemble_data["fold_states"]
    geo_dim = ensemble_data["geo_dim"]
    num_classes = ensemble_data["num_classes"]

    models = []
    for state in fold_states:
        model = FusionModelV14(geo_dim, num_classes).to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    # 加载数据
    manifest_path = REAL_DATA_DIR / "manifest.json"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    geo_extractor = GeometricFeatureExtractor()

    errors = []
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

        # 集成预测
        all_probs = []
        individual_preds = []
        for model in models:
            with torch.no_grad():
                out = model(img_tensor, geo_tensor)
                probs = torch.softmax(out, dim=1)
                all_probs.append(probs)
                individual_preds.append(out.argmax(1).item())

        avg_probs = torch.stack(all_probs).mean(dim=0)
        pred_label = avg_probs.argmax(1).item()
        confidence = avg_probs[0, pred_label].item()

        if pred_label != true_label:
            errors.append({
                "file": item["file"],
                "true": id_to_label[true_label],
                "pred": id_to_label[pred_label],
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "true_prob": avg_probs[0, true_label].item(),
                "all_probs": {id_to_label[j]: avg_probs[0, j].item() for j in range(num_classes)},
                "individual_preds": [id_to_label[p] for p in individual_preds],
                "geo_features": geo.tolist()
            })

    logger.info(f"\n错误总数: {len(errors)}")
    logger.info("\n详细错误分析:")
    logger.info("=" * 60)

    for i, e in enumerate(errors):
        logger.info(f"\n错误 {i+1}:")
        logger.info(f"  文件: {e['file']}")
        logger.info(f"  真实类别: {e['true']}")
        logger.info(f"  预测类别: {e['pred']}")
        logger.info(f"  预测置信度: {e['confidence']:.2%}")
        logger.info(f"  正确类概率: {e['true_prob']:.2%}")
        logger.info(f"  各类别概率:")
        for cat, prob in sorted(e['all_probs'].items(), key=lambda x: -x[1]):
            logger.info(f"    {cat}: {prob:.2%}")
        logger.info(f"  5个模型的预测: {e['individual_preds']}")

        # 分析几何特征
        geo = e['geo_features']
        logger.info(f"  几何特征摘要:")
        logger.info(f"    LINE比例: {geo[0]:.3f}")
        logger.info(f"    CIRCLE比例: {geo[1]:.3f}")
        logger.info(f"    ARC比例: {geo[2]:.3f}")
        logger.info(f"    实体总数(log): {geo[12]:.3f}")
        logger.info(f"    宽高比: {geo[15]:.3f}")

    # 保存错误案例
    with open("claudedocs/v14_errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    logger.info(f"\n错误详情已保存至: claudedocs/v14_errors.json")

    return errors


if __name__ == "__main__":
    analyze_v14_errors()
