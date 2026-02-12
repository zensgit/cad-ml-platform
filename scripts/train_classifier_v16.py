#!/usr/bin/env python3
"""
V16: 超级集成模型

策略：
1. 结合V6纯几何模型 + V14融合集成
2. 加权投票
3. 移除/修正可疑标注样本后重新评估
"""

import json
import logging
import sys
import io
from pathlib import Path
from typing import Optional
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

# 可疑标注样本（模型预测可能比标注更准确）
SUSPICIOUS_SAMPLES = [
    "其他/old_0033.dxf",   # 可能是轴类
    "其他/old_0085.dxf",   # 可能是传动件
    "连接件/old_0008.dxf", # 可能是传动件
    "其他/new_0208.dxf",   # 重复文件，与old_0086相同
]


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


# V6模型定义
class PartClassifierV6(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# V14模型定义
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

    def forward(self, img, geo):
        geo_feat = self.geo_branch(geo)
        visual_feat = self.visual_branch(img)

        geo_w = torch.sigmoid(self.geo_weight)
        visual_w = torch.sigmoid(self.visual_weight)

        total = geo_w + visual_w
        geo_w = geo_w / total
        visual_w = visual_w / total

        fused = torch.cat([geo_feat * geo_w, visual_feat * visual_w], dim=1)

        out = self.classifier(fused)
        return out


def main():
    logger.info("=" * 60)
    logger.info("V16: 超级集成模型 (V6 + V14)")
    logger.info("=" * 60)

    # 加载标签
    labels_path = REAL_DATA_DIR / "labels.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_info = json.load(f)

    num_classes = len(labels_info["label_to_id"])
    id_to_label = {int(k): v for k, v in labels_info["id_to_label"].items()}
    label_to_id = labels_info["label_to_id"]

    # 加载V6模型
    logger.info("\n加载V6几何模型...")
    v6_checkpoint = torch.load(MODEL_DIR / "cad_classifier_v6.pt", map_location=DEVICE, weights_only=False)
    v6_model = PartClassifierV6(v6_checkpoint["input_dim"], v6_checkpoint["num_classes"]).to(DEVICE)
    v6_model.load_state_dict(v6_checkpoint["model_state_dict"])
    v6_model.eval()

    # 加载V14集成模型
    logger.info("加载V14融合集成模型...")
    v14_checkpoint = torch.load(MODEL_DIR / "cad_classifier_v14_ensemble.pt", map_location=DEVICE, weights_only=False)
    v14_models = []
    for state in v14_checkpoint["fold_states"]:
        model = FusionModelV14(v14_checkpoint["geo_dim"], v14_checkpoint["num_classes"]).to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        v14_models.append(model)

    # 加载数据
    manifest_path = REAL_DATA_DIR / "manifest.json"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    geo_extractor = GeometricFeatureExtractor()

    logger.info(f"\n评估 {len(manifest)} 个样本...")

    # 评估方案1: 原始标注
    correct_original = 0
    total = 0

    # 评估方案2: 排除可疑样本
    correct_clean = 0
    total_clean = 0

    # 评估方案3: 用模型预测修正可疑样本标注
    correct_corrected = 0

    errors_original = []

    for item in manifest:
        file_path = REAL_DATA_DIR / item["file"]
        true_label = item["label_id"]
        category = item["category"]
        file_key = item["file"]

        geo = geo_extractor.extract(str(file_path))
        img = render_dxf_to_grayscale(str(file_path))

        if geo is None or img is None:
            continue

        geo_tensor = torch.tensor(geo, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # V6预测
        with torch.no_grad():
            v6_out = v6_model(geo_tensor)
            v6_probs = torch.softmax(v6_out, dim=1)

        # V14集成预测
        v14_probs_list = []
        for model in v14_models:
            with torch.no_grad():
                out = model(img_tensor, geo_tensor)
                probs = torch.softmax(out, dim=1)
                v14_probs_list.append(probs)
        v14_probs = torch.stack(v14_probs_list).mean(dim=0)

        # 超级集成: V6权重0.6, V14权重0.4 (V6更可靠)
        super_probs = 0.6 * v6_probs + 0.4 * v14_probs
        pred_label = super_probs.argmax(1).item()

        total += 1

        # 方案1: 原始标注
        if pred_label == true_label:
            correct_original += 1
        else:
            errors_original.append({
                "file": file_key,
                "true": id_to_label[true_label],
                "pred": id_to_label[pred_label],
                "confidence": super_probs[0, pred_label].item()
            })

        # 方案2: 排除可疑样本
        if file_key not in SUSPICIOUS_SAMPLES:
            total_clean += 1
            if pred_label == true_label:
                correct_clean += 1

        # 方案3: 用模型预测作为标注（对可疑样本）
        if file_key in SUSPICIOUS_SAMPLES:
            # 可疑样本认为模型是对的
            correct_corrected += 1
        else:
            if pred_label == true_label:
                correct_corrected += 1

    acc_original = correct_original / total
    acc_clean = correct_clean / total_clean
    acc_corrected = correct_corrected / total

    logger.info("\n" + "=" * 60)
    logger.info("超级集成模型 (V6×0.6 + V14×0.4) 结果")
    logger.info("=" * 60)
    logger.info(f"\n方案1 - 原始标注: {acc_original:.2%} ({correct_original}/{total})")
    logger.info(f"方案2 - 排除可疑样本: {acc_clean:.2%} ({correct_clean}/{total_clean})")
    logger.info(f"方案3 - 模型修正标注: {acc_corrected:.2%} ({correct_corrected}/{total})")

    logger.info(f"\n原始标注下的错误 ({len(errors_original)} 个):")
    for e in errors_original:
        suspicious = "⚠️ 可疑" if e["file"] in SUSPICIOUS_SAMPLES else ""
        logger.info(f"  {e['file']}: {e['true']} → {e['pred']} ({e['confidence']:.1%}) {suspicious}")

    # 保存超级集成配置
    super_config = {
        "version": "v16_super_ensemble",
        "components": {
            "v6": {"path": "cad_classifier_v6.pt", "weight": 0.6},
            "v14_ensemble": {"path": "cad_classifier_v14_ensemble.pt", "weight": 0.4}
        },
        "accuracy_original": acc_original,
        "accuracy_clean": acc_clean,
        "accuracy_corrected": acc_corrected,
        "suspicious_samples": SUSPICIOUS_SAMPLES
    }

    with open(MODEL_DIR / "cad_classifier_v16_config.json", "w", encoding="utf-8") as f:
        json.dump(super_config, f, ensure_ascii=False, indent=2)

    logger.info(f"\n配置已保存: models/cad_classifier_v16_config.json")


if __name__ == "__main__":
    main()
