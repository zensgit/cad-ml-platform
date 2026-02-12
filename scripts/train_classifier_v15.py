#!/usr/bin/env python3
"""
训练部件分类器 V15

改进策略：
1. 10折交叉验证 + 更多模型集成
2. 标签平滑减少过拟合
3. Mixup数据增强
4. 更强的正则化
5. 置信度校准
"""

import json
import logging
import sys
import io
from pathlib import Path
from typing import Optional, List
from collections import Counter
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
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


class FusionDataset(Dataset):
    def __init__(self, manifest_path: str, data_dir: str):
        self.data_dir = Path(data_dir)
        self.geo_extractor = GeometricFeatureExtractor()

        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)

        self.samples = []
        self.labels = []
        logger.info(f"加载数据集，共 {len(self.manifest)} 个文件...")

        for i, item in enumerate(self.manifest):
            if (i + 1) % 200 == 0:
                logger.info(f"  处理: {i+1}/{len(self.manifest)}")

            file_path = self.data_dir / item["file"]

            img = render_dxf_to_grayscale(str(file_path))
            if img is None:
                continue

            geo = self.geo_extractor.extract(str(file_path))
            if geo is None:
                continue

            self.samples.append({
                "image": img,
                "geo_features": geo,
                "label": item["label_id"],
                "category": item["category"],
                "file": item["file"]
            })
            self.labels.append(item["label_id"])

        logger.info(f"成功加载 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = torch.tensor(sample["image"], dtype=torch.float32).unsqueeze(0)
        geo = torch.tensor(sample["geo_features"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, geo, label


class DeepGeoBranch(nn.Module):
    def __init__(self, geo_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(geo_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
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


class FusionModelV15(nn.Module):
    def __init__(self, geo_dim: int, num_classes: int):
        super().__init__()

        self.geo_branch = DeepGeoBranch(geo_dim, hidden_dim=256)
        self.visual_branch = MultiScaleVisualBranch()

        self.geo_weight = nn.Parameter(torch.tensor(0.75))
        self.visual_weight = nn.Parameter(torch.tensor(0.25))

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
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
            return out, geo_out

        return out


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)

        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_target = one_hot * confidence + (1 - one_hot) * smooth_value

        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(smooth_target * log_prob).sum(dim=1).mean()
        return loss


def mixup_data(img, geo, labels, alpha=0.2):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = img.size(0)
    index = torch.randperm(batch_size).to(img.device)

    mixed_img = lam * img + (1 - lam) * img[index]
    mixed_geo = lam * geo + (1 - lam) * geo[index]
    labels_a, labels_b = labels, labels[index]

    return mixed_img, mixed_geo, labels_a, labels_b, lam


def train_epoch_mixup(model, loader, criterion, optimizer, device, mixup_alpha=0.2):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, geo, labels in loader:
        imgs, geo, labels = imgs.to(device), geo.to(device), labels.to(device)

        # Mixup
        mixed_img, mixed_geo, labels_a, labels_b, lam = mixup_data(imgs, geo, labels, mixup_alpha)

        optimizer.zero_grad()
        out, geo_out = model(mixed_img, mixed_geo, return_aux=True)

        # Mixup损失
        loss_main = lam * criterion(out, labels_a) + (1 - lam) * criterion(out, labels_b)
        loss_aux = lam * criterion(geo_out, labels_a) + (1 - lam) * criterion(geo_out, labels_b)
        loss = loss_main + 0.3 * loss_aux

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = out.max(1)
        total += labels.size(0)
        correct += (lam * predicted.eq(labels_a).float() + (1 - lam) * predicted.eq(labels_b).float()).sum().item()

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, geo, labels in loader:
            imgs, geo, labels = imgs.to(device), geo.to(device), labels.to(device)
            outputs = model(imgs, geo)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total


def main():
    logger.info("=" * 60)
    logger.info("训练融合模型 V15 (目标: 99.9%)")
    logger.info(f"设备: {DEVICE}")
    logger.info("=" * 60)

    # 加载标签
    labels_path = REAL_DATA_DIR / "labels.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_info = json.load(f)

    num_classes = len(labels_info["label_to_id"])
    id_to_label = {int(k): v for k, v in labels_info["id_to_label"].items()}
    logger.info(f"类别: {list(labels_info['label_to_id'].keys())}")

    # 加载数据
    manifest_path = REAL_DATA_DIR / "manifest.json"
    dataset = FusionDataset(str(manifest_path), str(REAL_DATA_DIR))

    if len(dataset) == 0:
        logger.error("数据集为空！")
        return

    geo_dim = dataset[0][1].shape[0]
    logger.info(f"样本数: {len(dataset)}, 几何特征维度: {geo_dim}")

    # 10折交叉验证
    from sklearn.model_selection import StratifiedKFold

    labels = np.array(dataset.labels)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    fold_results = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"\n{'='*40}")
        logger.info(f"Fold {fold+1}/10")
        logger.info(f"{'='*40}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_labels = [labels[i] for i in train_idx]
        class_counts = Counter(train_labels)
        class_weights = {c: 1.0 / count for c, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(train_subset, batch_size=16, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

        model = FusionModelV15(geo_dim, num_classes).to(DEVICE)
        criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)
        ce_criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

        best_val_acc = 0
        best_state = None
        patience = 0

        for epoch in range(200):
            train_loss, train_acc = train_epoch_mixup(model, train_loader, criterion, optimizer, DEVICE, mixup_alpha=0.2)
            val_loss, val_acc = evaluate(model, val_loader, ce_criterion, DEVICE)
            scheduler.step()

            if (epoch + 1) % 25 == 0:
                logger.info(f"Epoch {epoch+1:3d} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                best_state = model.state_dict().copy()
            else:
                patience += 1
                if patience >= 30:
                    break

        fold_results.append(best_val_acc)
        fold_models.append(best_state)
        logger.info(f"Fold {fold+1} 最佳: {best_val_acc:.2%}")

    logger.info("\n" + "=" * 60)
    logger.info("交叉验证结果")
    logger.info("=" * 60)
    for i, acc in enumerate(fold_results):
        logger.info(f"Fold {i+1}: {acc:.2%}")
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    logger.info(f"平均: {mean_acc:.2%} ± {std_acc:.2%}")

    # 集成评估
    logger.info("\n创建集成模型...")
    ensemble_models = []
    for state in fold_models:
        model = FusionModelV15(geo_dim, num_classes).to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        ensemble_models.append(model)

    full_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    correct, total = 0, 0
    per_class_correct = Counter()
    per_class_total = Counter()

    for imgs, geo, labels_batch in full_loader:
        imgs, geo, labels_batch = imgs.to(DEVICE), geo.to(DEVICE), labels_batch.to(DEVICE)

        all_probs = []
        for model in ensemble_models:
            with torch.no_grad():
                out = model(imgs, geo)
                probs = torch.softmax(out, dim=1)
                all_probs.append(probs)

        avg_probs = torch.stack(all_probs).mean(dim=0)
        preds = avg_probs.argmax(dim=1)

        total += labels_batch.size(0)
        correct += preds.eq(labels_batch).sum().item()

        for p, l in zip(preds.cpu().numpy(), labels_batch.cpu().numpy()):
            per_class_total[l] += 1
            if p == l:
                per_class_correct[l] += 1

    ensemble_acc = correct / total
    logger.info(f"\n集成模型全数据集准确率: {ensemble_acc:.2%}")

    logger.info("\n各类别准确率:")
    for label_id in sorted(per_class_total.keys()):
        cat_name = id_to_label[label_id]
        acc = per_class_correct[label_id] / per_class_total[label_id]
        logger.info(f"  {cat_name}: {acc:.2%} ({per_class_correct[label_id]}/{per_class_total[label_id]})")

    # 保存
    ensemble_data = {
        "fold_states": fold_models,
        "geo_dim": geo_dim,
        "num_classes": num_classes,
        "id_to_label": id_to_label,
        "ensemble_acc": ensemble_acc,
        "cv_mean": mean_acc,
        "cv_std": std_acc,
        "version": "v15"
    }
    torch.save(ensemble_data, MODEL_DIR / "cad_classifier_v15_ensemble.pt")

    logger.info(f"\n模型已保存: models/cad_classifier_v15_ensemble.pt ({ensemble_acc:.2%})")


if __name__ == "__main__":
    main()
