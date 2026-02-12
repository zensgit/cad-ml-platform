#!/usr/bin/env python3
"""
训练部件分类器 V14

目标：达到99%+准确率

改进策略：
1. 更深的几何特征网络（V6的核心优势）
2. 更强的视觉特征提取
3. 多尺度融合
4. 集成学习（多模型投票）
5. 更强的正则化防止过拟合
6. 交叉验证选择最佳模型
"""

import json
import logging
import sys
import io
from pathlib import Path
from typing import Optional, List, Tuple
from collections import Counter
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler, Subset
import torchvision.transforms as transforms
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


class FusionDataset(Dataset):
    """融合数据集"""

    def __init__(self, manifest_path: str, data_dir: str, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.geo_extractor = GeometricFeatureExtractor()

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.3),
            ])
        else:
            self.transform = None

        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)

        self.samples = []
        self.labels = []
        logger.info(f"加载数据集，共 {len(self.manifest)} 个文件...")

        success = 0
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
            success += 1

        logger.info(f"成功加载 {success} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = torch.tensor(sample["image"], dtype=torch.float32).unsqueeze(0)

        if self.augment and self.transform:
            img = self.transform(img)

        geo = torch.tensor(sample["geo_features"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, geo, label


class DeepGeoBranch(nn.Module):
    """更深的几何特征分支 - 模仿V6的成功"""

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
    """多尺度视觉分支"""

    def __init__(self):
        super().__init__()

        # 浅层特征 (大尺度结构)
        self.shallow = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # 中层特征 (中尺度结构)
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

        # 深层特征 (细节)
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

        # 融合
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
    """V14融合模型 - 更强的特征提取和融合"""

    def __init__(self, geo_dim: int, num_classes: int):
        super().__init__()

        # 几何分支 (主分支，权重更大)
        self.geo_branch = DeepGeoBranch(geo_dim, hidden_dim=256)

        # 视觉分支 (辅助)
        self.visual_branch = MultiScaleVisualBranch()

        # 可学习的融合权重
        self.geo_weight = nn.Parameter(torch.tensor(0.7))
        self.visual_weight = nn.Parameter(torch.tensor(0.3))

        # 融合后分类
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

        # 独立的几何分类头（用于辅助损失）
        self.geo_classifier = nn.Linear(128, num_classes)

    def forward(self, img, geo, return_aux=False):
        geo_feat = self.geo_branch(geo)
        visual_feat = self.visual_branch(img)

        # 加权融合
        geo_w = torch.sigmoid(self.geo_weight)
        visual_w = torch.sigmoid(self.visual_weight)

        # 归一化权重
        total = geo_w + visual_w
        geo_w = geo_w / total
        visual_w = visual_w / total

        # 融合
        fused = torch.cat([geo_feat * geo_w, visual_feat * visual_w], dim=1)

        out = self.classifier(fused)

        if return_aux:
            geo_out = self.geo_classifier(geo_feat)
            return out, geo_out, (geo_w.item(), visual_w.item())

        return out


def train_epoch(model, loader, criterion, optimizer, device, aux_weight=0.3):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, geo, labels in loader:
        imgs, geo, labels = imgs.to(device), geo.to(device), labels.to(device)

        optimizer.zero_grad()
        out, geo_out, _ = model(imgs, geo, return_aux=True)

        # 主损失 + 辅助几何损失
        loss_main = criterion(out, labels)
        loss_aux = criterion(geo_out, labels)
        loss = loss_main + aux_weight * loss_aux

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = out.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

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


def k_fold_train(dataset, num_classes, geo_dim, id_to_label, k=5):
    """K折交叉验证训练"""
    from sklearn.model_selection import StratifiedKFold

    labels = np.array(dataset.labels)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_results = []
    best_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"\n{'='*40}")
        logger.info(f"Fold {fold+1}/{k}")
        logger.info(f"{'='*40}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # 类别权重
        train_labels = [labels[i] for i in train_idx]
        class_counts = Counter(train_labels)
        class_weights = {c: 1.0 / count for c, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(train_subset, batch_size=16, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

        model = FusionModelV14(geo_dim, num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        best_val_acc = 0
        best_state = None
        patience = 0

        for epoch in range(150):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1:3d} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                best_state = model.state_dict().copy()
            else:
                patience += 1
                if patience >= 25:
                    break

        fold_results.append(best_val_acc)
        best_models.append(best_state)
        logger.info(f"Fold {fold+1} 最佳: {best_val_acc:.2%}")

    return fold_results, best_models


class EnsembleModel:
    """集成模型"""

    def __init__(self, models: List[nn.Module], device):
        self.models = models
        self.device = device

    def predict(self, img, geo):
        all_probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                out = model(img, geo)
                probs = torch.softmax(out, dim=1)
                all_probs.append(probs)

        # 平均概率
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs.argmax(dim=1)


def main():
    logger.info("=" * 60)
    logger.info("训练融合模型 V14 (目标: 99%+)")
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
    dataset = FusionDataset(str(manifest_path), str(REAL_DATA_DIR), augment=True)

    if len(dataset) == 0:
        logger.error("数据集为空！")
        return

    geo_dim = dataset[0][1].shape[0]
    logger.info(f"样本数: {len(dataset)}, 几何特征维度: {geo_dim}")

    # K折交叉验证
    logger.info("\n开始5折交叉验证训练...")
    fold_results, fold_models = k_fold_train(dataset, num_classes, geo_dim, id_to_label, k=5)

    logger.info("\n" + "=" * 60)
    logger.info("交叉验证结果")
    logger.info("=" * 60)
    for i, acc in enumerate(fold_results):
        logger.info(f"Fold {i+1}: {acc:.2%}")
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    logger.info(f"平均: {mean_acc:.2%} ± {std_acc:.2%}")

    # 选择最佳模型
    best_fold = np.argmax(fold_results)
    best_state = fold_models[best_fold]

    # 创建集成模型
    logger.info("\n创建集成模型...")
    ensemble_models = []
    for state in fold_models:
        model = FusionModelV14(geo_dim, num_classes).to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        ensemble_models.append(model)

    # 在全数据集上评估集成
    dataset.augment = False
    full_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    ensemble = EnsembleModel(ensemble_models, DEVICE)
    correct, total = 0, 0
    per_class_correct = Counter()
    per_class_total = Counter()

    for imgs, geo, labels in full_loader:
        imgs, geo, labels = imgs.to(DEVICE), geo.to(DEVICE), labels.to(DEVICE)
        preds = ensemble.predict(imgs, geo)

        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
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

    # 保存最佳单模型
    save_data = {
        "model_state_dict": best_state,
        "geo_dim": geo_dim,
        "num_classes": num_classes,
        "id_to_label": id_to_label,
        "best_val_acc": fold_results[best_fold],
        "ensemble_acc": ensemble_acc,
        "cv_mean": mean_acc,
        "cv_std": std_acc,
        "version": "v14",
        "categories": list(labels_info["label_to_id"].keys()),
        "img_size": IMG_SIZE,
        "architecture": "FusionModelV14"
    }

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(save_data, MODEL_DIR / "cad_classifier_v14.pt")

    # 保存集成模型
    ensemble_data = {
        "fold_states": fold_models,
        "geo_dim": geo_dim,
        "num_classes": num_classes,
        "id_to_label": id_to_label,
        "ensemble_acc": ensemble_acc,
        "version": "v14_ensemble"
    }
    torch.save(ensemble_data, MODEL_DIR / "cad_classifier_v14_ensemble.pt")

    logger.info(f"\n模型已保存:")
    logger.info(f"  单模型: models/cad_classifier_v14.pt (最佳fold: {fold_results[best_fold]:.2%})")
    logger.info(f"  集成模型: models/cad_classifier_v14_ensemble.pt ({ensemble_acc:.2%})")


if __name__ == "__main__":
    main()
