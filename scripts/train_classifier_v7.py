#!/usr/bin/env python3
"""
训练部件分类器 V7

基于V6架构，使用更大数据集 (905样本)
改进:
- 更大数据集 (295 -> 905)
- 自动学习率调整
- 更强数据增强
- 模型集成 (多次训练取最佳)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional
from collections import Counter
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/training_v7")
MODEL_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class EnhancedFeatureExtractorV4:
    """增强版特征提取器 V4 - 48维特征"""

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
            text_heights = []
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
                        if hasattr(entity.dxf, 'height'):
                            text_heights.append(entity.dxf.height)
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

            # 1. 实体类型比例 (12)
            for etype in ["LINE", "CIRCLE", "ARC", "LWPOLYLINE", "POLYLINE",
                          "SPLINE", "ELLIPSE", "TEXT", "MTEXT", "DIMENSION",
                          "HATCH", "INSERT"]:
                ratio = type_counts.get(etype, 0) / max(total_entities, 1)
                features.append(ratio)

            # 2. 基础几何 (4)
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

            # 3. 圆/弧统计 (6)
            if circle_radii:
                features.append(np.log1p(np.mean(circle_radii)) / 5)
                features.append(np.log1p(np.std(circle_radii)) / 5 if len(circle_radii) > 1 else 0)
                features.append(len(circle_radii) / max(total_entities, 1))
            else:
                features.extend([0, 0, 0])

            if arc_radii:
                features.append(np.log1p(np.mean(arc_radii)) / 5)
                features.append(np.mean(arc_angles) / 180 if arc_angles else 0)
                features.append(len(arc_radii) / max(total_entities, 1))
            else:
                features.extend([0, 0, 0])

            # 4. 线段统计 (4)
            if line_lengths:
                features.append(np.log1p(np.mean(line_lengths)) / 5)
                features.append(np.log1p(np.std(line_lengths)) / 5 if len(line_lengths) > 1 else 0)
                features.append(np.log1p(np.max(line_lengths)) / 5)
                features.append(np.log1p(np.min(line_lengths)) / 5)
            else:
                features.extend([0, 0, 0, 0])

            # 5. 图层特征 (6)
            unique_layers = len(set(layer_names))
            features.append(np.log1p(unique_layers) / 3)
            layer_lower = [l.lower() for l in layer_names]
            features.append(1.0 if any('dim' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('text' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('center' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('hidden' in l or 'hid' in l for l in layer_lower) else 0.0)
            features.append(1.0 if any('section' in l or 'cut' in l for l in layer_lower) else 0.0)

            # 6. 复杂度特征 (4)
            arc_circle_ratio = (type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0)) / max(total_entities, 1)
            features.append(arc_circle_ratio)
            text_ratio = (type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0)) / max(total_entities, 1)
            features.append(text_ratio)
            insert_ratio = type_counts.get("INSERT", 0) / max(total_entities, 1)
            features.append(insert_ratio)
            dim_ratio = dimension_count / max(total_entities, 1)
            features.append(dim_ratio)

            # 7. 空间分布 (4)
            if all_points and len(all_points) > 1:
                xs = np.array([p[0] for p in all_points])
                ys = np.array([p[1] for p in all_points])
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                density = len(all_points) / max(area, 0.001)
                features.append(np.log1p(density) / 10)
                features.append(np.std(xs) / max(max(xs) - min(xs), 0.001))
                features.append(np.std(ys) / max(max(ys) - min(ys), 0.001))
                center_x = (max(xs) + min(xs)) / 2
                center_y = (max(ys) + min(ys)) / 2
                mean_x = np.mean(xs)
                mean_y = np.mean(ys)
                center_offset = np.sqrt((mean_x - center_x)**2 + (mean_y - center_y)**2)
                features.append(np.log1p(center_offset) / 5)
            else:
                features.extend([0, 0.5, 0.5, 0])

            # 8. 形状复杂度 (4)
            if polyline_vertex_counts:
                features.append(np.log1p(np.mean(polyline_vertex_counts)) / 3)
                features.append(np.log1p(np.max(polyline_vertex_counts)) / 4)
            else:
                features.extend([0, 0])
            features.append(hatch_count / max(total_entities, 1))
            unique_blocks = len(set(block_names))
            features.append(np.log1p(unique_blocks) / 3)

            # 9. 类型特征 (4)
            curved = type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0) + type_counts.get("ELLIPSE", 0) + type_counts.get("SPLINE", 0)
            straight = type_counts.get("LINE", 0)
            curve_straight_ratio = curved / max(straight, 1)
            features.append(np.clip(curve_straight_ratio, 0, 5) / 5)
            annotation = type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0) + dimension_count
            features.append(annotation / max(total_entities, 1))
            geometry = straight + curved + type_counts.get("LWPOLYLINE", 0) + type_counts.get("POLYLINE", 0)
            geo_annotation_ratio = geometry / max(annotation, 1)
            features.append(np.clip(geo_annotation_ratio, 0, 20) / 20)
            symmetry_indicator = len(circle_radii) / max(total_entities, 1)
            features.append(symmetry_indicator)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            return None


class CADDatasetV7(Dataset):
    def __init__(self, manifest_path: str, data_dir: str, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.extractor = EnhancedFeatureExtractorV4()
        self.augment = augment

        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)

        self.samples = []
        logger.info(f"加载数据集，共 {len(self.manifest)} 个文件...")

        for item in self.manifest:
            file_path = self.data_dir / item["file"]
            features = self.extractor.extract(str(file_path))
            if features is not None:
                self.samples.append({
                    "features": features,
                    "label": item["label_id"],
                    "category": item["category"],
                    "file": item["file"]
                })

        if self.samples:
            logger.info(f"成功加载 {len(self.samples)} 个样本，特征维度: {self.samples[0]['features'].shape[0]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = sample["features"].copy()
        
        # 数据增强
        if self.augment:
            # 随机噪声
            noise = np.random.normal(0, 0.02, features.shape).astype(np.float32)
            features = features + noise
            # 随机缩放
            scale = np.random.uniform(0.95, 1.05)
            features = features * scale
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(sample["label"], dtype=torch.long)
        )


class ImprovedClassifierV7(nn.Module):
    """V7分类器 - 更深更宽"""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.6),
            
            # Layer 4
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.4),
            
            # Output
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes: int, smoothing: float = 0.1, weight=None):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.weight = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        loss = -true_dist * pred
        if self.weight is not None:
            weight = self.weight[target]
            loss = loss.sum(dim=-1) * weight
            return loss.mean()
        return loss.sum(dim=-1).mean()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


def oversample_minority(dataset, target_per_class: int):
    """过采样少数类"""
    class_samples = {}
    for i, sample in enumerate(dataset.samples):
        label = sample["label"]
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(sample)
    
    augmented = []
    for label, samples in class_samples.items():
        augmented.extend(samples)
        if len(samples) < target_per_class:
            need = target_per_class - len(samples)
            for _ in range(need):
                base = random.choice(samples)
                aug_sample = {
                    "features": base["features"] + np.random.normal(0, 0.03, base["features"].shape).astype(np.float32),
                    "label": base["label"],
                    "category": base["category"],
                    "file": base["file"] + "_aug"
                }
                augmented.append(aug_sample)
    
    dataset.samples = augmented
    return dataset


def main():
    logger.info("=" * 60)
    logger.info("训练部件分类器 V7")
    logger.info(f"设备: {DEVICE}")
    logger.info("=" * 60)

    # 加载标签
    labels_path = DATA_DIR / "labels.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_info = json.load(f)

    num_classes = len(labels_info["label_to_id"])
    id_to_label = {int(k): v for k, v in labels_info["id_to_label"].items()}
    logger.info(f"类别数: {num_classes}")
    logger.info(f"类别: {list(labels_info['label_to_id'].keys())}")

    # 加载数据
    manifest_path = DATA_DIR / "manifest.json"
    dataset = CADDatasetV7(str(manifest_path), str(DATA_DIR), augment=False)

    if len(dataset) == 0:
        logger.error("数据集为空！")
        return

    input_dim = dataset[0][0].shape[0]
    logger.info(f"原始样本数: {len(dataset)}")
    logger.info(f"特征维度: {input_dim}")

    # 过采样
    class_counts = Counter(s["label"] for s in dataset.samples)
    max_count = max(class_counts.values())
    target_per_class = min(max_count, 300)  # 最多300个/类
    
    logger.info(f"\n过采样目标: {target_per_class}/类")
    dataset = oversample_minority(dataset, target_per_class)
    logger.info(f"过采样后: {len(dataset.samples)} 个样本")

    # 划分数据集
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"训练集: {train_size}, 验证集: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 多次训练，选择最佳
    num_runs = 5
    best_overall_acc = 0
    best_model_state = None

    for run in range(num_runs):
        logger.info(f"\n{'='*40}")
        logger.info(f"训练轮次 {run+1}/{num_runs}")
        logger.info(f"{'='*40}")

        # 创建模型
        hidden_dim = 256
        model = ImprovedClassifierV7(input_dim, hidden_dim, num_classes, dropout=0.5).to(DEVICE)

        # 类别权重
        class_counts = Counter(s["label"] for s in dataset.samples)
        weights = torch.tensor([1.0 / class_counts.get(i, 1) for i in range(num_classes)], dtype=torch.float32)
        weights = weights / weights.sum() * num_classes
        criterion = LabelSmoothingLoss(num_classes, smoothing=0.1, weight=weights.to(DEVICE))

        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=200, steps_per_epoch=len(train_loader)
        )

        best_val_acc = 0
        patience = 30
        patience_counter = 0

        for epoch in range(200):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1:3d} | "
                    f"Train: {train_loss:.4f}/{train_acc:.2%} | "
                    f"Val: {val_loss:.4f}/{val_acc:.2%}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                if val_acc > best_overall_acc:
                    best_overall_acc = val_acc
                    best_model_state = {
                        "model_state_dict": model.state_dict(),
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dim,
                        "num_classes": num_classes,
                        "id_to_label": id_to_label,
                        "best_val_acc": val_acc,
                        "version": "v7",
                        "categories": list(labels_info["label_to_id"].keys())
                    }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        logger.info(f"Run {run+1} 最佳验证准确率: {best_val_acc:.2%}")

    # 保存最佳模型
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "cad_classifier_v7.pt"
    torch.save(best_model_state, model_path)

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info(f"最佳验证准确率: {best_overall_acc:.2%}")
    logger.info(f"模型保存至: {model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
