#!/usr/bin/env python3
"""
训练部件分类器 V6

优化策略：
1. 数据增强 - 特征扰动
2. 更强的正则化
3. Label Smoothing
4. 多次训练取最佳
5. 更大的隐藏层
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

DATA_DIR = Path("data/training_v5")
MODEL_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EnhancedFeatureExtractorV4:
    """特征提取器 (48维)"""

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
            return None


class AugmentedDataset(Dataset):
    """带数据增强的数据集"""

    def __init__(self, manifest_path: str, data_dir: str, augment: bool = True, aug_factor: int = 3):
        self.data_dir = Path(data_dir)
        self.extractor = EnhancedFeatureExtractorV4()
        self.augment = augment
        self.aug_factor = aug_factor

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
                })

        # 数据增强 - 对小类过采样
        if augment:
            self._oversample_minority_classes()

        if self.samples:
            logger.info(f"加载 {len(self.samples)} 个样本 (含增强)，特征维度: {self.samples[0]['features'].shape[0]}")

    def _oversample_minority_classes(self):
        """对小类进行过采样"""
        class_counts = Counter(s["label"] for s in self.samples)
        max_count = max(class_counts.values())

        augmented = []
        for sample in self.samples:
            label = sample["label"]
            # 计算需要复制的次数
            current_count = class_counts[label]
            oversample_ratio = max_count // current_count

            # 添加原始样本
            augmented.append(sample)

            # 添加扰动样本
            for _ in range(min(oversample_ratio - 1, self.aug_factor)):
                aug_features = self._augment_features(sample["features"])
                augmented.append({
                    "features": aug_features,
                    "label": sample["label"],
                    "category": sample["category"],
                })

        self.samples = augmented

    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """特征扰动增强"""
        noise = np.random.normal(0, 0.02, features.shape).astype(np.float32)
        aug_features = features + noise
        # 裁剪到合理范围
        aug_features = np.clip(aug_features, 0, 2)
        return aug_features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = sample["features"]

        # 训练时添加少量噪声
        if self.augment and np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, features.shape).astype(np.float32)
            features = features + noise
            features = np.clip(features, 0, 2)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(sample["label"], dtype=torch.long)
        )


class ImprovedClassifierV6(nn.Module):
    """改进版分类器 V6 - 更宽更深"""

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


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Loss"""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)

        # 创建平滑标签
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = -true_dist * log_preds

        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)

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

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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


def train_single_run(seed, labels_info, num_classes, id_to_label):
    """单次训练"""
    set_seed(seed)

    # 创建数据集
    manifest_path = DATA_DIR / "manifest.json"
    dataset = AugmentedDataset(str(manifest_path), str(DATA_DIR), augment=True, aug_factor=2)

    if len(dataset) == 0:
        return 0, None

    input_dim = dataset[0][0].shape[0]

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 模型
    hidden_dim = 256
    model = ImprovedClassifierV6(input_dim, hidden_dim, num_classes).to(DEVICE)

    # 类别权重
    class_counts = Counter(s["label"] for s in dataset.samples)
    weights = torch.tensor([1.0 / np.sqrt(class_counts.get(i, 1)) for i in range(num_classes)], dtype=torch.float32)
    weights = weights / weights.sum() * num_classes

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=weights.to(DEVICE))

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=200,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )

    # 训练
    best_val_acc = 0
    patience = 40
    patience_counter = 0
    best_state = None

    for epoch in range(200):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_acc, best_state


def main():
    logger.info("=" * 60)
    logger.info("训练部件分类器 V6 (优化版)")
    logger.info("=" * 60)

    labels_path = DATA_DIR / "labels.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_info = json.load(f)

    num_classes = len(labels_info["label_to_id"])
    id_to_label = {int(k): v for k, v in labels_info["id_to_label"].items()}
    logger.info(f"类别数: {num_classes}")
    logger.info(f"类别: {list(labels_info['label_to_id'].keys())}")

    # 多次训练取最佳
    num_runs = 5
    best_overall_acc = 0
    best_overall_state = None

    logger.info(f"\n进行 {num_runs} 次训练，取最佳...")
    for run in range(num_runs):
        seed = 42 + run * 10
        acc, state = train_single_run(seed, labels_info, num_classes, id_to_label)
        logger.info(f"Run {run+1}/{num_runs}: 准确率 = {acc:.2%}")

        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_state = state

    # 保存最佳模型
    if best_overall_state:
        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / "cad_classifier_v6.pt"

        # 重建模型获取参数
        dataset = AugmentedDataset(str(DATA_DIR / "manifest.json"), str(DATA_DIR), augment=False)
        input_dim = dataset[0][0].shape[0]

        torch.save({
            "model_state_dict": best_overall_state,
            "input_dim": input_dim,
            "hidden_dim": 256,
            "num_classes": num_classes,
            "id_to_label": id_to_label,
            "best_val_acc": best_overall_acc,
            "version": "v6",
            "categories": list(labels_info["label_to_id"].keys())
        }, model_path)

        logger.info(f"\n最佳模型准确率: {best_overall_acc:.2%}")
        logger.info(f"模型保存至: {model_path}")

    # 最终评估
    logger.info("\n" + "=" * 60)
    logger.info(f"训练完成！最佳验证准确率: {best_overall_acc:.2%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
