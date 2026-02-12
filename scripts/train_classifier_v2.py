#!/usr/bin/env python3
"""
改进版训练脚本 - 使用更丰富的图形特征
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/training")
MODEL_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnhancedDXFFeatureExtractor:
    """增强版DXF特征提取器"""

    def extract(self, dxf_path: str) -> Optional[np.ndarray]:
        """提取更丰富的特征向量"""
        try:
            import ezdxf
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            # 实体统计
            entity_types = []
            layer_names = []
            all_points = []
            circle_radii = []
            arc_angles = []
            line_lengths = []
            text_heights = []

            for entity in msp:
                etype = entity.dxftype()
                entity_types.append(etype)

                # 获取图层
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
                        angle = abs(entity.dxf.end_angle - entity.dxf.start_angle)
                        arc_angles.append(angle)

                    elif etype in ["TEXT", "MTEXT"]:
                        if hasattr(entity.dxf, 'insert'):
                            all_points.append((entity.dxf.insert.x, entity.dxf.insert.y))
                        if hasattr(entity.dxf, 'height'):
                            text_heights.append(entity.dxf.height)

                    elif etype in ["LWPOLYLINE", "POLYLINE"]:
                        if hasattr(entity, 'get_points'):
                            pts = list(entity.get_points())
                            for pt in pts:
                                all_points.append((pt[0], pt[1]))

                    elif etype == "INSERT":
                        if hasattr(entity.dxf, 'insert'):
                            all_points.append((entity.dxf.insert.x, entity.dxf.insert.y))

                except Exception:
                    pass

            # 实体类型统计
            type_counts = Counter(entity_types)
            total_entities = len(entity_types)

            # 计算特征
            features = []

            # 1. 实体类型比例 (12个特征)
            for etype in ["LINE", "CIRCLE", "ARC", "LWPOLYLINE", "POLYLINE",
                          "SPLINE", "ELLIPSE", "TEXT", "MTEXT", "DIMENSION",
                          "HATCH", "INSERT"]:
                ratio = type_counts.get(etype, 0) / max(total_entities, 1)
                features.append(ratio)

            # 2. 几何统计特征 (8个特征)
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

            # 圆/弧统计
            if circle_radii:
                features.append(np.log1p(np.mean(circle_radii)) / 5)
                features.append(np.log1p(np.std(circle_radii)) / 5)
            else:
                features.extend([0, 0])

            # 线段长度统计
            if line_lengths:
                features.append(np.log1p(np.mean(line_lengths)) / 5)
                features.append(np.log1p(np.std(line_lengths)) / 5)
            else:
                features.extend([0, 0])

            # 3. 图层特征 (4个特征)
            unique_layers = len(set(layer_names))
            features.append(np.log1p(unique_layers) / 3)

            # 常见图层名称检测
            layer_lower = [l.lower() for l in layer_names]
            features.append(1.0 if any('dim' in l for l in layer_lower) else 0.0)  # 尺寸图层
            features.append(1.0 if any('text' in l for l in layer_lower) else 0.0)  # 文字图层
            features.append(1.0 if any('center' in l for l in layer_lower) else 0.0)  # 中心线图层

            # 4. 复杂度特征 (4个特征)
            # 圆弧比例 (轴承、法兰等圆形零件会有更多圆弧)
            arc_ratio = (type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0)) / max(total_entities, 1)
            features.append(arc_ratio)

            # 文字密度 (有些零件标注更多)
            text_ratio = (type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0)) / max(total_entities, 1)
            features.append(text_ratio)

            # 块引用比例 (组件类通常有更多块引用)
            insert_ratio = type_counts.get("INSERT", 0) / max(total_entities, 1)
            features.append(insert_ratio)

            # 尺寸标注比例
            dim_ratio = type_counts.get("DIMENSION", 0) / max(total_entities, 1)
            features.append(dim_ratio)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.warning(f"提取特征失败 {dxf_path}: {e}")
            return None


class CADDatasetV2(Dataset):
    """改进版CAD图纸数据集"""

    def __init__(self, manifest_path: str, data_dir: str):
        self.data_dir = Path(data_dir)
        self.extractor = EnhancedDXFFeatureExtractor()

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

        logger.info(f"成功加载 {len(self.samples)} 个样本，特征维度: {self.samples[0]['features'].shape[0]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["features"], dtype=torch.float32),
            torch.tensor(sample["label"], dtype=torch.long)
        )


class ImprovedClassifier(nn.Module):
    """改进版分类器"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


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


def main():
    logger.info("=" * 60)
    logger.info("训练改进版2D图分类器")
    logger.info("=" * 60)

    # 加载标签
    labels_path = DATA_DIR / "labels.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_info = json.load(f)

    num_classes = len(labels_info["label_to_id"])
    id_to_label = {int(k): v for k, v in labels_info["id_to_label"].items()}
    logger.info(f"类别数: {num_classes}")
    logger.info(f"类别: {list(labels_info['label_to_id'].keys())}")

    # 创建数据集
    manifest_path = DATA_DIR / "manifest.json"
    dataset = CADDatasetV2(str(manifest_path), str(DATA_DIR))

    if len(dataset) == 0:
        logger.error("数据集为空！")
        return

    # 获取特征维度
    input_dim = dataset[0][0].shape[0]
    logger.info(f"特征维度: {input_dim}")

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"训练集: {train_size}, 验证集: {val_size}")

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 模型
    hidden_dim = 128
    model = ImprovedClassifier(input_dim, hidden_dim, num_classes).to(DEVICE)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 使用类别权重处理不平衡
    class_counts = Counter(s["label"] for s in dataset.samples)
    weights = torch.tensor([1.0 / class_counts.get(i, 1) for i in range(num_classes)], dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 训练
    num_epochs = 200
    best_val_acc = 0
    patience = 30
    patience_counter = 0

    logger.info("\n开始训练...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            MODEL_DIR.mkdir(exist_ok=True)
            model_path = MODEL_DIR / "cad_classifier_v2.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_classes": num_classes,
                "id_to_label": id_to_label,
                "best_val_acc": best_val_acc
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # 最终评估
    checkpoint = torch.load(MODEL_DIR / "cad_classifier_v2.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_acc, preds, labels = evaluate(model, val_loader, criterion, DEVICE)

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info(f"最佳验证准确率: {best_val_acc:.2%}")
    logger.info(f"模型保存至: {MODEL_DIR / 'cad_classifier_v2.pt'}")

    # 打印混淆情况
    logger.info("\n验证集预测结果:")
    for pred, label in zip(preds, labels):
        pred_name = id_to_label[pred]
        label_name = id_to_label[label]
        status = "✓" if pred == label else "✗"
        logger.info(f"  {status} 预测: {pred_name}, 实际: {label_name}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
