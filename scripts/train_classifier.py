#!/usr/bin/env python3
"""
训练2D图分类器

基于DXF图纸的几何特征训练分类模型
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cad.geometry import GeometryExtractor, BoundingBox

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置
DATA_DIR = Path("data/training")
MODEL_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DXFFeatureExtractor:
    """从DXF文件提取特征向量"""

    def __init__(self):
        self.extractor = GeometryExtractor()

    def extract(self, dxf_path: str) -> Optional[np.ndarray]:
        """提取DXF文件的特征向量"""
        try:
            import ezdxf
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            # 统计各类实体
            entity_counts = {
                "LINE": 0,
                "CIRCLE": 0,
                "ARC": 0,
                "LWPOLYLINE": 0,
                "POLYLINE": 0,
                "SPLINE": 0,
                "ELLIPSE": 0,
                "TEXT": 0,
                "MTEXT": 0,
                "DIMENSION": 0,
                "HATCH": 0,
                "INSERT": 0,
            }

            all_points = []

            for entity in msp:
                etype = entity.dxftype()
                if etype in entity_counts:
                    entity_counts[etype] += 1

                # 收集点坐标计算边界框
                try:
                    if hasattr(entity, 'dxf'):
                        if hasattr(entity.dxf, 'start'):
                            all_points.append((entity.dxf.start.x, entity.dxf.start.y))
                        if hasattr(entity.dxf, 'end'):
                            all_points.append((entity.dxf.end.x, entity.dxf.end.y))
                        if hasattr(entity.dxf, 'center'):
                            all_points.append((entity.dxf.center.x, entity.dxf.center.y))
                        if hasattr(entity.dxf, 'insert'):
                            all_points.append((entity.dxf.insert.x, entity.dxf.insert.y))
                except:
                    pass

            # 计算边界框特征
            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                width = max(xs) - min(xs) if xs else 0
                height = max(ys) - min(ys) if ys else 0
                aspect_ratio = width / height if height > 0 else 1.0
            else:
                width, height, aspect_ratio = 0, 0, 1.0

            # 总实体数
            total_entities = sum(entity_counts.values())

            # 构建特征向量
            features = [
                # 实体计数 (归一化)
                entity_counts["LINE"] / max(total_entities, 1),
                entity_counts["CIRCLE"] / max(total_entities, 1),
                entity_counts["ARC"] / max(total_entities, 1),
                entity_counts["LWPOLYLINE"] / max(total_entities, 1),
                entity_counts["POLYLINE"] / max(total_entities, 1),
                entity_counts["SPLINE"] / max(total_entities, 1),
                entity_counts["ELLIPSE"] / max(total_entities, 1),
                entity_counts["TEXT"] / max(total_entities, 1),
                entity_counts["MTEXT"] / max(total_entities, 1),
                entity_counts["DIMENSION"] / max(total_entities, 1),
                entity_counts["HATCH"] / max(total_entities, 1),
                entity_counts["INSERT"] / max(total_entities, 1),
                # 几何特征
                np.log1p(total_entities) / 10,  # 归一化的总实体数
                np.log1p(width) / 10,  # 归一化的宽度
                np.log1p(height) / 10,  # 归一化的高度
                np.clip(aspect_ratio, 0, 10) / 10,  # 归一化的宽高比
            ]

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.warning(f"提取特征失败 {dxf_path}: {e}")
            return None


class CADDataset(Dataset):
    """CAD图纸数据集"""

    def __init__(self, manifest_path: str, data_dir: str):
        self.data_dir = Path(data_dir)
        self.extractor = DXFFeatureExtractor()

        # 加载清单
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)

        # 预提取特征
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

        logger.info(f"成功加载 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["features"], dtype=torch.float32),
            torch.tensor(sample["label"], dtype=torch.long)
        )


class SimpleClassifier(nn.Module):
    """简单的MLP分类器"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
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
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total


def main():
    logger.info("=" * 60)
    logger.info("开始训练2D图分类器")
    logger.info("=" * 60)

    # 加载标签映射
    labels_path = DATA_DIR / "labels.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_info = json.load(f)

    num_classes = len(labels_info["label_to_id"])
    id_to_label = labels_info["id_to_label"]
    logger.info(f"类别数: {num_classes}")
    logger.info(f"类别: {list(labels_info['label_to_id'].keys())}")

    # 创建数据集
    manifest_path = DATA_DIR / "manifest.json"
    dataset = CADDataset(str(manifest_path), str(DATA_DIR))

    if len(dataset) == 0:
        logger.error("数据集为空！")
        return

    # 划分训练集和验证集 (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"训练集: {train_size}, 验证集: {val_size}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 创建模型
    input_dim = 16  # 特征维度
    hidden_dim = 64
    model = SimpleClassifier(input_dim, hidden_dim, num_classes).to(DEVICE)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 训练
    num_epochs = 100
    best_val_acc = 0
    patience = 20
    patience_counter = 0

    logger.info("\n开始训练...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}"
            )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # 保存模型
            MODEL_DIR.mkdir(exist_ok=True)
            model_path = MODEL_DIR / "cad_classifier.pt"
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

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info(f"最佳验证准确率: {best_val_acc:.2%}")
    logger.info(f"模型保存至: {MODEL_DIR / 'cad_classifier.pt'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
