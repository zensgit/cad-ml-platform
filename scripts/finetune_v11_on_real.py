#!/usr/bin/env python3
"""
微调V11视觉模型

使用合成数据预训练的模型，在真实数据上微调
"""

import json
import logging
import sys
import io
from pathlib import Path
from typing import Optional
from collections import Counter
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
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
                    start_angle = entity.dxf.start_angle
                    end_angle = entity.dxf.end_angle
                    from matplotlib.patches import Arc
                    arc = Arc((cx, cy), 2*r, 2*r, angle=0, theta1=start_angle, theta2=end_angle, color='k', linewidth=0.8)
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

    except Exception as e:
        return None


class RealCADDataset(Dataset):
    """真实CAD数据集"""

    def __init__(self, manifest_path: str, data_dir: str, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.augment = augment

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])
        else:
            self.transform = None

        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)

        self.samples = []
        self.labels = []
        logger.info(f"加载真实数据集，共 {len(self.manifest)} 个文件...")

        success = 0
        for i, item in enumerate(self.manifest):
            if (i + 1) % 100 == 0:
                logger.info(f"  处理: {i+1}/{len(self.manifest)}")

            file_path = self.data_dir / item["file"]

            img = render_dxf_to_grayscale(str(file_path))
            if img is None:
                continue

            self.samples.append({
                "image": img,
                "label": item["label_id"],
                "category": item["category"],
                "file": item["file"]
            })
            self.labels.append(item["label_id"])
            success += 1

        logger.info(f"成功加载 {success} 个样本")

        # 统计类别分布
        label_counts = Counter(self.labels)
        logger.info("类别分布:")
        for label, count in sorted(label_counts.items()):
            logger.info(f"  类别 {label}: {count} 个")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = torch.tensor(sample["image"], dtype=torch.float32).unsqueeze(0)

        if self.augment and self.transform:
            img = self.transform(img)

        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, label


class SimpleCNN(nn.Module):
    """简单CNN - 与V11相同架构"""

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
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
    total_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
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
    logger.info("微调V11视觉模型 (真实数据)")
    logger.info(f"设备: {DEVICE}")
    logger.info("=" * 60)

    # 加载真实数据标签
    labels_path = REAL_DATA_DIR / "labels.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_info = json.load(f)

    num_classes = len(labels_info["label_to_id"])
    id_to_label = {int(k): v for k, v in labels_info["id_to_label"].items()}
    logger.info(f"类别: {list(labels_info['label_to_id'].keys())}")

    # 加载真实数据
    manifest_path = REAL_DATA_DIR / "manifest.json"
    dataset = RealCADDataset(str(manifest_path), str(REAL_DATA_DIR), augment=True)

    if len(dataset) == 0:
        logger.error("数据集为空！")
        return

    logger.info(f"样本数: {len(dataset)}")

    # 划分数据集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"训练集: {train_size}, 验证集: {val_size}")

    # 类别权重 (处理不平衡)
    train_labels = [dataset.labels[i] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 方案1: 从头训练
    logger.info("\n" + "=" * 40)
    logger.info("方案1: 从头训练 (仅真实数据)")
    logger.info("=" * 40)

    model_scratch = SimpleCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_scratch.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_scratch_acc = 0
    best_scratch_state = None
    patience = 0

    for epoch in range(150):
        train_loss, train_acc = train_epoch(model_scratch, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model_scratch, val_loader, criterion, DEVICE)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:3d} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")

        if val_acc > best_scratch_acc:
            best_scratch_acc = val_acc
            patience = 0
            best_scratch_state = model_scratch.state_dict().copy()
        else:
            patience += 1
            if patience >= 25:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"从头训练最佳验证准确率: {best_scratch_acc:.2%}")

    # 方案2: 加载预训练模型微调
    logger.info("\n" + "=" * 40)
    logger.info("方案2: 微调预训练模型 (合成数据预训练)")
    logger.info("=" * 40)

    pretrained_path = MODEL_DIR / "cad_classifier_v11.pt"
    if pretrained_path.exists():
        checkpoint = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)

        model_finetune = SimpleCNN(num_classes).to(DEVICE)

        # 尝试加载预训练权重 (可能类别数不同)
        pretrained_state = checkpoint["model_state_dict"]
        model_state = model_finetune.state_dict()

        # 只加载匹配的层
        matched_layers = 0
        for name, param in pretrained_state.items():
            if name in model_state and model_state[name].shape == param.shape:
                model_state[name] = param
                matched_layers += 1

        model_finetune.load_state_dict(model_state)
        logger.info(f"加载了 {matched_layers} 个预训练层")

        # 冻结特征提取层，只训练分类器
        for param in model_finetune.features.parameters():
            param.requires_grad = False

        # 使用较小的学习率
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model_finetune.parameters()),
            lr=0.0005, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        best_finetune_acc = 0
        best_finetune_state = None
        patience = 0

        # 第一阶段: 只训练分类器
        logger.info("阶段1: 只训练分类器层...")
        for epoch in range(50):
            train_loss, train_acc = train_epoch(model_finetune, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, _, _ = evaluate(model_finetune, val_loader, criterion, DEVICE)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")

            if val_acc > best_finetune_acc:
                best_finetune_acc = val_acc
                patience = 0
                best_finetune_state = model_finetune.state_dict().copy()
            else:
                patience += 1
                if patience >= 15:
                    break

        # 第二阶段: 解冻全部，微调
        logger.info("\n阶段2: 全模型微调...")
        for param in model_finetune.features.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(model_finetune.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        patience = 0

        for epoch in range(50):
            train_loss, train_acc = train_epoch(model_finetune, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, _, _ = evaluate(model_finetune, val_loader, criterion, DEVICE)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")

            if val_acc > best_finetune_acc:
                best_finetune_acc = val_acc
                patience = 0
                best_finetune_state = model_finetune.state_dict().copy()
            else:
                patience += 1
                if patience >= 15:
                    break

        logger.info(f"微调最佳验证准确率: {best_finetune_acc:.2%}")
    else:
        logger.warning("预训练模型不存在，跳过微调")
        best_finetune_acc = 0
        best_finetune_state = None

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("训练结果汇总")
    logger.info("=" * 60)
    logger.info(f"方案1 (从头训练): {best_scratch_acc:.2%}")
    if best_finetune_state:
        logger.info(f"方案2 (微调预训练): {best_finetune_acc:.2%}")

    # 选择最佳模型保存
    if best_finetune_state and best_finetune_acc > best_scratch_acc:
        best_state = best_finetune_state
        best_acc = best_finetune_acc
        method = "finetune"
    else:
        best_state = best_scratch_state
        best_acc = best_scratch_acc
        method = "scratch"

    # 保存
    save_data = {
        "model_state_dict": best_state,
        "num_classes": num_classes,
        "id_to_label": id_to_label,
        "best_val_acc": best_acc,
        "version": "v12",
        "categories": list(labels_info["label_to_id"].keys()),
        "img_size": IMG_SIZE,
        "architecture": "SimpleCNN",
        "training_method": method
    }

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(save_data, MODEL_DIR / "cad_classifier_v12.pt")
    logger.info(f"\n最佳模型 ({method}): {best_acc:.2%}")
    logger.info(f"已保存为: models/cad_classifier_v12.pt")


if __name__ == "__main__":
    main()
