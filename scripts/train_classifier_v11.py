#!/usr/bin/env python3
"""
训练部件分类器 V11

使用增强版合成数据训练
- 简单CNN架构（从头训练，不使用ImageNet预训练）
- 适合CAD线条图的灰度输入
- 数据增强：旋转、缩放、平移
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
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 使用合成数据V2
DATA_DIR = Path("data/synthetic_v2")
REAL_DATA_DIR = Path("data/training_v7")  # 真实数据用于验证
MODEL_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

IMG_SIZE = 128  # 较小的图像尺寸，加快训练


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
        img = Image.open(buf).convert('L')  # 灰度图
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0

        plt.close(fig)
        buf.close()

        return img_array

    except Exception as e:
        return None


class CADDatasetV11(Dataset):
    """V11数据集 - 仅图像特征"""

    def __init__(self, manifest_path: str, data_dir: str, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.augment = augment

        # 数据增强
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
        logger.info(f"加载数据集，共 {len(self.manifest)} 个文件...")

        success = 0
        for i, item in enumerate(self.manifest):
            if (i + 1) % 500 == 0:
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
            success += 1

        logger.info(f"成功加载 {success} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = torch.tensor(sample["image"], dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        if self.augment and self.transform:
            img = self.transform(img)

        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, label


class SimpleCNN(nn.Module):
    """简单CNN - 专为CAD线条图设计"""

    def __init__(self, num_classes: int):
        super().__init__()

        # 卷积层 - 提取线条特征
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2: 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 3: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            # Block 4: 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1)
        )

        # 分类器
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

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total


def evaluate_on_real_data(model, device, id_to_label):
    """在真实数据上评估"""
    if not REAL_DATA_DIR.exists():
        logger.warning("真实数据目录不存在，跳过真实数据验证")
        return None

    manifest_path = REAL_DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.warning("真实数据manifest不存在，跳过真实数据验证")
        return None

    logger.info("\n在真实数据上评估...")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # 加载真实数据的标签映射
    real_labels_path = REAL_DATA_DIR / "labels.json"
    with open(real_labels_path, 'r', encoding='utf-8') as f:
        real_labels = json.load(f)

    real_label_to_id = real_labels["label_to_id"]

    # 创建合成数据标签到真实数据标签的映射
    synth_to_real = {}
    for synth_id, synth_name in id_to_label.items():
        if synth_name in real_label_to_id:
            synth_to_real[synth_id] = real_label_to_id[synth_name]

    model.eval()
    correct, total = 0, 0
    per_class_correct = Counter()
    per_class_total = Counter()

    with torch.no_grad():
        for item in manifest:
            file_path = REAL_DATA_DIR / item["file"]
            real_label_id = item["label_id"]
            category = item["category"]

            img = render_dxf_to_grayscale(str(file_path))
            if img is None:
                continue

            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            output = model(img_tensor)
            _, predicted = output.max(1)
            pred_label = predicted.item()

            # 检查预测的合成标签对应的真实标签
            if pred_label in synth_to_real:
                pred_real = synth_to_real[pred_label]
                is_correct = pred_real == real_label_id
            else:
                is_correct = False

            total += 1
            if is_correct:
                correct += 1
                per_class_correct[category] += 1
            per_class_total[category] += 1

    if total == 0:
        return None

    accuracy = correct / total
    logger.info(f"真实数据准确率: {accuracy:.2%} ({correct}/{total})")

    logger.info("各类别准确率:")
    for cat in sorted(per_class_total.keys()):
        cat_acc = per_class_correct[cat] / per_class_total[cat] if per_class_total[cat] > 0 else 0
        logger.info(f"  {cat}: {cat_acc:.2%} ({per_class_correct[cat]}/{per_class_total[cat]})")

    return accuracy


def main():
    logger.info("=" * 60)
    logger.info("训练部件分类器 V11 (合成数据 CNN)")
    logger.info(f"设备: {DEVICE}")
    logger.info("=" * 60)

    # 加载标签
    labels_path = DATA_DIR / "labels.json"
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_info = json.load(f)

    num_classes = len(labels_info["label_to_id"])
    id_to_label = {int(k): v for k, v in labels_info["id_to_label"].items()}
    logger.info(f"类别: {list(labels_info['label_to_id'].keys())}")

    # 加载数据
    manifest_path = DATA_DIR / "manifest.json"
    dataset = CADDatasetV11(str(manifest_path), str(DATA_DIR), augment=True)

    if len(dataset) == 0:
        logger.error("数据集为空！")
        return

    logger.info(f"样本数: {len(dataset)}")

    # 划分数据集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 验证集不使用数据增强
    val_dataset.dataset.augment = False

    logger.info(f"训练集: {train_size}, 验证集: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 训练
    model = SimpleCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    logger.info("\n开始训练...")

    for epoch in range(150):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "id_to_label": id_to_label,
                "best_val_acc": val_acc,
                "version": "v11",
                "categories": list(labels_info["label_to_id"].keys()),
                "img_size": IMG_SIZE,
                "architecture": "SimpleCNN"
            }
        else:
            patience_counter += 1
            if patience_counter >= 25:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"\n合成数据验证准确率: {best_val_acc:.2%}")

    # 加载最佳模型
    model.load_state_dict(best_state["model_state_dict"])

    # 在真实数据上评估
    real_acc = evaluate_on_real_data(model, DEVICE, id_to_label)

    # 保存模型
    MODEL_DIR.mkdir(exist_ok=True)
    if real_acc is not None:
        best_state["real_data_acc"] = real_acc

    model_path = MODEL_DIR / "cad_classifier_v11.pt"
    torch.save(best_state, model_path)
    logger.info(f"\n模型已保存: {model_path}")

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info(f"合成数据验证准确率: {best_val_acc:.2%}")
    if real_acc is not None:
        logger.info(f"真实数据准确率: {real_acc:.2%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
