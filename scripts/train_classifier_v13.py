#!/usr/bin/env python3
"""
训练部件分类器 V13

融合模型：视觉特征 + 几何特征
- 视觉分支：简单CNN提取图像特征
- 几何分支：48维几何特征
- 融合层：拼接后分类
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


class GeometricFeatureExtractor:
    """48维几何特征提取器 (与V6相同)"""

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
    """融合数据集 - 图像 + 几何特征"""

    def __init__(self, manifest_path: str, data_dir: str, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.geo_extractor = GeometricFeatureExtractor()

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
        logger.info(f"加载数据集，共 {len(self.manifest)} 个文件...")

        success = 0
        for i, item in enumerate(self.manifest):
            if (i + 1) % 100 == 0:
                logger.info(f"  处理: {i+1}/{len(self.manifest)}")

            file_path = self.data_dir / item["file"]

            # 提取图像
            img = render_dxf_to_grayscale(str(file_path))
            if img is None:
                continue

            # 提取几何特征
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


class FusionModelV13(nn.Module):
    """融合模型V13：视觉CNN + 几何MLP"""

    def __init__(self, geo_dim: int, num_classes: int):
        super().__init__()

        # 视觉分支 - 轻量CNN
        self.visual_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        visual_out_dim = 128

        # 几何分支 - MLP
        self.geo_branch = nn.Sequential(
            nn.Linear(geo_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
        )
        geo_out_dim = 128

        # 融合层
        fusion_dim = visual_out_dim + geo_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        # 注意力权重 (学习两个分支的重要性)
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, img, geo):
        # 视觉特征
        visual_feat = self.visual_branch(img)

        # 几何特征
        geo_feat = self.geo_branch(geo)

        # 拼接
        fused = torch.cat([visual_feat, geo_feat], dim=1)

        # 注意力加权
        attn = self.attention(fused)
        visual_weighted = visual_feat * attn[:, 0:1]
        geo_weighted = geo_feat * attn[:, 1:2]
        fused_weighted = torch.cat([visual_weighted, geo_weighted], dim=1)

        # 分类
        out = self.fusion(fused_weighted)
        return out


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, geo, labels in loader:
        imgs, geo, labels = imgs.to(device), geo.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, geo)
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
    per_class_correct = Counter()
    per_class_total = Counter()

    with torch.no_grad():
        for imgs, geo, labels in loader:
            imgs, geo, labels = imgs.to(device), geo.to(device), labels.to(device)
            outputs = model(imgs, geo)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for p, l in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                per_class_total[l] += 1
                if p == l:
                    per_class_correct[l] += 1

    return total_loss / len(loader), correct / total, per_class_correct, per_class_total


def main():
    logger.info("=" * 60)
    logger.info("训练融合模型 V13 (视觉CNN + 几何特征)")
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

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"训练集: {train_size}, 验证集: {val_size}")

    # 类别权重
    train_labels = [dataset.labels[i] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 训练融合模型
    model = FusionModelV13(geo_dim, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_acc = 0
    best_state = None
    patience = 0

    logger.info("\n开始训练融合模型...")

    for epoch in range(150):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, per_class_correct, per_class_total = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:3d} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "geo_dim": geo_dim,
                "num_classes": num_classes,
                "id_to_label": id_to_label,
                "best_val_acc": val_acc,
                "version": "v13",
                "categories": list(labels_info["label_to_id"].keys()),
                "img_size": IMG_SIZE,
                "architecture": "FusionModelV13"
            }
        else:
            patience += 1
            if patience >= 25:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型评估
    model.load_state_dict(best_state["model_state_dict"])
    _, final_acc, per_class_correct, per_class_total = evaluate(model, val_loader, criterion, DEVICE)

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info(f"最佳验证准确率: {best_val_acc:.2%}")
    logger.info("\n各类别准确率:")
    for label_id in sorted(per_class_total.keys()):
        cat_name = id_to_label[label_id]
        acc = per_class_correct[label_id] / per_class_total[label_id] if per_class_total[label_id] > 0 else 0
        logger.info(f"  {cat_name}: {acc:.2%} ({per_class_correct[label_id]}/{per_class_total[label_id]})")
    logger.info("=" * 60)

    # 保存
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(best_state, MODEL_DIR / "cad_classifier_v13.pt")
    logger.info(f"模型已保存: models/cad_classifier_v13.pt")


if __name__ == "__main__":
    main()
