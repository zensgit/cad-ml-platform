#!/usr/bin/env python3
"""
训练部件分类器 V9

创新: CNN视觉特征 + 几何特征融合
- 将DXF渲染为图像
- 使用预训练ResNet提取视觉特征
- 与48维几何特征融合
- 双通道分类器
"""

import json
import logging
import sys
import io
from pathlib import Path
from typing import Optional, Tuple
from collections import Counter
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/training_v7")  # 使用V7数据集
MODEL_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 图像参数
IMG_SIZE = 224


def render_dxf_to_image(dxf_path: str, size: int = IMG_SIZE) -> Optional[np.ndarray]:
    """将DXF渲染为图像"""
    try:
        import ezdxf
        from ezdxf.addons.drawing import matplotlib as dxf_matplotlib
        
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        # 创建图像
        fig = plt.figure(figsize=(size/100, size/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_aspect('equal')
        
        # 收集所有点来确定边界
        all_x, all_y = [], []
        
        for entity in msp:
            try:
                etype = entity.dxftype()
                if etype == "LINE":
                    x1, y1 = entity.dxf.start.x, entity.dxf.start.y
                    x2, y2 = entity.dxf.end.x, entity.dxf.end.y
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)
                    all_x.extend([x1, x2])
                    all_y.extend([y1, y2])
                elif etype == "CIRCLE":
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r = entity.dxf.radius
                    circle = plt.Circle((cx, cy), r, fill=False, color='k', linewidth=0.5)
                    ax.add_patch(circle)
                    all_x.extend([cx-r, cx+r])
                    all_y.extend([cy-r, cy+r])
                elif etype == "ARC":
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r = entity.dxf.radius
                    start_angle = entity.dxf.start_angle
                    end_angle = entity.dxf.end_angle
                    from matplotlib.patches import Arc
                    arc = Arc((cx, cy), 2*r, 2*r, angle=0, theta1=start_angle, theta2=end_angle, color='k', linewidth=0.5)
                    ax.add_patch(arc)
                    all_x.extend([cx-r, cx+r])
                    all_y.extend([cy-r, cy+r])
                elif etype in ["LWPOLYLINE", "POLYLINE"]:
                    if hasattr(entity, 'get_points'):
                        pts = list(entity.get_points())
                        if len(pts) >= 2:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            ax.plot(xs, ys, 'k-', linewidth=0.5)
                            all_x.extend(xs)
                            all_y.extend(ys)
            except:
                pass
        
        if not all_x or not all_y:
            plt.close(fig)
            return None
        
        # 设置边界
        margin = 0.1
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_range = x_max - x_min or 1
        y_range = y_max - y_min or 1
        ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        ax.axis('off')
        ax.set_facecolor('white')
        
        # 转换为numpy数组
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='white', edgecolor='none', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        
        plt.close(fig)
        buf.close()
        
        return img_array
        
    except Exception as e:
        return None


class EnhancedFeatureExtractorV4:
    """几何特征提取器 (48维)"""
    
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


class CADDatasetV9(Dataset):
    """V9数据集 - 图像 + 几何特征"""
    
    def __init__(self, manifest_path: str, data_dir: str):
        self.data_dir = Path(data_dir)
        self.geo_extractor = EnhancedFeatureExtractorV4()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        
        self.samples = []
        logger.info(f"加载数据集，共 {len(self.manifest)} 个文件...")
        
        success = 0
        for i, item in enumerate(self.manifest):
            if (i + 1) % 100 == 0:
                logger.info(f"  处理: {i+1}/{len(self.manifest)}")
            
            file_path = self.data_dir / item["file"]
            
            # 提取几何特征
            geo_features = self.geo_extractor.extract(str(file_path))
            if geo_features is None:
                continue
            
            # 渲染图像
            img = render_dxf_to_image(str(file_path))
            if img is None:
                continue
            
            self.samples.append({
                "image": img,
                "geo_features": geo_features,
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
        img_tensor = self.transform(sample["image"])
        geo_tensor = torch.tensor(sample["geo_features"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return img_tensor, geo_tensor, label


class FusionClassifierV9(nn.Module):
    """V9融合分类器 - CNN + MLP"""
    
    def __init__(self, geo_dim: int, num_classes: int, cnn_features: int = 512):
        super().__init__()
        
        # CNN分支 (使用预训练ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的FC层
        
        # 冻结CNN前几层
        for param in list(self.cnn_backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # CNN特征投影
        self.cnn_proj = nn.Sequential(
            nn.Linear(512, cnn_features),
            nn.BatchNorm1d(cnn_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 几何特征分支
        self.geo_branch = nn.Sequential(
            nn.Linear(geo_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 融合层
        fusion_dim = cnn_features + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, img, geo):
        # CNN分支
        cnn_out = self.cnn_backbone(img)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        cnn_features = self.cnn_proj(cnn_out)
        
        # 几何特征分支
        geo_features = self.geo_branch(geo)
        
        # 融合
        fused = torch.cat([cnn_features, geo_features], dim=1)
        out = self.fusion(fused)
        
        return out


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for img, geo, labels in loader:
        img, geo, labels = img.to(device), geo.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(img, geo)
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
        for img, geo, labels in loader:
            img, geo, labels = img.to(device), geo.to(device), labels.to(device)
            outputs = model(img, geo)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), correct / total


def main():
    logger.info("=" * 60)
    logger.info("训练部件分类器 V9 (CNN + 几何特征融合)")
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
    dataset = CADDatasetV9(str(manifest_path), str(DATA_DIR))
    
    if len(dataset) == 0:
        logger.error("数据集为空！")
        return
    
    geo_dim = dataset[0][1].shape[0]
    logger.info(f"样本数: {len(dataset)}, 几何特征维度: {geo_dim}")
    
    # 划分数据集
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"训练集: {train_size}, 验证集: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 训练
    num_runs = 3
    best_acc = 0
    best_state = None
    
    for run in range(num_runs):
        logger.info(f"\n训练轮次 {run+1}/{num_runs}")
        
        model = FusionClassifierV9(geo_dim, num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        best_run_acc = 0
        patience_counter = 0
        
        for epoch in range(100):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")
            
            if val_acc > best_run_acc:
                best_run_acc = val_acc
                patience_counter = 0
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {
                        "model_state_dict": model.state_dict(),
                        "geo_dim": geo_dim,
                        "num_classes": num_classes,
                        "id_to_label": id_to_label,
                        "best_val_acc": val_acc,
                        "version": "v9",
                        "categories": list(labels_info["label_to_id"].keys())
                    }
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
        
        logger.info(f"Run {run+1} 最佳: {best_run_acc:.2%}")
    
    # 保存
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(best_state, MODEL_DIR / "cad_classifier_v9.pt")
    logger.info(f"\n训练完成！最佳验证准确率: {best_acc:.2%}")


if __name__ == "__main__":
    main()
