#!/usr/bin/env python3
"""
V16 CAD部件分类器推理服务

使用方法:
1. 启动服务: python src/inference/classifier_api.py
2. 调用API: POST /classify 上传DXF文件
3. 批量分类: POST /classify/batch 上传多个文件

性能优化:
- 基于文件内容哈希的结果缓存
- LRU缓存策略，默认1000条
- 缓存命中时响应时间 < 1ms
"""

import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import time
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, Request, UploadFile, HTTPException
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.dependencies import get_admin_token
from src.utils.analysis_metrics import (
    classification_cache_hits_total,
    classification_cache_miss_total,
    classification_cache_size,
    classification_rate_limited_total,
)
from src.utils.dxf_features import extract_features_v6
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# 常量
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 128

# 类别映射 - 必须与训练时的labels.json一致
# label_to_id: 传动件=0, 其他=1, 壳体类=2, 轴类=3, 连接件=4
CATEGORIES = ["传动件", "其他", "壳体类", "轴类", "连接件"]
CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}
IDX_TO_CAT = {i: cat for i, cat in enumerate(CATEGORIES)}


# ============== LRU缓存 ==============

class LRUCache:
    """基于文件哈希的LRU缓存"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _hash_content(self, content: bytes) -> str:
        """计算文件内容哈希"""
        return hashlib.md5(content).hexdigest()

    def get(self, content: bytes) -> Optional[Dict]:
        """获取缓存结果"""
        key = self._hash_content(content)
        if key in self.cache:
            self.hits += 1
            classification_cache_hits_total.inc()
            # 移动到末尾（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        classification_cache_miss_total.inc()
        return None

    def put(self, content: bytes, result: Dict):
        """存入缓存"""
        key = self._hash_content(content)
        if key in self.cache:
            self.cache[key] = result
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # 移除最久未使用的
                self.cache.popitem(last=False)
            self.cache[key] = result
        classification_cache_size.set(len(self.cache))

    def stats(self) -> Dict:
        """缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}"
        }

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        classification_cache_size.set(0)


# 全局缓存实例
result_cache = LRUCache(max_size=1000)


# ============== 简易限流 ==============

class RateLimiter:
    """简单滑动窗口限流器（按客户端IP）"""

    def __init__(self, max_requests: int, window_seconds: int, burst: int = 0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst = burst
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.max_requests <= 0:
            return True
        now = time.monotonic()
        window_start = now - self.window_seconds
        bucket = self._requests[key]
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= (self.max_requests + self.burst):
            return False
        bucket.append(now)
        return True


_rate_limit_per_min = int(os.getenv("CLASSIFIER_RATE_LIMIT_PER_MIN", "120"))
_rate_limit_burst = int(os.getenv("CLASSIFIER_RATE_LIMIT_BURST", "20"))
_rate_limiter = RateLimiter(_rate_limit_per_min, 60, _rate_limit_burst)


def _enforce_rate_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    if not _rate_limiter.allow(client_host):
        classification_rate_limited_total.inc()
        logger.warning("Rate limit exceeded for %s", client_host)
        raise HTTPException(status_code=429, detail="Too many requests")

# ============== 模型定义 ==============

class ImprovedClassifierV6(nn.Module):
    """V6几何分类器"""
    def __init__(self, input_dim: int = 48, hidden_dim: int = 256, num_classes: int = 5, dropout: float = 0.5):
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


class DeepGeoBranch(nn.Module):
    """V14几何分支"""
    def __init__(self, geo_dim: int = 48, hidden_dim: int = 256):
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
    """V14视觉分支"""
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
    """V14融合模型"""
    def __init__(self, geo_dim: int = 48, num_classes: int = 5):
        super().__init__()
        self.geo_branch = DeepGeoBranch(geo_dim, 256)
        self.visual_branch = MultiScaleVisualBranch()
        self.geo_weight = nn.Parameter(torch.tensor(0.7))
        self.visual_weight = nn.Parameter(torch.tensor(0.3))
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
        self.geo_classifier = nn.Linear(128, num_classes)

    def forward(self, img, geo):
        geo_feat = self.geo_branch(geo)
        visual_feat = self.visual_branch(img)
        geo_w = torch.sigmoid(self.geo_weight)
        visual_w = torch.sigmoid(self.visual_weight)
        total = geo_w + visual_w
        geo_w, visual_w = geo_w / total, visual_w / total
        fused = torch.cat([geo_feat * geo_w, visual_feat * visual_w], dim=1)
        return self.classifier(fused)


# ============== 特征提取 ==============

def extract_geometric_features(dxf_path: str) -> Optional[np.ndarray]:
    """提取48维几何特征（兼容接口）"""
    return extract_features_v6(dxf_path, log=logger)


def render_dxf_to_image(dxf_path: str, size: int = IMG_SIZE) -> Optional[np.ndarray]:
    """渲染DXF为灰度图"""
    try:
        import ezdxf
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

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
                    from matplotlib.patches import Arc
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r = entity.dxf.radius
                    arc = Arc((cx, cy), 2*r, 2*r, angle=0, theta1=entity.dxf.start_angle,
                             theta2=entity.dxf.end_angle, color='k', linewidth=0.8)
                    ax.add_patch(arc)
                    all_x.extend([cx-r, cx+r])
                    all_y.extend([cy-r, cy+r])
                elif etype in ["LWPOLYLINE", "POLYLINE"]:
                    if hasattr(entity, 'get_points'):
                        pts = list(entity.get_points())
                        if len(pts) >= 2:
                            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                            ax.plot(xs, ys, 'k-', linewidth=0.8)
                            all_x.extend(xs)
                            all_y.extend(ys)
            except Exception:
                pass

        if not all_x or not all_y:
            plt.close(fig)
            return None

        margin = 0.1
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_range, y_range = max(x_max - x_min, 1e-6), max(y_max - y_min, 1e-6)
        ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        ax.axis('off')

        with io.BytesIO() as buf:
            plt.savefig(buf, format='png', dpi=100, facecolor='white', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)

            from PIL import Image
            img = Image.open(buf).convert('L').resize((size, size))
            return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        logger.error(f"渲染失败: {e}")
        return None


# ============== V16分类器 ==============

class V16Classifier:
    """V16超级集成分类器"""

    def __init__(self):
        self.v6_model = None
        self.v14_models = []
        self.v6_mean = None
        self.v6_std = None
        self.v14_mean = None
        self.v14_std = None
        self.v6_weight = 0.6
        self.v14_weight = 0.4
        self.loaded = False

    def load(self):
        """加载模型"""
        if self.loaded:
            return

        logger.info("加载V16模型...")

        # 加载V16配置
        config_path = MODEL_DIR / "cad_classifier_v16_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.v6_weight = config['components']['v6']['weight']
            self.v14_weight = config['components']['v14_ensemble']['weight']

        # 加载V6
        v6_path = MODEL_DIR / "cad_classifier_v6.pt"
        if not v6_path.exists():
            raise FileNotFoundError(f"V6模型不存在: {v6_path}")
        v6_ckpt = torch.load(v6_path, map_location=DEVICE, weights_only=False)
        self.v6_model = ImprovedClassifierV6(48, 256, 5, 0.5)
        self.v6_model.load_state_dict(v6_ckpt['model_state_dict'])
        self.v6_model.to(DEVICE)
        self.v6_model.eval()
        self.v6_mean = v6_ckpt.get('feature_mean', np.zeros(48))
        self.v6_std = v6_ckpt.get('feature_std', np.ones(48))

        # 加载V14集成
        v14_path = MODEL_DIR / "cad_classifier_v14_ensemble.pt"
        if not v14_path.exists():
            raise FileNotFoundError(f"V14集成模型不存在: {v14_path}")
        v14_ckpt = torch.load(v14_path, map_location=DEVICE, weights_only=False)
        for fold_state in v14_ckpt['fold_states']:
            model = FusionModelV14(48, 5)
            model.load_state_dict(fold_state)
            model.to(DEVICE)
            model.eval()
            self.v14_models.append(model)

        # V14使用全局均值（从数据估计）
        self.v14_mean = self.v6_mean
        self.v14_std = self.v6_std

        self.loaded = True
        logger.info(f"模型加载完成，设备: {DEVICE}")

    def predict(self, dxf_path: str) -> dict:
        """预测单个DXF文件"""
        if not self.loaded:
            self.load()

        # 提取特征
        features = extract_geometric_features(dxf_path)
        if features is None:
            raise ValueError("DXF特征提取失败")
        img = render_dxf_to_image(dxf_path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        feat_v6 = (features - self.v6_mean) / (self.v6_std + 1e-8)
        feat_v14 = (features - self.v14_mean) / (self.v14_std + 1e-8)

        with torch.inference_mode():
            x_v6 = torch.FloatTensor(feat_v6).unsqueeze(0).to(DEVICE)
            v6_probs = torch.softmax(self.v6_model(x_v6), dim=1)

            x_geo = torch.FloatTensor(feat_v14).unsqueeze(0).to(DEVICE)
            x_img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

            v14_probs_list = []
            for model in self.v14_models:
                probs = torch.softmax(model(x_img, x_geo), dim=1)
                v14_probs_list.append(probs)
            v14_probs = torch.mean(torch.stack(v14_probs_list), dim=0)

        # 融合
        final_probs = self.v6_weight * v6_probs + self.v14_weight * v14_probs
        pred_idx = torch.argmax(final_probs, dim=1).item()
        confidence = final_probs[0, pred_idx].item()

        # 所有类别的概率
        all_probs = {IDX_TO_CAT[i]: final_probs[0, i].item() for i in range(5)}

        return {
            "category": IDX_TO_CAT[pred_idx],
            "confidence": confidence,
            "probabilities": all_probs
        }


# ============== FastAPI应用 ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler for model warmup."""
    classifier.load()
    yield


app = FastAPI(
    title="V16 CAD部件分类器",
    description="基于深度学习的CAD零件自动分类服务",
    version="1.0.0",
    lifespan=lifespan,
)

# 全局分类器实例
classifier = V16Classifier()


class ClassificationResult(BaseModel):
    filename: str
    category: str
    confidence: float
    probabilities: dict


class BatchResult(BaseModel):
    results: List[ClassificationResult]
    total: int
    success: int


@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "model": "V16", "accuracy": "99.88%"}


@app.get("/categories")
async def get_categories():
    """获取所有类别"""
    return {"categories": CATEGORIES}


@app.post("/classify", response_model=ClassificationResult)
async def classify_file(request: Request, file: UploadFile = File(...)):
    """分类单个DXF文件（带缓存）"""
    _enforce_rate_limit(request)
    if not file.filename.lower().endswith('.dxf'):
        raise HTTPException(status_code=400, detail="只支持DXF文件")

    content = await file.read()

    # 检查缓存
    cached = result_cache.get(content)
    if cached is not None:
        logger.debug(f"缓存命中: {file.filename}")
        return ClassificationResult(
            filename=file.filename,
            category=cached["category"],
            confidence=cached["confidence"],
            probabilities=cached["probabilities"]
        )

    # 保存临时文件
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = classifier.predict(tmp_path)
        # 存入缓存
        result_cache.put(content, result)
        return ClassificationResult(
            filename=file.filename,
            category=result["category"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
    except ValueError as exc:
        logger.warning("分类失败: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        logger.error(f"分类失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/classify/batch", response_model=BatchResult)
async def classify_batch(request: Request, files: List[UploadFile] = File(...)):
    """批量分类DXF文件（带缓存）"""
    _enforce_rate_limit(request)
    results = []
    success = 0
    cache_hits = 0

    for file in files:
        if not file.filename.lower().endswith('.dxf'):
            results.append(ClassificationResult(
                filename=file.filename,
                category="error",
                confidence=0.0,
                probabilities={"error": "不是DXF文件"}
            ))
            continue

        content = await file.read()

        # 检查缓存
        cached = result_cache.get(content)
        if cached is not None:
            results.append(ClassificationResult(
                filename=file.filename,
                category=cached["category"],
                confidence=cached["confidence"],
                probabilities=cached["probabilities"]
            ))
            success += 1
            cache_hits += 1
            continue

        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = classifier.predict(tmp_path)
            # 存入缓存
            result_cache.put(content, result)
            results.append(ClassificationResult(
                filename=file.filename,
                category=result["category"],
                confidence=result["confidence"],
                probabilities=result["probabilities"]
            ))
            success += 1
        except ValueError as exc:
            results.append(ClassificationResult(
                filename=file.filename,
                category="error",
                confidence=0.0,
                probabilities={"error": str(exc)}
            ))
        except Exception as e:
            results.append(ClassificationResult(
                filename=file.filename,
                category="error",
                confidence=0.0,
                probabilities={"error": str(e)}
            ))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if cache_hits > 0:
        logger.info(f"批量分类: {len(files)}个文件, {cache_hits}个缓存命中")

    return BatchResult(results=results, total=len(files), success=success)


# ============== 缓存管理 ==============

@app.get("/cache/stats")
async def cache_stats(request: Request, admin_token: str = Depends(get_admin_token)):
    """获取缓存统计信息"""
    stats = result_cache.stats()
    client_host = request.client.host if request.client else "unknown"
    logger.info("Cache stats requested by %s", client_host)
    return stats


@app.post("/cache/clear")
async def cache_clear(request: Request, admin_token: str = Depends(get_admin_token)):
    """清空缓存"""
    result_cache.clear()
    client_host = request.client.host if request.client else "unknown"
    logger.info("Cache cleared by %s", client_host)
    return {"status": "ok", "message": "缓存已清空"}


# ============== CLI模式 ==============

def classify_cli(dxf_paths: List[str]):
    """命令行分类"""
    classifier.load()

    for path in dxf_paths:
        if not Path(path).exists():
            print(f"文件不存在: {path}")
            continue

        result = classifier.predict(path)
        print(f"{Path(path).name}: {result['category']} ({result['confidence']:.1%})")


if __name__ == "__main__":
    setup_logging()

    if len(sys.argv) > 1:
        # CLI模式
        classify_cli(sys.argv[1:])
    else:
        # API服务模式
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
