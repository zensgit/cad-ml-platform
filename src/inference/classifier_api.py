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

import asyncio
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import time
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, Request, UploadFile, HTTPException
from pydantic import BaseModel, ConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.dependencies import get_admin_token  # noqa: E402
from src.utils.analysis_metrics import (  # noqa: E402
    classification_cache_hits_total,
    classification_cache_miss_total,
    classification_cache_size,
    classification_rate_limited_total,
)
from src.utils.dxf_features import extract_features_v6  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

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
    """基于文件哈希的LRU缓存（L1内存缓存）"""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size: int = max_size
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def _hash_content(self, content: bytes) -> str:
        """计算文件内容哈希"""
        return hashlib.md5(content).hexdigest()

    def get(self, content: bytes) -> Optional[Dict[str, Any]]:
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

    def get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """通过key直接获取（不计统计）"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, content: bytes, result: Dict[str, Any]) -> None:
        """存入缓存"""
        key = self._hash_content(content)
        self.put_by_key(key, result)

    def put_by_key(self, key: str, result: Dict[str, Any]) -> None:
        """通过key直接存入"""
        if key in self.cache:
            self.cache[key] = result
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # 移除最久未使用的
                self.cache.popitem(last=False)
            self.cache[key] = result
        classification_cache_size.set(len(self.cache))

    def stats(self) -> Dict[str, Any]:
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

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        classification_cache_size.set(0)


class HybridCache:
    """混合缓存：L1内存 + L2 Redis

    特性：
    - L1: 本地LRU缓存，毫秒级响应
    - L2: Redis缓存，支持分布式部署
    - 自动降级：Redis不可用时仅用L1
    """

    REDIS_PREFIX: str = "clf:v16:"
    REDIS_TTL: int = 3600 * 24  # 24小时

    def __init__(self, l1_max_size: int = 1000) -> None:
        self.l1: LRUCache = LRUCache(max_size=l1_max_size)
        self._redis_client: Any = None
        self._redis_available: bool = False
        self._init_redis()

    def _init_redis(self) -> None:
        """初始化Redis连接"""
        try:
            from src.utils.cache import get_sync_client
            self._redis_client = get_sync_client()
            if self._redis_client:
                self._redis_client.ping()
                self._redis_available = True
                logger.info("分类器缓存: Redis L2已启用")
        except Exception as e:
            logger.info(f"分类器缓存: Redis不可用，仅使用L1内存缓存 ({e})")
            self._redis_available = False

    def _make_key(self, content: bytes) -> str:
        """生成缓存key"""
        return self.REDIS_PREFIX + hashlib.md5(content).hexdigest()

    def get(self, content: bytes) -> Optional[Dict[str, Any]]:
        """获取缓存结果（L1 -> L2）"""
        key = self._make_key(content)

        # L1查找
        result = self.l1.get_by_key(key)
        if result is not None:
            self.l1.hits += 1
            classification_cache_hits_total.inc()
            return result

        # L2查找
        if self._redis_available and self._redis_client:
            try:
                raw = self._redis_client.get(key)
                if raw:
                    result = json.loads(raw)
                    # 回填L1
                    self.l1.put_by_key(key, result)
                    self.l1.hits += 1
                    classification_cache_hits_total.inc()
                    return result
            except Exception as e:
                logger.debug(f"Redis get failed: {e}")

        self.l1.misses += 1
        classification_cache_miss_total.inc()
        return None

    def put(self, content: bytes, result: Dict[str, Any]) -> None:
        """存入缓存（同时写L1和L2）"""
        key = self._make_key(content)

        # 写L1
        self.l1.put_by_key(key, result)

        # 写L2
        if self._redis_available and self._redis_client:
            try:
                self._redis_client.setex(key, self.REDIS_TTL, json.dumps(result))
            except Exception as e:
                logger.debug(f"Redis set failed: {e}")

    def stats(self) -> Dict[str, Any]:
        """缓存统计"""
        stats = self.l1.stats()
        stats["redis_enabled"] = self._redis_available
        if self._redis_available and self._redis_client:
            try:
                # 统计Redis中分类器缓存的key数量
                keys = self._redis_client.keys(self.REDIS_PREFIX + "*")
                stats["redis_keys"] = len(keys) if keys else 0
            except Exception:
                stats["redis_keys"] = -1
        return stats

    def clear(self) -> None:
        """清空缓存（L1和L2）"""
        self.l1.clear()

        if self._redis_available and self._redis_client:
            try:
                keys = self._redis_client.keys(self.REDIS_PREFIX + "*")
                if keys:
                    self._redis_client.delete(*keys)
                logger.info(f"Redis缓存已清空: {len(keys) if keys else 0}个key")
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")


# 全局缓存实例 - 使用混合缓存支持Redis
_classifier_cache_max_size = int(os.getenv("CLASSIFIER_CACHE_MAX_SIZE", "1000"))
result_cache = HybridCache(l1_max_size=_classifier_cache_max_size)

# 线程池用于并行批处理（模型推理是CPU/GPU密集型）
_executor = ThreadPoolExecutor(max_workers=4)


# ============== 简易限流 ==============

class RateLimiter:
    """简单滑动窗口限流器（按客户端IP）"""

    def __init__(self, max_requests: int, window_seconds: int, burst: int = 0) -> None:
        self.max_requests: int = max_requests
        self.window_seconds: int = window_seconds
        self.burst: int = burst
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

    def __init__(self) -> None:
        self.v6_model: Optional[ImprovedClassifierV6] = None
        self.v14_models: List[FusionModelV14] = []
        self.v6_mean: Optional[np.ndarray] = None
        self.v6_std: Optional[np.ndarray] = None
        self.v14_mean: Optional[np.ndarray] = None
        self.v14_std: Optional[np.ndarray] = None
        self.v6_weight: float = 0.6
        self.v14_weight: float = 0.4
        self.loaded: bool = False
        self.use_half: bool = False  # 是否使用FP16半精度

    def load(self, use_half: Optional[bool] = None) -> None:
        """加载模型

        Args:
            use_half: 是否使用FP16半精度（减少约50%内存）
                      默认: CUDA/MPS设备自动启用
        """
        if self.loaded:
            return

        logger.info("加载V16模型...")

        # 自动决定是否使用半精度
        if use_half is None:
            use_half = DEVICE.type in ('cuda', 'mps')
        self.use_half = use_half

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
        if self.use_half:
            self.v6_model.half()
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
            if self.use_half:
                model.half()
            model.eval()
            self.v14_models.append(model)

        # V14使用全局均值（从数据估计）
        self.v14_mean = self.v6_mean
        self.v14_std = self.v6_std

        self.loaded = True
        precision = "FP16" if self.use_half else "FP32"
        logger.info(f"模型加载完成，设备: {DEVICE}, 精度: {precision}")

    def predict(self, dxf_path: str) -> Dict[str, Any]:
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

        # 选择数据类型
        dtype = torch.float16 if self.use_half else torch.float32

        with torch.inference_mode():
            x_v6 = torch.tensor(feat_v6, dtype=dtype).unsqueeze(0).to(DEVICE)
            v6_probs = torch.softmax(self.v6_model(x_v6), dim=1).float()

            x_geo = torch.tensor(feat_v14, dtype=dtype).unsqueeze(0).to(DEVICE)
            x_img = torch.tensor(img, dtype=dtype).unsqueeze(0).unsqueeze(0).to(DEVICE)

            v14_probs_list = []
            for model in self.v14_models:
                probs = torch.softmax(model(x_img, x_geo), dim=1).float()
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

def _warmup_model() -> None:
    """预热模型 - 执行一次空推理预热GPU/CPU缓存"""
    try:
        # 创建虚拟输入
        dummy_geo = np.zeros(48, dtype=np.float32)
        dummy_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # V6预热
        with torch.inference_mode():
            x_v6 = torch.FloatTensor(dummy_geo).unsqueeze(0).to(DEVICE)
            _ = classifier.v6_model(x_v6)

            # V14预热
            x_geo = torch.FloatTensor(dummy_geo).unsqueeze(0).to(DEVICE)
            x_img = torch.FloatTensor(dummy_img).unsqueeze(0).unsqueeze(0).to(DEVICE)
            for model in classifier.v14_models:
                _ = model(x_img, x_geo)

        logger.info("模型预热完成")
    except Exception as e:
        logger.warning(f"模型预热失败: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler for model warmup."""
    classifier.load()
    _warmup_model()
    yield
    # 关闭线程池
    _executor.shutdown(wait=False)


app = FastAPI(
    title="V16 CAD部件分类器",
    description="""
## CAD零件自动分类服务

基于深度学习的V16超级集成分类器，准确率99.67%。

### 功能特性
- **5类零件分类**: 传动件、其他、壳体类、轴类、连接件
- **高性能缓存**: LRU缓存，重复文件毫秒级响应
- **并行批处理**: 多文件并行推理，提升吞吐量
- **置信度输出**: 返回各类别概率分布

### 技术架构
- V6几何特征分类器 (权重0.6)
- V14视觉+几何融合集成 (权重0.4)
""",
    version="1.0.0",
    lifespan=lifespan,
)

# 全局分类器实例
classifier = V16Classifier()


class ClassificationResult(BaseModel):
    """分类结果"""
    filename: str
    category: str
    confidence: float
    probabilities: dict

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "filename": "part_001.dxf",
            "category": "壳体类",
            "confidence": 0.92,
            "probabilities": {
                "传动件": 0.02,
                "其他": 0.02,
                "壳体类": 0.92,
                "轴类": 0.03,
                "连接件": 0.01
            }
        }
    })


class BatchResult(BaseModel):
    """批量分类结果"""
    results: List[ClassificationResult]
    total: int
    success: int

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "results": [],
            "total": 10,
            "success": 9
        }
    })


@app.get("/", tags=["健康检查"])
async def root():
    """
    健康检查端点

    返回服务状态、模型版本和准确率。
    """
    return {"status": "ok", "model": "V16", "accuracy": "99.67%"}


@app.get("/categories", tags=["元数据"])
async def get_categories():
    """
    获取所有分类类别

    返回模型支持的5个零件类别列表。
    """
    return {"categories": CATEGORIES}


@app.post("/classify", response_model=ClassificationResult, tags=["分类"])
async def classify_file(request: Request, file: UploadFile = File(...)):
    """
    分类单个DXF文件

    上传DXF文件进行零件分类，支持缓存加速。

    - **file**: DXF格式的CAD文件
    - 返回: 类别、置信度、各类别概率
    """
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


def _predict_single(args: Tuple[int, str, bytes, str]) -> Tuple[int, str, Optional[Dict[str, Any]], Optional[str]]:
    """同步预测单个文件（用于线程池）

    Args:
        args: (index, filename, content, tmp_path)

    Returns:
        (index, filename, result_dict, error_msg)
    """
    index, filename, content, tmp_path = args
    try:
        result = classifier.predict(tmp_path)
        return (index, filename, result, None)
    except Exception as e:
        return (index, filename, None, str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/classify/batch", response_model=BatchResult, tags=["分类"])
async def classify_batch(request: Request, files: List[UploadFile] = File(...)):
    """
    批量分类DXF文件

    并行处理多个DXF文件，支持缓存加速。

    - **files**: 多个DXF格式的CAD文件
    - 返回: 每个文件的分类结果、总数、成功数
    - 特性: 缓存命中的文件直接返回，未缓存文件并行处理
    """
    _enforce_rate_limit(request)

    # 第一阶段：读取文件并检查缓存
    cached_results: Dict[int, ClassificationResult] = {}
    pending_files: List[Tuple[int, str, bytes]] = []
    pending_content: Dict[int, bytes] = {}
    errors: Dict[int, ClassificationResult] = {}

    for index, file in enumerate(files):
        if not file.filename.lower().endswith('.dxf'):
            errors[index] = ClassificationResult(
                filename=file.filename,
                category="error",
                confidence=0.0,
                probabilities={"error": "不是DXF文件"}
            )
            continue

        content = await file.read()

        # 检查缓存
        cached = result_cache.get(content)
        if cached is not None:
            cached_results[index] = ClassificationResult(
                filename=file.filename,
                category=cached["category"],
                confidence=cached["confidence"],
                probabilities=cached["probabilities"]
            )
        else:
            pending_files.append((index, file.filename, content))
            pending_content[index] = content

    # 第二阶段：并行处理未缓存的文件
    predicted_results: Dict[str, ClassificationResult] = {}

    if pending_files:
        # 准备临时文件
        tasks = []
        for index, filename, content in pending_files:
            tmp = tempfile.NamedTemporaryFile(suffix='.dxf', delete=False)
            tmp.write(content)
            tmp.close()
            tasks.append((index, filename, content, tmp.name))

        # 使用线程池并行预测
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: list(_executor.map(_predict_single, tasks))
        )

        # 处理结果
        for (_, _, _), (index, filename, result, error) in zip(pending_files, results):
            if error:
                predicted_results[index] = ClassificationResult(
                    filename=filename,
                    category="error",
                    confidence=0.0,
                    probabilities={"error": error}
                )
            else:
                content = pending_content.get(index)
                # 存入缓存
                if content is not None:
                    result_cache.put(content, result)
                predicted_results[index] = ClassificationResult(
                    filename=filename,
                    category=result["category"],
                    confidence=result["confidence"],
                    probabilities=result["probabilities"]
                )

    # 第三阶段：按原始顺序组装结果
    results = []
    success = 0
    cache_hits = len(cached_results)

    for index, file in enumerate(files):
        if index in cached_results:
            results.append(cached_results[index])
            success += 1
        elif index in predicted_results:
            r = predicted_results[index]
            results.append(r)
            if r.category != "error":
                success += 1
        elif index in errors:
            results.append(errors[index])
        else:
            results.append(
                ClassificationResult(
                    filename=file.filename,
                    category="error",
                    confidence=0.0,
                    probabilities={"error": "未知错误"},
                )
            )

    if cache_hits > 0 or len(pending_files) > 0:
        logger.info(f"批量分类: {len(files)}文件, 缓存命中{cache_hits}, 并行处理{len(pending_files)}")

    return BatchResult(results=results, total=len(files), success=success)


# ============== 缓存管理 ==============

@app.get("/cache/stats", tags=["缓存管理"])
async def cache_stats(request: Request, admin_token: str = Depends(get_admin_token)):
    """
    获取缓存统计信息

    需要管理员令牌。返回缓存大小、命中率等指标。
    """
    stats = result_cache.stats()
    client_host = request.client.host if request.client else "unknown"
    logger.info("Cache stats requested by %s", client_host)
    return stats


@app.post("/cache/clear", tags=["缓存管理"])
async def cache_clear(request: Request, admin_token: str = Depends(get_admin_token)):
    """
    清空缓存

    需要管理员令牌。清除所有缓存的分类结果。
    """
    result_cache.clear()
    client_host = request.client.host if request.client else "unknown"
    logger.info("Cache cleared by %s", client_host)
    return {"status": "ok", "message": "缓存已清空"}


# ============== CLI模式 ==============

def classify_cli(dxf_paths: List[str]) -> None:
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
