# CAD ML Platform 详细改进方案 V2

**基于代码库深度分析的集成策略**

---

## 一、当前系统架构分析

### 1.1 特征提取架构 (`src/core/feature_extractor.py`)

**当前状态：**
```
特征版本演进:
v1 (7维)   → 基础几何+语义
v2 (12维)  → +归一化尺寸+长宽比
v3 (22维)  → +实体种类频率
v4 (24维)  → +surface_count, shape_entropy
v5 (26维)  → +旋转/缩放不变特征 (shape signature, axis invariants, topological)
v6 (32维)  → +矩不变量 (惯性张量特征值, 距离分布)
v7 (160维) → +视觉嵌入 (128维 CNN embedding) ← 已有骨架
v8 (1184维)→ +PointNet++ (1024维) ← 已有骨架
```

**关键发现：**
- ✅ v7/v8 框架已存在，但 `renderer.py` 返回零向量（DummyRenderer）
- ✅ `MatplotlibRenderer` 已实现基础渲染，但生成的是简化的像素平均值，非CNN特征
- ✅ Metric Learning 集成已完成 (`MetricEmbedder`)
- ❌ 缺少真正的CNN特征提取器（EfficientNet/ResNet）

### 1.2 向量存储架构 (`src/core/similarity.py`)

**当前状态：**
```
后端选择: VECTOR_STORE_BACKEND = memory | redis
索引类型:
├── InMemoryVectorStore  → 全量扫描 + heap top-k
├── FaissVectorStore     → IndexFlatIP (内积 ≈ 余弦)
└── MilvusVectorStore    → 分布式向量数据库 (可选)

降级机制:
├── 自动降级检测 (FAISS → Memory)
├── 恢复回退策略 (Exponential backoff)
└── Flapping suppression (防抖动)
```

**关键发现：**
- ✅ 支持 metadata 过滤 (material, complexity)
- ✅ TTL 自动清理
- ❌ **只支持单一向量索引**，无法做混合检索
- ❌ 缺少稀疏索引（Elasticsearch/倒排索引）

### 1.3 分类系统 (`src/core/knowledge/`)

**当前状态：**
```
知识模块融合架构:
├── MechanicalPartKnowledgeBase  (基础几何+OCR模式)
├── MaterialKnowledgeBase        (材料-零件关联)
├── PrecisionKnowledgeBase       (公差/精度识别)
├── StandardsKnowledgeBase       (行业标准)
├── FunctionalKnowledgeBase      (功能特征)
├── AssemblyKnowledgeBase        (装配关系)
├── ManufacturingKnowledgeBase   (制造工艺)
└── GeometryPatterns             (动态知识库)

融合权重 (DEFAULT_WEIGHTS):
├── base_classifier: 0.20
├── geometry: 0.22  ← 形状识别权重最高
├── material: 0.10
├── precision: 0.10
├── standards: 0.10
├── functional: 0.10
├── assembly: 0.08
└── manufacturing: 0.10
```

**关键发现：**
- ✅ 支持动态知识库 (`use_dynamic_knowledge=True`)
- ✅ 自适应权重学习已实现 (`AdaptiveWeightManager`)
- ✅ 21种零件类型支持
- ❌ **纯规则引擎**，无LLM推理能力
- ❌ OCR文本仅做关键词匹配，缺乏语义理解

### 1.4 OCR解析 (`src/core/ocr/parsing/dimension_parser.py`)

**当前状态：**
```
解析能力:
├── 直径 (Φ/⌀/∅)
├── 半径 (R)
├── 螺纹 (M<num>x<pitch>)
├── 表面粗糙度 (Ra)
├── 双向公差 (+/-值)
└── GD&T符号 (垂直度/平行度等)
```

**关键发现：**
- ✅ 正则表达式解析成熟
- ❌ **对OCR识别错误敏感**（如 M1O vs M10）
- ❌ 无语义推理能力（无法从上下文推断）

---

## 二、改进方案详细设计

### 模块一：视觉感知增强 (Visual Perception Enhancement)

#### 2.1.1 问题诊断

当前 `renderer.py` 的 `MatplotlibRenderer` 存在问题：
```python
# 当前实现 - 仅做像素平均，非真正CNN特征
raw_data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
embedding = [float(np.mean(raw_data[i:i+step])) / 255.0 ...]
```

这只是将图像像素值平均成128维向量，**缺乏语义特征提取能力**。

#### 2.1.2 改进方案

**方案A：引入预训练CNN (推荐)**

```python
# 新增: src/core/visual_extractor.py

from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CNNVisualExtractor:
    """基于预训练CNN的视觉特征提取器."""

    SUPPORTED_BACKBONES = ["efficientnet_b0", "resnet18", "mobilenet_v3"]

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        output_dim: int = 128,
        device: str = "cpu",
        model_path: Optional[str] = None,
    ):
        self.backbone = backbone
        self.output_dim = output_dim
        self.device = device
        self.model = None
        self.transform = None
        self._init_model(model_path)

    def _init_model(self, model_path: Optional[str] = None):
        """初始化CNN模型."""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            # 选择骨干网络
            if self.backbone == "efficientnet_b0":
                weights = models.EfficientNet_B0_Weights.DEFAULT
                base_model = models.efficientnet_b0(weights=weights)
                # 移除分类头，获取特征
                self.feature_dim = base_model.classifier[1].in_features  # 1280
                base_model.classifier = torch.nn.Identity()
            elif self.backbone == "resnet18":
                weights = models.ResNet18_Weights.DEFAULT
                base_model = models.resnet18(weights=weights)
                self.feature_dim = base_model.fc.in_features  # 512
                base_model.fc = torch.nn.Identity()
            elif self.backbone == "mobilenet_v3":
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
                base_model = models.mobilenet_v3_small(weights=weights)
                self.feature_dim = base_model.classifier[0].in_features
                base_model.classifier = torch.nn.Identity()
            else:
                raise ValueError(f"Unsupported backbone: {self.backbone}")

            # 添加降维层
            self.model = torch.nn.Sequential(
                base_model,
                torch.nn.Linear(self.feature_dim, self.output_dim),
                torch.nn.LayerNorm(self.output_dim),
            )

            # 加载微调权重 (如果有)
            if model_path:
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    logger.info(f"Loaded visual model from {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load weights: {e}")

            self.model.to(self.device)
            self.model.eval()

            # 图像预处理
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            logger.info(f"CNNVisualExtractor initialized: {self.backbone} -> {self.output_dim}D")

        except ImportError as e:
            logger.warning(f"PyTorch/torchvision not available: {e}")
            self.model = None

    def extract_from_image(self, image: "PIL.Image.Image") -> np.ndarray:
        """从单张图像提取特征."""
        if self.model is None:
            return np.zeros(self.output_dim, dtype=np.float32)

        import torch

        # 预处理
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            features = self.model(tensor)

        return features.cpu().numpy().flatten()

    def extract_multiview(
        self,
        images: List["PIL.Image.Image"],
        pooling: str = "max",
    ) -> np.ndarray:
        """多视图特征提取与池化.

        Args:
            images: 多个视图的图像列表
            pooling: 池化方式 ("max", "mean", "attention")

        Returns:
            聚合后的特征向量 (output_dim,)
        """
        if not images or self.model is None:
            return np.zeros(self.output_dim, dtype=np.float32)

        # 提取每个视图的特征
        features = [self.extract_from_image(img) for img in images]
        features = np.stack(features, axis=0)  # (N, output_dim)

        # 池化
        if pooling == "max":
            return np.max(features, axis=0)
        elif pooling == "mean":
            return np.mean(features, axis=0)
        elif pooling == "attention":
            # 简单注意力池化
            weights = np.exp(np.linalg.norm(features, axis=1))
            weights = weights / weights.sum()
            return (features * weights[:, np.newaxis]).sum(axis=0)
        else:
            return np.max(features, axis=0)


class MultiViewRenderer:
    """多视图渲染器 - 生成标准视角的深度图/轮廓图."""

    # 标准视角定义 (正方体6面 + 6个45度角)
    STANDARD_VIEWS = [
        (0, 0, 1),     # 前
        (0, 0, -1),    # 后
        (1, 0, 0),     # 右
        (-1, 0, 0),    # 左
        (0, 1, 0),     # 上
        (0, -1, 0),    # 下
        (1, 1, 1),     # 右上前
        (-1, 1, 1),    # 左上前
        (1, -1, 1),    # 右下前
        (-1, -1, 1),   # 左下前
        (1, 1, -1),    # 右上后
        (-1, 1, -1),   # 左上后
    ]

    def __init__(self, image_size: int = 224, use_trimesh: bool = True):
        self.image_size = image_size
        self.use_trimesh = use_trimesh
        self._trimesh = None
        self._pyrender = None

        if use_trimesh:
            try:
                import trimesh
                self._trimesh = trimesh
                try:
                    import pyrender
                    self._pyrender = pyrender
                except ImportError:
                    logger.warning("pyrender not available, using software rendering")
            except ImportError:
                logger.warning("trimesh not available, falling back to matplotlib")
                self.use_trimesh = False

    def render_views(self, doc: "CadDocument", num_views: int = 12) -> List["PIL.Image.Image"]:
        """渲染多个视角的图像.

        Args:
            doc: CAD文档
            num_views: 视图数量 (max 12)

        Returns:
            PIL图像列表
        """
        from PIL import Image

        views = self.STANDARD_VIEWS[:num_views]
        images = []

        if self.use_trimesh and self._trimesh and doc.sample_points:
            # 使用trimesh渲染点云
            images = self._render_trimesh(doc, views)
        else:
            # 回退到matplotlib
            images = self._render_matplotlib(doc, views)

        return images

    def _render_trimesh(self, doc: "CadDocument", views: List[tuple]) -> List["PIL.Image.Image"]:
        """使用trimesh渲染."""
        from PIL import Image
        import numpy as np

        images = []
        points = np.array(doc.sample_points)

        # 创建点云
        cloud = self._trimesh.PointCloud(points)

        for view_dir in views:
            # 创建场景
            scene = self._trimesh.Scene()
            scene.add_geometry(cloud)

            # 设置相机
            scene.set_camera(angles=view_dir)

            try:
                # 渲染到图像
                png = scene.save_image(resolution=(self.image_size, self.image_size))
                img = Image.open(png)
                images.append(img.convert("RGB"))
            except Exception:
                # 回退到空白图像
                images.append(Image.new("RGB", (self.image_size, self.image_size), "white"))

        return images

    def _render_matplotlib(self, doc: "CadDocument", views: List[tuple]) -> List["PIL.Image.Image"]:
        """使用matplotlib渲染."""
        from PIL import Image
        import matplotlib.pyplot as plt
        from io import BytesIO
        import numpy as np

        images = []

        for view_dir in views:
            fig = plt.figure(figsize=(2.24, 2.24), dpi=100)

            if doc.sample_points:
                from mpl_toolkits.mplot3d import Axes3D
                ax = fig.add_subplot(111, projection='3d')
                pts = np.array(doc.sample_points)

                # 旋转点云到视角
                # (简化：只渲染原始点云)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c='k')
                ax.view_init(elev=view_dir[1]*30, azim=view_dir[0]*30)
            else:
                ax = fig.add_subplot(111)
                # 渲染2D实体
                for entity in doc.entities:
                    self._draw_entity_2d(ax, entity)

            ax.axis('off')

            # 保存到内存
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            images.append(Image.open(buf).convert("RGB"))

        return images

    def _draw_entity_2d(self, ax, entity):
        """绘制2D实体."""
        import matplotlib.pyplot as plt

        kind = entity.kind.upper()
        attrs = entity.attributes

        if kind == "LINE":
            start = attrs.get("start", [0, 0])[:2]
            end = attrs.get("end", [0, 0])[:2]
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1)
        elif kind == "CIRCLE":
            center = attrs.get("center", [0, 0])[:2]
            radius = attrs.get("radius", 1.0)
            circle = plt.Circle(center, radius, fill=False, color='k')
            ax.add_patch(circle)
```

**集成到 feature_extractor.py v7 分支:**

```python
# 修改 src/core/feature_extractor.py v7 分支

if version == "v7":
    # ... 现有 v6 特征计算 ...

    # 6. Visual Embedding (128 dims) - 使用真正的CNN
    try:
        from src.core.visual_extractor import CNNVisualExtractor, MultiViewRenderer

        # 获取渲染器和特征提取器 (单例)
        renderer = MultiViewRenderer()
        extractor = get_visual_extractor()  # 使用工厂函数获取单例

        # 渲染多视图
        images = renderer.render_views(doc, num_views=12)

        # 提取并池化特征
        visual_embedding = extractor.extract_multiview(images, pooling="max")

    except Exception as e:
        logger.warning(f"Visual feature extraction failed: {e}")
        visual_embedding = [0.0] * 128
```

#### 2.1.3 配置参数

```bash
# 新增环境变量
VISUAL_BACKBONE=efficientnet_b0  # efficientnet_b0 | resnet18 | mobilenet_v3
VISUAL_OUTPUT_DIM=128
VISUAL_POOLING=max  # max | mean | attention
VISUAL_MODEL_PATH=models/visual_finetuned.pth  # 微调模型路径
VISUAL_RENDER_BACKEND=trimesh  # trimesh | matplotlib
```

#### 2.1.4 预期效果

| 指标 | 当前 | 改进后 |
|------|------|--------|
| 视觉特征维度 | 128 (像素平均) | 128 (CNN语义) |
| 异形件识别率 | ~40% | ~70-80% |
| 特征提取延迟 | <10ms | ~50-100ms |

---

### 模块二：LLM语义推理引擎

#### 2.2.1 问题诊断

当前OCR解析和知识库分类存在以下问题：
1. **OCR错误敏感**: `M1O` (误识别) vs `M10` (正确) 无法区分
2. **纯规则匹配**: 无法处理模糊情况
3. **缺乏推理能力**: 无法从上下文推断零件类型

#### 2.2.2 改进方案

**新增: `src/core/llm_reasoner.py`**

```python
"""LLM-based reasoning engine for CAD analysis.

Provides semantic reasoning capabilities:
1. OCR Error Correction (M1O -> M10)
2. Part Type Inference from context
3. Explainable Classification with reasoning chain
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """LLM推理结果."""

    # 主要输出
    part_type: str
    confidence: float
    reasoning_chain: List[str]

    # OCR纠正
    ocr_corrections: Dict[str, str] = field(default_factory=dict)

    # 备选项
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    # 元数据
    model_used: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0


class LLMReasoner:
    """LLM推理引擎.

    支持多种LLM后端:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Local (DeepSeek, Llama via vLLM/Ollama)
    """

    # 系统提示词
    SYSTEM_PROMPT = """你是一个资深机械设计工程师，专门分析CAD图纸。

你的任务是：
1. 分析提供的CAD特征数据
2. 纠正可能的OCR识别错误
3. 推断零件类型并给出置信度
4. 提供推理链条解释你的判断

输出格式要求（严格JSON）：
{
    "part_type": "零件类型英文名",
    "confidence": 0.0-1.0之间的数值,
    "reasoning": ["推理步骤1", "推理步骤2", ...],
    "ocr_corrections": {"原始文本": "纠正后文本"},
    "alternatives": [{"type": "备选类型", "confidence": 0.0-1.0}]
}

支持的零件类型：
shaft(轴), gear(齿轮), bearing(轴承), bolt(螺栓), flange(法兰),
housing(壳体), plate(板), washer(垫圈), spring(弹簧), pulley(皮带轮),
coupling(联轴器), bracket(支架), bushing(衬套), pin(销钉), cam(凸轮),
cover(盖板), nut(螺母), connecting_rod(连杆), piston(活塞), valve(阀门),
seal(密封圈), unknown(未知)
"""

    # 分析提示词模板
    ANALYSIS_PROMPT = """请分析以下CAD图纸数据：

## 几何特征
- 边界盒: 宽={bbox_width}mm, 高={bbox_height}mm, 深={bbox_depth}mm
- 长宽比: {aspect_ratio}
- 体积估计: {volume}mm³
- 实体数量: {entity_count}
- 圆形实体比例: {circle_ratio}

## OCR识别文本（可能有错误）
{ocr_raw}

## 图层信息
{layers}

## 知识库预分类结果
- 预测类型: {kb_prediction}
- 置信度: {kb_confidence}
- 信号: {kb_signals}

请综合分析并输出JSON格式结果。"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.3,
        timeout: int = 30,
        fallback_enabled: bool = True,
    ):
        """初始化LLM推理器.

        Args:
            provider: LLM提供商 (openai, anthropic, local)
            model: 模型名称
            temperature: 生成温度
            timeout: 超时时间(秒)
            fallback_enabled: 是否启用规则引擎回退
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.fallback_enabled = fallback_enabled

        self._client = None
        self._init_client()

        # 历史记录缓冲区 (用于上下文学习)
        self._history_buffer: List[Dict] = []
        self._max_history = 10

    def _init_client(self):
        """初始化LLM客户端."""
        try:
            if self.provider == "openai":
                import openai
                self._client = openai.OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=self.timeout,
                )
            elif self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                )
            elif self.provider == "local":
                # 使用本地模型 (vLLM/Ollama 兼容接口)
                import openai
                self._client = openai.OpenAI(
                    base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1"),
                    api_key="dummy",
                    timeout=self.timeout,
                )
            else:
                logger.warning(f"Unknown provider: {self.provider}")
        except ImportError as e:
            logger.warning(f"LLM client library not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")

    def reason(
        self,
        geometric_features: Dict[str, Any],
        ocr_data: Dict[str, Any],
        entity_counts: Dict[str, int],
        kb_result: Optional[Dict[str, Any]] = None,
        layers: Optional[List[str]] = None,
    ) -> ReasoningResult:
        """执行LLM推理.

        Args:
            geometric_features: 几何特征字典
            ocr_data: OCR数据
            entity_counts: 实体计数
            kb_result: 知识库预分类结果
            layers: 图层列表

        Returns:
            ReasoningResult
        """
        import time
        start_time = time.time()

        # 构建提示词
        prompt = self._build_prompt(
            geometric_features, ocr_data, entity_counts, kb_result, layers
        )

        try:
            # 调用LLM
            response = self._call_llm(prompt)

            # 解析响应
            result = self._parse_response(response)
            result.latency_ms = (time.time() - start_time) * 1000
            result.model_used = self.model

            # 添加到历史
            self._add_to_history(prompt, result)

            return result

        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")

            # 回退到知识库结果
            if self.fallback_enabled and kb_result:
                return ReasoningResult(
                    part_type=kb_result.get("part_type", "unknown"),
                    confidence=kb_result.get("confidence", 0.5) * 0.8,  # 降权
                    reasoning_chain=["LLM unavailable, using knowledge base fallback"],
                    model_used="fallback",
                    latency_ms=(time.time() - start_time) * 1000,
                )

            return ReasoningResult(
                part_type="unknown",
                confidence=0.0,
                reasoning_chain=[f"LLM error: {str(e)}"],
                model_used="error",
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _build_prompt(
        self,
        geometric_features: Dict[str, Any],
        ocr_data: Dict[str, Any],
        entity_counts: Dict[str, int],
        kb_result: Optional[Dict[str, Any]],
        layers: Optional[List[str]],
    ) -> str:
        """构建分析提示词."""

        # 计算圆形比例
        total_entities = sum(entity_counts.values()) or 1
        circle_count = entity_counts.get("CIRCLE", 0) + entity_counts.get("ARC", 0)
        circle_ratio = circle_count / total_entities

        # 提取OCR原始文本
        ocr_raw = []
        if ocr_data:
            raw_text = ocr_data.get("raw_text", "")
            if raw_text:
                ocr_raw.append(f"原始文本: {raw_text}")

            dimensions = ocr_data.get("dimensions", [])
            for dim in dimensions[:10]:  # 限制数量
                ocr_raw.append(f"- {dim.get('type', 'unknown')}: {dim.get('value', '')} {dim.get('raw', '')}")

        return self.ANALYSIS_PROMPT.format(
            bbox_width=geometric_features.get("bbox_width", 0),
            bbox_height=geometric_features.get("bbox_height", 0),
            bbox_depth=geometric_features.get("bbox_depth", 0),
            aspect_ratio=round(geometric_features.get("aspect_variance", 0), 2),
            volume=geometric_features.get("bbox_volume_estimate", 0),
            entity_count=sum(entity_counts.values()),
            circle_ratio=round(circle_ratio, 2),
            ocr_raw="\n".join(ocr_raw) if ocr_raw else "无OCR数据",
            layers=", ".join(layers[:10]) if layers else "无图层信息",
            kb_prediction=kb_result.get("part_type", "unknown") if kb_result else "无",
            kb_confidence=kb_result.get("confidence", 0) if kb_result else 0,
            kb_signals=json.dumps(kb_result.get("signals", {}), ensure_ascii=False) if kb_result else "{}",
        )

    def _call_llm(self, prompt: str) -> str:
        """调用LLM API."""
        if self._client is None:
            raise RuntimeError("LLM client not initialized")

        if self.provider in ("openai", "local"):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _parse_response(self, response: str) -> ReasoningResult:
        """解析LLM响应."""
        try:
            data = json.loads(response)

            return ReasoningResult(
                part_type=data.get("part_type", "unknown"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning_chain=data.get("reasoning", []),
                ocr_corrections=data.get("ocr_corrections", {}),
                alternatives=[
                    {"type": alt.get("type", ""), "confidence": alt.get("confidence", 0)}
                    for alt in data.get("alternatives", [])
                ],
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # 尝试提取文本中的类型
            for part_type in ["shaft", "gear", "bearing", "bolt"]:
                if part_type in response.lower():
                    return ReasoningResult(
                        part_type=part_type,
                        confidence=0.5,
                        reasoning_chain=["Extracted from malformed response"],
                    )
            raise

    def _add_to_history(self, prompt: str, result: ReasoningResult):
        """添加到历史缓冲区."""
        self._history_buffer.append({
            "prompt_hash": hash(prompt),
            "result": result.part_type,
            "confidence": result.confidence,
        })

        if len(self._history_buffer) > self._max_history:
            self._history_buffer.pop(0)

    def correct_ocr(self, raw_text: str) -> Dict[str, str]:
        """专门的OCR纠错接口.

        Args:
            raw_text: OCR原始文本

        Returns:
            纠正映射 {原始: 纠正后}
        """
        if not self._client:
            return {}

        prompt = f"""请检查以下OCR识别文本中可能的错误并纠正：

原始文本: {raw_text}

常见错误类型：
- 数字/字母混淆: O/0, I/1, l/1, S/5, B/8
- 螺纹规格: M1O -> M10, M8x1,25 -> M8x1.25
- 公差: ±O.02 -> ±0.02

请输出JSON格式: {{"corrections": {{"错误文本": "正确文本"}}}}"""

        try:
            response = self._call_llm(prompt)
            data = json.loads(response)
            return data.get("corrections", {})
        except Exception as e:
            logger.warning(f"OCR correction failed: {e}")
            return {}


# 工厂函数和单例
_llm_reasoner_instance: Optional[LLMReasoner] = None

def get_llm_reasoner() -> Optional[LLMReasoner]:
    """获取LLM推理器单例."""
    global _llm_reasoner_instance

    if not os.getenv("LLM_REASONING_ENABLED", "").lower() == "true":
        return None

    if _llm_reasoner_instance is None:
        _llm_reasoner_instance = LLMReasoner(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-4-turbo-preview"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
        )

    return _llm_reasoner_instance
```

#### 2.2.3 集成到分类流程

修改 `src/core/knowledge/enhanced_classifier.py`:

```python
def classify(self, ...):
    # ... 现有分类逻辑 ...

    # LLM增强 (可选)
    llm_result = None
    if os.getenv("LLM_REASONING_ENABLED", "").lower() == "true":
        try:
            from src.core.llm_reasoner import get_llm_reasoner
            reasoner = get_llm_reasoner()
            if reasoner:
                llm_result = reasoner.reason(
                    geometric_features=geometric_features,
                    ocr_data=ocr_data,
                    entity_counts=entity_counts,
                    kb_result={
                        "part_type": best_part,
                        "confidence": best_score,
                        "signals": score_breakdown.get(best_part, {}),
                    },
                    layers=list(ocr_data.get("layers", [])),
                )

                # 融合LLM结果
                if llm_result.confidence > 0.7:
                    best_part = llm_result.part_type
                    best_score = (best_score + llm_result.confidence) / 2
        except Exception as e:
            logger.warning(f"LLM reasoning failed: {e}")

    return EnhancedClassificationResult(
        # ... 现有字段 ...
        metadata={
            # ... 现有元数据 ...
            "llm_reasoning": llm_result.reasoning_chain if llm_result else None,
            "llm_corrections": llm_result.ocr_corrections if llm_result else None,
        },
    )
```

#### 2.2.4 配置参数

```bash
# LLM配置
LLM_REASONING_ENABLED=true
LLM_PROVIDER=openai           # openai | anthropic | local
LLM_MODEL=gpt-4-turbo-preview # 或 claude-3-sonnet | deepseek-v3
LLM_API_KEY=sk-...
LLM_TEMPERATURE=0.3
LLM_TIMEOUT=30
LLM_FALLBACK_ENABLED=true
```

---

### 模块三：混合检索系统

#### 2.3.1 问题诊断

当前向量检索存在的问题：
1. **形状相似但规格不同**: 两个齿轮形状相似但模数不同
2. **只有稠密向量索引**: 无法做精确的参数过滤
3. **无语义重排序**: 检索结果未经过语义校验

#### 2.3.2 改进方案

**新增: `src/core/hybrid_search.py`**

```python
"""Hybrid Search Engine - 混合检索系统.

结合多种检索方式:
1. Dense Vector Search (Faiss/Milvus) - 形状相似性
2. Sparse Index (Elasticsearch/内存倒排) - 参数精确匹配
3. Re-ranking (LLM/Cross-encoder) - 语义重排序
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """检索结果."""
    doc_id: str
    score: float
    vector_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridSearchConfig:
    """混合检索配置."""

    # 权重配置
    vector_weight: float = 0.6
    sparse_weight: float = 0.3
    rerank_weight: float = 0.1

    # 检索参数
    vector_top_k: int = 50    # 向量检索召回数
    sparse_top_k: int = 50    # 稀疏检索召回数
    final_top_k: int = 10     # 最终返回数

    # 功能开关
    enable_sparse: bool = True
    enable_rerank: bool = False

    # 硬过滤
    pre_filter: Dict[str, Any] = field(default_factory=dict)


class SparseIndex:
    """内存稀疏索引 (简化版倒排索引)."""

    def __init__(self):
        self._inverted_index: Dict[str, Dict[str, set]] = {
            "material": {},      # material -> {doc_ids}
            "part_type": {},     # part_type -> {doc_ids}
            "thread_spec": {},   # thread规格 -> {doc_ids}
            "diameter": {},      # 直径范围 -> {doc_ids}
        }
        self._doc_store: Dict[str, Dict] = {}  # doc_id -> full metadata

    def add(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """添加文档到索引."""
        self._doc_store[doc_id] = metadata

        # 索引各字段
        for field in self._inverted_index:
            value = metadata.get(field)
            if value:
                if isinstance(value, (int, float)):
                    # 数值分桶
                    bucket = self._bucket_numeric(field, value)
                    if bucket not in self._inverted_index[field]:
                        self._inverted_index[field][bucket] = set()
                    self._inverted_index[field][bucket].add(doc_id)
                else:
                    # 字符串精确匹配
                    value_str = str(value).lower()
                    if value_str not in self._inverted_index[field]:
                        self._inverted_index[field][value_str] = set()
                    self._inverted_index[field][value_str].add(doc_id)

    def _bucket_numeric(self, field: str, value: float) -> str:
        """数值分桶."""
        if field == "diameter":
            # 直径按5mm分桶
            bucket = int(value // 5) * 5
            return f"{bucket}-{bucket+5}"
        return str(int(value))

    def search(
        self,
        query: Dict[str, Any],
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """执行稀疏检索.

        Args:
            query: 查询条件 {field: value}
            top_k: 返回数量

        Returns:
            [(doc_id, score), ...]
        """
        if not query:
            return []

        # 计算每个文档的匹配分数
        doc_scores: Dict[str, float] = {}

        for field, value in query.items():
            if field not in self._inverted_index:
                continue

            # 查找匹配的文档
            if isinstance(value, (int, float)):
                bucket = self._bucket_numeric(field, value)
                matching_docs = self._inverted_index[field].get(bucket, set())
            else:
                value_str = str(value).lower()
                matching_docs = self._inverted_index[field].get(value_str, set())

            # 累加分数
            for doc_id in matching_docs:
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0

        # 归一化并排序
        max_score = len(query)
        results = [
            (doc_id, score / max_score)
            for doc_id, score in doc_scores.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def remove(self, doc_id: str) -> None:
        """从索引中移除文档."""
        if doc_id not in self._doc_store:
            return

        metadata = self._doc_store.pop(doc_id)

        for field in self._inverted_index:
            value = metadata.get(field)
            if value:
                if isinstance(value, (int, float)):
                    bucket = self._bucket_numeric(field, value)
                    if bucket in self._inverted_index[field]:
                        self._inverted_index[field][bucket].discard(doc_id)
                else:
                    value_str = str(value).lower()
                    if value_str in self._inverted_index[field]:
                        self._inverted_index[field][value_str].discard(doc_id)


class HybridSearchEngine:
    """混合检索引擎."""

    def __init__(self, config: Optional[HybridSearchConfig] = None):
        self.config = config or HybridSearchConfig()
        self._sparse_index = SparseIndex()
        self._vector_store = None
        self._reranker = None

        self._init_components()

    def _init_components(self):
        """初始化组件."""
        # 向量存储
        from src.core.similarity import get_vector_store
        self._vector_store = get_vector_store()

        # 重排序器 (可选)
        if self.config.enable_rerank:
            try:
                from src.core.llm_reasoner import get_llm_reasoner
                self._reranker = get_llm_reasoner()
            except Exception as e:
                logger.warning(f"Reranker not available: {e}")

    def register(self, doc_id: str, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """注册文档到混合索引."""
        from src.core.similarity import register_vector

        # 注册到向量存储
        success = register_vector(doc_id, vector, metadata)

        # 注册到稀疏索引
        if success and self.config.enable_sparse:
            self._sparse_index.add(doc_id, metadata)

        return success

    def search(
        self,
        query_vector: List[float],
        query_metadata: Optional[Dict[str, Any]] = None,
        config: Optional[HybridSearchConfig] = None,
    ) -> List[SearchResult]:
        """执行混合检索.

        Args:
            query_vector: 查询向量
            query_metadata: 查询元数据 (用于稀疏检索)
            config: 可选的配置覆盖

        Returns:
            检索结果列表
        """
        cfg = config or self.config

        # 1. 向量检索
        vector_results = self._vector_store.query(
            query_vector,
            top_k=cfg.vector_top_k,
        )
        vector_scores = {doc_id: score for doc_id, score in vector_results}

        # 2. 稀疏检索 (如果启用)
        sparse_scores = {}
        if cfg.enable_sparse and query_metadata:
            sparse_results = self._sparse_index.search(
                query_metadata,
                top_k=cfg.sparse_top_k,
            )
            sparse_scores = {doc_id: score for doc_id, score in sparse_results}

        # 3. 融合分数 (RRF)
        all_doc_ids = set(vector_scores.keys()) | set(sparse_scores.keys())
        fused_results = []

        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            s_score = sparse_scores.get(doc_id, 0.0)

            # 加权融合
            final_score = (
                cfg.vector_weight * v_score +
                cfg.sparse_weight * s_score
            )

            fused_results.append(SearchResult(
                doc_id=doc_id,
                score=final_score,
                vector_score=v_score,
                sparse_score=s_score,
                metadata=self._sparse_index._doc_store.get(doc_id, {}),
            ))

        # 排序
        fused_results.sort(key=lambda x: x.score, reverse=True)

        # 4. 重排序 (如果启用)
        if cfg.enable_rerank and self._reranker:
            top_candidates = fused_results[:cfg.final_top_k * 2]
            fused_results = self._rerank(query_vector, query_metadata, top_candidates)

        return fused_results[:cfg.final_top_k]

    def _rerank(
        self,
        query_vector: List[float],
        query_metadata: Optional[Dict],
        candidates: List[SearchResult],
    ) -> List[SearchResult]:
        """使用LLM重排序."""
        # 简化实现：使用LLM判断相关性
        # 实际生产环境建议使用Cross-Encoder
        return candidates


# 单例
_hybrid_engine: Optional[HybridSearchEngine] = None

def get_hybrid_search_engine() -> HybridSearchEngine:
    """获取混合检索引擎单例."""
    global _hybrid_engine

    if _hybrid_engine is None:
        config = HybridSearchConfig(
            vector_weight=float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6")),
            sparse_weight=float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.3")),
            rerank_weight=float(os.getenv("HYBRID_RERANK_WEIGHT", "0.1")),
            enable_sparse=os.getenv("HYBRID_SPARSE_ENABLED", "true").lower() == "true",
            enable_rerank=os.getenv("HYBRID_RERANK_ENABLED", "false").lower() == "true",
        )
        _hybrid_engine = HybridSearchEngine(config)

    return _hybrid_engine
```

#### 2.3.3 配置参数

```bash
# 混合检索配置
HYBRID_SEARCH_ENABLED=true
HYBRID_VECTOR_WEIGHT=0.6
HYBRID_SPARSE_WEIGHT=0.3
HYBRID_RERANK_WEIGHT=0.1
HYBRID_SPARSE_ENABLED=true
HYBRID_RERANK_ENABLED=false
```

---

### 模块四：主动学习闭环

#### 2.4.1 改进方案

**新增: `src/core/active_learning.py`**

```python
"""Active Learning Loop - 主动学习闭环.

实现:
1. 不确定性采样 (Uncertainty Sampling)
2. 用户反馈收集
3. 自动微调触发
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """用户反馈记录."""
    doc_id: str
    predicted_type: str
    true_type: str
    confidence: float
    timestamp: float
    user_id: Optional[str] = None
    feedback_type: str = "correction"  # correction | confirmation | rejection


@dataclass
class UncertaintySample:
    """不确定性样本."""
    doc_id: str
    predicted_type: str
    confidence: float
    alternatives: List[Dict[str, float]]
    features: List[float]
    timestamp: float


class ActiveLearner:
    """主动学习管理器."""

    # 不确定性阈值
    UNCERTAINTY_LOW = 0.4
    UNCERTAINTY_HIGH = 0.7

    # 重训练阈值
    RETRAIN_MIN_SAMPLES = 100

    def __init__(
        self,
        feedback_store: str = "redis",  # redis | postgres | memory
        uncertainty_threshold: Tuple[float, float] = (0.4, 0.7),
        retrain_threshold: int = 100,
    ):
        self.feedback_store = feedback_store
        self.uncertainty_low, self.uncertainty_high = uncertainty_threshold
        self.retrain_threshold = retrain_threshold

        # 内存缓冲
        self._feedback_buffer: List[FeedbackRecord] = []
        self._uncertainty_buffer: List[UncertaintySample] = []

        # 统计
        self._total_feedback = 0
        self._corrections_since_retrain = 0

        self._init_store()

    def _init_store(self):
        """初始化存储后端."""
        if self.feedback_store == "redis":
            try:
                from src.utils.cache import get_client
                self._redis = get_client()
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self._redis = None

    def should_flag_uncertain(self, confidence: float) -> bool:
        """判断是否应标记为不确定样本."""
        return self.uncertainty_low <= confidence <= self.uncertainty_high

    def record_prediction(
        self,
        doc_id: str,
        predicted_type: str,
        confidence: float,
        alternatives: List[Dict[str, float]],
        features: List[float],
    ) -> Optional[UncertaintySample]:
        """记录预测结果，返回是否需要人工审核."""

        if self.should_flag_uncertain(confidence):
            sample = UncertaintySample(
                doc_id=doc_id,
                predicted_type=predicted_type,
                confidence=confidence,
                alternatives=alternatives,
                features=features,
                timestamp=time.time(),
            )
            self._uncertainty_buffer.append(sample)

            # 存储到Redis
            if self._redis:
                self._redis.lpush(
                    "active_learning:uncertain",
                    json.dumps({
                        "doc_id": doc_id,
                        "predicted": predicted_type,
                        "confidence": confidence,
                        "alternatives": alternatives,
                        "ts": sample.timestamp,
                    })
                )

            return sample

        return None

    def submit_feedback(
        self,
        doc_id: str,
        predicted_type: str,
        true_type: str,
        confidence: float,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """提交用户反馈."""

        is_correction = (predicted_type != true_type)
        feedback_type = "correction" if is_correction else "confirmation"

        record = FeedbackRecord(
            doc_id=doc_id,
            predicted_type=predicted_type,
            true_type=true_type,
            confidence=confidence,
            timestamp=time.time(),
            user_id=user_id,
            feedback_type=feedback_type,
        )

        self._feedback_buffer.append(record)
        self._total_feedback += 1

        if is_correction:
            self._corrections_since_retrain += 1

        # 存储到Redis
        if self._redis:
            self._redis.lpush(
                "active_learning:feedback",
                json.dumps({
                    "doc_id": doc_id,
                    "predicted": predicted_type,
                    "true": true_type,
                    "confidence": confidence,
                    "type": feedback_type,
                    "user_id": user_id,
                    "ts": record.timestamp,
                })
            )

        # 检查是否需要触发重训练
        should_retrain = self._corrections_since_retrain >= self.retrain_threshold

        return {
            "status": "recorded",
            "feedback_type": feedback_type,
            "total_feedback": self._total_feedback,
            "corrections_since_retrain": self._corrections_since_retrain,
            "should_trigger_retrain": should_retrain,
        }

    def get_uncertain_samples(self, limit: int = 20) -> List[Dict]:
        """获取待审核的不确定样本."""
        samples = []

        if self._redis:
            raw_samples = self._redis.lrange("active_learning:uncertain", 0, limit - 1)
            for raw in raw_samples:
                samples.append(json.loads(raw))
        else:
            for sample in self._uncertainty_buffer[-limit:]:
                samples.append({
                    "doc_id": sample.doc_id,
                    "predicted": sample.predicted_type,
                    "confidence": sample.confidence,
                    "alternatives": sample.alternatives,
                    "ts": sample.timestamp,
                })

        return samples

    def export_training_data(self) -> Dict[str, Any]:
        """导出用于微调的训练数据."""

        # 收集Triplet数据
        triplets = []

        # 从反馈中构建
        # Anchor: 当前样本
        # Positive: 同类确认样本
        # Negative: 用户纠错的异类样本

        corrections = [f for f in self._feedback_buffer if f.feedback_type == "correction"]
        confirmations = [f for f in self._feedback_buffer if f.feedback_type == "confirmation"]

        # 按true_type分组
        by_type: Dict[str, List[FeedbackRecord]] = {}
        for record in self._feedback_buffer:
            t = record.true_type
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(record)

        # 构建triplets
        for part_type, records in by_type.items():
            if len(records) < 2:
                continue

            for i, anchor in enumerate(records):
                # Positive: 同类
                for j, positive in enumerate(records):
                    if i == j:
                        continue

                    # Negative: 异类
                    for other_type, other_records in by_type.items():
                        if other_type == part_type:
                            continue
                        for negative in other_records[:1]:  # 限制数量
                            triplets.append({
                                "anchor_id": anchor.doc_id,
                                "positive_id": positive.doc_id,
                                "negative_id": negative.doc_id,
                                "anchor_type": part_type,
                            })

        return {
            "triplets": triplets,
            "total_feedback": self._total_feedback,
            "corrections_count": len(corrections),
            "confirmations_count": len(confirmations),
            "part_type_distribution": {k: len(v) for k, v in by_type.items()},
            "exported_at": time.time(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息."""
        return {
            "total_feedback": self._total_feedback,
            "corrections_since_retrain": self._corrections_since_retrain,
            "pending_uncertain_samples": len(self._uncertainty_buffer),
            "retrain_threshold": self.retrain_threshold,
            "progress_to_retrain": self._corrections_since_retrain / self.retrain_threshold,
        }


# 单例
_active_learner: Optional[ActiveLearner] = None

def get_active_learner() -> ActiveLearner:
    """获取主动学习管理器单例."""
    global _active_learner

    if _active_learner is None:
        _active_learner = ActiveLearner(
            feedback_store=os.getenv("ACTIVE_LEARNING_STORE", "redis"),
            retrain_threshold=int(os.getenv("ACTIVE_LEARNING_RETRAIN_THRESHOLD", "100")),
        )

    return _active_learner
```

---

## 三、实施路线图

### Phase 3.1 (Week 1-2): LLM辅助推理

**优先级**: P0 (立即可用，无需训练)

```yaml
任务:
  - 创建 src/core/llm_reasoner.py
  - 集成到 enhanced_classifier.py
  - 添加OCR纠错API端点
  - 配置LLM环境变量

预期效果:
  - OCR识别错误纠正率: 90%+
  - 边缘案例分类改进: 20-30%

风险:
  - LLM API延迟 (~500ms)
  - Token成本 (~$0.01-0.05/分析)

缓解:
  - 启用缓存
  - 设置超时和回退
```

### Phase 3.2 (Week 3-4): 混合检索升级

**优先级**: P0 (用户体验直接改进)

```yaml
任务:
  - 创建 src/core/hybrid_search.py
  - 实现内存稀疏索引
  - 修改 /v1/vectors/similarity 端点
  - 添加metadata预过滤

预期效果:
  - 检索准确率提升: 20-30%
  - "形似但规格不同"问题解决

风险:
  - 内存占用增加

缓解:
  - 分桶策略优化
  - 可选Elasticsearch后端
```

### Phase 3.3 (Month 2): 视觉特征增强

**优先级**: P1 (需要模型集成)

```yaml
任务:
  - 创建 src/core/visual_extractor.py
  - 集成预训练CNN (EfficientNet)
  - 升级renderer.py
  - 微调视觉模型 (可选)

预期效果:
  - 异形件识别率: 40% -> 75%

风险:
  - PyTorch依赖增加
  - 推理延迟 (~100ms)

缓解:
  - 可选CPU/GPU模式
  - 特征缓存
```

### Phase 3.4 (Month 3+): 主动学习闭环

**优先级**: P1 (长期价值)

```yaml
任务:
  - 创建 src/core/active_learning.py
  - 实现反馈API端点
  - 开发标注界面 (Label Studio集成)
  - 自动微调脚本

预期效果:
  - 月度准确率提升: 10-15%
  - 用户参与度提升

风险:
  - 反馈噪声
  - 数据标注成本

缓解:
  - 验证规则
  - 专家审核门控
```

---

## 四、配置汇总

### 4.1 新增环境变量

```bash
# ===== LLM 配置 =====
LLM_REASONING_ENABLED=true
LLM_PROVIDER=openai                    # openai | anthropic | local
LLM_MODEL=gpt-4-turbo-preview
LLM_API_KEY=sk-...
LLM_TEMPERATURE=0.3
LLM_TIMEOUT=30
LLM_FALLBACK_ENABLED=true

# ===== 视觉特征配置 =====
VISUAL_BACKBONE=efficientnet_b0        # efficientnet_b0 | resnet18 | mobilenet_v3
VISUAL_OUTPUT_DIM=128
VISUAL_POOLING=max                     # max | mean | attention
VISUAL_MODEL_PATH=models/visual.pth
VISUAL_RENDER_BACKEND=matplotlib       # trimesh | matplotlib

# ===== 混合检索配置 =====
HYBRID_SEARCH_ENABLED=true
HYBRID_VECTOR_WEIGHT=0.6
HYBRID_SPARSE_WEIGHT=0.3
HYBRID_RERANK_WEIGHT=0.1
HYBRID_SPARSE_ENABLED=true
HYBRID_RERANK_ENABLED=false

# ===== 主动学习配置 =====
ACTIVE_LEARNING_ENABLED=true
ACTIVE_LEARNING_STORE=redis            # redis | postgres | memory
ACTIVE_LEARNING_RETRAIN_THRESHOLD=100
UNCERTAINTY_LOW=0.4
UNCERTAINTY_HIGH=0.7
```

### 4.2 新增依赖

```txt
# requirements-ml-enhanced.txt

# LLM
openai>=1.3.0
anthropic>=0.8.0

# Visual
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
trimesh>=3.20.0    # 可选
pyrender>=0.1.45   # 可选

# Search
elasticsearch>=8.0.0  # 可选，生产环境推荐

# Active Learning
label-studio-sdk>=0.0.32  # 可选
```

---

## 五、成功指标

| 指标 | 当前基线 | Phase 3.1后 | Phase 3.2后 | Phase 3.3后 | Phase 3.4后 |
|------|---------|------------|------------|------------|------------|
| 分类准确率 | ~65% | 75% | 80% | 85% | 90%+ |
| 检索Recall@10 | ~60% | 65% | 80% | 85% | 90% |
| OCR纠错率 | 0% | 90% | 90% | 90% | 90% |
| 平均延迟 | ~200ms | ~700ms | ~800ms | ~900ms | ~1000ms |
| 月度改进率 | 0% | 5% | 8% | 10% | 15% |

---

**文档版本**: V2.0
**更新日期**: 2025-12-02
**作者**: Claude Code Analysis
