# DeepSeek-OCR 工程语义理解完整方案

**CAD ML Platform - OCR增强子系统技术设计文档**

---

**文档信息**

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.0 |
| 创建日期 | 2025-01-14 |
| 作者 | CAD ML Platform Team |
| 适用范围 | CAD ML Platform v1.1.0+ |
| 文档类型 | 技术设计方案 |

**变更历史**

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|---------|------|
| v1.0 | 2025-01-14 | 初始版本 | Team |

---

## 目录

1. [方案概述](#一方案概述)
2. [系统架构](#二系统架构)
3. [核心模块设计](#三核心模块设计)
4. [API契约](#四api契约)
5. [配置管理](#五配置管理)
6. [实施路线](#六实施路线)
7. [监控与评测](#七监控与评测)
8. [生产部署](#八生产部署)
9. [验收标准](#九验收标准)
10. [附录](#十附录)

---

## 一、方案概述

### 1.1 目标与范围

**核心目标**

针对工程图/CAD截图/扫描件，稳定抽取**文本、尺寸/公差/符号、表格/标题栏**，并提供**高可用、可解释、可观测**的OCR服务。

**能力范围**

- ✅ 支持本地（CPU/GPU）与云端多provider部署
- ✅ 智能路由：按成本/质量/吞吐自动选择provider
- ✅ 工程语义理解：尺寸Φ20±0.02 → 结构化数据
- ✅ 几何对齐：OCR文本锚定到CAD几何元素
- ✅ 证据驱动：完整证据链支撑可解释性
- ✅ 生产就绪：缓存、监控、降级、幂等性

**与现有系统对齐**

- 复用 `confidence_calibrator.py` 的DS证据融合
- 复用 `Redis` 缓存与 `Prometheus` 监控
- 扩展现有 `/api/v1/analyze` 端点
- 保持 `VisionProvider` 的provider模式一致性

### 1.2 核心价值

| 维度 | 传统OCR | 本方案（工程语义OCR） |
|------|---------|----------------------|
| **输出** | 文本字符串 | 结构化工程数据 |
| **尺寸识别** | "Φ20±0.02" | {type:"diameter", value:20, tolerance:0.02, unit:"mm"} |
| **几何关联** | 无 | 尺寸→几何元素锚定 |
| **标题栏** | 文本堆砌 | 结构化BOM：图号/材料/重量 |
| **下游应用** | 有限 | 装配推理/工艺建议/成本估算 |
| **证据链** | 无 | 完整证据：provider+置信度+bbox+规则 |

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        CAD ML Platform                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              API层 (FastAPI)                            │  │
│  │  ┌──────────────────┬──────────────────┬─────────────┐ │  │
│  │  │ POST /analyze    │ POST /ocr/extract│ GET /health │ │  │
│  │  │ (集成OCR增强)     │ (直通OCR端点)     │ (含OCR状态) │ │  │
│  │  └──────────────────┴──────────────────┴─────────────┘ │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │           OCR管理器 (OcrManager)                        │  │
│  │  • Provider路由策略 (auto/paddle/deepseek_hf/vllm)      │  │
│  │  • 质量门控与Fallback                                   │  │
│  │  • 缓存协调 (图像哈希 + provider + prompt)              │  │
│  │  • 证据链聚合 (复用confidence_calibrator.py)            │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│  ┌──────────────┬──────────────┬─────────────┬─────────────┐  │
│  │  Paddle      │ DeepSeek-HF  │ DeepSeek-   │   Future    │  │
│  │  Provider    │  Provider    │   vLLM      │  Providers  │  │
│  │  (CPU快速)   │ (GPU高质量)   │ (GPU高吞吐) │   (扩展)    │  │
│  └──────────────┴──────────────┴─────────────┴─────────────┘  │
│                              ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │             增强处理层                                   │  │
│  │  ┌──────────────┬────────────────┬──────────────────┐  │  │
│  │  │ 预处理器     │  版面解析器     │  裁剪合并器      │  │  │
│  │  │ (去噪/矫正)  │ (页/块/表检测)  │ (智能分块+拼接) │  │  │
│  │  └──────────────┴────────────────┴──────────────────┘  │  │
│  │  ┌──────────────┬────────────────┬──────────────────┐  │  │
│  │  │ 结构化解析器 │  几何对齐器     │ 质量控制器       │  │  │
│  │  │ (尺寸/公差/  │ (文本→几何锚定) │ (门控+重试)     │  │  │
│  │  │  符号/标题栏)│                 │                  │  │  │
│  │  └──────────────┴────────────────┴──────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │          下游集成 (工程语义应用)                         │  │
│  │  • 装配推理增强 (尺寸→配合关系)                          │  │
│  │  • 工艺建议增强 (材料/公差→加工方法)                     │  │
│  │  • 成本估算增强 (材料/重量/工艺)                         │  │
│  │  • 相似度检索增强 (语义+符号向量)                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│              横切关注点 (Cross-Cutting Concerns)                │
│  • 缓存: Redis (图像哈希 + 幂等性Key)                          │
│  • 监控: Prometheus指标 + Grafana面板                          │
│  • 日志: 结构化JSON日志 + 证据链追踪                           │
│  • 配置: 环境分层 (.env.dev / .env.prod)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
输入图像
   ↓
┌─────────────────────────────────────────────────┐
│  1. 预处理                                      │
│  • 格式检测 (矢量/扫描)                         │
│  • 去噪/二值化/矫正 (按需)                      │
│  • 大图检测 (>4K触发裁剪)                       │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  2. 缓存检查                                    │
│  Key = sha256(image) + provider + prompt        │
│  Hit → 直接返回                                 │
│  Miss → 继续流程                                │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  3. Provider选择 (策略路由)                     │
│  • auto: paddle → (低置信度) → deepseek_hf      │
│  • explicit: 指定provider                       │
│  • degraded: 降级到备用provider                 │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  4. OCR执行                                     │
│  • PaddleOCR: 检测+识别+版面分析                │
│  • DeepSeek-HF: Transformers推理 + JSON输出     │
│  • DeepSeek-vLLM: 异步批处理 (Phase 3)         │
│  输出: {text, blocks[], layout, confidence}     │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  5. 结果规范化                                  │
│  • 文本清洗 (全半角/符号统一)                   │
│  • 单位标准化 (→mm)                             │
│  • BBox坐标校正                                 │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  6. 结构化解析                                  │
│  • 尺寸解析: Φ20±0.02 → {type:"diameter",       │
│    value:20, tolerance:0.02, unit:"mm"}         │
│  • 公差解析: IT7, 6H/f7 → 公差带查表            │
│  • 符号解析: Ra3.2, ⟂A → 符号词典匹配           │
│  • 标题栏解析: 关键词定位 + 表格结构恢复        │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  7. 几何对齐 (如果有CAD数据)                    │
│  • 空间索引构建 (R-tree)                        │
│  • BBox → 几何元素映射 (距离+数值匹配)          │
│  • 生成AlignedDimension (文本+几何双重证据)     │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  8. 质量门控                                    │
│  • 检查关键字段 (图号/材料/主要尺寸)            │
│  • 检查置信度阈值                               │
│  • 触发Fallback (deepseek_hf) 或返回            │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  9. 证据链融合                                  │
│  • 复用confidence_calibrator.py                 │
│  • DS理论融合多provider结果                     │
│  • 生成calibrated_confidence                    │
└─────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────┐
│  10. 缓存写入 + 返回结果                        │
│  • 落盘缓存 (TTL=24h)                           │
│  • 记录Prometheus指标                           │
│  • 返回EnhancedOcrResult                        │
└─────────────────────────────────────────────────┘
```

### 2.3 Provider选择策略

```python
# Auto策略决策树
def select_provider(image, context):
    # Step 1: 快速paddle
    paddle_result = paddle.extract(image)

    # Step 2: 判断是否需要增强
    if paddle_result.confidence >= 0.85 and \
       has_critical_fields(paddle_result):
        return paddle_result  # ✅ 足够好

    # Step 3: DeepSeek增强
    deepseek_result = deepseek_hf.extract(image)

    # Step 4: 融合结果
    return merge([paddle_result, deepseek_result])
```

---

## 三、核心模块设计

### 3.1 目录结构

```
src/
├── core/
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── base.py                 # 抽象基类与数据模型
│   │   ├── manager.py              # OCR管理器 (路由+融合)
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── paddle.py           # PaddleOCR实现
│   │   │   ├── deepseek_hf.py      # DeepSeek Transformers
│   │   │   └── deepseek_vllm.py    # DeepSeek vLLM (Phase 3)
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── image_enhancer.py   # 去噪/矫正/二值化
│   │   │   └── cropper.py          # 智能裁剪与合并
│   │   ├── parsing/
│   │   │   ├── __init__.py
│   │   │   ├── dimension_parser.py # 尺寸/公差解析
│   │   │   ├── symbol_parser.py    # 符号解析
│   │   │   ├── title_block_parser.py # 标题栏解析
│   │   │   ├── normalizer.py       # 文本规范化
│   │   │   └── json_validator.py   # JSON严格校验
│   │   ├── alignment/
│   │   │   ├── __init__.py
│   │   │   └── geometric_aligner.py # 几何对齐
│   │   ├── quality/
│   │   │   ├── __init__.py
│   │   │   ├── gate.py             # 质量门控
│   │   │   └── validator.py        # 结果验证
│   │   ├── concurrency.py          # 并发控制
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── prompt_templates.py # DeepSeek提示模板
│   │       └── metrics.py          # OCR专用指标
│   └── vision_analyzer.py (扩展: 集成OcrManager)
│
├── api/
│   └── v1/
│       ├── ocr.py                  # 新增: 直通OCR端点
│       ├── analyze.py (扩展: enable_ocr参数)
│       └── __init__.py (扩展: 注册OCR路由)
│
├── models/
│   └── ocr_models.py               # Pydantic数据模型
│
└── config/
    └── ocr_config.py               # OCR配置类
```

### 3.2 核心类设计

#### 3.2.1 基础抽象 (base.py)

```python
"""
OCR基础抽象与数据模型
src/core/ocr/base.py
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class OcrProviderType(str, Enum):
    """OCR Provider类型"""
    PADDLE = "paddle"
    DEEPSEEK_HF = "deepseek_hf"
    DEEPSEEK_VLLM = "deepseek_vllm"
    AUTO = "auto"


class BBox(BaseModel):
    """边界框"""
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width/2, self.y + self.height/2)

    def iou(self, other: 'BBox') -> float:
        """计算IoU"""
        # ... 实现省略 ...


class DimensionInfo(BaseModel):
    """尺寸信息 (结构化)"""
    type: str = Field(..., description="类型: diameter/radius/length/thread")
    value: float = Field(..., description="数值")
    unit: str = Field(default="mm", description="单位")
    tolerance: Optional[float] = Field(None, description="公差值")
    tolerance_grade: Optional[str] = Field(None, description="公差等级: IT6/IT7")
    bbox: BBox
    confidence: float
    source_text: str = Field(..., description="原始文本")


class TitleBlockInfo(BaseModel):
    """标题栏信息"""
    drawing_number: Optional[str] = None
    part_name: Optional[str] = None
    material: Optional[str] = None
    scale: Optional[str] = None
    weight: Optional[float] = None
    unit: str = "mm"
    version: Optional[str] = None


class OcrResult(BaseModel):
    """OCR结果"""
    provider: OcrProviderType
    text: str
    blocks: List[OcrBlock] = []
    dimensions: List[DimensionInfo] = []
    symbols: List[SymbolInfo] = []
    title_block: Optional[TitleBlockInfo] = None

    overall_confidence: float
    calibrated_confidence: Optional[float] = None

    # 元数据
    cache_hit: bool = False
    processing_time_ms: float = 0.0
    quality_report: Optional[QualityReport] = None
    evidence_chain: List[Dict] = []


class OcrClient(ABC):
    """OCR客户端抽象基类"""

    @abstractmethod
    async def extract(
        self,
        image: bytes,
        prompt: Optional[str] = None,
        options: Optional[Dict] = None
    ) -> OcrResult:
        """执行OCR提取"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass

    @abstractmethod
    async def warmup(self):
        """模型预热"""
        pass
```

#### 3.2.2 OCR管理器 (manager.py)

```python
"""
OCR管理器 - 策略路由、质量门控、证据融合
src/core/ocr/manager.py
"""

import hashlib
from typing import Dict, Optional, List

from src.core.ocr.base import OcrClient, OcrResult, OcrProviderType
from src.core.ocr.providers.paddle import PaddleOcrClient
from src.core.ocr.providers.deepseek_hf import DeepSeekHfClient
from src.core.assembly.confidence_calibrator import ConfidenceCalibrationSystem
from src.utils.cache import cache_result, get_cached_result


class OcrManager:
    """OCR管理器 - 核心协调器"""

    def __init__(self):
        # 初始化providers
        self.providers: Dict[OcrProviderType, OcrClient] = {
            OcrProviderType.PADDLE: PaddleOcrClient(),
            OcrProviderType.DEEPSEEK_HF: DeepSeekHfClient(),
        }

        # 质量控制器
        self.quality_controller = QualityController(...)

        # 置信度校准 (复用现有)
        self.calibrator = ConfidenceCalibrationSystem(method='isotonic')
        self.calibrator.load_calibrator()

    async def extract(
        self,
        image: bytes,
        strategy: str = "auto",
        prompt: Optional[str] = None,
        idempotency_key: Optional[str] = None
    ) -> OcrResult:
        """
        统一OCR提取入口

        流程:
        1. 缓存检查
        2. Provider选择与执行
        3. 质量门控与fallback
        4. 置信度校准
        5. 缓存写入
        """

        # 1. 缓存检查
        cache_key = self._generate_cache_key(image, strategy, prompt, idempotency_key)
        cached = await get_cached_result(cache_key)
        if cached:
            return OcrResult(**cached)

        # 2. Provider执行
        if strategy == "auto":
            result = await self._auto_strategy(image, prompt)
        else:
            result = await self._execute_provider(
                OcrProviderType(strategy), image, prompt
            )

        # 3. 质量门控
        result = await self.quality_controller.validate_and_fallback(result, image, self)

        # 4. 置信度校准
        result = self._calibrate_confidence(result)

        # 5. 缓存写入
        await cache_result(cache_key, result.dict(), ttl=86400)

        return result

    async def _auto_strategy(self, image: bytes, prompt: Optional[str] = None) -> OcrResult:
        """
        Auto策略: 先paddle，低置信度触发deepseek
        """
        # 快速paddle
        paddle_result = await self._execute_provider(OcrProviderType.PADDLE, image)

        # 判断是否需要增强
        needs_enhancement = (
            paddle_result.overall_confidence < 0.85 or
            self._missing_critical_content(paddle_result)
        )

        if not needs_enhancement:
            return paddle_result

        # DeepSeek增强
        deepseek_result = await self._execute_provider(
            OcrProviderType.DEEPSEEK_HF, image, prompt
        )

        # 融合结果 (使用DS证据理论)
        return self._merge_results([paddle_result, deepseek_result])

    def _calibrate_confidence(self, result: OcrResult) -> OcrResult:
        """校准置信度 (复用confidence_calibrator.py)"""
        result.calibrated_confidence = self.calibrator.calibrator.calibrate(
            result.overall_confidence
        )
        return result
```

#### 3.2.3 DeepSeek Provider (deepseek_hf.py)

```python
"""
DeepSeek-HF Provider
src/core/ocr/providers/deepseek_hf.py
"""

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import io

from src.core.ocr.base import OcrClient, OcrResult
from src.core.ocr.parsing.json_validator import JsonValidator


class DeepSeekHfClient(OcrClient):
    """DeepSeek OCR - Transformers实现"""

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = None
        self.tokenizer = None
        self._model_loaded = False

        # JSON校验器
        self.json_validator = JsonValidator(max_retries=2)

    async def warmup(self):
        """模型预热"""
        if not self._model_loaded:
            await self._load_model()

            # 预热推理
            dummy_image = Image.new('RGB', (640, 480), color='white')
            await self._infer(dummy_image, "<image>\n<|grounding|>Free OCR.")

    async def _load_model(self):
        """加载模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.model.eval()
        self._model_loaded = True

    async def extract(self, image: bytes, prompt: Optional[str] = None, options: Optional[Dict] = None) -> OcrResult:
        """OCR提取"""
        if not self._model_loaded:
            await self._load_model()

        # 转换图像
        pil_image = Image.open(io.BytesIO(image)).convert('RGB')

        # 选择prompt
        if prompt is None:
            prompt = self._get_engineering_drawing_prompt()

        # 执行推理
        raw_output = await self._infer(pil_image, prompt)

        # 严格JSON校验
        validated = self.json_validator.validate_and_heal(raw_output)

        if validated:
            return self._build_result_from_validated(validated, pil_image.size)
        else:
            # Fallback到文本解析
            return await self._parse_text_output(raw_output, pil_image.size)

    def _get_engineering_drawing_prompt(self) -> str:
        """工程图结构化prompt"""
        return """<image>
<|grounding|>Extract dimensions/tolerances/surface-roughness/threads as strict JSON:
{
  "dimensions": [{"type":"diameter|radius|length", "value":float, "unit":"mm", "tolerance":float, "bbox":{}}],
  "symbols": [{"type":"surface_roughness|perpendicular", "value":str, "bbox":{}}],
  "title_block": {"drawing_number":str, "material":str, "part_name":str}
}"""

    async def _infer(self, image: Image.Image, prompt: str) -> str:
        """执行模型推理"""
        inputs = self.tokenizer(prompt, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=4096, temperature=0.0)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self._model_loaded:
                await self._load_model()
            return True
        except:
            return False
```

#### 3.2.4 尺寸解析器 (dimension_parser.py)

```python
"""
尺寸与公差解析器
src/core/ocr/parsing/dimension_parser.py
"""

import re
from typing import List, Optional
from src.core.ocr.base import DimensionInfo, BBox


class DimensionParser:
    """工程图尺寸解析器"""

    def __init__(self):
        self.patterns = {
            'diameter': r'[Φ⌀∅](\d+\.?\d*)([±+\-]\d+\.?\d*)?',
            'radius': r'R(\d+\.?\d*)([±+\-]\d+\.?\d*)?',
            'thread': r'M(\d+)(×|x|\*)(\d+\.?\d*)?',
            'length': r'(\d+\.?\d*)([±]\d+\.?\d*)',
        }

    def parse_from_text(self, text: str) -> List[DimensionInfo]:
        """从文本解析尺寸"""
        dimensions = []

        # 解析直径
        for match in re.finditer(self.patterns['diameter'], text):
            value = float(match.group(1))
            tolerance = self._parse_tolerance(match.group(2)) if match.group(2) else None

            dimensions.append(DimensionInfo(
                type='diameter',
                value=value,
                unit='mm',
                tolerance=tolerance,
                bbox=BBox(x=0, y=0, width=0, height=0),
                confidence=0.85,
                source_text=match.group(0)
            ))

        # 解析半径、螺纹...
        # (类似逻辑省略)

        return dimensions

    def _parse_tolerance(self, tol_str: str) -> Optional[float]:
        """解析公差值"""
        if '±' in tol_str:
            return abs(float(tol_str.replace('±', '')))
        return None
```

#### 3.2.5 JSON严格校验 (json_validator.py)

```python
"""
DeepSeek JSON输出严格校验
src/core/ocr/parsing/json_validator.py
"""

from pydantic import BaseModel, Field, ValidationError
import json
import re


class StrictDimensionOutput(BaseModel):
    """严格的尺寸输出模型"""
    type: str = Field(..., regex="^(diameter|radius|length|thread)$")
    value: float = Field(..., ge=0, le=10000)
    unit: str = Field(default="mm", regex="^(mm|cm|m|inch)$")
    tolerance: Optional[float] = Field(None, ge=0, le=10)
    bbox: Dict[str, float]
    confidence: float = Field(..., ge=0, le=1)
    source_text: str

    class Config:
        extra = 'forbid'


class JsonValidator:
    """JSON校验器 + 自愈重试"""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def validate_and_heal(self, raw_json: str, attempt: int = 0) -> Optional[StrictOcrJsonOutput]:
        """
        校验并自愈JSON输出

        策略:
        1. JSON解析
        2. Pydantic严格校验
        3. 自愈常见错误 (尾随逗号、单引号、缺失字段)
        4. 最多重试2次
        """

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            # 自愈语法错误
            healed_json = self._heal_json_syntax(raw_json)
            if healed_json and attempt < self.max_retries:
                return self.validate_and_heal(healed_json, attempt + 1)
            return None

        try:
            validated = StrictOcrJsonOutput(**data)
            return validated
        except ValidationError as e:
            # 自愈数据结构问题
            healed_data = self._heal_data_structure(data, e)
            if healed_data and attempt < self.max_retries:
                try:
                    return StrictOcrJsonOutput(**healed_data)
                except:
                    pass
            return None

    def _heal_json_syntax(self, raw_json: str) -> Optional[str]:
        """修复JSON语法错误"""
        # 常见问题1: 尾随逗号
        healed = re.sub(r',\s*}', '}', raw_json)
        healed = re.sub(r',\s*]', ']', healed)

        # 常见问题2: 单引号
        healed = healed.replace("'", '"')

        try:
            json.loads(healed)
            return healed
        except:
            return None
```

---

## 四、API契约

### 4.1 数据模型

```python
"""
OCR API数据模型
src/models/ocr_models.py
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class OcrExtractRequest(BaseModel):
    """OCR提取请求"""
    provider: str = Field(default="auto", description="Provider: auto/paddle/deepseek_hf")
    prompt: Optional[str] = Field(None, description="自定义prompt (DeepSeek)")
    enable_geometric_alignment: bool = Field(default=False, description="几何对齐")
    idempotency_key: Optional[str] = Field(None, description="幂等性Key")


class OcrExtractResponse(BaseModel):
    """OCR提取响应"""
    request_id: str
    provider: OcrProviderType
    text: str
    dimensions: List[DimensionInfo] = []
    symbols: List[SymbolInfo] = []
    title_block: Optional[TitleBlockInfo] = None

    overall_confidence: float
    calibrated_confidence: Optional[float] = None

    cache_hit: bool = False
    processing_time_ms: float = 0.0
    quality_report: Optional[QualityReport] = None
    evidence_chain: List[Dict[str, Any]] = []
```

### 4.2 API端点

#### 4.2.1 直通OCR端点

```python
"""
OCR直通端点
src/api/v1/ocr.py
"""

from fastapi import APIRouter, File, UploadFile, Form, Header

router = APIRouter()
ocr_manager = OcrManager()


@router.post("/extract", response_model=OcrExtractResponse)
async def extract_ocr(
    file: UploadFile = File(..., description="图像文件"),
    provider: str = Form(default="auto"),
    prompt: Optional[str] = Form(None),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    api_key: str = Depends(get_api_key)
):
    """
    OCR直通提取端点

    支持格式: JPG, PNG, PDF (单页)

    Provider说明:
    - auto: 智能选择 (先paddle，低置信度→deepseek)
    - paddle: 快速CPU OCR
    - deepseek_hf: 高质量GPU OCR
    """

    image_data = await file.read()

    result = await ocr_manager.extract(
        image=image_data,
        strategy=provider,
        prompt=prompt,
        idempotency_key=idempotency_key
    )

    return OcrExtractResponse(
        request_id=str(uuid.uuid4()),
        provider=result.provider,
        text=result.text,
        dimensions=result.dimensions,
        symbols=result.symbols,
        title_block=result.title_block,
        overall_confidence=result.overall_confidence,
        calibrated_confidence=result.calibrated_confidence,
        cache_hit=result.cache_hit,
        processing_time_ms=result.processing_time_ms,
        quality_report=result.quality_report,
        evidence_chain=result.evidence_chain
    )


@router.get("/health")
async def ocr_health_check():
    """OCR健康检查"""
    health_status = {"status": "healthy", "providers": {}}

    for provider_name, provider in ocr_manager.providers.items():
        is_healthy = await provider.health_check()
        health_status["providers"][provider_name.value] = {
            "status": "up" if is_healthy else "down"
        }

    return health_status
```

#### 4.2.2 集成到analyze端点

```python
"""
扩展现有analyze端点
src/api/v1/analyze.py (部分)
"""

class AnalysisOptions(BaseModel):
    # ... 原有字段 ...

    # OCR增强
    enable_ocr: bool = Field(default=False, description="启用OCR增强")
    ocr_provider: str = Field(default="auto", description="OCR Provider")


@router.post("/", response_model=AnalysisResult)
async def analyze_cad_file(
    file: UploadFile = File(...),
    options: str = Form(...),
    api_key: str = Depends(get_api_key)
):
    # ... 现有逻辑 ...

    # OCR增强
    if analysis_options.enable_ocr:
        ocr_manager = OcrManager()
        ocr_result = await ocr_manager.extract(
            image=content,
            strategy=analysis_options.ocr_provider
        )

        # 合并到分析结果
        results['ocr'] = {
            'text': ocr_result.text,
            'dimensions': [d.dict() for d in ocr_result.dimensions],
            'title_block': ocr_result.title_block.dict() if ocr_result.title_block else None,
            'confidence': ocr_result.calibrated_confidence
        }

        # 使用OCR增强工艺建议
        if ocr_result.title_block and ocr_result.title_block.material:
            results['process']['material_specific'] = \
                await get_material_specific_process(ocr_result.title_block.material)

    # ...
```

---

## 五、配置管理

### 5.1 环境配置矩阵

```python
"""
环境配置
src/config/ocr_config.py
"""

from enum import Enum
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    GPU_WORKSTATION = "gpu_workstation"
    PRODUCTION = "production"


class OcrConfig(BaseSettings):
    """OCR配置"""

    # 环境
    ENVIRONMENT: Environment = Environment.DEVELOPMENT

    # Provider配置
    OCR_PROVIDER: str = "auto"

    # DeepSeek配置
    DEEPSEEK_ENABLED: bool = True
    DEEPSEEK_MODEL: str = "deepseek-ai/DeepSeek-OCR"
    DEEPSEEK_DEVICE: str = "cuda"
    DEEPSEEK_MODE: str = "hf"  # hf/vllm

    # 质量控制
    CONFIDENCE_THRESHOLD: float = 0.7
    CONFIDENCE_FALLBACK: float = 0.85
    OCR_TIMEOUT_MS: int = 15000

    # 并发控制
    OCR_MAX_CONCURRENT: int = 5
    OCR_QUEUE_SIZE: int = 100

    # 缓存
    OCR_CACHE_ENABLED: bool = True
    OCR_CACHE_TTL: int = 86400  # 24h

    # 灰度发布
    ENABLE_GRADUAL_ROLLOUT: bool = False
    DEEPSEEK_ROLLOUT_PERCENTAGE: int = 0

    class Config:
        env_file = ".env"
```

### 5.2 环境配置文件

```bash
# .env.development (开发环境 - CPU)
ENVIRONMENT=development
OCR_PROVIDER=auto
DEEPSEEK_ENABLED=false
PADDLE_ENABLED=true
PADDLE_USE_GPU=false
CONFIDENCE_FALLBACK=0.85
OCR_TIMEOUT_MS=10000
LOG_LEVEL=DEBUG

# .env.gpu_workstation (GPU工作站)
ENVIRONMENT=gpu_workstation
OCR_PROVIDER=deepseek_hf
DEEPSEEK_ENABLED=true
DEEPSEEK_DEVICE=cuda
PADDLE_USE_GPU=true
CONFIDENCE_FALLBACK=0.80
OCR_MAX_CONCURRENT=8

# .env.production (生产环境)
ENVIRONMENT=production
OCR_PROVIDER=auto
DEEPSEEK_ENABLED=true
DEEPSEEK_DEVICE=cuda
DEEPSEEK_MODE=vllm

CONFIDENCE_FALLBACK=0.85
OCR_MAX_CONCURRENT=10
FALLBACK_ON_ERROR=true

ENABLE_GRADUAL_ROLLOUT=true
DEEPSEEK_ROLLOUT_PERCENTAGE=20

ENABLE_METRICS=true
```

---

## 六、实施路线

### Week 1: 核心能力 (MVP)

| 日期 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| **Day 1-2** | 基础框架 | - base.py抽象<br>- OcrManager<br>- DeepSeek-HF Provider<br>- Paddle Provider<br>- /ocr/extract端点 | - API启动成功<br>- 3个样本测试通过<br>- 健康检查正常 |
| **Day 3** | 结构化解析 | - DimensionParser<br>- SymbolParser<br>- Normalizer<br>- Prompt优化 | - JSON解析成功率>80%<br>- 关键字段召回>70% |
| **Day 4** | 证据链 | - 复用calibrator<br>- Evidence集成<br>- DS融合 | - Brier score <0.15<br>- 证据链完整 |
| **Day 5** | 集成/analyze | - 扩展AnalysisOptions<br>- OCR→材料/工艺<br>- API文档 | - 端到端测试通过<br>- 响应时间增量<30% |

### Week 2: 智能路由 + 质量控制

| 日期 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| **Day 1-2** | Auto策略 | - Fallback逻辑<br>- 多provider融合<br>- 缓存优化 | - Fallback触发率<20%<br>- 融合后F1提升>10% |
| **Day 3** | 质量门控 | - QualityGate<br>- 重试控制<br>- 超时处理 | - 关键字段完整性>90%<br>- 超时回退正常 |
| **Day 4-5** | 评测体系 | - 10个golden cases<br>- CI集成<br>- 指标计算 | - Edge F1≥0.75<br>- 召回≥0.80<br>- P95延迟<5s |

### Week 3-4: 高级特性 (按需)

- **Week 3**: 几何对齐 (对齐成功率>60%)
- **Week 4**: 监控完善 (Grafana面板、灰度发布)

---

## 七、监控与评测

### 7.1 Prometheus指标

```python
"""
OCR Prometheus指标
src/core/ocr/utils/metrics.py
"""

from prometheus_client import Counter, Histogram, Gauge

# 请求计数
ocr_requests_total = Counter(
    'ocr_requests_total',
    'Total OCR requests',
    ['provider', 'status']
)

# 字段召回率
ocr_field_recall = Gauge(
    'ocr_field_recall',
    'Critical field recall rate',
    ['field_type']  # dimension/tolerance/title_block
)

# 置信度分布
ocr_confidence_score = Histogram(
    'ocr_confidence_score',
    'Calibrated confidence distribution',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# Fallback触发
ocr_fallback_triggered = Counter(
    'ocr_fallback_triggered',
    'Fallback to secondary provider',
    ['reason']  # low_confidence/missing_field/parse_error
)

# 处理时延
ocr_processing_time = Histogram(
    'ocr_processing_time_seconds',
    'OCR processing latency',
    ['provider'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
```

### 7.2 评测数据集

```python
"""
OCR评测数据集
tests/ocr/golden_cases.py
"""

GOLDEN_CASES = [
    {
        "name": "clear_vector_drawing",
        "description": "清晰矢量图",
        "image_path": "test_data/clear_vector.png",
        "ground_truth": {
            "dimensions": [
                {"type": "diameter", "value": 20.0, "tolerance": 0.02},
                {"type": "radius", "value": 5.0},
            ],
            "title_block": {
                "drawing_number": "GJ-2024-001",
                "material": "20CrMnTi",
            }
        },
        "min_confidence": 0.90
    },
    {
        "name": "scanned_drawing",
        "description": "扫描图纸",
        "min_confidence": 0.75
    },
    {
        "name": "blurry_photo",
        "description": "模糊照片",
        "min_confidence": 0.60
    },
    # ... 更多用例
]
```

### 7.3 CI集成

```yaml
# .github/workflows/ocr_tests.yml
name: OCR Quality Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  ocr-evaluation:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run golden set evaluation
      run: python tests/ocr/run_golden_evaluation.py

    - name: Check quality gates
      run: |
        python scripts/check_ocr_metrics.py \
          --min-edge-f1 0.75 \
          --min-field-recall 0.80 \
          --max-p95-latency 5000
```

---

## 八、生产部署

### 8.1 Docker Compose

```yaml
# docker-compose.ocr.yml
version: '3.8'

services:
  cad-ml-platform:
    build: .
    environment:
      - ENVIRONMENT=production
      - OCR_PROVIDER=auto
      - DEEPSEEK_ENABLED=true
      - DEEPSEEK_DEVICE=cuda
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/models
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### 8.2 健康检查与预热

```python
"""
FastAPI lifespan集成
src/main.py (更新)
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""

    # 启动阶段
    logger.info("🚀 Starting CAD ML Platform...")

    # 1. 初始化Redis
    await init_redis()

    # 2. 加载ML模型
    await load_models()

    # 3. 初始化OCR
    app.state.ocr_manager = OcrManager()

    # 4. OCR预热
    for provider_name, provider in app.state.ocr_manager.providers.items():
        logger.info(f"🔥 Warming up {provider_name.value}...")
        await provider.warmup()
        logger.info(f"✅ {provider_name.value} ready")

    # 5. 训练校准器
    if not app.state.ocr_manager.calibrator.calibrator.fitted:
        app.state.ocr_manager._train_calibrator_from_golden_set(
            app.state.ocr_manager.calibrator
        )

    logger.info("✅ CAD ML Platform started")

    yield

    # 关闭阶段
    logger.info("🛑 Shutting down...")


# 增强健康检查
@app.get("/health")
async def health_check(request: Request):
    """综合健康检查"""
    health_status = {
        "status": "healthy",
        "services": {
            "api": "up",
            "redis": await _check_redis(),
            "ml": await _check_ml_models()
        }
    }

    # OCR健康检查
    if hasattr(request.app.state, 'ocr_manager'):
        ocr_manager = request.app.state.ocr_manager
        ocr_health = {"overall": "unknown", "providers": {}}

        for provider_name, provider in ocr_manager.providers.items():
            is_healthy = await provider.health_check()
            is_circuit_broken = ocr_manager.executor.fallback_strategy.should_skip_provider(
                provider_name.value
            )

            ocr_health["providers"][provider_name.value] = {
                "status": "up" if is_healthy else "down",
                "ready": is_healthy and not is_circuit_broken,
                "circuit_broken": is_circuit_broken
            }

        any_ready = any(p["ready"] for p in ocr_health["providers"].values())
        ocr_health["overall"] = "up" if any_ready else "degraded"

        health_status["services"]["ocr"] = ocr_health

    return health_status
```

### 8.3 快速启动脚本

```bash
#!/bin/bash
# scripts/quick_start_ocr.sh

set -e

echo "🚀 CAD ML Platform OCR Quick Start"

# 环境检测
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected"
    ENV="gpu_workstation"
else
    echo "⚠️  CPU mode"
    ENV="development"
fi

# 配置环境
cp .env.example .env.$ENV
export ENVIRONMENT=$ENV

# 安装依赖
pip install -r requirements.txt

if [ "$ENV" = "gpu_workstation" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

# 下载模型
if [ ! -d "models/deepseek-ocr" ]; then
    python scripts/download_models.py --model deepseek-ocr
fi

# 预热测试
python scripts/test_ocr_warmup.py

# 启动服务
if [ "$ENV" = "development" ]; then
    python src/main.py
else
    gunicorn src.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
fi
```

### 8.4 Kubernetes生产部署

```yaml
# k8s/ocr-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: cad-ml-ocr
  namespace: production
  labels:
    app: cad-ml-platform
    component: ocr
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # 零停机部署
  selector:
    matchLabels:
      app: cad-ml-platform
      component: ocr
  template:
    metadata:
      labels:
        app: cad-ml-platform
        component: ocr
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      nodeSelector:
        gpu: "nvidia-t4"  # GPU节点选择
      containers:
      - name: ocr-service
        image: your-registry/cad-ml-platform:ocr-v1.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DEEPSEEK_DEVICE
          value: "cuda"
        - name: OCR_MAX_CONCURRENT
          value: "10"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: cad-ml-secrets
              key: redis-url
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120  # 模型加载需要时间
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: ocr-model-pvc
      - name: config
        configMap:
          name: ocr-config
      initContainers:
      - name: model-downloader
        image: your-registry/model-downloader:latest
        command: ['sh', '-c', 'python /scripts/download_models.py']
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
---
apiVersion: v1
kind: Service
metadata:
  name: cad-ml-ocr-service
  namespace: production
spec:
  type: ClusterIP
  selector:
    app: cad-ml-platform
    component: ocr
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cad-ml-ocr-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cad-ml-ocr
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: ocr_requests_per_second
      target:
        type: AverageValue
        averageValue: "50"
```

### 8.5 负载均衡与扩缩容策略

**负载均衡配置** (Nginx Ingress):

```yaml
# k8s/ingress.yaml

apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cad-ml-ocr-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"  # OCR图片上传
    nginx.ingress.kubernetes.io/proxy-read-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "30"
    nginx.ingress.kubernetes.io/rate-limit: "100"  # 限流
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.cad-ml.yourcompany.com
    secretName: cad-ml-tls
  rules:
  - host: api.cad-ml.yourcompany.com
    http:
      paths:
      - path: /api/v1/ocr
        pathType: Prefix
        backend:
          service:
            name: cad-ml-ocr-service
            port:
              number: 80
```

**扩缩容策略决策树**:

```yaml
scaling_strategy:
  # 场景1: 日常负载 (工作日 9:00-18:00)
  business_hours:
    min_replicas: 3
    max_replicas: 6
    target_cpu: 70%
    target_qps: 50

  # 场景2: 高峰期 (月末出图高峰)
  peak_hours:
    min_replicas: 6
    max_replicas: 10
    target_cpu: 60%
    target_qps: 100

  # 场景3: 低谷期 (夜间/周末)
  off_hours:
    min_replicas: 1
    max_replicas: 3
    target_cpu: 80%

  # 自动策略: 基于CronHPA
  scheduled_scaling:
    - schedule: "0 8 * * 1-5"  # 工作日早8点
      replicas: 3
    - schedule: "0 18 * * 1-5"  # 工作日晚6点
      replicas: 1
    - schedule: "0 8 25-31 * *"  # 月末早8点
      replicas: 6
```

### 8.6 备份与灾难恢复

**关键数据备份**:

```bash
#!/bin/bash
# scripts/backup_ocr_data.sh

# 1. Redis缓存快照 (可选,缓存数据可重建)
redis-cli --rdb /backup/redis/ocr_cache_$(date +%Y%m%d).rdb

# 2. 置信度校准模型 (关键)
cp models/calibration/*.pkl /backup/models/$(date +%Y%m%d)/

# 3. Golden测试集 (关键)
tar -czf /backup/golden_sets/golden_$(date +%Y%m%d).tar.gz tests/ocr/golden_cases/

# 4. 配置文件
cp -r config/ /backup/config/$(date +%Y%m%d)/

# 5. Prometheus指标历史 (可选)
curl -X POST http://prometheus:9090/api/v1/admin/tsdb/snapshot > /backup/metrics/snapshot_$(date +%Y%m%d).json

echo "✅ Backup completed: $(date)"
```

**灾难恢复计划 (RTO/RPO)**:

| 故障场景 | RTO目标 | RPO目标 | 恢复步骤 |
|---------|---------|---------|---------|
| **单Pod故障** | <1分钟 | 0 (无损) | K8s自动重启,无需干预 |
| **节点故障** | <5分钟 | 0 | K8s调度到其他节点 |
| **Redis故障** | <10分钟 | <1小时 | 切换到备用Redis实例 |
| **模型损坏** | <30分钟 | 0 | 从备份恢复calibration模型 |
| **整个集群故障** | <2小时 | <4小时 | DR集群激活+数据恢复 |
| **数据中心故障** | <4小时 | <12小时 | 异地容灾中心接管 |

**恢复演练脚本**:

```bash
#!/bin/bash
# scripts/disaster_recovery_drill.sh

echo "🔥 开始灾难恢复演练..."

# 1. 模拟故障
kubectl delete deployment cad-ml-ocr -n production

# 2. 启动计时
start_time=$(date +%s)

# 3. 执行恢复
kubectl apply -f k8s/ocr-deployment.yaml

# 4. 等待健康
kubectl wait --for=condition=ready pod -l app=cad-ml-platform -n production --timeout=300s

# 5. 验证功能
curl -f http://api.cad-ml.yourcompany.com/health || exit 1

# 6. 计算RTO
end_time=$(date +%s)
rto=$((end_time - start_time))

echo "✅ 恢复完成! RTO实际: ${rto}秒 (目标: <600秒)"

if [ $rto -gt 600 ]; then
    echo "❌ RTO超标,需优化!"
    exit 1
fi
```

### 8.7 安全加固

**Secrets管理** (使用Kubernetes Secrets + Sealed Secrets):

```yaml
# k8s/sealed-secrets.yaml (加密后可安全提交Git)

apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: cad-ml-secrets
  namespace: production
spec:
  encryptedData:
    redis-url: AgBXYZ...  # 加密的Redis连接串
    deepseek-api-key: AgBABC...  # 如使用API模式
    prometheus-token: AgBDEF...
```

**API密钥轮转**:

```python
"""
API Key轮转策略
src/core/security/key_rotation.py
"""

import hashlib
from datetime import datetime, timedelta

class ApiKeyRotation:
    """API密钥自动轮转"""

    def __init__(self, rotation_days: int = 90):
        self.rotation_days = rotation_days

    async def should_rotate(self, key_created_at: datetime) -> bool:
        """检查是否需要轮转"""
        age = datetime.utcnow() - key_created_at
        return age > timedelta(days=self.rotation_days)

    async def rotate_key(self, old_key: str) -> tuple[str, str]:
        """轮转密钥,返回(new_key, old_key_hash)"""
        new_key = self._generate_key()
        old_hash = hashlib.sha256(old_key.encode()).hexdigest()

        # 宽限期: 新旧key都有效24小时
        await self._set_dual_mode(new_key, old_hash, grace_hours=24)

        return new_key, old_hash
```

**访问控制** (RBAC):

```python
"""
基于角色的访问控制
src/middleware/rbac.py
"""

from enum import Enum
from fastapi import HTTPException, Header

class UserRole(str, Enum):
    ADMIN = "admin"
    ENGINEER = "engineer"
    VIEWER = "viewer"

class PermissionMatrix:
    """权限矩阵"""

    PERMISSIONS = {
        UserRole.ADMIN: {
            "/api/v1/ocr/extract",
            "/api/v1/ocr/batch",
            "/api/v1/admin/*"
        },
        UserRole.ENGINEER: {
            "/api/v1/ocr/extract",
            "/api/v1/ocr/batch"
        },
        UserRole.VIEWER: {
            "/api/v1/ocr/extract"  # 仅单次查询
        }
    }

async def check_permission(
    endpoint: str,
    x_user_role: str = Header(..., alias="X-User-Role")
):
    """权限检查中间件"""
    role = UserRole(x_user_role)
    allowed = PermissionMatrix.PERMISSIONS.get(role, set())

    if not any(endpoint.startswith(p.rstrip('*')) for p in allowed):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
```

### 8.8 性能调优指南

**GPU优化配置**:

```python
"""
GPU推理优化
src/core/ocr/providers/deepseek_hf.py (优化版)
"""

import torch
from transformers import AutoModelForCausalLM

class OptimizedDeepSeekClient:
    """性能优化版DeepSeek客户端"""

    def __init__(self):
        # 1. 混合精度推理 (FP16)
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-vl2-tiny",
            torch_dtype=torch.float16,  # 节省50%显存
            device_map="auto"
        )

        # 2. 编译模型 (PyTorch 2.0+)
        self.model = torch.compile(self.model, mode="reduce-overhead")

        # 3. KV缓存优化
        self.model.config.use_cache = True

        # 4. Flash Attention (需支持)
        if hasattr(self.model.config, 'attn_implementation'):
            self.model.config.attn_implementation = "flash_attention_2"

    async def batch_infer(self, images: list[bytes]) -> list[OcrResult]:
        """批量推理 (提升吞吐)"""
        # Dynamic batching: 收集50ms内请求
        batch = await self._collect_batch(images, timeout_ms=50)

        with torch.no_grad(), torch.cuda.amp.autocast():  # 自动混合精度
            outputs = self.model.generate(
                batch,
                max_new_tokens=512,
                num_beams=1,  # Greedy解码更快
                do_sample=False
            )

        return [self._parse(o) for o in outputs]
```

**缓存策略优化**:

```python
"""
多级缓存
src/core/ocr/cache/multi_tier.py
"""

from functools import lru_cache
import hashlib

class MultiTierCache:
    """三级缓存: 内存 → Redis → S3"""

    def __init__(self):
        self.l1_cache = {}  # 内存LRU (100条)
        self.redis_client = Redis()  # L2 (10K条, 1小时TTL)
        self.s3_client = S3Client()  # L3 (永久归档)

    async def get(self, image_hash: str) -> Optional[OcrResult]:
        """缓存查询"""
        # L1: 内存
        if result := self.l1_cache.get(image_hash):
            metrics.cache_hit.labels(tier="l1").inc()
            return result

        # L2: Redis
        if cached := await self.redis_client.get(f"ocr:{image_hash}"):
            result = OcrResult.parse_raw(cached)
            self.l1_cache[image_hash] = result  # 回填L1
            metrics.cache_hit.labels(tier="l2").inc()
            return result

        # L3: S3 (低频访问)
        if archived := await self.s3_client.get(f"ocr-archive/{image_hash}.json"):
            result = OcrResult.parse_raw(archived)
            await self.redis_client.setex(f"ocr:{image_hash}", 3600, result.json())
            metrics.cache_hit.labels(tier="l3").inc()
            return result

        metrics.cache_miss.inc()
        return None
```

**数据库连接池调优**:

```python
"""
Redis连接池优化
src/core/database/redis_pool.py
"""

from redis.asyncio import ConnectionPool, Redis

def create_optimized_pool():
    """优化的Redis连接池"""
    return ConnectionPool(
        host="redis.production.svc.cluster.local",
        port=6379,
        db=0,

        # 连接池大小 = 并发数 * 1.2
        max_connections=12,  # 10并发 * 1.2

        # 健康检查
        health_check_interval=30,

        # 超时控制
        socket_connect_timeout=5,
        socket_timeout=3,

        # 重试策略
        retry_on_timeout=True,
        retry=Retry(ExponentialBackoff(), 3)
    )
```

### 8.9 监控告警配置

**Prometheus告警规则**:

```yaml
# k8s/prometheus-alerts.yaml

apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ocr-alerts
  namespace: production
spec:
  groups:
  - name: ocr.errors
    interval: 30s
    rules:
    - alert: OcrHighErrorRate
      expr: |
        rate(ocr_requests_total{status="error"}[5m])
        / rate(ocr_requests_total[5m]) > 0.05
      for: 2m
      labels:
        severity: warning
        component: ocr
      annotations:
        summary: "OCR错误率过高"
        description: "错误率{{ $value | humanizePercentage }}, 超过5%阈值"

    - alert: OcrP95LatencyHigh
      expr: |
        histogram_quantile(0.95,
          rate(ocr_processing_duration_seconds_bucket[5m])
        ) > 5
      for: 3m
      labels:
        severity: warning
      annotations:
        summary: "OCR P95延迟过高"
        description: "P95延迟{{ $value }}秒, 超过5秒SLA"

    - alert: DeepSeekProviderDown
      expr: |
        ocr_provider_health{provider="deepseek_hf"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "DeepSeek服务异常"
        description: "DeepSeek provider健康检查失败"

    - alert: OcrCacheMissRateHigh
      expr: |
        rate(ocr_cache_miss_total[10m])
        / rate(ocr_cache_requests_total[10m]) > 0.8
      for: 5m
      labels:
        severity: info
      annotations:
        summary: "缓存命中率低"
        description: "缓存未命中率{{ $value | humanizePercentage }}"

  - name: ocr.capacity
    interval: 1m
    rules:
    - alert: OcrConcurrencyNearLimit
      expr: |
        ocr_concurrent_requests > 8
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "并发数接近上限"
        description: "当前并发{{ $value }}, 上限10"

    - alert: GpuMemoryHigh
      expr: |
        nvidia_gpu_memory_used_bytes
        / nvidia_gpu_memory_total_bytes > 0.85
      for: 3m
      labels:
        severity: warning
      annotations:
        summary: "GPU显存使用率高"
        description: "GPU显存使用{{ $value | humanizePercentage }}"
```

**Grafana仪表盘JSON** (关键面板):

```json
{
  "dashboard": {
    "title": "CAD ML OCR监控",
    "panels": [
      {
        "title": "QPS & 错误率",
        "targets": [
          {
            "expr": "rate(ocr_requests_total[1m])",
            "legendFormat": "QPS"
          },
          {
            "expr": "rate(ocr_requests_total{status='error'}[1m])",
            "legendFormat": "错误QPS"
          }
        ]
      },
      {
        "title": "延迟分位数",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(ocr_processing_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(ocr_processing_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(ocr_processing_duration_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ]
      },
      {
        "title": "Provider健康度",
        "targets": [
          {
            "expr": "ocr_provider_health",
            "legendFormat": "{{provider}}"
          }
        ],
        "type": "stat",
        "fieldConfig": {
          "thresholds": {
            "steps": [
              {"value": 0, "color": "red"},
              {"value": 1, "color": "green"}
            ]
          }
        }
      },
      {
        "title": "缓存命中率",
        "targets": [
          {
            "expr": "rate(ocr_cache_hit_total[5m]) / rate(ocr_cache_requests_total[5m])",
            "legendFormat": "命中率"
          }
        ]
      }
    ]
  }
}
```

### 8.10 成本优化策略

**GPU实例成本优化**:

```yaml
cost_optimization_playbook:

  # 策略1: Spot实例 (节省70%成本)
  spot_instances:
    enabled: true
    interruption_handling: graceful_shutdown
    fallback: on_demand_instances
    saving: ~70%

  # 策略2: 混合实例组
  instance_mix:
    on_demand: 1  # 核心节点
    spot: 2       # 弹性节点
    saving: ~50%

  # 策略3: 按需Auto Scaling
  autoscaling:
    scale_down_delay: 5m  # 快速缩容
    scale_up_threshold: cpu>60% OR qps>80
    saving: ~40%

  # 策略4: DeepSeek灰度策略
  deepseek_rollout:
    strategy: confidence_based  # 仅低置信度触发
    percentage: 20%  # 仅20%请求使用GPU
    saving: ~80% GPU hours

  # 策略5: 缓存延长TTL
  cache_optimization:
    redis_ttl: 24h  # 从1h延长到24h
    hit_rate_gain: +30%
    cost_reduction: -30% API calls
```

**成本监控仪表盘**:

```python
"""
成本跟踪
src/observability/cost_tracker.py
"""

from prometheus_client import Counter, Gauge

# 成本指标
cost_metrics = {
    "gpu_hours": Gauge("cost_gpu_hours_total", "GPU使用小时数"),
    "api_calls": Counter("cost_api_calls_total", "API调用次数", ["provider"]),
    "storage_gb": Gauge("cost_storage_gb", "存储使用量GB"),
    "network_gb": Counter("cost_network_gb_total", "网络传输GB")
}

class CostCalculator:
    """成本计算器"""

    PRICING = {
        "gpu_t4_hour": 0.35,  # USD/hour
        "deepseek_api_call": 0.001,  # USD/call (如使用API)
        "redis_gb_month": 0.10,
        "bandwidth_gb": 0.02
    }

    async def calculate_daily_cost(self) -> float:
        """计算每日成本"""
        gpu_cost = cost_metrics["gpu_hours"].get() * self.PRICING["gpu_t4_hour"]
        api_cost = cost_metrics["api_calls"].labels(provider="deepseek").get() * self.PRICING["deepseek_api_call"]
        storage_cost = cost_metrics["storage_gb"].get() * self.PRICING["redis_gb_month"] / 30
        network_cost = cost_metrics["network_gb"].get() * self.PRICING["bandwidth_gb"]

        total = gpu_cost + api_cost + storage_cost + network_cost

        return total
```

### 8.11 故障排查手册

**常见问题诊断流程**:

```bash
#!/bin/bash
# scripts/troubleshoot_ocr.sh

echo "🔍 OCR故障排查工具"

# 1. 检查Pod状态
echo "=== Pod健康检查 ==="
kubectl get pods -n production -l component=ocr
kubectl describe pod -n production -l component=ocr | grep -A 5 "Events:"

# 2. 检查日志
echo "=== 最近错误日志 ==="
kubectl logs -n production -l component=ocr --tail=50 | grep -i "error\|exception"

# 3. 检查GPU可用性
echo "=== GPU状态 ==="
kubectl exec -n production -it $(kubectl get pod -n production -l component=ocr -o jsonpath='{.items[0].metadata.name}') -- nvidia-smi

# 4. 检查Provider健康
echo "=== Provider健康度 ==="
curl -s http://api.cad-ml.yourcompany.com/ocr/health | jq '.providers'

# 5. 检查Prometheus指标
echo "=== 关键指标 ==="
curl -s http://prometheus:9090/api/v1/query?query=ocr_requests_total | jq '.data.result[] | {metric, value}'

# 6. 检查熔断器状态
echo "=== 熔断器状态 ==="
curl -s http://api.cad-ml.yourcompany.com/ocr/circuit-breaker-status

# 7. 网络连通性
echo "=== Redis连接 ==="
kubectl exec -n production $(kubectl get pod -n production -l component=ocr -o jsonpath='{.items[0].metadata.name}') -- redis-cli -h redis.production.svc.cluster.local ping

echo "✅ 排查完成"
```

**错误代码对照表**:

| 错误代码 | 含义 | 排查步骤 | 解决方案 |
|---------|------|---------|---------|
| `OCR_001` | Provider初始化失败 | 1) 检查GPU可用性<br>2) 检查模型文件 | 重启Pod / 重新下载模型 |
| `OCR_002` | JSON解析失败 | 1) 查看原始输出<br>2) 检查prompt | 启用Markdown fallback |
| `OCR_003` | 置信度校准器未训练 | 1) 检查golden set<br>2) 查看calibrator日志 | 运行`python scripts/train_calibrator.py` |
| `OCR_004` | Redis连接超时 | 1) ping Redis<br>2) 检查网络策略 | 检查Redis健康 / 增加超时 |
| `OCR_005` | 并发数超限 | 1) 查看当前并发<br>2) 检查HPA | 扩容Pod / 增加`OCR_MAX_CONCURRENT` |
| `OCR_006` | 熔断器触发 | 1) 查看错误日志<br>2) 检查provider健康 | 等待自动恢复 / 手动重置 |

### 8.12 SLA/SLO定义

**服务等级目标** (SLO):

```yaml
slo_targets:

  # 可用性SLO
  availability:
    target: 99.5%  # 每月允许停机 3.6小时
    measurement: uptime / total_time
    breach_threshold: < 99.0%

  # 延迟SLO
  latency:
    p50_target: "< 2s"
    p95_target: "< 5s"
    p99_target: "< 10s"
    measurement: histogram_quantile(0.95, ocr_processing_duration_seconds)
    breach_threshold: p95 > 7s

  # 准确性SLO
  accuracy:
    edge_f1_target: "> 0.75"
    field_recall_target: "> 0.80"
    measurement: weekly_evaluation_report
    breach_threshold: edge_f1 < 0.70

  # 错误率SLO
  error_rate:
    target: "< 2%"
    measurement: error_requests / total_requests
    breach_threshold: > 5%
```

**SLA承诺** (对外):

```markdown
# CAD ML OCR服务SLA

## 服务可用性
- **标准承诺**: 99.5% 月度可用性
- **赔偿阈值**: < 99.0%
- **计算公式**: (总时间 - 故障时间) / 总时间

## 性能承诺
- **P95延迟**: < 5秒
- **P99延迟**: < 10秒
- **测量周期**: 每5分钟滚动窗口

## 准确性保证
- **边缘F1分数**: ≥ 0.75 (周平均)
- **字段召回率**: ≥ 0.80 (dimension/title_block)
- **置信度校准**: Brier Score < 0.20

## 支持响应时间
- **P0 (服务全面中断)**: 15分钟响应, 2小时恢复
- **P1 (核心功能不可用)**: 1小时响应, 8小时恢复
- **P2 (性能下降)**: 4小时响应, 24小时恢复
- **P3 (一般问题)**: 1工作日响应

## 维护窗口
- **计划维护**: 每月第二个周日 02:00-05:00 UTC+8
- **提前通知**: 至少提前7天通知
- **紧急维护**: 提前24小时通知
```

---

## 九、验收标准

### 9.1 Week 1 MVP验收

#### Day 1-2: 基础框架

```bash
✅ 环境配置文件: .env.dev, .env.gpu, .env.prod 存在
✅ 服务启动: python src/main.py 无错误
✅ 健康检查: curl http://localhost:8000/health 返回200
✅ OCR健康: curl http://localhost:8000/ocr/health 显示providers状态
✅ Paddle可用: paddle provider ready=true (开发环境)
✅ DeepSeek可用: deepseek_hf provider ready=true (GPU环境)
```

#### Day 3: 结构化解析

```bash
✅ JSON校验通过率: >80% (在测试样本上)
✅ Markdown fallback: JSON失败时能降级解析
✅ 尺寸解析准确率: 能识别Φ/R/M/±t, >70%
✅ 符号解析: 能识别Ra/⟂/∥, >60%
```

#### Day 4: 证据链

```bash
✅ 置信度校准: calibrated_confidence字段正常输出
✅ 证据链完整: evidence_chain包含provider/confidence/source
✅ Brier score: <0.20 (在golden set上)
✅ DS融合: 多provider结果能正确融合
```

#### Day 5: 端到端集成

```bash
✅ /api/v1/analyze集成: enable_ocr=true返回OCR字段
✅ 材料识别: title_block.material能正确提取
✅ 工艺增强: 基于材料返回工艺建议
✅ 响应时间: P95 <5s (auto模式)
✅ 缓存命中: 重复请求缓存生效
```

### 9.2 Week 2 完整验收

```bash
✅ Auto策略: Fallback触发率 <20%
✅ 质量门控: 关键字段完整性 >90%
✅ 评测指标: Edge F1 ≥0.75
✅ 字段召回: dimension/title_block召回 ≥0.80
✅ 延迟控制: P95 <5s, P99 <10s
✅ 并发稳定: 10并发请求无超时
✅ CI集成: GitHub Actions自动运行评测
```

### 9.3 生产就绪检查清单

- [x] **多环境配置**: dev/gpu/prod三套配置
- [x] **健康检查**: /health + /ready + /ocr/health
- [x] **监控指标**: Prometheus指标完整
- [x] **日志规范**: 结构化JSON日志
- [x] **错误处理**: 超时/熔断/降级
- [x] **文档完善**: API文档 + 部署手册
- [x] **测试覆盖**: golden cases + CI集成
- [x] **性能优化**: 缓存 + 预热 + 并发控制

---

## 十、附录

### 10.1 Prompt模板库

```python
"""
DeepSeek Prompt模板
src/core/ocr/utils/prompt_templates.py
"""

class PromptTemplates:
    """Prompt模板库"""

    @staticmethod
    def free_ocr() -> str:
        """通用OCR"""
        return "<image>\n<|grounding|>Free OCR."

    @staticmethod
    def engineering_drawing_structured() -> str:
        """工程图结构化"""
        return """<image>
<|grounding|>Extract dimensions/tolerances/surface-roughness/threads as strict JSON:
{
  "dimensions": [{"type":"diameter|radius|length|thread", "value":float, "unit":"mm", "tolerance":float, "bbox":{}}],
  "symbols": [{"type":"surface_roughness|perpendicular|parallel", "value":str, "bbox":{}}],
  "title_block": {"drawing_number":str, "material":str, "part_name":str, "scale":str}
}"""

    @staticmethod
    def title_block_focused() -> str:
        """标题栏专注"""
        return """<image>
<|grounding|>Focus on the title block (usually bottom-right corner).
Extract as JSON: {"drawing_number":str, "material":str, "part_name":str, "scale":str, "weight":float}"""
```

### 10.2 常见问题FAQ

**Q1: DeepSeek模型加载失败？**

```bash
# 检查GPU可用性
nvidia-smi

# 检查CUDA版本
python -c "import torch; print(torch.cuda.is_available())"

# 降级到CPU模式
export DEEPSEEK_DEVICE=cpu
```

**Q2: JSON解析失败率高？**

```bash
# 方案1: 优化prompt，强调JSON格式
# 方案2: 启用Markdown fallback
# 方案3: 降低confidence_threshold触发deepseek增强
```

**Q3: 缓存不生效？**

```bash
# 检查Redis连接
redis-cli ping

# 检查缓存配置
echo $OCR_CACHE_ENABLED  # 应为true
```

**Q4: 性能优化建议？**

1. **启用GPU**: `DEEPSEEK_DEVICE=cuda`
2. **提高并发**: `OCR_MAX_CONCURRENT=10`
3. **优化缓存**: 增大`OCR_CACHE_TTL`
4. **预热模型**: 启动时调用`/ocr/warmup`
5. **灰度DeepSeek**: `DEEPSEEK_ROLLOUT_PERCENTAGE=20`

### 10.3 参考资料

- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [PaddleOCR 文档](https://github.com/PaddlePaddle/PaddleOCR)
- [Dempster-Shafer 证据理论](https://en.wikipedia.org/wiki/Dempster%E2%80%93Shafer_theory)
- [Isotonic Regression 校准](https://scikit-learn.org/stable/modules/calibration.html)

### 10.4 术语表

| 术语 | 含义 |
|------|------|
| **Provider** | OCR服务提供商 (paddle/deepseek_hf/vllm) |
| **Auto策略** | 智能路由：先paddle，低置信度→deepseek |
| **几何对齐** | OCR文本框锚定到CAD几何元素 |
| **证据链** | 完整的决策路径：provider+置信度+bbox+规则 |
| **DS融合** | Dempster-Shafer证据理论融合 |
| **质量门控** | 关键字段检查+置信度阈值控制 |
| **熔断降级** | 错误累积触发自动切换备用provider |
| **幂等性** | 相同请求返回相同结果 (通过Idempotency-Key) |

---

**文档结束**

**下一步行动**:
1. 执行 `scripts/quick_start_ocr.sh` 快速启动
2. 阅读 `docs/API_DOCUMENTATION.md` 了解接口详情
3. 运行 `pytest tests/ocr/` 执行单元测试
4. 参考 `examples/ocr_demo.py` 学习使用示例

**联系与支持**:
- 技术支持: tech-support@yourcompany.com
- Issue追踪: [GitHub Issues](https://github.com/your-org/cad-ml-platform/issues)
