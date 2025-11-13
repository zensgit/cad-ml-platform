# 💡 视觉识别成本优化策略

> 如何用最低成本实现最好的CAD图纸识别效果

---

## 📊 成本对比分析

### 场景1：纯付费API方案
```
日处理量：1000张图纸
使用GPT-4 Vision：1000 × $0.02 = $20/天 = $600/月
年成本：$7,200 (约5万人民币)
```

### 场景2：纯开源方案
```
日处理量：1000张图纸
使用开源模型：$0
服务器成本：$200/月 (GPU服务器)
年成本：$2,400 (约1.7万人民币)
```

### 场景3：智能混合方案（推荐）
```
日处理量：1000张图纸
- 80% 用开源（800张）：$0
- 15% 用Claude（150张）：$1.5
- 5% 用GPT-4（50张）：$1
日成本：$2.5 = $75/月
年成本：$900 (约6300人民币)
```

---

## 🎯 免费/低成本实现方案

### 方案A：100%免费开源方案

#### 技术架构
```python
class FreeVisionSystem:
    """
    完全免费的视觉识别系统
    准确率：80-85%
    成本：$0（除服务器外）
    """

    def __init__(self):
        # 1. OCR引擎
        self.ocr = TesseractOCR()  # 或 PaddleOCR

        # 2. 对象检测
        self.detector = YOLOv8()  # 免费预训练模型

        # 3. 图像分类
        self.classifier = CustomCNN()  # 自训练模型

        # 4. 特征提取
        self.feature_extractor = CLIP()  # OpenAI开源
```

#### 实现步骤

**Step 1: 安装免费工具**
```bash
# OCR工具
sudo apt-get install tesseract-ocr
pip install pytesseract

# 或使用PaddleOCR（中文更好）
pip install paddlepaddle paddleocr

# 对象检测
pip install ultralytics  # YOLO

# 图像理解
pip install transformers  # CLIP, BLIP等
```

**Step 2: 构建识别管道**
```python
# src/vision/free_vision.py
import pytesseract
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import numpy as np

class FreeCADVision:
    """完全免费的CAD视觉识别"""

    def __init__(self):
        # 初始化免费模型
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        self.yolo = YOLO('yolov8x.pt')  # 预训练模型

    def analyze_drawing(self, image_path):
        """分析CAD图纸"""

        # 1. OCR提取文字
        ocr_result = self.ocr.ocr(image_path, cls=True)
        text_info = self._parse_ocr(ocr_result)

        # 2. 对象检测
        detection_result = self.yolo(image_path)
        objects = self._parse_detection(detection_result)

        # 3. 特征提取（使用OpenCV）
        features = self._extract_features(image_path)

        # 4. 规则推理
        part_type = self._infer_part_type(text_info, objects, features)

        return {
            "part_type": part_type,
            "dimensions": text_info['dimensions'],
            "materials": text_info['materials'],
            "confidence": 0.8
        }

    def _parse_ocr(self, result):
        """解析OCR结果"""
        dimensions = []
        materials = []

        for line in result[0]:
            text = line[1][0]

            # 识别尺寸
            if 'Φ' in text or 'R' in text or 'mm' in text:
                dimensions.append(text)

            # 识别材料
            if any(m in text for m in ['钢', '铝', '铁', '铜']):
                materials.append(text)

        return {
            'dimensions': dimensions,
            'materials': materials
        }

    def _extract_features(self, image_path):
        """使用OpenCV提取特征"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 轮廓检测
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 霍夫圆检测
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20)

        return {
            'edge_count': len(edges),
            'contour_count': len(contours),
            'circle_count': 0 if circles is None else len(circles[0])
        }

    def _infer_part_type(self, text_info, objects, features):
        """基于规则推断零件类型"""

        # 简单规则推理
        if features['circle_count'] > 5:
            return "齿轮"
        elif 'Φ' in str(text_info['dimensions']):
            return "轴类零件"
        elif features['contour_count'] > 10:
            return "复杂零件"
        else:
            return "板材"
```

**Step 3: 训练自己的模型（可选）**
```python
# 使用免费的标注数据训练
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class CADClassifier(nn.Module):
    """自定义CAD分类器"""

    def __init__(self, num_classes=10):
        super().__init__()
        # 使用预训练的ResNet作为基础
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.base_model(x)

# 免费训练
def train_free_model():
    model = CADClassifier()

    # 使用免费的CAD数据集
    # 1. ABC Dataset (开源)
    # 2. Thingiverse (开源3D模型)
    # 3. 自己标注的数据

    # 训练代码...
```

---

### 方案B：低成本混合方案（推荐）

#### 智能路由策略
```python
class SmartVisionRouter:
    """
    智能选择最合适的识别服务
    月成本：< $100
    准确率：90%+
    """

    def __init__(self):
        self.free_vision = FreeCADVision()
        self.paid_vision = None  # 按需初始化

    def analyze(self, image, importance="normal"):
        """
        根据重要性选择服务

        importance级别：
        - low: 纯免费方案
        - normal: 免费为主，必要时付费
        - high: 直接使用付费API
        """

        if importance == "low":
            # 批量处理、初步筛选
            return self.free_vision.analyze(image)

        elif importance == "normal":
            # 先用免费识别
            free_result = self.free_vision.analyze(image)

            # 如果置信度低，才用付费API
            if free_result['confidence'] < 0.7:
                return self.use_paid_api(image, provider="claude")  # Claude更便宜

            return free_result

        else:  # high importance
            # 重要图纸直接用最好的
            return self.use_paid_api(image, provider="gpt4")

    def use_paid_api(self, image, provider="claude"):
        """按需调用付费API"""

        if provider == "claude":
            # Claude API - $0.008/图
            return self.claude_analyze(image)
        elif provider == "gpt4":
            # GPT-4 Vision - $0.02/图
            return self.gpt4_analyze(image)
        else:
            # Google Vision - $0.0015/图
            return self.google_vision_analyze(image)
```

#### 成本控制策略
```python
class CostController:
    """成本控制器"""

    def __init__(self, monthly_budget=100):
        self.budget = monthly_budget
        self.current_cost = 0
        self.api_usage = {
            'free': 0,
            'google': 0,
            'claude': 0,
            'gpt4': 0
        }

    def select_service(self, priority=1):
        """根据预算选择服务"""

        remaining_budget = self.budget - self.current_cost

        if remaining_budget <= 0:
            # 预算用完，只用免费
            return 'free'

        if priority == 1:  # 低优先级
            return 'free'
        elif priority == 2:  # 中等
            if remaining_budget > 50:
                return 'google'  # 最便宜的付费
            return 'free'
        else:  # 高优先级
            if remaining_budget > 20:
                return 'claude' if remaining_budget < 50 else 'gpt4'
            return 'free'
```

---

## 🚀 推荐实施路径

### 第一阶段：先用免费方案验证（0成本）
```python
# Week 1-2: 纯开源实现
1. Tesseract/PaddleOCR - 文字识别
2. YOLO - 对象检测
3. OpenCV - 特征提取
4. 规则引擎 - 零件分类

准确率：75-80%
成本：$0
```

### 第二阶段：引入低成本API（$50/月）
```python
# Week 3-4: 混合方案
1. 80% 使用免费方案
2. 20% 使用Google Cloud Vision
3. 缓存结果避免重复调用

准确率：85-88%
成本：$50/月
```

### 第三阶段：智能混合优化（$100/月）
```python
# Week 5-6: 生产方案
1. 70% 免费方案
2. 20% Google Vision
3. 8% Claude API
4. 2% GPT-4 (仅重要图纸)

准确率：90-92%
成本：$100/月
```

---

## 💡 节省成本的技巧

### 1. 缓存策略
```python
# 使用Redis缓存识别结果
import hashlib
import redis

class VisionCache:
    def __init__(self):
        self.redis = redis.Redis()

    def get_or_analyze(self, image):
        # 计算图片hash
        image_hash = hashlib.md5(image).hexdigest()

        # 检查缓存
        cached = self.redis.get(image_hash)
        if cached:
            return json.loads(cached)

        # 分析并缓存
        result = analyze_image(image)
        self.redis.setex(image_hash, 86400, json.dumps(result))  # 缓存24小时
        return result
```

### 2. 批处理优化
```python
# 批量处理降低API调用
def batch_process(images):
    # 先用免费方案批量过滤
    free_results = [free_analyze(img) for img in images]

    # 只对低置信度的使用付费API
    need_paid = [img for img, res in zip(images, free_results)
                  if res['confidence'] < 0.7]

    # 批量调用API（某些API支持批量，更便宜）
    if need_paid:
        paid_results = batch_api_call(need_paid)
```

### 3. 模型微调
```python
# 使用少量付费API结果改进免费模型
def improve_free_model():
    # 1. 用付费API标注100张图片
    labeled_data = []
    for img in sample_images[:100]:
        label = gpt4_vision_analyze(img)  # 成本：$2
        labeled_data.append((img, label))

    # 2. 微调免费模型
    free_model.fine_tune(labeled_data)

    # 3. 提升免费模型准确率：75% -> 85%
```

---

## 📊 不同预算的最佳方案

| 月预算 | 推荐方案 | 预期效果 |
|--------|----------|----------|
| **$0** | 纯开源（Tesseract + YOLO + OpenCV） | 准确率75-80% |
| **$50** | 开源 + Google Vision(20%) | 准确率85% |
| **$100** | 开源 + 混合API | 准确率90% |
| **$200** | 智能路由 + 缓存优化 | 准确率92% |
| **$500+** | 高级API为主 + 开源辅助 | 准确率95%+ |

---

## 🎯 立即开始的免费方案

```bash
# 1. 安装免费工具（5分钟）
pip install pytesseract paddleocr ultralytics opencv-python

# 2. 下载预训练模型（10分钟）
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

# 3. 运行免费识别（立即可用）
python free_vision_demo.py

# 成本：$0
# 准确率：75-80%
# 可立即开始！
```

---

## 🏆 最终建议

### 如果您是创业公司/个人项目：
→ **先用100%免费方案**，验证产品可行性
→ 有收入后逐步引入付费API

### 如果您是中小企业：
→ **采用混合方案**，月预算$100以内
→ 重要客户用付费API，常规处理用免费

### 如果您是大企业：
→ **部署混合架构**，付费API保证质量
→ 同时训练自己的模型降低长期成本

---

**结论**：您完全可以从$0开始，通过免费开源方案实现基础的CAD图纸识别功能！