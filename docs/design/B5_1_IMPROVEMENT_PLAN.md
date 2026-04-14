# B5.1 提升计划：DXF 文字内容融合

**日期**: 2026-04-14  
**基线**: B5.0 — 整体 acc ≥ 92%，无文件名 90.5%+  
**目标**: 无文件名场景 acc ≥ 93%（文字内容作为额外信号）  
**核心**: 提取 DXF 实体内文字（TEXT/MTEXT），构建 TextContentClassifier 并融入 HybridClassifier

---

## 1. 动机与分析

### 1.1 为什么需要文字融合

B5.0 的 90.5%+ 精度主要来自图结构（形状/拓扑）。但有些类别图形上相似，文字上有明显区分：

| 易混淆对 | 图形区别 | 文字区别 |
|---------|---------|---------|
| 箱体 vs 轴承座 | 轮廓相近 | 标注："轴承座孔 Φ80" |
| 阀门 vs 传动件 | 旋转体相似 | 标注："DN50 球阀" |
| 换热器 vs 筒体 | 容器结构相近 | 标注："管板" / "折流板" |
| 过滤器 vs 罐体 | 容器相近 | 标注："过滤精度 50μ" |

### 1.2 DXF 文字来源

| 文字来源 | ezdxf 实体类型 | 典型内容 |
|---------|--------------|---------|
| 标注文字 | TEXT, MTEXT | 零件名、材料、技术要求 |
| 图框标题栏 | ATTDEF, ATTRIB | 图号、名称、比例 |
| 引线注释 | LEADER + MTEXT | 工艺要求、热处理 |
| 尺寸标注文字 | DIMENSION | Φ100, M12×1.5 |
| 块属性 | INSERT + ATTRIB | 标准件规格 |

---

## 2. 技术架构

### 2.1 TextContentClassifier

```python
# src/ml/text_classifier.py

class TextContentClassifier:
    """Classify DXF by text content using keyword matching + TF-IDF."""
    
    # 24 类关键词字典（优先级降序）
    KEYWORDS: dict[str, list[str]] = {
        "法兰": ["法兰", "法兰盘", "flange", "PN", "连接盘", "对焊法兰"],
        "轴类": ["轴", "主轴", "花键轴", "阶梯轴", "shaft", "传动轴"],
        "箱体": ["箱体", "壳体", "机壳", "齿轮箱", "housing", "减速箱"],
        "轴承座": ["轴承座", "轴承支座", "bearing", "轴承孔", "轴承盖"],
        "传动件": ["传动", "齿轮", "皮带轮", "链轮", "联轴器", "transmission"],
        "阀门": ["阀门", "球阀", "蝶阀", "闸阀", "截止阀", "valve", "DN"],
        "罐体": ["罐体", "储罐", "压力容器", "tank", "容积"],
        "过滤器": ["过滤器", "滤芯", "过滤精度", "filter", "滤网"],
        "换热器": ["换热器", "管板", "折流板", "换热管", "heat exchanger"],
        "泵": ["泵体", "叶轮", "泵壳", "pump", "离心泵", "螺杆泵"],
        "搅拌器": ["搅拌", "搅拌桨", "agitator", "mixer", "叶片"],
        "弹簧": ["弹簧", "碟簧", "spring", "弹力", "刚度"],
        "分离器": ["分离器", "旋风", "沉降", "separator", "分离"],
        "筒体": ["筒体", "筒节", "壳程", "cylinder", "圆筒"],
        "封头": ["封头", "椭圆封头", "半球", "head", "封口"],
        "支架": ["支架", "支座", "底座", "bracket", "stand"],
        "盖罩": ["盖板", "端盖", "防护罩", "cover", "盖"],
        "液压组件": ["液压", "液压缸", "活塞", "hydraulic", "油缸"],
        "板类": ["板", "平板", "隔板", "plate", "挡板"],
        "旋转组件": ["转子", "旋转", "rotor", "转盘", "回转"],
        "锥体": ["锥体", "锥管", "锥形", "cone", "变径"],
        "紧固件": ["螺栓", "螺母", "螺钉", "fastener", "六角"],
        "进出料装置": ["进料", "出料", "加料", "排料", "inlet", "outlet"],
        "人孔": ["人孔", "手孔", "manhole", "人孔盖", "检查孔"],
    }
    
    def predict_probs(self, text: str) -> dict[str, float]:
        """Return softmax-normalized class scores from keyword matching."""
        if not text.strip():
            return {}  # 无文字 → 不提供信号
        
        scores: dict[str, float] = {}
        for cls, keywords in self.KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw.lower() in text.lower())
            scores[cls] = hits / max(len(keywords), 1)
        
        total = sum(scores.values())
        if total < 1e-6:
            return {}  # 无匹配 → 不提供信号
        
        # Softmax-like normalization
        return {k: v / total for k, v in scores.items()}
```

### 2.2 DXFTextExtractor

```python
# src/ml/text_extractor.py

class DXFTextExtractor:
    """Extract all text content from DXF modelspace entities."""
    
    TEXT_TYPES = {"TEXT", "MTEXT", "ATTDEF", "ATTRIB", "DIMENSION"}
    
    def extract(self, dxf_bytes: bytes) -> str:
        """Return concatenated text from all text entities."""
        import ezdxf
        import io
        
        try:
            doc = ezdxf.read(io.BytesIO(dxf_bytes))
        except Exception:
            return ""
        
        texts: list[str] = []
        for entity in doc.modelspace():
            etype = entity.dxftype()
            if etype == "MTEXT":
                t = entity.plain_mtext()
            elif etype in ("TEXT", "ATTDEF", "ATTRIB"):
                t = entity.dxf.get("text", "")
            elif etype == "DIMENSION":
                t = entity.dxf.get("text", "")
            else:
                continue
            if t:
                texts.append(t.strip())
        
        return " ".join(texts)
```

### 2.3 HybridClassifier 集成

```python
# src/ml/hybrid_config.py 新增

class TextContentConfig:
    enabled: bool = True
    fusion_weight: float = 0.15
    min_text_length: int = 5    # 少于 5 字符视为无效文字

# 权重重新平衡
class Graph2DConfig:
    fusion_weight: float = 0.45  # 略降（从 0.50 到 0.45）

class FilenameClassifierConfig:
    fusion_weight: float = 0.40  # 略降（从 0.45 到 0.40）
```

融合方程（无文件名场景）：

```
score(cls) = g2d_w × P_graph2d(cls)
           + txt_w × P_text(cls)     [如果文字非空]
           + tb_w  × P_titleblock(cls) [如果标题栏可解析]
```

---

## 3. 验证方案

### 3.1 文字提取基准测试

```python
# 先统计文字覆盖率
def audit_text_coverage(manifest_csv: str) -> dict:
    """统计各类别的文字覆盖率和关键词命中率"""
    ...

# 期望：
# - 约 60-70% 的 DXF 文件含有可用文字
# - 关键词命中率 ≥ 40%（在有文字的文件中）
```

### 3.2 场景测试集定义

| 场景 | 样本构成 | 评估指标 |
|------|---------|---------|
| A: 有文件名 + 有文字 | 正常文件 | acc ≥ 99% |
| B: 无文件名 + 有文字 | UUID 文件名 + 原文字 | acc ≥ 93% |
| C: 无文件名 + 无文字 | UUID 文件名 + 清空文字 | acc ≥ 90% |
| D: 错误文件名 + 有文字 | 对抗文件名 | acc ≥ 75% |

### 3.3 消融实验

```
配置 1: graph2d only (baseline)               → 90.5%
配置 2: graph2d + titleblock                 → ?%
配置 3: graph2d + text_content               → ?%（目标 ≥ 92%）
配置 4: graph2d + titleblock + text_content  → ?%（目标 ≥ 93%）
```

---

## 4. 实现步骤

### Step 1: 文字提取与分析（1 天）

```bash
# 统计文字覆盖率
python3 scripts/audit_text_coverage.py \
    --manifest data/graph_cache/cache_manifest.csv \
    --output docs/design/B5_1_TEXT_AUDIT.md
```

关注：
- 有效文字的 DXF 比例（期望 ≥ 60%）
- 各类别平均文字长度
- 关键词命中率（验证字典完整性）

### Step 2: 实现 TextContentClassifier（1 天）

```
src/ml/text_extractor.py      # DXF 文字提取
src/ml/text_classifier.py     # 关键词分类器
tests/unit/test_text_cls.py   # 单元测试
```

关键设计决策：
- 优先精确匹配，再模糊匹配
- 无关键词命中 → 返回空（不参与融合）
- 防止文字噪声（数字/标点不参与分类）

### Step 3: 融合集成与权重搜索（1 天）

```python
# 扩展 search_hybrid_weights.py 支持三路融合
fn_weights  = [0.35, 0.40, 0.45]
g2d_weights = [0.40, 0.45, 0.50]
txt_weights = [0.10, 0.15, 0.20]
```

### Step 4: 场景专项测试（1 天）

```bash
# 构建测试集 A/B/C/D
python3 scripts/build_scenario_testsets.py \
    --manifest data/graph_cache/cache_manifest.csv \
    --output data/scenario_tests/

# 评估四个场景
python3 scripts/evaluate_scenarios.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --testsets data/scenario_tests/ \
    --output docs/design/B5_1_SCENARIO_RESULTS.md
```

---

## 5. 验收标准

| 检查项 | 目标 | 判定方式 |
|--------|------|---------|
| 文字覆盖率 | ≥ 60% DXF 含有效文字 | `audit_text_coverage.py` |
| 场景 B（无文件名+有文字）acc | **≥ 93%** | 场景测试集 |
| 场景 C（纯 Graph2D）acc | ≥ 90%（不退步） | 场景测试集 |
| 关键词命中率 | ≥ 40%（有文字样本中） | 消融分析 |
| 推理延迟增加 | < 20ms（文字提取额外开销） | 基准测试 |
| 测试套件 | 全通过（Superpass gate） | CI |

---

## 6. 关键文件规划

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/ml/text_extractor.py` | 待创建 | DXF 文字提取器 |
| `src/ml/text_classifier.py` | 待创建 | 关键词文字分类器 |
| `scripts/audit_text_coverage.py` | 待创建 | 文字覆盖率分析 |
| `scripts/build_scenario_testsets.py` | 待创建 | 场景测试集构建 |
| `scripts/evaluate_scenarios.py` | 待创建 | 多场景评估脚本 |
| `src/ml/hybrid_config.py` | 待更新 | 添加 TextContentConfig |
| `src/ml/hybrid_classifier.py` | 待更新 | 集成 TextContentClassifier |
| `scripts/search_hybrid_weights.py` | 待扩展 | 支持三路权重搜索 |

---

## 7. 预期影响分析

### 7.1 收益场景

| 类别 | 关键词覆盖度 | 预期 acc 提升 |
|------|------------|-------------|
| 阀门 | 高（"DN50", "球阀"） | +10-15pp |
| 换热器 | 高（"管板", "折流板"） | +8-12pp |
| 轴承座 | 中（"轴承孔", "轴承座"） | +5-8pp |
| 过滤器 | 中（"过滤精度", "滤芯"） | +5-8pp |
| 筒体 | 低（文字稀少） | +2-5pp |

### 7.2 风险场景

| 风险 | 概率 | 缓解 |
|------|------|------|
| 文字被图形遮挡，提取不完整 | 中 | 多字段尝试 + 容错返回 |
| 关键词跨类共享（如"轴"） | 高 | 加权词典 + 上下文判断 |
| 文字提取增加 20ms+ 延迟 | 低 | 异步提取 / 缓存文字内容 |
| 错误文字导致 acc 下降 | 低 | 置信度阈值 + 仅高置信文字参与融合 |

---

## 8. 里程碑节点

```
B5.0 完成（训练）   [2026-04-14 进行中]
B5.1 Step 1 文字分析 [2026-04-15]
B5.1 Step 2 TextCls  [2026-04-16]
B5.1 Step 3 融合集成 [2026-04-17]
B5.1 Step 4 场景测试 [2026-04-18]
B5.1 报告生成        [2026-04-18]
```

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
