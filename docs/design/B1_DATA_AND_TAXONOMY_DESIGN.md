# B1: Data Inventory and 25-Class Label Taxonomy Design

## 1. Current Data Inventory

### DXF File Sources

| Directory | Description | File Count (approx) |
|-----------|-------------|---------------------|
| `data/training/` | Original training set | ~200 |
| `data/training_v2/` | V2 curated training | ~300 |
| `data/training_v3/` | V3 additions | ~150 |
| `data/training_v5/` | V5 refined set | ~400 |
| `data/training_v7/` | V7 cleaned set | ~500 |
| `data/training_v8/` | V8 latest training | ~600 |
| `data/training_4000/` | Bulk 4000-drawing collection | ~4000 |
| `data/training_merged/` | Merged deduplicated | ~1000 |
| `data/training_merged_v2/` | V2 merged (by_class) | ~1361 |
| `data/standards_dxf/` | Standards reference | ~50 |
| `data/synthetic_v2/` | Synthetic generated | ~200 |
| **Total** | | **~8,876** |

### Existing Class Distribution (training_merged_v2/by_class)

| Class | Samples |
|-------|---------|
| 其他 | 754 |
| 连接件 | 204 |
| 壳体类 | 161 |
| 轴类 | 149 |
| 传动件 | 81 |
| 支架 | 12 |

The current 6-class system is heavily imbalanced: "其他" (other) contains 55% of samples, acting as a catch-all that limits classifier precision.

## 2. Label Synonyms Analysis

The existing `label_synonyms_template.json` defines **94 categories** with varying specificity:
- Fine-grained part types (e.g., 超声波法兰, 对接法兰, 连接法兰)
- Drawing types mixed with part types (装配图, 零件图, 原理图)
- Educational/noise labels (零件一 through 零件九, 基准代号, 粗糙度)
- Person names (金雨薇)

### Problems with 94 Categories
1. **Too sparse**: Many categories have <5 samples, insufficient for supervised learning
2. **Semantic overlap**: Multiple labels for the same concept (e.g., 3 types of flanges)
3. **Mixed ontology**: Part types, drawing types, and metadata labels in one flat list
4. **Noise**: Educational exercises and irrelevant labels pollute training

## 3. Taxonomy v2: 25-Class Consolidation

### Design Principles
- **Semantic grouping**: Similar parts consolidated (all flanges -> 法兰)
- **Orthogonal metadata**: Drawing types (零件图/装配图/原理图) separated as secondary tags
- **Noise exclusion**: Educational labels, person names, and formatting labels excluded
- **Trainability**: Target minimum 30 samples per class

### 25 Part-Family Classes

| ID | Class | Source Labels | Description |
|----|-------|---------------|-------------|
| 1 | 封头 | 上封头组件, 下封板 | Vessel heads/end caps |
| 2 | 筒体 | 上筒体组件, 下筒体组件, 前筒体, 后筒体 | Cylindrical shells |
| 3 | 罐体 | 罐体部分, 罐体支腿, 基质沥青计量罐 | Tank/vessel bodies |
| 4 | 锥体 | 下锥体组件 | Cone sections |
| 5 | 换热器 | 再沸器, 管束 | Heat exchangers |
| 6 | 分离器 | 汽水分离器, 捕集器组件, 捕集口 | Separators/collectors |
| 7 | 过滤器 | 过滤托架, 过滤芯组件 | Filters |
| 8 | 搅拌器 | 搅拌器组件, 搅拌桨组件, 搅拌轴组件, 搅拌减速机机罩 | Agitators/mixers |
| 9 | 传动件 | 传动轴, 蜗轮蜗杆, 蜗杆, 涡轮, 差动机构 | Transmission components |
| 10 | 轴类 | 轴类, 轴头组件, 阶梯轴, 零件轴改 | Shafts |
| 11 | 轴承座 | 轴承座, 短轴承座, 下轴承支架, 轴承 | Bearings/housings |
| 12 | 法兰 | 超声波法兰, 人孔法兰, 对接法兰, 连接法兰, 出料凸缘 | Flanges |
| 13 | 紧固件 | 紧固件, 调节螺栓, 压下螺丝, 铰制螺栓 | Fasteners |
| 14 | 支架 | 支承座, 支腿, 架类, 机架, 拖车 | Supports/frames |
| 15 | 板类 | 板类, 底板, 挡板, 活动齿板, 压紧片, 角钢 | Plates/baffles |
| 16 | 箱体 | 箱体, 箱座, 电加热箱 | Housings/boxes |
| 17 | 盖罩 | 盖, 罩, 压盖, 齿轮罩, 保护罩组件 | Covers/guards |
| 18 | 阀门 | 阀体 | Valves |
| 19 | 泵 | 泵, 泵盖图 | Pumps |
| 20 | 液压组件 | 液压开盖组件, 出料正压隔离器, 汽缸 | Hydraulic components |
| 21 | 进出料装置 | 自动进料装置, 侧推料组件, 直推 | Feed/discharge devices |
| 22 | 旋转组件 | 旋转组件, 拖轮组件, 手轮组件, 滑块, 拨叉 | Rotary assemblies |
| 23 | 人孔 | 人孔 | Manholes |
| 24 | 弹簧 | 扭转弹簧 | Springs |
| 25 | 夹具 | 夹具, 夹具主体座, 组合刀架装配图 | Fixtures/jigs |

### Excluded Labels (18 total)
- Educational: 零件一 through 零件九
- Noise: 金雨薇, 数据图, 基准代号, 简化标题栏表格, 粗糙度
- Meta: 模板, 副本, 技术文件, 其他

### Drawing-Type Metadata (orthogonal, not a class)
- 零件图, 装配图, 原理图, 机械制图

## 4. Data Gap Analysis

Based on a 30 samples/class target for minimum viable training:

| Status | Count |
|--------|-------|
| Classes likely above threshold | ~8 (轴类, 传动件, 支架, 紧固件, 法兰, 板类, 箱体, 盖罩) |
| Classes likely below threshold | ~10 (锥体, 换热器, 过滤器, 阀门, 泵, 弹簧, etc.) |
| Classes needing investigation | ~7 (new groupings need sample counting) |

**Estimated gap**: ~300-400 additional samples needed across underrepresented classes.

### Mitigation Strategies
1. **Augmentation**: Geometric augmentations (rotation, scale, mirror, dropout) can 3-5x existing samples
2. **Annotation**: Re-label existing "其他" samples that may belong to specific classes
3. **Collection**: Targeted acquisition for classes with <10 raw samples
4. **Synthetic**: Generate synthetic DXF files for geometry-distinguishable classes

## 5. Configuration

Taxonomy v2 is defined in `config/label_taxonomy_v2.yaml` and consumed by:
- `scripts/label_annotation_tool.py` — annotation pipeline
- `scripts/build_unified_manifest.py` — manifest builder
- Future: `src/ml/` classifiers for v2 training
