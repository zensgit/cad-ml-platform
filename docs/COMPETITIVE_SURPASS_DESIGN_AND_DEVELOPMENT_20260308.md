# Competitive Surpass Design And Development 2026-03-08

## 目标

本阶段的目标不是继续堆单一模型，而是把现有 CAD AI 能力收敛成一条
更稳定、可解释、可复核、可运维的产品主线，对标并超越典型图纸理解产品。

当前的主产品判断已经明确为：

- `Hybrid` 作为主分类链
- `Graph2D` 作为弱信号、冲突信号、拒识信号
- `OCR / titleblock / filename / history / knowledge` 作为可解释证据层
- `review-pack / active-learning / benchmark scorecard` 作为闭环层
- `Qdrant` 作为可观测、可迁移、可治理的向量底座

## 目标架构

### 1. 决策主链

- `fine_part_type`
- `coarse_part_type`
- `final_decision_source`
- `decision_path`
- `source_contributions`
- `rejection_reason`

这层已经不再依赖单一路径，而是围绕稳定 coarse/fine 契约统一输出。

### 2. 解释与复核

- assistant evidence / explainability
- OCR review guidance
- active-learning review queue / export
- rejection review pack / CI summary

目标是让系统不仅能给答案，还能说明为什么给这个答案、哪里需要人工复核。

### 3. 运维与向量底座

- Qdrant native metadata / search / filters / mutations
- migration status / pending / plan / readiness
- vectors stats / distribution / maintenance observability

目标是让向量层从“存储”升级成“可运维资产”。

## 已合入主线的关键能力

### 输出与分类稳定化

- coarse / fine 双层标签契约
- analyze / batch-classify / similarity / vector search 一致化输出
- provider coarse contract
- hybrid explanation / decision trace

### 真实验证与样例验证

- DXF 真数据 hybrid / graph2d 验证链
- `.h5` 在线样例 smoke
- `STEP/B-Rep` 在线样例 smoke
- Apple Silicon `micromamba + pythonocc-core` 本机 3D 环境自举

### 复核与闭环

- review-pack strict / non-strict gate
- active-learning export context
- review queue / export
- benchmark 文档与阶段执行计划

### Qdrant 主线

- native search / list / topk
- native register / update / delete
- migration status / preview / trends / summary / pending / pending run / plan
- migration recommendations / advisories / coverage / readiness

## 当前并行 PR 栈

以下能力已经完成本地验证并处于并行推进状态：

### `#163`

- active-learning review queue report 接入 evaluation CI
- 输出 artifact / summary / PR comment 扩展

### `#164`

- benchmark scorecard 增加 OCR 组件
- 支持 OCR review summary 驱动 benchmark 状态

### `#165`

- assistant/query 稳定 explainability contract
- 包含：
  - `summary`
  - `decision_path`
  - `source_contributions`
  - `alternative_labels`
  - `uncertainty`

### `#166`

- vectors stats / distribution 增加 Qdrant readiness
- 包含：
  - `indexed_ratio`
  - `unindexed_vectors_count`
  - `scan_truncated`
  - `readiness`
  - `readiness_hints`

### `#167`

- Qdrant store 增加只读 `inspect_collection()`
- maintenance stats 增加精简版 Qdrant observability
- vectors stats 改为优先复用只读 inspect 快照

## 为什么这条路线能超越对标产品

### 1. 不只是抽字段

很多竞品强在 OCR / 标题栏 / 标注抽取。
当前主线已经在向“工程语义 + 决策 + 复核”推进：

- coarse/fine 语义稳定
- 知识证据与解释输出
- review queue / feedback / retrain 飞轮

### 2. 不只是有向量库

很多系统只把向量库存起来。
当前主线已经补到了：

- metadata contract
- qdrant native data plane
- migration readiness
- maintenance / stats observability

这让向量层真正可运维。

### 3. 不只是模型更强

当前的优势正在转向：

- 拒识能力
- explainability
- review closure
- benchmark scorecard
- operational readiness

这比单纯拉高一个分类模型分数更接近真实生产竞争力。

## 下一阶段建议

### 收口优先级

1. 收口 `#163/#164/#165`
2. 收口 `#166/#167`
3. 把 benchmark scorecard 接入更多真实运维信号

### 下一批低冲突增强

- benchmark scorecard 接 Qdrant readiness / observability
- maintenance stats 的 qdrant error taxonomy
- assistant explainability 与 review-pack 联动
- OCR review guidance 与 review queue priority 联动

## 结论

当前主线已经从“分类器集合”进入“可解释、可复核、可运维的 CAD AI 平台”阶段。

对标目标不再只是识别精度，而是：

- 更稳定的 coarse/fine 契约
- 更好的审阅与拒识体验
- 更强的向量底座可观测性
- 更完整的人工反馈闭环
