# Benchmark 对标与超越当前交付状态总设计/开发文档

## 1. 文档定位

- 基线代码: `origin/main@c76a0fa7f917628a4cd1ba63d884742070edeb78`
- 基线时间: `2026-03-08`
- 文档目的: 基于当前 `main` 已合入内容，统一说明 benchmark 对标与超越相关的当前交付状态、已完成能力、主要差距与下一阶段路线。
- 文档边界: 只盘点已存在于 `origin/main` 的代码、脚本、接口与验证记录，不把未合入分支能力写成既成事实。

## 2. 执行结论

当前 `main` 已经具备“对标”所需的基础工程闭环，但距离“可证明超越 benchmark”仍有明确差距。

可以确认已经交付的，不再是单点模型能力，而是一套可持续演进的工程化能力组合：

- `Hybrid` 已成为 DXF 主决策路径，`Graph2D` 被收敛为弱信号、冲突检测和复核输入。
- DXF API 已补齐粗细粒度输出、分支冲突信息、知识摘要输出，具备面向 benchmark 统计和下游消费的稳定字段。
- `Graph2D` 的训练指标、盲测诊断、低置信度统计已经产品化，能客观证明其真实上限和当前短板。
- `History Sequence` 已完成参考迁移、数据集/编码器/评估脚本基础建设，并补齐 coarse benchmark 合同，但真实数据验证仍明显不足。
- `Active Learning`、review pack、CI 严格门控、Qdrant 迁移观测与迁移计划接口已经形成运维闭环。

因此，当前最准确的定位不是“已经全面超越 benchmark”，而是：

- 已完成 benchmark-ready 的工程基线。
- 已在工程闭环、可观测性、治理与迁移能力上部分超越单纯 benchmark 系统。
- 尚未在 `Graph2D`、`History Sequence`、`3D/B-Rep` 的真实数据效果上形成足够强的超越性证据。

## 3. 当前交付状态总览

| 工作流 | 当前状态 | `main` 上已交付内容 | 结论 |
| --- | --- | --- | --- |
| Benchmark 评估脚手架 | 已交付 | `scripts/vision_cad_feature_benchmark.py` 及基线报告/对比报告/导出链路；`Graph2D` 训练与诊断脚本；`History Sequence` coarse 评估脚本 | 具备跑 benchmark、留基线、做 diff 的基础能力 |
| DXF 主决策路径 | 已交付 | `Hybrid` 主路径、`coarse_*` 输出、`final_decision_source`、`has_branch_conflict`、知识摘要输出 | 已不再依赖单一 `Graph2D` 结果 |
| Graph2D 训练与诊断 | 已交付但效果不足 | `--metrics-out`、blind diagnose、`low_conf_rate`、运行时门控与 override 配置 | 工具链完整，模型效果仍弱 |
| Active Learning / Review Closure | 已交付 | `true_fine_type` / `true_coarse_type`、导出上下文增强、review pack 与 CI 严格门控 | 已具备闭环样本沉淀能力 |
| History Sequence | 部分交付 | `.h5` 发现、HPSketch 数据集、序列编码器、coarse eval 合同、在线 smoke | 基础设施完成，真实 benchmark 证据不足 |
| 向量后端迁移治理 | 已交付 | pending summary、recommendations、migration plan、partial scan 风险显式化 | 运维治理强于单纯模型 benchmark |
| 3D/B-Rep | 预备态 | analyze/eval 输出已有 prep 字段，但真实数据覆盖仍为 0 | 尚未进入可对标阶段 |

## 4. 已完成能力

### 4.1 Benchmark 可运行、可对比、可追溯

当前 `main` 已不是“人工口头对比”，而是具备可执行 benchmark 合同：

- CAD 特征 benchmark：
  - `scripts/vision_cad_feature_benchmark.py`
  - `scripts/vision_cad_feature_baseline_report.py`
  - `scripts/vision_cad_feature_compare_report.py`
  - `scripts/vision_cad_feature_compare_export.py`
- `Graph2D` benchmark：
  - `scripts/train_2d_graph.py --metrics-out`
  - `scripts/diagnose_graph2d_on_dxf_dir.py`
  - 支持 `--strip-text-entities`、`--mask-filename` 的 blind path
  - 输出 `low_conf_rate`
- `Hybrid DXF` benchmark：
  - `scripts/eval_hybrid_dxf_manifest.py`
  - 可同时观察 `graph2d`、`filename`、`titleblock`、`hybrid`、`final`
- `History Sequence` benchmark：
  - `scripts/eval_history_sequence_classifier.py`
  - 已补齐 `coarse_accuracy_overall`、`coarse_macro_f1_overall`、mismatch 摘要

这意味着当前仓库已经具备三个关键能力：

- 能冻结 baseline。
- 能跑对比。
- 能把差距定位到具体分支、具体字段、具体诊断信号。

### 4.2 DXF 对标主线已完成从“模型导向”到“产品导向”的切换

`main` 上最重要的完成项不是单个模型提升，而是主决策架构已经切换：

- `Hybrid` 成为主决策路径。
- `Graph2D` 明确降级为弱信号和 review/rejection 输入。
- analyze 输出补齐：
  - `coarse_part_type`
  - `coarse_hybrid_label`
  - `coarse_graph2d_label`
  - `coarse_filename_label`
  - `coarse_titleblock_label`
  - `final_decision_source`
  - `branch_conflicts`
  - `has_branch_conflict`
- 知识输出补齐：
  - `knowledge_checks`
  - `violations`
  - `standards_candidates`
  - `knowledge_hints`

这一步的价值在于，benchmark 不再只看最终标签，而是能看：

- 主决策来自哪条分支。
- 分支之间是否冲突。
- 结果是否带有工程知识解释。
- 下游是否可以消费稳定 coarse taxonomy。

### 4.3 已把 `Graph2D` 的真实短板显性化，而不是继续被它误导

`main` 上的 `Graph2D` 交付不是“精度已经解决”，而是“问题已经被稳定测量并安全收敛”。

基于 `docs/REAL_DATA_GRAPH2D_VALIDATION_20260306.md` 与
`docs/HYBRID_REAL_DATA_VALIDATION_20260306.md`，当前事实很明确：

- 真实 DXF 110 文件集上，`Graph2D` 诊断准确率约为 `0.1182`。
- `low_conf_rate = 1.0`。
- 预测塌缩到少数标签。
- 去掉文本与文件名后并未明显更差，说明当前模型本身没有形成强泛化能力。

但与早期阶段不同，当前 `main` 已经把这类风险产品化处理掉：

- 运行时有 `GRAPH2D_MIN_CONF`、`GRAPH2D_FUSION_ENABLED`、override 配置。
- blind diagnose 能区分结构信号与辅助信号。
- review/CI 可以围绕 `Graph2D` 弱信号做阻断、分流与复核。

换句话说，`Graph2D` 当前虽然没有“超越 benchmark”，但已经被纳入一个安全、可观测、不会主导错误决策的位置。

### 4.4 Active Learning 与 Review 治理已经闭环

当前 `main` 已完成“有 benchmark 结果”到“能把 hard cases 变成训练资产”的转变：

- `ActiveLearningSample` 已增加：
  - `predicted_coarse_type`
  - `true_fine_type`
  - `true_coarse_type`
  - `true_is_coarse_label`
- 反馈 API 保持 `true_type` 向后兼容，同时新增 coarse/fine 归一化写入。
- review pack 与导出样本已保留：
  - coarse/fine 标签上下文
  - rejection / knowledge conflict 上下文
  - triage priority

这部分能力的意义不是直接提分，而是使 benchmark 不再是一次性评估，而能回流成为训练和治理输入。

### 4.5 History Sequence 已完成“可接入 benchmark”的第一阶段

围绕 `HPSketch` / `SketchGraphs` / `DeepCAD` 的参考迁移已经完成第一轮落地：

- `src/ml/history_sequence_tools.py`
- `src/ml/train/sequence_encoder.py`
- `src/ml/train/hpsketch_dataset.py`
- `src/ml/history_sequence_classifier.py`
- `scripts/eval_history_sequence_classifier.py`

已经具备的能力包括：

- `.h5` 数据发现与加载
- 命令序列 token 化
- dataset/label map 构建
- 轻量级 sequence encoder/classifier
- coarse/fine 统一评估合同
- 在线 smoke 验证 CLI 通路

这说明 `History Sequence` 已经从“调研想法”进入“可 benchmark、可 shadow、可继续接 Hybrid”的阶段。

### 4.6 运维与迁移能力已经开始超越单纯 benchmark 系统

benchmark 系统通常只回答“谁准”。当前 `main` 已开始回答“怎么稳定迁、怎么安全发、怎么灰度跑”：

- `GET /api/v1/vectors/migrate/pending/summary`
- `GET /api/v1/vectors/migrate/plan`
- 推荐字段：
  - `recommended_from_versions`
  - `largest_pending_from_version`
  - `largest_pending_count`
  - `suggested_run_limit`
  - `allow_partial_scan_required`

这类能力本身不提高模型分数，但会直接决定新特征版本和新 benchmark 结论能否真正进入生产。

## 5. 当前差距

### 5.1 还没有形成“可证明超越”的统一 scorecard

现状：

- benchmark 能力分散在 CAD feature、Hybrid DXF、Graph2D、History Sequence、Qdrant 迁移观测多个脚本和验证文档中。
- 仓库里已经有大量验证记录，但缺少单一总表来统一回答：
  - 当前主线 benchmark 是什么
  - 基线是哪一版
  - 哪些分支已经超过基线
  - 哪些只适合 shadow

影响：

- 对内难以快速判定“本周是否真的更强”。
- 对外无法严谨宣称“已经超越 benchmark”。

### 5.2 `Graph2D` 仍未达到可独立对标的强度

现状：

- `best_val_acc = 0.1538`
- 真实 DXF 集 `accuracy = 0.1182`
- `low_conf_rate = 1.0`

影响：

- 不能把 `Graph2D` 包装为当前主价值分支。
- 如果继续以单模型叙事对外表达，会和真实数据结论冲突。

结论：

- 现阶段应继续把 `Graph2D` 定位为弱信号、异常检测和 review 排序输入。

### 5.3 `Hybrid` 的强分支已经确认，但最终消费契约仍在迁移中

`docs/HYBRID_REAL_DATA_VALIDATION_20260306.md` 给出的结果非常关键：

- `filename_label = 0.8727`
- `titleblock_label = 0.8727`
- `hybrid_label = 0.8727`
- `final_part_type = 0.5545`

问题并不是 `Hybrid` 分支不强，而是旧的 `part_type` 语义仍混合了细粒度标签和规则形态。

当前 `main` 已经补齐 `coarse_*` 输出，这是重要修复；但在“benchmark 对标与超越”的视角下，仍有一段下游迁移工作没有完成：

- benchmark 报表要优先使用 coarse 契约。
- 下游消费者要从旧 `part_type` 迁移到新字段。
- 所有验收结论要避免把旧字段误当 coarse ground truth。

### 5.4 `History Sequence` 仍缺少真实数据规模验证

现状：

- 基础工具链和 coarse 合同已经就位。
- 在线 `.h5` smoke 只有 `1` 个样本。
- 真实标注 `.h5` 数据集仍未形成稳定 benchmark 资产。

影响：

- 现在只能证明“管道可跑、合同正确”，不能证明“分支有效贡献生产精度”。
- 暂时还不能把 `History Sequence` 写成已经完成的超越性能力。

### 5.5 `3D/B-Rep` 还处于 benchmark 准备态

当前主线已经补了 `brep_feature_hints`、`brep_embedding_dim` 等 prep 字段，但真实数据验证里：

- `brep_valid_3d_count = 0`
- `brep_feature_hints_count = 0`

这说明 3D 路线仍处于接口和评估预埋阶段，尚未进入有效对标。

### 5.6 缺少与外部 benchmark 的同口径复现实验

仓库中已有竞品与外部工程图 benchmark 分析文档，但当前 `main` 还没有形成统一、可复现实验来直接回答：

- 与外部多模态模型基线同一数据、同一指标、同一口径比较，当前到底领先多少。
- 当前“超越”究竟发生在识别质量、工程治理、还是全链路交付上。

因此，现阶段最稳妥的说法应是：

- 已 benchmark-informed。
- 已局部 benchmark-ready。
- 还未 benchmark-proven beyond on a unified external scorecard。

## 6. 下一阶段路线

### 阶段 A: 统一 benchmark 记分板与验收口径

目标：先统一“怎么比”，再讨论“是否超越”。

建议交付：

- 固定一份主线 scorecard，至少覆盖：
  - DXF `graph2d`
  - `filename`
  - `titleblock`
  - `hybrid`
  - `final`
  - `history`
  - `3d/brep prep`
  - 迁移治理状态
- 所有评估统一输出：
  - exact / coarse accuracy
  - low confidence rate
  - branch conflict rate
  - coverage / input resolved rate
  - 数据集规模与标签分布
- 把基线冻结为单一 artifact，而不是散落在多份临时验证文档中。

阶段出口：

- 能用一份报告直接回答“本周主线相对基线是升还是降”。
- 能区分“模型效果提升”和“治理/运维能力提升”。

### 阶段 B: 修复当前最短板的弱分支

目标：把当前明显拖后腿但仍有战略价值的分支拉到可用区间。

优先级建议：

1. `Graph2D`
   - 继续冻结数据集与标签口径
   - 把 `other`/模糊类重新整理
   - 通过训练配方 sweep 和低置信度校准验证是否还有继续投资价值
   - 若无法显著超出当前 `0.1182` 水平，则明确长期定位为 review-only
2. `History Sequence`
   - 建立真实标注 `.h5` manifest
   - 用 coarse/fine 双口径跑非 smoke 的批量评估
   - 在 `Hybrid` 中先 shadow 接入，再决定是否进入主决策
3. 输出契约迁移
   - 统一让 benchmark、导出和下游消费以 `coarse_*` 字段为准
   - 把旧 `part_type` 降为兼容字段，而不是主 benchmark 字段

阶段出口：

- `Graph2D` 要么证明自己能贡献可观增益，要么正式降级为长期弱信号。
- `History Sequence` 从“可跑”升级到“有真实样本证据”。
- `Hybrid` 强分支优势能被最终消费者完整接收。

### 阶段 C: 形成“超越 benchmark”的正式证据链

目标：把当前工程优势转成可对内对外复述的正式结论。

建议交付：

- 一套同口径外部/内部对照实验：
  - 外部多模态模型或竞品基线
  - 当前 DXF `Hybrid` 主线
  - `Graph2D` / `History` / `3D` 作为辅助或下一阶段分支
- 一份正式超越报告，分开陈述三类超越：
  - 识别质量超越
  - 工程治理超越
  - 生产可迁移性超越
- 结合 Qdrant 迁移计划能力，补齐新特征版本进入生产的执行验证。

阶段出口：

- “超越 benchmark”不再是方向判断，而是有固定数据集、固定脚本、固定 scorecard 支撑的发布结论。

## 7. 建议的当前对外口径

如果今天必须给出一句准确表述，建议使用下面这个版本：

> 当前主线已经完成 benchmark-ready 的工程交付，`Hybrid + coarse output + active learning + migration governance` 构成了新的生产基线；其中工程闭环能力已部分超越单点 benchmark 方案，但 `Graph2D`、`History Sequence`、`3D/B-Rep` 的真实数据效果仍处于补证阶段，尚不宜宣称全面超越。

## 8. 本文依据的主线材料

- `README.md`
- `docs/HYBRID_REAL_DATA_VALIDATION_20260306.md`
- `docs/REAL_DATA_GRAPH2D_VALIDATION_20260306.md`
- `docs/AI_PARALLEL_DELIVERY_FINAL_VALIDATION_20260306.md`
- `docs/ACTIVE_LEARNING_COARSE_FEEDBACK_VALIDATION_20260307.md`
- `docs/HISTORY_SEQUENCE_COARSE_EVAL_VALIDATION_20260307.md`
- `docs/SEQUENCE_REFERENCE_MIGRATION_20260306.md`
- `docs/QDRANT_MIGRATION_PENDING_SUMMARY_VALIDATION_20260308.md`
- `docs/QDRANT_MIGRATION_RECOMMENDATIONS_VALIDATION_20260308.md`
- `docs/QDRANT_MIGRATION_PLAN_VALIDATION_20260308.md`
- `src/api/v1/analyze.py`
- `src/api/v1/active_learning.py`
- `src/api/v1/vectors.py`
- `src/core/active_learning.py`
- `src/core/knowledge/analysis_summary.py`
- `src/ml/history_sequence_tools.py`
- `src/ml/train/sequence_encoder.py`
- `src/ml/train/hpsketch_dataset.py`
- `src/ml/history_sequence_classifier.py`
- `scripts/eval_history_sequence_classifier.py`
- `scripts/train_2d_graph.py`
- `scripts/diagnose_graph2d_on_dxf_dir.py`
