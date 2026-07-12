# CAD-ML-Platform 定位与路线图 — Design (for-review)

> **Historical decision record.** The current repo-level product and market strategy is
> [`PRODUCT_STRATEGY.md`](PRODUCT_STRATEGY.md). This dated document remains the code
> inventory, engineering-convergence record, and Phase 0 authorization; where product
> direction conflicts, the canonical strategy wins.

**编制日期**: 2026-07-06
**性质**: for-review 提案 / 设计锁。**本 PR 无 runtime、不改代码、不删代码**——它锁定的是*方向*。承重决策(角色 A/B、三仓边界)留给 owner ratify;merge 即授权 Phase 0 动刀。
**依据**: 一次 5-路代码级审计(vision 实质 / ~60-dir 脚手架引用图 / 实际服务面 / 反馈飞轮 / 消费者),以 **file/path + line 证据**为准(部分为路径级/聚合级,非严格逐行;复算命令见**附录 A**),**不以 README 自述为准**。

---

## 0. 一句话结论

代码里,本仓**不是**"为 ERP/MES/PLM 提供统一智能分析的 CAD ML 平台"。它是:**一个 CAD 图纸查重 + 分类引擎,外面裹了一层由 agent 车队生成的巨量"企业级微服务"脚手架**;真核不大但**评测/治理骨架罕见地强**,且当前 B-Rep 溯源正对准真瓶颈(数据)。**唯一真实、部署级的消费者是 DedupCAD;PLM/ERP/MES/metasheet 在 `src/` 里零集成代码。**

一个定调数据点:`git` 2226 commits、单作者、单日峰值 235 → 这是 agent-fleet 的 built-ahead 产物,膨胀由此而来。

---

## 1. 诊断:真核 vs 泡沫(证据表)

| 层 | 代码级结论(file/path + line 证据) | 归类 |
|---|---|---|
| **查重/相似度引擎** | 生产级**古典几何**:Hungarian 指派、RANSAC/Kabsch、Fréchet、Procrustes、空间直方图(`dedupcad_precision/vendor/scoring.py`、`vendor/entities_match.py`);真 FAISS+Qdrant(`src/core/similarity.py`、`vector_stores/qdrant_store.py`);真 PR 标定(5886 对扫描,`data/dedup_threshold_scan_*`)。**启发式,非学习。** | ✅ 真核 (4/5) |
| **分类 ML** | 真训练权重:graph2d GNN、PointNet(`models/pointnet_synthetic_v1.pth` 28MB)、ExtraTrees(`models/extratrees_*.joblib` 78MB)、cad_classifier;真金标 `data/manifests/golden_val_set.csv`(915)/`golden_train_set.csv`(3661);热重载 `src/ml/classifier.py:227` 真、带回滚。 | ✅ 真核(平台唯一真 ML) |
| **评测/治理** | 真 harness `src/ml/evaluation/`;CI `.github/workflows/evaluation-report.yml`(push/PR/nightly/quarterly);blind/release 门 `GRAPH2D_BLIND_GATE`/`HYBRID_BLIND_GATE`/`FORWARD_SCORECARD_RELEASE_GATE`;**硬治理门** `governance-gates.yml`(只有人工核验样本可训练)。 | ✅ 真核(稀有强项);短板:精度门默认软(`continue-on-error` / `*_STRICT='false'`) |
| **次级实用件** | cost(规则,`src/core/cost/estimator.py`,读 `config/cost_model.yaml`)、materials(~16K,`MATERIAL_DATABASE`)、diff(DXF 改版 Diff+ECN,`src/core/diff/`)、drift(真 PSI 接进 classifier,`src/ml/monitoring/prediction_monitor.py`,`hybrid_classifier.py:284`)。 | ✅ 真、保留 |
| **B-Rep 溯源** | 活跃工作流:manifest provenance/license 门、证据 worklist(近期 #495)。**正对数据瓶颈。** golden 目前是 informational 占位,待 50–100 个真 STEP/IGES(`brep-golden-eval.yml`)。 | ✅ 战略正确 |
| **反馈飞轮** | **未闭合**:`feedback.py:229` 自认 JSONL 占位;唯一自适应权重环 `src/ml/learning/feedback_loop.py` **孤儿**(仅 `__init__.py`+tests 引用,classifier 从不读回);`scripts/auto_retrain.sh` 真但**无 CI/cron 触发**、读另一个 store;active-learning 默认关(`active_learning_policy.py:10`)+ 内存(`active_learning.py:305`);`benchmark/feedback_flywheel.py:14` 自诊断 `"passive_feedback_only"`。 | ⚠️ 2/5 脱线管道 |
| **Copilot/assistant** | 真 RAG+多 provider+function-calling(`src/core/assistant/`,14.5K+7.5K 测试),**但休眠**:任何 `requirements*.txt` 都无 LLM SDK、默认 offline;**Claude 默认模型 `claude-3-sonnet-20240229`(`llm_providers.py:22`、`assistant.py:54`)已退役——调用即 404**;`function_calling.py:105` 的 `claude-sonnet-4-20250514` 已弃用。 | ⚠️ 真但死火 |
| **`src/core/vision/`(82K,占 core 1/3)** | **~90% 脚手架/桩**:68 个 `*VisionProvider` 装饰器(Chaos/Saga/EventSourced/MultiRegion…,不做任何视觉)+ `experimental/automl_engine.py` 用 `random.uniform` **伪造**训练指标;默认 provider 是硬编码桩(`providers/deepseek_stub.py`)。**0 本地 ML**(无 torch/cv2/faiss)。**外部仅 3 处引用**,其中 `dedupcad_vision.py` 只用了它的 `circuit_breaker`。真代码仅 ~5–8K(VLM API 胶水 + phash + PIL)。 | ❌ 泡沫 |
| **~40-dir 企业模式层(~58K LOC)** | **~82% 运行期不可达**;仅 7 dir 承重(`providers`/`resilience`(仅 adaptive_decorator)/`vector_stores`/`twin`/`cost`/`storage`/`logging`);~20 dir 只被测试引用(`cqrs` 里是虚构 `CreateOrderCommand`/`AddItemCommand` 电商 demo);**13 dir 零引用死代码 ~7.1K LOC**。 | ❌ 泡沫(且测试给假绿覆盖) |
| **v2 REST / gRPC / pointcloud `/similar`** | v2(`src/api/v2/endpoints.py`)、gRPC(`src/api/grpc/server.py`)未挂载、返回 mock;pointcloud `/similar` 硬编码空。 | ❌ 桩,交付零 |

---

## 2. 锁定的定位决策(owner ratify)

### 2.1 [锁-A|默认推荐] 角色 = DedupCAD 家族的 canonical「CAD 查重 + 分类引擎」

代码、数据、部署、评测**四样全指向 DedupCAD**:唯一真集成消费者(已发布镜像 `ghcr.io/zensgit/dedupcad-vision`、E2E `make test-dedupcad-vision`、出站计数 `src/utils/analysis_metrics.py:818`、JSON `contracts/`)。`PLM`/`ERP`/`MES` 在 `src/` word-boundary grep 为空;`metasheet` 全仓 0 次;`Yuantus` 仅 1 注释。

**决策:承认真身,脱掉"平台"戏服。** 做那个引擎/中台,把这唯一深集成做到极致。

### 2.2 [开放问题|owner 拍板] 是否走 B(横向 CAD 智能中台)

若战略上确要横向服务 PLM/metasheet,请当**绿地集成**对待——今天一行对接代码都没有。**选一个真实第一消费者立项建 adapter + 量化 ROI**,别假设"天然是扩展"。本文建议:A 先钉死,B 作为 A 稳固后按 ROI 决定的**显式一跳**,不靠 README 默认成立。

### 2.3 [锁|三仓真相源] 消除 vendored scoring 重复(**独立 owner-decision + 跨仓轨,别塞进串行 Phase**)

**是两个不同层面,别混为一谈:** ① `dedupcad_precision/` **in-process vendor** 了 `dedupcad` 的 L4 几何 scoring(`__init__.py:1-6`);② 本仓是 `dedupcad-vision` :58001 的 **HTTP 客户端**做视觉召回(`dedupcad_vision.py:36`,`:370`)。→ 要消除的是 **①(in-process 拷贝)**;② 是正常服务调用,**保留**。决策:定 ① 的真相源(抽成共享 pip 包,两仓依赖它)。
**⚠️ 去 vendor 非零成本(深审补):** 那份 vendored scoring 是 **5886-对 PR 阈值标定所锁定的 pinned artifact**;直接 repoint 到 canonical 会**静默失效阈值(精度回退且无失败测试)**。故:pin dedupcad 到精确 tag/commit → 采纳时**重跑 5886-对扫描**,门控 `precision-delta ≤ ε` → 过 golden dedup 门前**保留 vendor 作 fallback**。因需跨仓协调 + owner 拍板,**单列一条并行轨,勿阻塞其它工作**。

### 2.4 [锁|横切铁律] AI 是"提议"非"权威" + 证据精确
① 每个模型输出带 置信度 + 来源 + human-in-the-loop(同 metasheet2 的 untrusted-write 原则)。② **别再让目录名/README 声称代码没有的能力**(vision 误名、"trained ML" 无权重、"feedback loop" 实为 passive)——把你们 B-Rep provenance 门的"证据优先"推广到整个平台的能力自述。

---

## 3. 路线图(**深审后 v2:3 轨并行,非 5 段串行**)

> 12-agent 对抗验证纠正了 v1 的串行结构:去泡沫 / 闭飞轮 / 数据三件事**互不依赖、归不同执行者**;串行会把**真瓶颈(数据)排到最后**、把**手动 hygiene 当 headline**([21])。改为三轨并行,在"评测门翻硬"处汇合。

### 轨 C · 数据(瓶颈,human-gated,**第一天就起——lead time 最长**)
- **B-Rep 真数据**:`data/brep_golden/` 已落 **~60 个真 NIST STEP/STP**(**非**单样例——深审纠正 v1 的"仅占位"说法),`brep-golden-eval.yml` 只等把它们 curate 成 release-eligible manifest + 标注。工具 100% 就绪,缺的是 curate+label,不是从零 sourcing。**外部 lead time 最长,必须最先并行,不能等代码轨([3][18])。**
- **建"反馈 SOURCE"([19],飞轮的真前置):** 今天 DedupCAD **出站单向**(`analysis_metrics.py:816` 仅出站计数),corrections 存量 ~40 行 → 飞轮**无燃料**。先建一条纠正回收通道(DedupCAD 查重工作流人工复核 / golden 标注)**产出 `low_conf.csv` 的人工核验行**。

### 轨 B · 飞轮(护城河,human-gated,与 C 汇合)
- **拆"接管道"与"喂燃料"([1]):** 主 spine **已在跑**——classifier 推理 `maybe_enqueue` 已把 low-conf 样本持久化到 `data/review_queue/low_conf.csv`,`auto_retrain.sh` 已读它、且被**真硬门** `governance-gates.yml`(0 `continue-on-error`)守着。
- 要做的只有:①**人工复核动作**(填 `reviewed_label/human_verified`,今空缺、未建)+ ②**排程触发** `auto_retrain.sh`(它 gate 在 `MIN_REVIEWED=200` 人工核验样本,需轨 C 的燃料);③**删重复死环**:`feedback_log.jsonl`(占位)+ 孤儿 `FeedbackLearningPipeline`(其 EMA 权重 classifier **从不读回**,是**删/重建**,非"接")。
- **两条重训 track 分开别混([9]):**(A)`ActiveLearner→finetune_from_feedback.py→classifier_vX.pkl→reload_model`(真热重载+回滚);(B)`low_conf.csv→auto_retrain.sh→graph2d .pth→GRAPH2D_MODEL_PATH+重启`(**今天无热重载**)。

### 轨 A · 诚实化 + 去泡沫(agent-fleet 可做,**但先设反膨胀硬门,再删**)
- **[linchpin·先做] 反膨胀门翻 blocking([13][14]):** `code-quality.yml` 每步 `|| true`(dead-code `vulture`:165、duplicate-code:123 全 fail-soft)而 "Code Quality" 是 watched-required → **strip 掉 `|| true` 并设 branch-protection 必需**;加 **prune-safety job**(见 §60)+ **scaffolding-budget job**(复用 `metrics-budget-check.yml` 的 exit-1,防 core 文件数反弹;`vulture` 先 commit baseline、只 fail 新增)。**没这道门,清理会被同一车队重新灌回——这是整份方案最该补的一条。**
- **[owner-gate·强制,[17]] 加 CODEOWNERS**:`.github/workflows/**`、`src/core/**` 删除、本 doc 需人工审——现**无 CODEOWNERS**,CLAUDE.md 又说分支保护可摘,持 merge 权的同一车队可自批删除/翻门 PR。
- **然后再删** 13 个零引用死 **顶层 scaffold dir**(~7.1K LOC)—— **完整路径,勿裸名(会与同名活模块混淆)**:
  `src/core/circuit_breaker`、`src/core/dead_letter_queue`、`src/core/outbox`、`src/core/message_bus`、`src/core/idempotency`、`src/core/api_versioning`、`src/core/rate_limiter`、`src/core/webhook`、`src/core/caching`、`src/core/batch_processing`、`src/core/event_sourcing`、`src/core/health_check`、`src/core/notifications`。
  - **⚠️ 不在删除范围(同名但活)——排除清单以附录 A 的 twin-scan 输出为准,勿手抄(手抄会漏)。** 已核实活 twin 至少含:`src/utils/{circuit_breaker,idempotency,rate_limiter}.py`(`idempotency` 被 `api/v1/ocr.py:28`+`drawing.py:34` 用)、`src/core/assistant/caching.py`(测试用)、**以及深审补漏的 5 个**:`src/core/{resilience,resilience_enhanced,gateway}/circuit_breaker.py`、`src/core/{resilience,gateway}/rate_limiter.py`。
  - **⚠️ 特例(先解耦再动 vision):** `src/core/vision/circuit_breaker.py` 被**唯一活消费者** `dedupcad_vision.py:18-24` import。故 Phase 0 顺序 = **先把它抽到中性模块并 repoint `dedupcad_vision.py` 的 import,再降级/改名 vision**,否则打断唯一真集成。(`dedupcad_vision.py` 用的 `dedupcad_vision_*` 指标在 `src/utils/analysis_metrics.py`,不受 vision 改动影响。)
  - **Phase 0 PR 不靠人工 grep——必须过一道 blocking `prune-safety` CI job**:对每个待删路径 grep 外部 importer(排除自身+tests),有残留即 `exit 1`;并跑一次 `import src.main` 运行时冒烟(**不加 `|| true`**)以抓 lazy-import(如 `health/self_healing.py` 的懒引入)。
- 卸/删未挂载面:`src/api/v2`、`src/api/grpc`、未注册的 `api/v1/batch.py`/`api/v1/websocket.py`、未 mount 的 `AuditMiddleware`;pointcloud `/similar` 桩下线;删 vision 的 68 个 `*VisionProvider` 装饰器 + `experimental/`(~90%),按 §60 特例**先解耦 `circuit_breaker`**。
- **honesty sweep(一次收齐"码声称了没有的能力",[20]):** `manufacturing_v2` 缺 encoder 权重→静默退化 TF-IDF(**今天就在骗**,不拖后期)、退役模型 ID(→`claude-sonnet-5`)、假绿测试、`vision/` 降级为 "VLM 描述 façade"、孤儿 metric-learning 模型——**全归这一轮**,符合 §2.4 铁律。
- **前向假绿护栏([16]):** blocking diff-coverage 门 + assertion-presence lint(零 assert 的测试文件 fail)。
- **里程碑**:core 从 626 收敛到 ~15–20 个模块(由 scaffolding-budget 门锁死防反弹);旗舰契约 `analyze→vector→similarity/dedup + classification`,cost/materials/diff 一等次级。

### 汇合点 · 评测门翻硬(**分级 + path filter,别一刀 red 全仓,[6]**)
- `evaluation-report.yml` 有 **70 处 `continue-on-error`**,且它与 `governance-gates.yml` 都在 `pull_request:[main]` **无 path filter** → 直接翻硬会 **red 掉每个在途 PR(含别窗口非-ML 线)**。故:①枚举 70 处软步 + 真 vs 占位 golden 覆盖;②**逐门翻硬 + 加 path filter(只对 ML/scoring PR 生效)**;③同步更新 soft-mode-smoke 契约;④**先对 open PR dry-run 再翻**。这本身是 owner-ratify 的分级变更。

> 身份/A-vs-B(§2.1/2.2)与三仓真相源(§2.3)不变;A-first、留干净服务边界让 B 廉价的判断,经深审对 B-overreach 的反驳(rejected 3 条 AvB/ROI 批评)后仍成立。

---

## 4. 飞轮现状 vs 缺口(**深审校正——v1 的"全断/纯接线"说法不准**)

```
[已在跑的 spine — 生产级]
classifier 推理 ─maybe_enqueue─► data/review_queue/low_conf.csv ─► auto_retrain.sh
    ─(governance 硬门: 只有 human_verified 样本可训, 0 continue-on-error)─► graph2d .pth ─► GRAPH2D_MODEL_PATH + 重启

[真正缺的 3 件]
1) 燃料([19]): 无反馈 SOURCE(DedupCAD 出站单向; corrections ~40 行) + 人工复核动作未建
   → low_conf 的 human_verified 行填不上 → auto_retrain 的 MIN_REVIEWED=200 永不满足
2) 触发: auto_retrain.sh 无排程/CI(今天手跑)
3) 删重复死环: feedback_log.jsonl(占位) + 孤儿 FeedbackLearningPipeline(EMA 权重 classifier 不读回)= 删, 非接

[另一条独立热重载 track, 别和上面混([9])]
ActiveLearner 导出 ─► finetune_from_feedback.py ─► classifier_vX.pkl ─► reload_model(热重载 + 回滚槽)
```
校正:**护城河的真前置是"反馈 SOURCE + 人工复核"(轨 C),不是接线**——spine 大体已生产级。若要统一 **4 个 store(4 套 schema/字段名,[11])**,那是"定 1 套 canonical schema 或选 1 个 store 作真相删其余 3",属 **invent(schema+迁移)**,别低估。

---

## 5. 本 PR 的边界 + Phase 0 授权前置(深审强化)
- **无 runtime、不删任何代码、不改 flag/CI**——纯 for-review 文档。
- Phase 0 删除**在 owner ratify 本文之后**,作为独立 for-review PR;且该 PR **必须**:①先落"反膨胀硬门"再删(轨 A linchpin);②过 **blocking `prune-safety` CI job**(非人工 grep)+ `import src.main` 运行时冒烟;③**先解耦 `vision/circuit_breaker`**(§60 特例,否则打断 `dedupcad_vision.py`);④逐路径附"零外部 import"证据;⑤**不自动合**。
- **owner-ratify 决策项**:§2.1(走 A)、§2.3(三仓真相源=独立跨仓轨 + 阈值重标定门)、**"评测门翻硬"分级方案**、**引入 CODEOWNERS**。本文把 **A + 三轨并行 + linchpin 反膨胀门** 列为推荐,决定权在 owner。

---

## 附录 A — 证据复算命令(5-probe 关键命令)

> 本文证据粒度到 **file/path + line**,非严格逐行。以下命令在 repo 根运行,可复算主要结论。

- **死 dir 零引用 + 同名活模块碰撞**(Phase 0 安全性,防误删):
  ```sh
  for n in circuit_breaker dead_letter_queue outbox message_bus idempotency api_versioning \
           rate_limiter webhook caching batch_processing event_sourcing health_check notifications; do
    echo "$n: core=$([ -d src/core/$n ] && echo ✓ || echo ✗)  same-named-live=$(
      { find src -type d -name "$n"; find src -type f -name "$n.py"; } | grep -vx "src/core/$n" | tr '\n' ' ')"
  done
  ```
- **脚手架层可达性**(外部 importer,排除自身/tests):`grep -rlE "src\.core\.<dir>" src/main.py src/api src/core | grep -v "src/core/<dir>/"`(逐 dir 统计)。
- **`vision/` 无本地 ML**:`grep -rlE "torch|tensorflow|cv2|faiss|sklearn|onnx" src/core/vision/`(应为空);默认桩 `src/core/vision/providers/deepseek_stub.py`。
- **真 ML 权重位置**:`ls models/*.pth models/*.joblib`;加载方在 `src/ml/*`、`src/core/classification/*`(**非** `src/core/vision/`)。
- **飞轮孤儿**:`grep -rl "FeedbackLearningPipeline" src tests`(应只有 `src/ml/learning/__init__.py` + `tests/unit/test_feedback_loop.py`);主 feedback 端点自述 `src/api/v1/feedback.py:229`。
- **消费者**:`grep -rwE "PLM|ERP|MES|metasheet|Yuantus" src/`(除 `Yuantus` 1 注释外为空);DedupCAD 真集成 `src/core/dedupcad_vision.py`、`contracts/`、出站计数 `src/utils/analysis_metrics.py:818`。
- **退役模型 ID**:`grep -rn "claude-3-sonnet-20240229\|claude-sonnet-4-20250514" src/core/assistant/`。
