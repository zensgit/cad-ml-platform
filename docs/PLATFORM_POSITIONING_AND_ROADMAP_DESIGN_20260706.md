# CAD-ML-Platform 定位与路线图 — Design (for-review)

**编制日期**: 2026-07-06
**性质**: for-review 提案 / 设计锁。**本 PR 无 runtime、不改代码、不删代码**——它锁定的是*方向*。承重决策(角色 A/B、三仓边界)留给 owner ratify;merge 即授权 Phase 0 动刀。
**依据**: 一次 5-路代码级审计(vision 实质 / ~60-dir 脚手架引用图 / 实际服务面 / 反馈飞轮 / 消费者),以 file:line 证据为准,**不以 README 自述为准**。

---

## 0. 一句话结论

代码里,本仓**不是**"为 ERP/MES/PLM 提供统一智能分析的 CAD ML 平台"。它是:**一个 CAD 图纸查重 + 分类引擎,外面裹了一层由 agent 车队生成的巨量"企业级微服务"脚手架**;真核不大但**评测/治理骨架罕见地强**,且当前 B-Rep 溯源正对准真瓶颈(数据)。**唯一真实、部署级的消费者是 DedupCAD;PLM/ERP/MES/metasheet 在 `src/` 里零集成代码。**

一个定调数据点:`git` 2226 commits、单作者、单日峰值 235 → 这是 agent-fleet 的 built-ahead 产物,膨胀由此而来。

---

## 1. 诊断:真核 vs 泡沫(证据表)

| 层 | 代码级结论(file:line 证据) | 归类 |
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

### 2.3 [锁|三仓真相源] 消除 vendored scoring 重复

现状:`dedupcad_precision/__init__.py` 明写"vendors the core … from the `dedupcad` repository";同时本仓是 `dedupcad-vision` 的 HTTP 客户端(`dedupcad_vision.py:36` `:58001`,`:370` `/api/v2/search`)。→ **同一 scoring 两处实现。** 决策:定**一个**真相源(本仓成 canonical、dedupcad 依赖它;或反之以依赖引入,停止 vendor 拷贝)。不除此债,后续每步双倍成本。

### 2.4 [锁|横切铁律] AI 是"提议"非"权威" + 证据精确
① 每个模型输出带 置信度 + 来源 + human-in-the-loop(同 metasheet2 的 untrusted-write 原则)。② **别再让目录名/README 声称代码没有的能力**(vision 误名、"trained ML" 无权重、"feedback loop" 实为 passive)——把你们 B-Rep provenance 门的"证据优先"推广到整个平台的能力自述。

---

## 3. 路线图(接住已在跑的 `phase1-slim-*` + B-Rep)

### Phase 0 — 诚实化 + 去泡沫(让真核可见)· 低风险可回滚,先做
- 删 13 个零引用死 dir(~7.1K LOC):`circuit_breaker`/`dead_letter_queue`/`outbox`/`message_bus`/`idempotency`/`api_versioning`/`rate_limiter`/`webhook`/`caching`/`batch_processing`/`event_sourcing`/`health_check`/`notifications`。
- 卸/删未挂载面:`src/api/v2`、`src/api/grpc`、未注册的 `api/v1/batch.py`/`api/v1/websocket.py`、未 mount 的 `AuditMiddleware`;pointcloud `/similar` 桩下线或显式标注。
- ~20 个 test-only 模式 dir 与 vision 装饰器动物园:**删,或隔离进显式 `experimental/` 边界**;修"假绿"测试(它们给死代码假覆盖)。
- **把 `vision/` 如实降级/改名为 "VLM 描述 façade"**;修 assistant 退役模型 ID(`claude-3-sonnet-20240229`→`claude-sonnet-5`,或整块 default-off 明确标休眠)。
- **目标里程碑**:core 从 626 文件收敛到真正的 ~15–20 个模块;给 `phase1-slim-*` 一个北极星。

### Phase 1 — 锐化身份 + 解三仓边界
- 落实 §2.3(消除 vendored scoring);锁旗舰契约:`analyze→vector→similarity/dedup + classification`,cost/materials/diff 为一等次级。

### Phase 2 — 闭合飞轮(接线,不是发明)· 真护城河
- 把 3–4 个断裂 store 接成一条线:**feedback API → active-learning 队列(打开 `ACTIVE_LEARNING_ENABLED` + 持久化)→ `auto_retrain.sh`(排程/CI 触发,`reload_model` 热重载已真)→ 评测门(精度门从软翻硬)**。
- 孤儿 `FeedbackLearningPipeline` 接上或删。**这一条把"瓶颈是数据"变成复利:真评测 × 真纠正 × 真重训 = 别人抄不走。**

### Phase 3 — ML 诚实化
- 补 `models/embeddings/manufacturing_v2/` 缺失的 encoder 权重(训/发或删掉声明——现状会静默退化到 TF-IDF);孤儿 metric-learning 模型接上或删;明说 dedup 是古典几何(强,没问题)、分类才是学习部分。

### Phase 4 — 溯源数据 + 一个灯塔消费者
- 继续 B-Rep 溯源(打数据瓶颈)。若走 B:给**一个**消费者出干净 adapter(metasheet2 行富集 或 PLM BOM auto-fill),量化 ROI 再泛化。**不一次对接全部。**

---

## 4. 飞轮接线图(Phase 2 目标态)

```
                        [现状:断裂]                         [目标:一条线]
POST /feedback ──► data/feedback_log.jsonl (死胡同)   POST /feedback ─┐
active-learning ──► memory(默认丢) ──► 人读            active-learning ─┼─► 统一样本库(持久化, 人工核验)
FeedbackLearningPipeline ──► 孤儿(仅 tests)                            │        │(治理门: 只有核验样本可训练)
auto_retrain.sh ──► 读另一个 store, 无触发                              ▼        ▼
                                                        排程/CI 触发 auto_retrain.sh ─► golden 评测门(硬)
                                                                                         │ pass
                                                                                         ▼
                                                                        reload_model (热重载, 带回滚)
```
零件全部已存在且多为生产级;缺的是**接线 + 排程 + 把精度门翻硬**,不是发明。

---

## 5. 本 PR 的边界
- **无 runtime、不删任何代码、不改 flag/CI**——纯 for-review 文档。
- Phase 0 的删除动作**在 owner ratify 本文之后**,作为独立 for-review PR 落地(带"零外部引用"证据 + CI 绿),**不自动合**。
- §2.1(A)、§2.2(A/B)、§2.3(三仓真相源)是 **owner-ratify 决策项**;本文把 A 列为推荐,决定权在 owner。
