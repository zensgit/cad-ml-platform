# CAD 组合级战略 — 一盘棋怎么下（Design, for-review）

**编制日期**: 2026-07-12
**性质**: for-review 提案。**本 PR 无 runtime、不改代码、不删代码、不碰任何 flag/CI** —— 它锁定的是*组合层面的方向*。承重决策（§6 三项）留给 owner ratify。
**范围**: 跨 9 个仓的组合（cad-ml-platform / dedupcad / dedupcad-vision / Yuantus / PLM / VemCAD / CADGameFusion / Athena / metasheet2）。
**依据**: 一次 4-路跨仓代码级审计（几何栈 / 查重家族 / PLM-ECM / 跨仓依赖图），以 **file/path + line 证据**为准，**不以 README 自述为准**。是 `PLATFORM_POSITIONING_AND_ROADMAP_DESIGN_20260706.md`（单仓级）在**组合层面**的续篇。
**证据强度声明（重要）**: cad-ml 是**内部深挖**（飞轮、金标、治理门都逐行核过）；**Yuantus / VemCAD / Athena 只到 README + 集成客户端 + contracts + commit 信号级别**（不在本 org，未逐行审）。故本文关于 Yuantus 的结论按"**结构上是中枢**"表述（已被证据支撑），而"**押注 Yuantus**"仅作**假设**，待 §6 两问回答后才成立。

---

## 0. 一句话结论

**这不是一堆互不相干的项目——它是一个真实连通的 CAD/PLM 栈，中枢是 Yuantus（元图PLM）。** 布局是对的（你自己早写下来了），棋子也都在盘上（7 条 LIVE 代码边）。

**但真正的"降维打击"已经发生了，且不是来自竞品：AI 把"造东西"的成本打到接近零，于是在没有一个已证实客户的情况下，造出了 9 个仓、3 套查重、2 个 PLM。** 瓶颈从来不是"能不能造"，而是**验证与分销**。

---

## 1. 真实依赖图（跨仓代码级，非 README 自述）

```
metasheet2 ─(flag,半接)→ Yuantus(元图PLM) ─→ cad-ml-platform ─→ dedupcad-vision
                              │  ├─→ Athena (CMIS 内容发布)
                              │  └─→ CADGameFusion ←─(submodule)─ VemCAD
                              └─ .NET AutoCAD/SolidWorks 插件 ──→ 直达设计师
```

### 1.1 LIVE 边（真接在代码/构建里，共 7 条）

| 边 | 证据 |
|---|---|
| `VemCAD → CADGameFusion` | git submodule，已 checkout：`VemCAD/.gitmodules:1`（`deps/cadgamefusion`）；代码直引 `VemCAD/apps/web/shared/runtime_bridge.js:13-14`；构建 `scripts/dev_build.sh:5,36` |
| `Yuantus → cad-ml-platform` | `Yuantus/src/yuantus/integrations/cad_ml.py:69`（`CadMLClient`）；任务 `meta_engine/tasks/cad_pipeline_tasks.py:1323`（`vision_analyze_sync`/`ocr_extract_sync`）；契约 `contracts/cad_ml_vision_analyze.schema.json` |
| `Yuantus → dedupcad-vision` | `integrations/dedup_vision.py:94`；任务 `cad_pipeline_tasks.py:1244`；契约 `contracts/dedupcad_vision_search_v2.schema.json` |
| `Yuantus → Athena` | `integrations/athena.py:78`；`meta_engine/ecm_publication/cmis_adapter.py:81`；`docker-compose.ecm-publish.yml` |
| `Yuantus → CADGameFusion` | `docker-compose.cadgf.yml:11` 挂 `/opt/cadgf`（convert CLI + DXF importer `.so`）；`settings.py:270` |
| `cad-ml → dedupcad-vision` | `src/core/dedupcad_vision.py:36,48`（httpx + 熔断 + 重试）；契约 `contracts/dedupcad_vision_search.schema.json` |
| `cad-ml → dedupcad`（vendored） | `src/core/dedupcad_precision/vendor/__init__.py:1` —— 把 dedupcad 的 L4 几何 scoring **拷进本仓**以省一次服务跳 |

### 1.2 默认关 / 半接 / 谱系边

- `dedupcad-vision → cad-ml`（L3"大脑"）：**默认关**且**被格式挡死**——`config/progressive_config.py:215`（`ML_PLATFORM_ENABLED` 默认 `"false"`）、`integrations/ml_platform.py:29,44`（`enabled=False`），且 `/api/v1/analyze` 拒收 PNG/JPG（正是 L3 会送的），见 `DEDUP2D_VISION_INTEGRATION_CONTRACT.md:62-72`。
- `dedupcad-vision → dedupcad`（L4 几何）：默认关（`config/progressive_config.py:222`，`GEOMETRIC_ENABLED` 默认 `"false"`），且**目标端点在 dedupcad 上不存在**（`geometric/client.py:246` 打 `/api/compare`，dedupcad 实际只有 `/compare`、`/api/visual-diff`）——**是过期的接线**。
- `metasheet2 ↔ Yuantus`：pact 与 TS 消费端为真（`Yuantus/contracts/pacts/metasheet2-yuantus-plm.json`），但 Yuantus 侧 bridge **是 flag 门控的惰性骨架**（`api/routers/metasheet_bridge.py:1-20` 自述 "inert by design"）。

### 1.3 组合级更正（推翻 `..._20260706.md` §2.1 的一处前提）

> **cad-ml 的真消费者是 Yuantus，不是 DedupCAD。**

原单仓审计得出"唯一真消费者是 DedupCAD"，在组合层面**不成立**：dedupcad-vision 出厂把 cad-ml 这层**关掉且格式挡死**（见 §1.2），即**在 dedupcad-vision 的出厂路径里 cad-ml 贡献为零**；而 Yuantus 是**真的在调** cad-ml 的 vision/OCR（§1.1）。这改变了 cad-ml 该为谁优化。

---

## 2. 组合级"泡沫"清单（同一种病，粒度从模块变成仓）

本 session 在 cad-ml 内剪掉 98 个脚手架模块（#504）——**组合层面是同一种病**：

| 冗余 | 事实 |
|---|---|
| **3 套查重实现** | ① `dedupcad`（作为服务已死：零出站、零 LIVE 入站，只剩**代码捐献**——precision core 被 cad-ml vendored、AutoCAD 插件被 Yuantus 收录、又被 PLM 重写为 `dedupcad2`）；② `dedupcad-vision` **内嵌自研** CV/ML（pHash→FAISS→ONNX MobileNetV3，出厂 ON）；③ cad-ml 的 `dedupcad_precision/vendor/`（LIVE） |
| **几何 IP 被重写两遍** | dedupcad-vision **把 dedupcad 的 entity-diff 几何 IP 原地重写**（`geometric/entity_diff.py` + 自己的 `/api/compare`、`/api/diff/entities`，CHANGELOG [1.2.0] 称之 "L4 Killer Feature"）——**спearhead 正在吸收被它依赖的那个仓的差异化** |
| **2 个 PLM** | Yuantus（正统，2,119 commits，活跃至 2026-07-09）vs PLM-standalone（126 commits，死于 2025-12，**无 GitHub remote**，正被 `Yuantus/docs/REUSE.md` 逐层收割；其自带 dedup 已被官方判定改用 dedupcad-vision） |

**这个思路你自己早就写下来了**（`VemCAD/docs/VEMCAD_RENDER_SERVICE_PLAN_20260610.md:137`）：
> "别开第四个 CAD 栈；Yuantus = 数据/编排中枢，我们做几何/渲染引擎，dedupcad-vision 做相似度引擎……PLM-standalone 和 Yuantus 重复建设，**需要定单一记录系统**。"

**战略是对的，问题在执行从未收敛。**

---

## 3. 真正的降维打击：不是竞品，是"造得太便宜"

| 仓 | commits | 客户 |
|---|---|---|
| cad-ml | 2,226（单人，单日峰值 235） | 0 |
| Yuantus | 2,119 | 0（私有交付/内部共享 dev，无外部付费客户证据） |
| Athena | 1,499 | 0（仅内部 UAT，README 自述 "deployment configuration is pre-production"） |
| dedupcad-vision | — | 0（已发 GHCR 镜像 + Helm + Release v1.1.1，**但无具名试点**；验收报告仍是空模板 `reports/DELIVERY_SUMMARY.md:14,43`） |
| dedupcad | — | 0（最强验证 = 67 对工业影子模式；`PROJECT_STATUS.md:468` "User Feedback \| TBD \| Pending"） |
| VemCAD | — | 0（v0.1.0 可打包，无安装包产物，无具名客户） |

> **AI 已经对你完成了一次降维打击——不是在产品上跟你竞争，而是让造东西便宜到你在没有一个客户的情况下造了 9 个。**

**跨全部 9 个仓：零个已证实的外部付费客户。** 这是头号生存风险，不是 AI 竞争。

---

## 4. AI 会打到这盘棋的哪一层

| 层 | 会被前沿模型降维打击？ | 依据 |
|---|---|---|
| 理解层（cad-ml vision / OCR / 分类**准确率**） | **会** | `vision/providers/qwen_vl.py:157-237` 就是"图转 base64 + prompt + 解析 JSON"，无专有视觉逻辑。**而它在出厂路径里本来就是关的**（§1.2） |
| 相似度 / 几何裁决 | **打不到** | 刚体变换容差下的等价性是**数学不是感知**：`dedupcad_precision/vendor/scoring.py:506,662,679`（Hungarian / Procrustes / RANSAC-Kabsch）、`entities_match.py:506`（Fréchet） |
| 几何内核（CADGameFusion） | **打不到**，但**也不是护城河** | 是真 C ABI + document model（`core/include/core/core_c_api.h`，~80 个 `cadgf_*` 符号），但**硬活是现成的**（`vcpkg.json`: clipper2 / earcut / eigen），且**只有 2D**（无 NURBS/B-Rep/参数化）——**不是 OCC/Parasolid 级内核** |
| **中枢 / 记录系统（Yuantus）** | **打不到，且是价值捕获层** | AI 不颠覆"记录系统 + 工作流 + 分销"；AI 只是它内部的一个功能 |

### 4.1 必须点破的自我欺骗

> **若 AI 让引擎变便宜，那竞争对手的 PLM 也能便宜地拿到同样的引擎。**

所以 **引擎质量是入场券，不是护城河**。真护城河只能是四样：
1. **记录系统的锁定**（BOM / ECO / 版本 / 权限 —— 换系统的迁移成本）；
2. **工作流沉淀的专有修正/标注数据**（设计师在真实工作流里产生的 corrections）；
3. **中文压力容器/工艺装备领域知识**（cad-ml 的 24 类 taxonomy：法兰/轴类/箱体/换热器/封头/人孔…）；
4. **切换成本**。

---

## 5. 飞轮：管道在，燃料不在（**条件式表述**）

cad-ml 的飞轮**此刻 0/200**：`data/review_queue/low_conf.csv` 有 211 行候选、**0 行人工核验**；`scripts/auto_retrain.sh` 的 `MIN_REVIEWED=200` 永不满足。整套 ML/治理骨架**在空转一个空油箱**。

**组合层面确实存在那条燃料管道**：Yuantus 通过 `.NET` AutoCAD/SolidWorks 插件（`Yuantus/clients/autocad-material-sync/CADDedupPlugin/DedupApiClient.cs`）直达设计师，在真实 CAD 工作流里驱动查重/分类——**设计师的纠正就是燃料**。

> **⚠️ 但别把"代码里有通道"洗成"有分销"。** Yuantus 至今**零外部客户**。正确表述是条件式的：
> **若 Yuantus 拿下真实设计师使用 → 把插件工作流里的人工纠正回流到 cad-ml 的 review queue → 飞轮才点得着。**
>
> "没有客户"这个问题**没有因为拉高视角而消失，它只是上移了一层。**

**附带诚实标注**（来自 cad-ml 深挖）：金标 train 与 **val 都约 2/3 是合成/增强**（val: 454 合成 + 162 增强 / 914）。故"91.5% golden-val 门"在很大程度上**在给自己的合成分布打分**——治理是好工程，但它守的评测**部分是自证循环**。

---

## 6. Owner-ratify 决策项（**只有 owner 能答，本文不替你决定**）

1. **`adharamans` 是不是你自己的新 org？**
   代码上看是：Yuantus **2,119 / 2,119 条 commit 全是 `zensgit`**（`git log --format='%ae' | sort -u`）。本文按"**仍是你的**"来分析。**请确认**——整盘棋都建立在"你拥有中枢"这个前提上。若不是，价值捕获层就不在你手里，其余 8 个仓就沦为别人产品的引擎供应商。

2. **有没有一个客户，痛到愿意把完整垂直切片（设计师 → PLM → 查重/分类 → 确定性记录）真的部署起来？**
   - **有** → 全力打通那条链（§7）。
   - **没有** → 你有 9 个仓的引擎和 0 个买家。**先找买家，别再写引擎。**

3. **单一记录系统选谁：Yuantus 还是 PLM-standalone？**
   你自己的 thesis 文档已标为待决（§2）。证据强烈指向 Yuantus 正统、PLM-standalone 归档。**定一个，杀一个。**

---

## 7. 建议路线（若 §6-2 答案为"有"）

1. **停止横向铺摊子。** 冻结引擎层功能开发；把引擎做**无聊、稳定、可换、收敛**。
2. **收敛冗余**（与剪 vision zoo 同理，粒度换成仓）：查重定**一个真相源**（建议 dedupcad-vision 的内嵌实现 + cad-ml 的 vendored precision 二选一，见 `..._20260706.md` §2.3 的去-vendor 轨与阈值重标定门）；PLM 定**一个记录系统**；dedupcad-top 降级为归档/代码捐献。
3. **只打通一条端到端垂直切片**：设计师在 AutoCAD → Yuantus 工作流 → 查重/分类 → 确定性记录 → **纠正被捕获** → 飞轮。**这条链每一段都已存在，但从未被端到端跑通过一次。**
4. **AI 用法定死**：
   - 引擎可换 —— 骑基座模型的进步曲线，**永不硬绑模型版本**（退役的 `claude-3-sonnet-20240229` 调用即 404，是本仓现成的反面教材）；
   - **AI 绝不进确定性裁决** —— 查重/BOM/ECO 的记录完整性**就是**护城河本身（"AI 提议，确定性组件裁决"，`src/api/v1/dedup.py:1-6` 的现有架构已站对位）；
   - AI 用来**打你自己的数据瓶颈** —— AI 辅助标注 → **人工核验**入库（正合本仓"只有 `human_verified` 可训"的治理门）；
   - 把治理/评测骨架重新定义为"**安全快速换模型的能力**"：别人换模型是赌博，你换模型有回归门。**这让 AI 商品化为你所用。**
5. **然后去找一个真客户。** 在此之前写的任何一行引擎代码，边际价值都接近零。

---

## 附录 A — 证据复算命令

```sh
# 跨仓真实边（在 ~/Downloads/Github 下跑）
grep -rn "CadMLClient\|DedupVisionClient" Yuantus/src/yuantus/integrations/
grep -rn "ML_PLATFORM_ENABLED\|GEOMETRIC_ENABLED" dedupcad-vision/src/caddedup_vision/config/progressive_config.py
sed -n '1,6p' cad-ml-platform/src/core/dedupcad_precision/vendor/__init__.py   # vendored 自述
cat VemCAD/.gitmodules                                                          # VemCAD → CADGameFusion

# Yuantus 归属（§6-1）
git -C Yuantus log --format='%ae' | sort -u        # 期望：仅 zensgit
git -C Yuantus remote -v                            # 期望：adharamans/yuantus-plm

# 飞轮燃料（§5）
awk -F, 'NR>1 && $7!="" {n++} END{print "human-reviewed rows:", n+0}' \
  cad-ml-platform/data/review_queue/low_conf.csv    # 期望：0（对比 MIN_REVIEWED=200）

# 金标合成占比（§5 诚实标注）
grep -c synthetic cad-ml-platform/data/manifests/golden_val_set.csv
```

---

## 附录 B — 本文不主张什么（防过度解读）

- **不主张** Yuantus 是"必然的押注"——只主张它**结构上是中枢**（7 条 LIVE 边里它占 4 条）。押注与否取决于 §6 的两个答案。
- **不主张** 确定性/可审计/离线是护城河——它们是**规则性行业的准入门槛**（竞品也能接确定性后处理）。真护城河见 §4.1。
- **不主张**任何仓"该被删"——PLM-standalone 与 dedupcad-top 的处置是 owner 决策（§6-3）。
- **不主张**已完成组合级尽调——Yuantus/VemCAD/Athena 仅到 README + 集成客户端级（见抬头证据强度声明）。
