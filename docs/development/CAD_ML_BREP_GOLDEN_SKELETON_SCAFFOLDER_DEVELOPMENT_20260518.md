# B-Rep Golden Manifest Skeleton Scaffolder — Development

Date: 2026-05-18
Stage: 2a (data preparation tooling)
Companion: `CAD_ML_BREP_GOLDEN_SKELETON_SCAFFOLDER_VERIFICATION_20260518.md`
Decision context: `CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_DEVELOPMENT_20260517.md` §4.4

## 0. 为什么

Stage 2a 需要 50–100 个 case 的 `brep_golden_manifest.real.json`。每 case ~14 字段
= 700–1400 个手填字段，结构错误（拼错 key、漏 `expected_topology`、
`public_nc` 误标 `release_eligible`）极易发生且 validator 只能事后报错。

`scripts/build_brep_golden_manifest_skeleton.py` 把"手填全部"降为"放文件 →
跑脚本 → 补 3 类人判断字段"，结构正确性由脚本保证。

## 1. 设计

| 维度 | 决策 |
| --- | --- |
| 输入 | `--root <dir>`，递归扫 `*.step/*.stp/*.iges/*.igs` |
| `source_type` 推断 | 取 `root` 下第一段路径，按约定桶映射（`internal/`→internal, `public_nc/`→public_nc, `vendor/` 等）；未知桶 → 默认 `internal` **且**打 `TODO-source-type` tag（不可静默错分进 release floor） |
| `release_eligible` 推断 | `source_type ∉ RELEASE_EXCLUDED_SOURCE_TYPES`（与 validator 同源常量） |
| `id` | 相对路径 slug 化；重名 stem 自动加序号去重 |
| `expected_behavior` | 默认 `parse_success` |
| `part_family` / `license` | 占位 `"TODO"` + tag |
| `expected_topology` | 保守占位（validator-valid，非真实下限）+ `TODO-topology` tag |
| 拓扑探测 | **不做** — 需 OpenCASCADE（CI-only）。真实下限在 OCC-provisioned `brep-golden-eval.yml` 里收紧 |
| str/Path 容错 | `build_skeleton` 内 `root = Path(root)`，公共 API 不因 str 崩 |

## 2. 输出

- skeleton JSON（`--output-json` 或 stdout）
- stderr summary：`case_count` / `release_eligible_count` /
  `source_type_counts` / `needs_source_type_review` /
  `all_need_part_family_license_topology`
- 0 文件 → exit 1 + warning（不静默产出空 manifest）

## 3. 与现有工具的边界

| 工具 | 职责 |
| --- | --- |
| `build_brep_golden_manifest_skeleton.py`（本次新增） | 扫目录 → 出结构正确骨架 |
| `validate_brep_golden_manifest.py` | 校验填好的 manifest（release floor / source_type / topology schema） |
| `scripts/ci/build_brep_golden_manifest_optional.sh` | CI wrapper：validate + OCC eval |
| `eval_brep_step_dir.py` | OCC 真实解析 + 拓扑评测 |

本脚本**不**重叠任何现有职责，是 manifest 构造前的"脚手架"前置步。

## 4. 人工填充工作流（脚本落地后）

```bash
# 1. 放文件
mkdir -p data/brep_golden/internal data/brep_golden/public_nc
cp <internal STEP/IGES> data/brep_golden/internal/
cp <ABC subset>         data/brep_golden/public_nc/

# 2. 生成骨架
python3 scripts/build_brep_golden_manifest_skeleton.py \
  --root data/brep_golden \
  --manifest-root . \
  --output-json config/brep_golden_manifest.real.json

# 3. 人工补：每 case 的 part_family / license / 真实 expected_topology
#    （grep '"TODO"' 与 tag TODO-* 定位）

# 4. 校验
python3 scripts/validate_brep_golden_manifest.py \
  --manifest config/brep_golden_manifest.real.json \
  --min-release-samples 50 --fail-on-not-release-ready
#    exit 0 ⇒ 进入 Stage 2a CI dispatch
```

## 5. 范围边界（显式排除）

- ✗ 不探拓扑（OCC-only，CI 职责）
- ✗ 不自动猜 `license`（法律判断，必须人填）
- ✗ 不自动猜 `part_family`（领域判断，必须人填）
- ✗ 不下载/获取数据（数据 sourcing 是人工，见 §4.4 决策）
- ✗ 不改 validator / wrapper / eval（职责分离）

## 6. 测试

`tests/unit/test_build_brep_golden_manifest_skeleton.py`，OCC-free，7 个用例：
source_type 推断、未知桶 flag、TODO 占位、重名去重、空目录、summary 计数、
**skeleton→填→真实 validator release_ready 往返**（证明产出的是 schema-correct
外壳而非看似合理的 JSON）。

## 7. 时间盒

| 项 | 估时 | 实际 |
| --- | --- | --- |
| 脚本 + 测试 | 1h | ✅ 完成 |
| str/Path robustness 修 | 0.1h | ✅（自测发现） |
| DEV + VERIFICATION MD | 0.3h | ✅ |
| 合入（cycle 7，与 validator public_nc 同 PR #479） | 用户授权 | 待 |
