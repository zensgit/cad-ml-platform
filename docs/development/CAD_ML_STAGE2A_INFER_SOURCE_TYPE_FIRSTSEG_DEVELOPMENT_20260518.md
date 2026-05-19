# Stage 2a — `_infer_source_type` First-Segment Alignment

Date: 2026-05-18
Type: 文档与实现对齐 + 防静默错分修正
Companion: `CAD_ML_STAGE2A_INFER_SOURCE_TYPE_FIRSTSEG_VERIFICATION_20260518.md`
Touches: `scripts/build_brep_golden_manifest_skeleton.py`, `tests/unit/test_build_brep_golden_manifest_skeleton.py`

## 0. 一句话

`_infer_source_type` 之前**遍历所有路径段**找已知桶，与 DESIGN §2.2
"source_type inferred from **first** path segment" 的契约不一致；这个
不一致是**静默错分**风险。改为只看 `rel_parts[0]`。

## 1. 缺陷

原实现：

```python
for part in rel_parts:                       # 遍历所有段
    bucket = KNOWN_SOURCE_BUCKETS.get(part.strip().lower())
    if bucket:
        return bucket, True                  # clean=True（不打 flag）
return "internal", False
```

`data/brep_golden/mystery/public_nc/x.step` →
`rel_parts = ("mystery", "public_nc")` → 遍历到第二段命中 `public_nc`
→ 返回 `("public_nc", True)`。

**后果（双向静默错分）**：

- `mystery/public_nc/x.step` 被判 `public_nc` 且 `clean=True` → **不打**
  `TODO-source-type` tag → 静默排除出 release floor（本该是需人确认的
  未知结构）。
- 对称风险：`mystery/internal/x.step` 被判 `internal` 且 `clean=True`
  → 静默**计入** release floor。若日后有人把 `mystery/` 重命名为
  `public_nc/`，release-eligibility 会无声翻转，没有任何 tag 提示复核。

设计文档 §2.2 写的是"first path segment"，实现却是"any segment"——
**文档与实现不一致，且不一致的那一侧有 release-gate 安全含义**。

## 2. 修正

```python
if rel_parts:
    bucket = KNOWN_SOURCE_BUCKETS.get(rel_parts[0].strip().lower())
    if bucket:
        return bucket, True
return "internal", False
```

只有**首段**是权威。深层出现的桶名一律忽略 →
`mystery/...` 一律 `("internal", False)` → 调用方打 `TODO-source-type`
→ 必须人确认才能定调，无法静默错分（任一方向）。

## 3. 行为对照

| 路径 | 旧 | 新 |
| --- | --- | --- |
| `internal/p.step` | internal, clean | internal, clean（不变） |
| `public_nc/abc.step` | public_nc, clean | public_nc, clean（不变） |
| `vendor/acme/shaft.stp` | vendor, clean（首段命中） | vendor, clean（不变） |
| `mystery/public_nc/x.step` | **public_nc, clean** ⚠️ | **internal, FLAGGED** ✅ |
| `mystery/internal/y.step` | **internal, clean** ⚠️ | **internal, FLAGGED** ✅ |
| `loose.step`（root 级，无子目录） | internal, flagged | internal, flagged（不变） |

仅"深层桶名"两行行为改变，且都是从"静默"变为"打 flag 要求复核"——
更安全，无功能退化。

## 4. 测试

新增 2 个用例（`tests/unit/test_build_brep_golden_manifest_skeleton.py`）：

- `test_known_bucket_only_recognised_in_first_segment`：
  `mystery/public_nc/x.step` 与 `mystery/internal/y.step` 均须
  `source_type=internal` **且** 带 `TODO-source-type`；显式断言
  `release_eligible` 真值，强调 tag 是防错分的唯一屏障。
- `test_root_level_file_defaults_internal_and_flagged`：root 级文件
  （空 `rel_parts`）须 default internal + flagged。

现有用例无回归（已 OCC-free 重放断言验证）：
`vendor/acme/shaft.stp` 仍 `vendor`（首段即桶）；summary 计数不变；
skeleton→fill→validator release_ready 往返不变；validator example
自洽（`insufficient_release_samples`, `errors=[]`）。

## 4b. Blocking finding 修正（review 复现后追加）

**Reviewer 复现**：PR head 上 50 个 `mystery/public_nc/*.step` → 生成
50 个带 `TODO-source-type` 的 case，validator 仍 `release_ready,
release_eligible_count=50, errors=[]`。`TODO-source-type` tag 是**建议性**
的——validator 只读 `release_eligible`，不看 tags——与测试注释"tag
forces human confirm before it counts"矛盾。**该 finding 阻塞合并。**

**两层修（纵深防御）**：

1. **scaffolder 默认收紧**（`build_brep_golden_manifest_skeleton.py:~115`）：
   `release_eligible = clean and source_type not in RELEASE_EXCLUDED_SOURCE_TYPES`。
   un-clean 推断（unknown/deep/root-level）→ `release_eligible=False`，
   不再凭猜进 release floor。tag 成为硬 gate。

2. **validator 独立拒绝**（`validate_brep_golden_manifest.py:~201`）：
   `release_eligible` 的 case 若仍带 `TODO-*` tag **或** `part_family`/
   `license` 字段值为 `TODO` → `errors` → manifest `invalid`。与
   scaffolder 解耦，scaffolder 回归或手写 manifest 都无法绕过。

**人工 signoff 契约**（修正后明确）：补字段值 **且** 清 `TODO-*` tag，
二者缺一不可——validator 强制。`test_skeleton_requires_full_human_signoff_to_pass_validator`
钉死此契约（仅填字段 → invalid；再清 tag → release_ready）。

**行为对照（修正后）**：

| 路径 | source_type | release_eligible | 备注 |
| --- | --- | --- | --- |
| `internal/p.step` | internal, clean | True | 不变 |
| `public_nc/abc.step` | public_nc, clean | False | release-excluded |
| `vendor/acme/shaft.stp` | vendor, clean | True | 首段命中 |
| `mystery/public_nc/x.step` | internal, FLAGGED | **False** | 修正：原 True（blocking） |
| `mystery/internal/y.step` | internal, FLAGGED | **False** | 修正：原 True |
| `loose.step`（root 级） | internal, FLAGGED | **False** | 修正：原 True |

**回归 guard**：`test_scaffolder_blocking_finding_regression_50_mystery_public_nc`
钉死 reviewer 的精确复现（50 mystery/public_nc → 0 eligible，非
release_ready）。validator 侧 +3 测试（TODO-tag 拒绝 / TODO-field 拒绝 /
非 release case 可保留 TODO）。

## 5. 范围

`_infer_source_type`（纯函数）+ scaffolder `release_eligible` 默认 +
validator TODO 硬 gate + 测试 + 本对 MD。example manifest 无 TODO
字段/tag，不受新 gate 影响（已验证自洽）。所有改动在离线脚手架 /
校验器，非运行时生产路径。

## 6. 验证

见 companion VERIFICATION MD §1–5。本地 py_compile + 单元级
`_infer_source_type` 真值断言 + skeleton 行为自测 + 现有测试重放 +
validator example 自洽，全绿。CI（3.11）权威。
