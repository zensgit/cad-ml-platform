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

## 5. 范围

仅一个纯函数 + 测试 + 本对 MD。validator / wrapper / eval /
example manifest **未触碰**。无生产路径行为依赖此函数的代码改动
（scaffolder 是离线脚手架，非运行时）。

## 6. 验证

见 companion VERIFICATION MD §1–5。本地 py_compile + 单元级
`_infer_source_type` 真值断言 + skeleton 行为自测 + 现有测试重放 +
validator example 自洽，全绿。CI（3.11）权威。
