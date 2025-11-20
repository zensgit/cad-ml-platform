# Observation Period Log

## Active Milestones

| Module | Tag | Baseline Date | Status |
|--------|-----|---------------|--------|
| Vision Golden B.1 | `vision-golden-b1` | 2025-11-16 | Observing |
| OCR Week 1 MVP | `ocr-week1-mvp` | 2025-11-16 | Observing |

---

## Daily Eval Notes

### Format
```
[Date] [Module]
- Eval result: [metric changes if any]
- Pain points: [what was annoying]
- Ideas: [what would be nice to have]
```

### 2025-11-16 (Day 0)
**Vision**:
- Baseline: AVG_HIT_RATE 66.7% (Easy 100%, Medium 60%, Hard 40%)
- Enhancement: Added git info (timestamp/branch/commit/tag) to eval output
- Git: main @ 031a204

**OCR**:
- Baseline: Recall 100%, Brier 0.025, Edge-F1 0.000
- Enhancement: Added git info + baseline diff comparison to eval output
- Diff indicator: [SAME] / [IMPROVED] / [REGRESSED] for each metric
- Git: main @ 031a204

### 2025-11-16 (Day 0 - Run 2)
**Vision** (29 tests, 1.38s):
- AVG_HIT_RATE: 66.7% (unchanged)
- Easy: 100%, Medium: 60%, Hard: 40%
- Git: main @ 031a204
- Pain point: No baseline diff comparison like OCR has

**OCR** (94 tests, 0.66s):
- dimension_recall: 1.000 [SAME]
- brier_score: 0.025 [SAME]
- edge_f1: 0.000 [SAME]
- Git: main @ 031a204
- All metrics stable vs baseline

**Observations**:
- Both eval scripts now show git info
- OCR baseline diff is very helpful for quick scan
- Vision could benefit from same pattern

### 2025-11-16 (Day 0 - Run 3)
**Vision**:
- Enhancement: Added baseline diff comparison
- AVG_HIT_RATE: 66.7% [SAME]
- MIN_HIT_RATE: 40.0% [SAME]
- MAX_HIT_RATE: 100.0% [SAME]
- Git: main @ 031a204
- Now consistent with OCR eval output format

**Pain point resolved**: Vision eval now has same baseline diff feature as OCR

### 2025-11-16 (Day 0 - Run 4) - Full Execution
**Vision** (29 tests, 0.48s):
- All tests passed
- AVG_HIT_RATE: 66.7% [SAME]
- MIN_HIT_RATE: 40.0% [SAME]
- MAX_HIT_RATE: 100.0% [SAME]
- Git: main @ 031a204

**OCR** (94 tests, 0.46s):
- All tests passed
- dimension_recall: 1.000 [SAME]
- brier_score: 0.025 [SAME]
- edge_f1: 0.000 [SAME]
- Git: main @ 031a204

**Summary**: All 123 tests passing, all metrics stable vs baseline. No regression detected.

### 2025-11-17 (Day 1) - 可观测性 Sprint 完成
**新增功能**:
- `make eval-history`: Eval 结果自动保存到 JSON
- `make health-check`: 一键输出系统健康状态
- `reports/eval_history/`: 历史记录目录

**验证结果**:
```
$ make eval-history
→ Saved: reports/eval_history/20251117_205534_main_031a204.json

$ make health-check
→ Git: main @ 031a204 (ocr-week1-mvp)
→ Latest: dimension_recall=1.0, brier_score=0.025, edge_f1=0.0
```

**修复**: `python` → `python3` (macOS 兼容)

### 2025-11-17 (Day 1) - 测试地图文档
**新增文档**: `docs/TEST_MAP.md`
- Vision 模块: 4 测试文件，29 个测试，详细覆盖说明
- OCR 模块: 15 测试文件，94 个测试，完整场景映射
- 依赖图、运行指南、覆盖率分析
- 维护指南（添加新测试时如何更新）

**痛点解决**: 现在可以快速了解每个测试文件测什么功能

---

## Wants But Not Done (Accumulate 3-5 → Next Sprint)

_Record during observation: "I wish I had X" / "I keep manually doing Y"_

1. [x] ~~Vision eval: Add baseline diff comparison (like OCR has)~~ **DONE**
2. [x] ~~**Eval 历史记录**~~ **DONE** - `make eval-history` 自动保存 JSON
3. [x] ~~**快速健康检查命令**~~ **DONE** - `make health-check` 一键汇总
4. [x] ~~**测试地图文档**~~ **DONE** - `docs/TEST_MAP.md` 完整覆盖说明
5. [ ] **真实测试样本**: samples 目录为空，全是 synthetic 1x1 PNG，无法测真实图纸

---

## Baseline Drift Check

**Warning signs** (if any appear, pause new features):
- [ ] Golden eval results diverging from baseline doc description
- [ ] Frequently writing ad-hoc scripts/shell pipelines
- [ ] Can't remember what certain tests verify
- [ ] Test count rising but clarity falling

**Current status**: Clean (Day 0)

---

## Next Sprint Candidates (Pick when observation complete)

### Direction A: Vision+OCR Combined Evaluation Mini
- Target: `scripts/evaluate_vision_ocr_combined.py`
- Prereq: Define unified sample ID + metadata schema
- Output: Combined score for before/after comparison
- **Priority signals**: Need to compare model/prompt changes across both modules

### Direction B: PDF Async Paging MVP
- Target: `src/core/ocr/pdf_loader.py` + 2-3 tests
- Output: Split multi-page PDF → page images → OCR pipeline
- **Priority signals**: Real-world PDF inputs becoming frequent

### Direction C: Assembly Evidence Skeleton (Day 1-2 only)
- Target: Evidence/Relation models + EvidenceCollector + unit tests
- Output: Stable anchor point for future geometry/rules
- **Priority signals**: Starting to think about reasoning chains

---

## Observation Period Rules

### What TO DO:
- Run `pytest` + `make eval-*-golden` after any module changes
- Add observation notes if metrics change noticeably
- Record "wants but not done" ideas as they surface
- Small additions: logs, tests, thin CLI scripts (more observable)

### What to AVOID:
- Large-scale refactoring during observation
- Deep architectural changes
- Batch renames without clear regression testing
- Adding features before baseline stabilizes

### When to STOP:
- Baseline drift detected
- Frequent manual workarounds accumulating
- Test clarity degrading
- Need to update docs/reports to match reality

---

## Rhythm

1. **Sprint (3-5 days)**: Focused development on one theme
2. **Milestone**: Report + baseline + git tag
3. **Observation (1-2+ days)**: Monitor, collect pain points, light tooling
4. **Decide**: Pick next sprint from accumulated insights

Current Phase: **Observation** (Vision + OCR dual)
