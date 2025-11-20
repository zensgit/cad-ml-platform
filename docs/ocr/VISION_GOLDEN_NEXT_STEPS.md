# Vision Golden - 观察期指南

**当前状态**: ✅ Stage B.1 完成，进入观察期
**观察期时长**: 1-2 天（弹性）
**开始日期**: 2025-01-16

---

## 📋 每日操作清单

### 当你改 Vision/OCR 代码时

```bash
# 1. 改代码
vim src/core/vision/xxx.py

# 2. 跑测试
pytest tests/vision -v

# 3. 跑评估（新习惯！）
make eval-vision-golden

# 4. 记录观察（1分钟）
vim docs/ocr/VISION_GOLDEN_OBSERVATIONS.md
# 填写当天的 Day N 部分
```

### 当你没改 Vision/OCR 代码时

**不需要做任何事** - 观察期是"需要时才用"，不用强制每天跑。

---

## 📝 如何记录观察

### 最简单方式（30秒）

打开 `docs/ocr/VISION_GOLDEN_OBSERVATIONS.md`，找到今天的日期：

```markdown
### 2025-01-17 - Day 1

**行动**:
- ✅ 改了 VisionManager 错误处理
- ✅ 运行了 make eval-vision-golden
- ✅ 结果: 66.7%，正常

**观察**:
- 使用体验: 命令顺手
- 调试定位: MIN/MAX 信息有用
- 需求想法: 无

**需求信号**: 无变化
```

### 重点记录三类信号

1. **使用体验类**
   - 命令是否顺手？
   - 输出是否合适？
   - 有什么不方便？

2. **调试定位类**
   - 能否快速找到问题样本？
   - 缺少什么信息？

3. **需求想法类**
   - 想要什么功能？
   - 想加什么样本？
   - 想要什么改进？

---

## 🎯 观察期结束条件

**满足任一条件即可结束**:

### 1. 明确需求出现（2+ 次）
```markdown
例如：
- "想要 JSON 输出" 出现 3 次 → 说明确实需要
- "手动对比 baseline 麻烦" 出现 2 次 → 需要 --save-report
```

### 2. 明确痛点出现
```markdown
例如：
- 样本太少，无法验证复杂场景
- 统计信息不够，需要更细粒度
```

### 3. 使用足够充分（3+ 次无问题）
```markdown
- 使用 3 次以上，都觉得顺手
- 没有明显痛点
→ 说明当前系统足够好
```

### 4. 时间到达（1-2 天）
```markdown
- 1-2 天后回顾记录
- 根据累积信号决定下一步
```

---

## 📊 观察期结束后如何决策

### Step 1: 回顾观察记录

打开 `docs/ocr/VISION_GOLDEN_OBSERVATIONS.md`，查看：
- 高频痛点（出现 2+ 次）
- 高价值需求（impact ≥ 4）
- 低成本改进（cost ≤ 2）

### Step 2: 使用决策树

文档里已有决策树：

```
问题 1: 是否频繁需要更详细的数据/分组/可视化？
├─ 是 → Stage B.2（增强指标、分类统计、JSON 输出）
└─ 否 → 问题 2

问题 2: 是否遇到"单看 Vision 不足以判断质量"？
├─ 是 → Stage C（Vision + OCR 联合评估）
└─ 否 → 问题 3

问题 3: 是否急于知道真实模型提升？
├─ 是 → Phase 3（DeepSeek-VL provider）
└─ 否 → 维持现状，偶尔扩样本
```

### Step 3: 需求优先级评估

使用表格评估（在 OBSERVATIONS.md 底部）：

| 需求 | Impact | Cost | Frequency | 优先级 |
|------|--------|------|-----------|--------|
| JSON 输出 | 4 | 2 | 4 | ⭐⭐⭐⭐ |
| 样本分类 | 3 | 3 | 2 | ⭐⭐ |

**优先级规则**:
- ⭐⭐⭐⭐⭐: Impact≥4 且 Cost≤2（立即做）
- ⭐⭐⭐⭐: Impact≥4 且 Cost=3（优先做）
- ⭐⭐⭐: Impact=3 且 Frequency≥3（考虑做）

---

## 🚀 快速参考

### 关键文档

| 文档 | 用途 |
|------|------|
| `VISION_GOLDEN_OBSERVATIONS.md` | 记录日常观察 ✏️ |
| `VISION_GOLDEN_STAGE_B1_COMPLETE.md` | Stage B.1 完成总结 |
| `HOW_TO_ADD_SAMPLE.md` | 扩展样本指南 |
| `reports/vision_golden_baseline.md` | Baseline 报告 |
| `reports/vision_golden_test_report_20250116.md` | 测试报告 |

### 关键命令

| 命令 | 用途 |
|------|------|
| `make eval-vision-golden` | 运行 Vision golden 评估 |
| `make eval-all-golden` | 运行 Vision + OCR 评估 |
| `pytest tests/vision -v` | 运行 Vision 测试 |
| `git log --oneline --decorate` | 查看 git 历史（含 tag） |

### Git 里程碑

| Tag | Commit | Description |
|-----|--------|-------------|
| `vision-golden-b1` | 98fcab4 | Stage B.1 baseline (66.7%) |

---

## ❓ 常见问题

### Q: 我今天没改 Vision 代码，需要跑评估吗？
**A**: 不需要！观察期是"需要时才用"，不用强制每天跑。

### Q: 我忘记记录了怎么办？
**A**: 没关系，下次记得记就行。观察期是帮你收集信号，不是强制作业。

### Q: 我想立即扩展样本，可以吗？
**A**: 可以！参考 `tests/vision/golden/HOW_TO_ADD_SAMPLE.md`。但建议先观察 1-2 天，确认真的需要。

### Q: 观察期可以提前结束吗？
**A**: 可以！如果你已经有明确需求（如想要 JSON 输出），可以立即开始实施，不用等满 1-2 天。

### Q: 观察期可以延长吗？
**A**: 可以！如果 1-2 天后还没有明确信号，可以继续观察。观察期是弹性的。

---

## 🎯 今天（2025-01-16）已完成

- ✅ Stage B.1 所有任务完成
- ✅ Baseline 建立（66.7%）
- ✅ 完整测试通过（112/112）
- ✅ Git 里程碑创建（vision-golden-b1）
- ✅ 测试报告生成
- ✅ 第一条观察记录完成
- ✅ 观察期正式开始

---

## 🔜 明天（2025-01-17）

**如果你改 Vision/OCR 代码**:
1. 跑测试
2. 跑评估（`make eval-vision-golden`）
3. 记录观察（1 分钟）

**如果你不改相关代码**:
- 什么都不用做，继续其他工作即可

---

**Last Updated**: 2025-01-16
**Status**: 观察期进行中（Day 0 完成）
**Next Review**: 2025-01-17 或 2025-01-18
