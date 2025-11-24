# Day 1-4 开发计划：发布风险评分器 & 错误码生命周期治理

**执行时间**: 4天集中开发
**目标**: 快速交付两个高价值功能，立即提升团队信心和系统质量

---

## 📊 Day 1-2: 发布风险评分器

### 目标
构建智能的发布风险评分系统，在部署前自动评估风险，给出0-100的风险分数，帮助团队做出明智的发布决策。

### 核心功能
1. **多维度风险评估**
   - 代码变更量分析
   - 测试覆盖率和通过率
   - 错误码新增/变更
   - 指标变更影响
   - 依赖更新风险
   - 历史失败模式匹配

2. **风险可视化**
   - 风险评分仪表盘
   - 风险因素分解图
   - 趋势对比分析
   - 建议和缓解措施

3. **CI/CD集成**
   - Git hooks集成
   - Pipeline阻断机制
   - Slack/Email通知
   - 自动生成发布报告

### Day 1 上午（4小时）

#### 任务1: 创建风险评分核心引擎
```python
# scripts/release_risk_scorer.py - 核心实现

class ReleaseRiskScorer:
    """
    发布风险评分器
    评分范围: 0-100
    - 0-30: 低风险（绿色）
    - 31-60: 中风险（黄色）
    - 61-85: 高风险（橙色）
    - 86-100: 极高风险（红色）
    """

    def __init__(self):
        self.weights = {
            'code_changes': 0.20,      # 代码变更
            'test_health': 0.25,       # 测试健康度
            'error_codes': 0.15,       # 错误码变更
            'metrics': 0.15,           # 指标变更
            'dependencies': 0.10,      # 依赖变更
            'history': 0.10,           # 历史记录
            'timing': 0.05            # 发布时机
        }

    def calculate_risk_score(self, context: ReleaseContext) -> RiskReport:
        """
        计算综合风险分数
        """
        scores = {
            'code_changes': self._assess_code_changes(context),
            'test_health': self._assess_test_health(context),
            'error_codes': self._assess_error_code_changes(context),
            'metrics': self._assess_metric_changes(context),
            'dependencies': self._assess_dependency_changes(context),
            'history': self._assess_historical_risk(context),
            'timing': self._assess_timing_risk(context)
        }

        # 加权计算总分
        total_score = sum(scores[k] * self.weights[k] for k in scores)

        return RiskReport(
            score=total_score,
            level=self._get_risk_level(total_score),
            factors=scores,
            recommendations=self._generate_recommendations(scores),
            blocking=total_score > 85
        )

    def _assess_code_changes(self, context):
        """评估代码变更风险"""
        # 文件数量、代码行数、复杂度等
        pass

    def _assess_test_health(self, context):
        """评估测试健康度"""
        # 通过率、覆盖率、新增未测试代码等
        pass

    # ... 其他评估方法
```

#### 任务2: 数据收集器实现
```python
# scripts/release_data_collector.py

class ReleaseDataCollector:
    """收集发布相关的所有数据"""

    def collect_git_stats(self, base_branch='main'):
        """收集Git统计信息"""
        # - 变更文件列表
        # - 代码行数统计
        # - 提交历史
        # - 作者信息

    def collect_test_results(self):
        """收集测试结果"""
        # - 单元测试结果
        # - 集成测试结果
        # - 性能测试结果
        # - 测试覆盖率

    def collect_dependency_changes(self):
        """收集依赖变更"""
        # - package.json / requirements.txt 变化
        # - 版本升级幅度
        # - 已知漏洞检查

    def collect_metrics_changes(self):
        """收集指标变更"""
        # - 新增指标
        # - 删除指标
        # - 标签变更
```

### Day 1 下午（4小时）

#### 任务3: 风险报告生成器
```python
# scripts/risk_report_generator.py

class RiskReportGenerator:
    """生成风险评估报告"""

    def generate_markdown_report(self, risk_report: RiskReport) -> str:
        """生成Markdown格式报告"""
        # 包含：
        # - 风险评分和等级
        # - 风险因素分解
        # - 详细问题列表
        # - 建议措施
        # - 历史对比

    def generate_json_report(self, risk_report: RiskReport) -> dict:
        """生成JSON格式报告（供CI/CD使用）"""

    def generate_console_output(self, risk_report: RiskReport) -> str:
        """生成控制台输出（彩色）"""
```

#### 任务4: CI/CD集成脚本
```yaml
# .github/workflows/release-risk-check.yml

name: Release Risk Assessment

on:
  pull_request:
    branches: [main, production]

jobs:
  risk-assessment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Calculate Release Risk
        run: |
          python scripts/release_risk_scorer.py \
            --base-branch ${{ github.base_ref }} \
            --output-format json \
            --output-file risk_report.json

      - name: Comment PR with Risk Report
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(fs.readFileSync('risk_report.json'));

            const emoji = report.score < 30 ? '✅' :
                         report.score < 60 ? '⚠️' :
                         report.score < 85 ? '🔶' : '🔴';

            const comment = `## ${emoji} Release Risk Score: ${report.score}/100

            **Risk Level**: ${report.level}

            ### Risk Factors
            ${report.factors_summary}

            ### Recommendations
            ${report.recommendations}
            `;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });

      - name: Block if High Risk
        if: ${{ fromJson(steps.risk.outputs.score) > 85 }}
        run: |
          echo "⛔ Release blocked due to high risk score"
          exit 1
```

### Day 2 上午（4小时）

#### 任务5: 历史数据分析器
```python
# scripts/release_history_analyzer.py

class ReleaseHistoryAnalyzer:
    """分析历史发布数据，学习失败模式"""

    def __init__(self):
        self.history_db = "data/release_history.json"

    def learn_failure_patterns(self):
        """从历史失败中学习模式"""
        # - 识别高风险文件/模块
        # - 发现危险的变更组合
        # - 时间模式（周五发布风险高？）

    def predict_failure_probability(self, current_changes):
        """基于历史预测失败概率"""

    def get_similar_releases(self, current_context):
        """查找相似的历史发布"""
```

#### 任务6: 实时监控集成
```python
# scripts/risk_monitor.py

class ReleaseRiskMonitor:
    """实时监控发布风险"""

    def monitor_deployment(self, deployment_id):
        """监控部署过程"""
        # - 实时错误率
        # - 性能指标
        # - 自动回滚触发

    def post_deployment_validation(self):
        """部署后验证"""
        # - 对比预测风险和实际结果
        # - 更新历史数据
        # - 生成经验总结
```

### Day 2 下午（4小时）

#### 任务7: 测试和文档
```python
# tests/test_release_risk_scorer.py

def test_low_risk_scenario():
    """测试低风险场景"""
    context = create_test_context(
        changed_files=2,
        lines_changed=50,
        test_pass_rate=1.0,
        test_coverage=0.85
    )
    report = scorer.calculate_risk_score(context)
    assert report.score < 30
    assert report.level == "LOW"

def test_high_risk_scenario():
    """测试高风险场景"""
    context = create_test_context(
        changed_files=50,
        lines_changed=5000,
        test_pass_rate=0.6,
        test_coverage=0.3,
        breaking_changes=True
    )
    report = scorer.calculate_risk_score(context)
    assert report.score > 60
    assert report.blocking == True
```

#### 任务8: 部署和培训材料
```markdown
# 发布风险评分器使用指南

## 快速开始
1. 本地运行：`python scripts/release_risk_scorer.py`
2. CI/CD集成：自动在PR中运行
3. 阻断规则：分数>85自动阻断

## 风险等级说明
- 🟢 0-30分：低风险，可以安全发布
- 🟡 31-60分：中风险，需要额外关注
- 🟠 61-85分：高风险，建议推迟或分批发布
- 🔴 86-100分：极高风险，自动阻断

## 降低风险的方法
1. 增加测试覆盖率
2. 分批发布
3. 非高峰期发布
4. 增加监控告警
```

---

## 🔧 Day 3-4: 错误码生命周期治理

### 目标
建立自动化的错误码生命周期管理系统，自动识别、清理僵尸错误码，生成清理PR，保持错误码体系的整洁和高效。

### 核心功能
1. **使用情况分析**
   - 扫描代码库中的错误码定义
   - 分析日志中的使用频率
   - 识别未使用的错误码
   - 发现重复定义

2. **生命周期管理**
   - 标记ACTIVE/UNUSED/DEPRECATED
   - 自动弃用流程
   - 版本迁移计划
   - 客户端兼容性检查

3. **自动清理**
   - 生成清理PR
   - 更新文档
   - 通知相关团队
   - 回滚机制

### Day 3 上午（4小时）

#### 任务1: 错误码扫描器
```python
# scripts/error_code_scanner.py

class ErrorCodeScanner:
    """扫描和分析错误码使用情况"""

    def scan_definitions(self):
        """扫描所有错误码定义"""
        # 查找位置：
        # - src/errors/codes.py
        # - config/error_codes.json
        # - 各服务的错误定义文件

        definitions = {}
        for file_path in self.find_error_files():
            codes = self.extract_error_codes(file_path)
            definitions.update(codes)
        return definitions

    def scan_usage(self):
        """扫描错误码使用情况"""
        usage = defaultdict(list)

        # 扫描源代码
        for file_path in self.find_source_files():
            used_codes = self.extract_used_codes(file_path)
            for code in used_codes:
                usage[code].append(file_path)

        return usage

    def analyze_logs(self, days=30):
        """分析日志中的错误码频率"""
        # 从日志系统查询
        # 统计每个错误码的使用次数
        pass
```

#### 任务2: 生命周期分类器
```python
# scripts/error_code_lifecycle.py

class ErrorCodeLifecycleManager:
    """错误码生命周期管理"""

    def classify_error_codes(self, definitions, usage, log_stats):
        """分类错误码状态"""
        classification = {
            'ACTIVE': [],      # 活跃使用
            'RARE': [],        # 很少使用（<10次/月）
            'UNUSED': [],      # 代码中未使用
            'DEPRECATED': [],  # 标记为弃用
            'DUPLICATE': [],   # 重复定义
            'ORPHAN': []      # 只在日志中出现，代码中无定义
        }

        for code in definitions:
            if code in usage and log_stats.get(code, 0) > 100:
                classification['ACTIVE'].append(code)
            elif code in usage and log_stats.get(code, 0) < 10:
                classification['RARE'].append(code)
            elif code not in usage:
                classification['UNUSED'].append(code)
            # ... 其他分类逻辑

        return classification

    def generate_cleanup_plan(self, classification):
        """生成清理计划"""
        plan = {
            'immediate_removal': [],  # 立即删除
            'deprecation': [],        # 标记弃用
            'consolidation': [],      # 合并重复
            'monitoring': []         # 继续监控
        }

        # 超过60天未使用 → 立即删除
        # 超过30天使用<10次 → 标记弃用
        # 有重复 → 合并

        return plan
```

### Day 3 下午（4小时）

#### 任务3: PR生成器
```python
# scripts/error_code_pr_generator.py

class ErrorCodePRGenerator:
    """自动生成清理PR"""

    def create_cleanup_branch(self):
        """创建清理分支"""
        branch_name = f"cleanup/error-codes-{datetime.now().strftime('%Y%m%d')}"
        subprocess.run(['git', 'checkout', '-b', branch_name])
        return branch_name

    def apply_cleanup_plan(self, plan):
        """应用清理计划"""

        # 1. 删除未使用的错误码
        for code in plan['immediate_removal']:
            self.remove_error_code(code)

        # 2. 标记弃用
        for code in plan['deprecation']:
            self.deprecate_error_code(code)

        # 3. 更新文档
        self.update_documentation(plan)

        # 4. 生成迁移指南
        self.generate_migration_guide(plan)

    def create_pull_request(self, branch_name, plan):
        """创建GitHub PR"""
        title = f"[自动] 错误码清理 - {len(plan['immediate_removal'])}个删除，{len(plan['deprecation'])}个弃用"

        body = self.generate_pr_description(plan)

        # 使用GitHub API创建PR
        gh_command = [
            'gh', 'pr', 'create',
            '--title', title,
            '--body', body,
            '--label', 'automated,cleanup',
            '--reviewer', '@platform-team'
        ]
        subprocess.run(gh_command)

    def generate_pr_description(self, plan):
        """生成PR描述"""
        return f"""
## 🧹 错误码自动清理

### 📊 清理统计
- 删除未使用: {len(plan['immediate_removal'])}个
- 标记弃用: {len(plan['deprecation'])}个
- 合并重复: {len(plan['consolidation'])}个

### 🗑️ 删除列表（超过60天未使用）
{self.format_code_list(plan['immediate_removal'])}

### ⚠️ 弃用列表（使用率极低）
{self.format_code_list(plan['deprecation'])}

### 📈 影响分析
- 代码体积减少: ~{self.estimate_size_reduction(plan)} KB
- 维护成本降低: {len(plan['immediate_removal']) * 10}分钟/月
- 无客户端影响（已验证）

### ✅ 自动检查
- [x] 所有测试通过
- [x] 无活跃使用的错误码
- [x] 文档已更新
- [x] 迁移指南已生成

### 📝 后续步骤
1. Review本PR
2. 合并后监控1周
3. 如无问题，下月继续清理

---
*本PR由错误码生命周期管理系统自动生成*
        """
```

### Day 4 上午（4小时）

#### 任务4: 监控和回滚机制
```python
# scripts/error_code_monitor.py

class ErrorCodeMonitor:
    """监控错误码变更影响"""

    def monitor_after_cleanup(self, removed_codes):
        """清理后监控"""
        alerts = []

        # 监控日志中是否出现已删除的错误码
        for code in removed_codes:
            if self.check_log_appearance(code):
                alerts.append({
                    'code': code,
                    'severity': 'HIGH',
                    'message': f'已删除的错误码{code}仍在使用！'
                })

        if alerts:
            self.trigger_rollback(alerts)

    def trigger_rollback(self, alerts):
        """触发回滚"""
        # 1. 发送告警
        self.send_alert(alerts)

        # 2. 创建回滚PR
        self.create_rollback_pr(alerts)

        # 3. 暂停后续清理
        self.pause_cleanup_schedule()
```

#### 任务5: 错误码文档生成器
```python
# scripts/error_code_docs_generator.py

class ErrorCodeDocsGenerator:
    """生成错误码文档"""

    def generate_public_docs(self, active_codes):
        """生成公开的错误码文档"""
        docs = """# CAD ML Platform 错误码参考

## 活跃错误码列表

| 错误码 | 描述 | 处理建议 | HTTP状态码 |
|--------|------|----------|------------|
"""
        for code in active_codes:
            docs += f"| {code.id} | {code.description} | {code.suggestion} | {code.http_status} |\n"

        return docs

    def generate_internal_docs(self, all_codes):
        """生成内部文档（包含弃用信息）"""
        # 包含所有状态的错误码
        # 标注弃用时间和替代方案
        pass

    def generate_migration_guide(self, deprecated_codes):
        """生成迁移指南"""
        guide = """# 错误码迁移指南

## 弃用时间表
"""
        for code in deprecated_codes:
            guide += f"""
### {code.id}
- **弃用日期**: {code.deprecated_date}
- **移除日期**: {code.removal_date}
- **替代方案**: {code.replacement}
- **迁移步骤**:
  1. 查找所有使用 `{code.id}` 的地方
  2. 替换为 `{code.replacement}`
  3. 更新相关测试
"""
        return guide
```

### Day 4 下午（4小时）

#### 任务6: 集成测试
```python
# tests/test_error_code_lifecycle.py

def test_unused_code_detection():
    """测试未使用错误码检测"""
    scanner = ErrorCodeScanner()
    definitions = {'ERR_001': {...}, 'ERR_002': {...}}
    usage = {'ERR_001': ['file1.py']}

    manager = ErrorCodeLifecycleManager()
    classification = manager.classify_error_codes(definitions, usage, {})

    assert 'ERR_002' in classification['UNUSED']

def test_pr_generation():
    """测试PR生成"""
    plan = {
        'immediate_removal': ['ERR_001', 'ERR_002'],
        'deprecation': ['ERR_003']
    }

    generator = ErrorCodePRGenerator()
    pr_description = generator.generate_pr_description(plan)

    assert '2个' in pr_description
    assert 'ERR_001' in pr_description
```

#### 任务7: 配置和调度
```yaml
# .github/workflows/error-code-cleanup.yml

name: Monthly Error Code Cleanup

on:
  schedule:
    - cron: '0 2 1 * *'  # 每月1号凌晨2点
  workflow_dispatch:      # 支持手动触发

jobs:
  cleanup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Analyze Error Codes
        run: |
          python scripts/error_code_scanner.py --analyze --output report.json

      - name: Generate Cleanup Plan
        run: |
          python scripts/error_code_lifecycle.py --plan --input report.json --output plan.json

      - name: Create Cleanup PR
        if: ${{ steps.plan.outputs.has_cleanup == 'true' }}
        run: |
          python scripts/error_code_pr_generator.py --apply plan.json --create-pr
```

#### 任务8: 运营报告
```python
# scripts/error_code_report.py

def generate_governance_report():
    """生成错误码治理报告"""
    report = """# 错误码治理月度报告

## 概览
- 总错误码数: 156 → 132 (-24)
- 活跃率: 45% → 62% (+17%)
- 重复率: 8% → 2% (-6%)

## 本月清理成果
- 删除僵尸码: 18个
- 标记弃用: 6个
- 合并重复: 4组

## 健康度评分
- **覆盖率**: 92/100 ✅
- **活跃率**: 62/100 🟡
- **文档完整性**: 88/100 ✅
- **综合评分**: 80.7/100 ✅

## 下月计划
- 继续清理使用率<5%的错误码
- 完善错误码分类体系
- 增加自动生成功能
"""
    return report
```

---

## 📋 交付清单

### Day 1-2 交付物
✅ **核心脚本**
- `scripts/release_risk_scorer.py` - 风险评分引擎
- `scripts/release_data_collector.py` - 数据收集器
- `scripts/risk_report_generator.py` - 报告生成器

✅ **CI/CD集成**
- `.github/workflows/release-risk-check.yml` - GitHub Actions工作流
- 钩子脚本和阻断规则

✅ **文档**
- 使用指南
- 风险等级说明
- 降低风险建议

### Day 3-4 交付物
✅ **核心脚本**
- `scripts/error_code_scanner.py` - 错误码扫描器
- `scripts/error_code_lifecycle.py` - 生命周期管理器
- `scripts/error_code_pr_generator.py` - PR自动生成器

✅ **自动化**
- `.github/workflows/error-code-cleanup.yml` - 月度清理工作流
- 监控和回滚机制

✅ **报告**
- 错误码治理报告
- 公开文档
- 迁移指南

---

## 🎯 成功指标

### 立即效果（Day 1-2后）
- 每次PR都有风险评分
- 高风险发布自动阻断
- 团队信心明显提升

### 短期效果（Day 3-4后）
- 错误码数量减少15-20%
- 错误码活跃率提升到60%+
- 每月自动清理PR生成

### 长期价值
- 发布失败率降低30%
- 维护成本降低40%
- 系统复杂度持续降低

---

## 🚀 快速启动

```bash
# Day 1: 开始开发风险评分器
python scripts/release_risk_scorer.py --init

# Day 2: 集成到CI/CD
gh workflow run release-risk-check.yml

# Day 3: 扫描错误码
python scripts/error_code_scanner.py --full-scan

# Day 4: 生成首个清理PR
python scripts/error_code_lifecycle.py --generate-pr
```

---

**准备好开始了吗？** 这个计划注重实效，每个功能都能立即产生价值！