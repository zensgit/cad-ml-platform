# CAD ML Platform - Makefile
# 统一的开发工作流

.PHONY: help install dev test lint format type-check clean run docs docker eval-history health-check eval-trend \
	observability-up observability-down observability-status self-check metrics-validate prom-validate \
	dashboard-import security-audit metrics-audit cardinality-check

# 默认目标
.DEFAULT_GOAL := help

# 变量定义
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
MYPY := $(PYTHON) -m mypy
FLAKE8 := $(PYTHON) -m flake8

# 项目路径
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
ASSEMBLY_MODULE := src/core/assembly

# 颜色输出
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## 显示帮助信息
	@echo "$(GREEN)CAD ML Platform - 开发命令$(NC)"
	@echo "----------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## 安装依赖
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

dev: ## 设置开发环境
	@echo "$(GREEN)Setting up development environment...$(NC)"
	$(PYTHON) -m venv venv
	. venv/bin/activate && $(PIP) install --upgrade pip
	. venv/bin/activate && $(MAKE) install
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

test: ## 运行测试
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

test-assembly: ## 运行装配模块测试
	@echo "$(GREEN)Running assembly module tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/assembly -v --cov=$(ASSEMBLY_MODULE)

test-baseline: ## 运行基线评测
	@echo "$(GREEN)Running baseline evaluation...$(NC)"
	$(PYTHON) scripts/run_baseline_evaluation.py

lint: ## 运行代码检查（仅 src/，使用 .flake8 配置）
	@echo "$(GREEN)Running linters (src only)...$(NC)"
	$(FLAKE8) $(SRC_DIR)
	@echo "$(GREEN)Linting passed!$(NC)"

lint-all: ## 运行全仓库代码检查（开发用，可能报较多告警）
	@echo "$(YELLOW)Running linters (full repo)...$(NC)"
	$(FLAKE8)

# 注意：有一个测试文件包含非 UTF-8 内容，Black 无法处理。
# 我们在格式化时排除该文件，避免开发流程中断。
BLACK_EXCLUDES := tests/vision/test_vision_ocr_integration.py

format: ## 格式化代码
	@echo "$(GREEN)Formatting code...$(NC)"
	$(BLACK) $(SRC_DIR) $(TEST_DIR) --line-length=100 --extend-exclude "$(BLACK_EXCLUDES)"
	$(ISORT) $(SRC_DIR) $(TEST_DIR) --profile black --line-length=100
	@echo "$(GREEN)Code formatted!$(NC)"

type-check: ## 类型检查（使用 mypy.ini 配置）
	@echo "$(GREEN)Type checking...$(NC)"
	$(MYPY) $(SRC_DIR)

clean: ## 清理临时文件
	@echo "$(RED)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	@echo "$(GREEN)Cleanup complete!$(NC)"

run: ## 启动服务
	@echo "$(GREEN)Starting CAD ML Platform...$(NC)"
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

run-demo: ## 运行演示
	@echo "$(GREEN)Running assembly demo...$(NC)"
	$(PYTHON) examples/assembly_demo.py

docs: ## 生成文档
	@echo "$(GREEN)Generating documentation...$(NC)"
	$(PYTHON) -m mkdocs build
	@echo "$(GREEN)Documentation built in site/$(NC)"

self-check: ## Run basic self-check
	@echo "$(GREEN)Running self-check...$(NC)"
	$(PYTHON) scripts/self_check.py

self-check-enhanced: ## Run comprehensive self-check
	@echo "$(GREEN)Running enhanced self-check...$(NC)"
	$(PYTHON) scripts/self_check_enhanced.py

docker-build: ## 构建Docker镜像
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t cad-ml-platform:latest .

docker-run: ## 运行Docker容器
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run -d -p 8000:8000 --name cad-ml-platform cad-ml-platform:latest

docker-stop: ## 停止Docker容器
	@echo "$(RED)Stopping Docker container...$(NC)"
	docker stop cad-ml-platform
	docker rm cad-ml-platform

# Golden 评估相关
eval-vision-golden: ## 运行 Vision 模块 Golden 评估
	@echo "$(GREEN)Running Vision Golden Evaluation...$(NC)"
	$(PYTHON) scripts/evaluate_vision_golden.py

eval-ocr-golden: ## 运行 OCR 模块 Golden 评估
	@echo "$(GREEN)Running OCR Golden Evaluation...$(NC)"
	$(PYTHON) tests/ocr/run_golden_evaluation.py

eval-all-golden: ## 运行所有 Golden 评估
	@echo "$(GREEN)Running All Golden Evaluations...$(NC)"
	@echo "$(YELLOW)=== Vision Golden Evaluation ===$(NC)"
	$(MAKE) eval-vision-golden
	@echo ""
	@echo "$(YELLOW)=== OCR Golden Evaluation ===$(NC)"
	$(MAKE) eval-ocr-golden

# ==================== OBSERVABILITY TARGETS ====================

observability-up: ## 启动完整的可观测性栈
	@echo "$(GREEN)Starting observability stack...$(NC)"
	docker-compose -f docker-compose.observability.yml up -d
	@echo "$(GREEN)Waiting for services to be ready...$(NC)"
	@sleep 10
	@echo "$(GREEN)Observability stack is running!$(NC)"
	@echo "  - Application: http://localhost:8000"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"
	@echo "  - Metrics: http://localhost:8000/metrics"

observability-down: ## 停止可观测性栈
	@echo "$(RED)Stopping observability stack...$(NC)"
	docker-compose -f docker-compose.observability.yml down
	@echo "$(GREEN)Observability stack stopped$(NC)"

observability-status: ## 检查可观测性栈状态
	@echo "$(GREEN)Checking observability stack status...$(NC)"
	@docker-compose -f docker-compose.observability.yml ps
	@echo ""
	@echo "$(YELLOW)Service Health:$(NC)"
	@curl -s localhost:8000/health | jq '.status' 2>/dev/null || echo "App: Not running"
	@curl -s localhost:9090/-/ready 2>/dev/null && echo "Prometheus: Ready" || echo "Prometheus: Not ready"
	@curl -s localhost:3000/api/health 2>/dev/null && echo "Grafana: Ready" || echo "Grafana: Not ready"

self-check-strict: ## 运行严格模式自检
	@echo "$(GREEN)Running strict self-check...$(NC)"
	SELF_CHECK_STRICT_METRICS=1 \
	SELF_CHECK_MIN_OCR_ERRORS=5 \
	SELF_CHECK_INCREMENT_COUNTERS=1 \
	$(PYTHON) scripts/self_check.py

self-check-json: ## 运行自检并输出JSON
	@echo "$(GREEN)Running self-check with JSON output...$(NC)"
	@$(PYTHON) scripts/self_check.py --json | $(PYTHON) -m json.tool

metrics-validate: ## 验证指标合约
	@echo "$(GREEN)Validating metrics contract...$(NC)"
	$(PYTEST) tests/test_metrics_contract.py -v
	$(PYTEST) tests/test_provider_error_mapping.py -v

prom-validate: ## 验证Prometheus录制规则
	@echo "$(GREEN)Validating Prometheus recording rules...$(NC)"
	$(PYTHON) scripts/validate_prom_rules.py --skip-promtool
	@echo ""
	@echo "$(YELLOW)Validating with promtool (Docker)...$(NC)"
	@docker run --rm -v $(PWD)/docs/prometheus:/rules:ro \
		prom/prometheus:latest \
		promtool check rules /rules/recording_rules.yml || echo "$(YELLOW)Promtool not available$(NC)"

promtool-validate-all: ## 使用 promtool 验证所有规则文件
	@echo "$(GREEN)Validating all Prometheus rules with promtool...$(NC)"
	bash scripts/validate_prometheus.sh

dashboard-import: ## 导入Grafana仪表板
	@echo "$(GREEN)Importing Grafana dashboard...$(NC)"
	@echo "Please ensure Grafana is running on http://localhost:3000"
	@echo "Login with admin/admin and import the dashboard from:"
	@echo "  docs/grafana/observability_dashboard.json"
	@open http://localhost:3000/dashboard/import || echo "Open http://localhost:3000/dashboard/import manually"

security-audit: ## 运行安全审计
	@echo "$(GREEN)Running security audit...$(NC)"
	@echo "$(YELLOW)Checking dependencies with pip-audit...$(NC)"
	-pip-audit
	@echo ""
	@echo "$(YELLOW)Checking with safety...$(NC)"
	-safety check
	@echo ""
	@echo "$(YELLOW)Running bandit security scan...$(NC)"
	-bandit -r $(SRC_DIR) -f json -o security-report.json

observability-test: ## 运行可观测性测试套件
	@echo "$(GREEN)Running observability test suite...$(NC)"
	$(PYTEST) tests/test_observability_suite.py -v

observability-logs: ## 查看可观测性栈日志
	@echo "$(GREEN)Showing observability stack logs...$(NC)"
	docker-compose -f docker-compose.observability.yml logs -f

observability-restart: ## 重启可观测性栈
	@echo "$(YELLOW)Restarting observability stack...$(NC)"
	$(MAKE) observability-down
	$(MAKE) observability-up

observability-clean: ## 清理可观测性数据
	@echo "$(RED)Cleaning observability data...$(NC)"
	docker-compose -f docker-compose.observability.yml down -v
	@echo "$(GREEN)All observability data cleaned$(NC)"

# ==================== METRICS AUDIT TARGETS ====================

metrics-audit: ## 运行指标基数审计
	@echo "$(GREEN)Running metrics cardinality audit...$(NC)"
	$(PYTHON) scripts/cardinality_audit.py --format markdown
	@echo "$(GREEN)Audit complete!$(NC)"

cardinality-check: ## 检查指标基数并生成报告
	@echo "$(GREEN)Checking metrics cardinality...$(NC)"
	$(PYTHON) scripts/cardinality_audit.py \
		--prometheus-url http://localhost:9090 \
		--warning-threshold 100 \
		--critical-threshold 1000 \
		--format json \
		--output reports/cardinality_report.json
	@echo "$(GREEN)Report saved to reports/cardinality_report.json$(NC)"

metrics-audit-watch: ## 持续监控指标基数
	@echo "$(GREEN)Starting continuous cardinality monitoring...$(NC)"
	@while true; do \
		clear; \
		$(PYTHON) scripts/cardinality_audit.py --format markdown; \
		sleep 60; \
	done

# 快速命令别名
obs-up: observability-up
obs-down: observability-down
obs-status: observability-status
	@echo "$(GREEN)All golden evaluations complete!$(NC)"

eval-combined: ## 运行 Vision+OCR 联合评估（计算 combined score）
	@echo "$(GREEN)Running Vision+OCR Combined Evaluation...$(NC)"
	$(PYTHON) scripts/evaluate_vision_ocr_combined.py

eval-combined-save: ## 联合评估并保存历史记录
	@echo "$(GREEN)Running Vision+OCR Combined Evaluation (with history)...$(NC)"
	$(PYTHON) scripts/evaluate_vision_ocr_combined.py --save-history

eval-report: ## 生成静态 HTML 评测报告
	@echo "$(GREEN)Generating Evaluation Report...$(NC)"
	@echo "Step 1/3: Running combined evaluation..."
	@$(MAKE) eval-combined-save || echo "$(YELLOW)Warning: eval-combined-save failed, continuing...$(NC)"
	@echo "Step 2/3: Generating trend charts..."
	@$(MAKE) eval-trend || echo "$(YELLOW)Warning: eval-trend failed, continuing...$(NC)"
	@echo "Step 3/3: Generating HTML report..."
	$(PYTHON) scripts/generate_eval_report.py
	@echo "$(GREEN)Report generated!$(NC)"
	@echo "Open: file://$(PWD)/reports/eval_history/report/index.html"

# 可观测性：评测历史与健康检查
eval-history: ## 保存评测结果到历史目录
	@echo "$(GREEN)Saving evaluation results to history...$(NC)"
	bash scripts/eval_with_history.sh

health-check: ## 一键输出系统关键健康状态
	@echo "$(GREEN)Quick health summary...$(NC)"
	python3 scripts/quick_health.py

test-map: ## 自动更新 TEST_MAP.md（同步测试统计）
	@echo "$(GREEN)Updating TEST_MAP.md...$(NC)"
	python3 scripts/list_tests.py --markdown > docs/TEST_MAP_AUTO.md
	@echo "Auto-generated test statistics saved to docs/TEST_MAP_AUTO.md"
	@echo "Review and merge into docs/TEST_MAP.md as needed"

test-map-overwrite: ## 覆盖更新 TEST_MAP.md（需要输入 'yes' 确认）
	@read -p "This will overwrite docs/TEST_MAP.md. Type 'yes' to confirm: " ans; \
	if [ "$$ans" = "yes" ]; then \
		python3 scripts/list_tests.py --markdown > docs/TEST_MAP.md; \
		echo "$(GREEN)docs/TEST_MAP.md updated.$(NC)"; \
	else \
		echo "$(YELLOW)Aborted. docs/TEST_MAP.md not changed.$(NC)"; \
		exit 1; \
	fi

# CI相关命令
ci-test: ## CI测试流程
	@echo "$(GREEN)Running CI tests...$(NC)"
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test
	$(MAKE) test-baseline
	@echo "$(GREEN)CI tests passed!$(NC)"

ci-check-metrics: ## 检查指标阈值
	@echo "$(GREEN)Checking metrics thresholds...$(NC)"
	$(PYTHON) scripts/check_metrics.py --min-f1 0.75 --min-confidence 0.7

ci-combined-check: ## CI 联合评估质量门禁（支持 MIN_COMBINED/MIN_VISION/MIN_OCR 覆盖）
	@echo "$(GREEN)Running CI Combined Check...$(NC)"
	@echo "Using thresholds: combined=$${MIN_COMBINED:-0.8}, vision=$${MIN_VISION:-0.65}, ocr=$${MIN_OCR:-0.9}"
	$(PYTHON) scripts/evaluate_vision_ocr_combined.py \
		--min-combined $${MIN_COMBINED:-0.8} \
		--min-vision $${MIN_VISION:-0.65} \
		--min-ocr $${MIN_OCR:-0.9}

# 数据库相关
db-migrate: ## 运行数据库迁移
	@echo "$(GREEN)Running database migrations...$(NC)"
	alembic upgrade head

db-rollback: ## 回滚数据库
	@echo "$(YELLOW)Rolling back database...$(NC)"
	alembic downgrade -1

# Redis相关
redis-start: ## 启动Redis
	@echo "$(GREEN)Starting Redis...$(NC)"
	redis-server --daemonize yes

redis-stop: ## 停止Redis
	@echo "$(RED)Stopping Redis...$(NC)"
	redis-cli shutdown

# 监控相关
grafana-import: ## 导入Grafana仪表板
	@echo "$(GREEN)Importing Grafana dashboard...$(NC)"
	$(PYTHON) scripts/import_grafana_dashboard.py

# 知识库相关
kb-validate: ## 验证知识库
	@echo "$(GREEN)Validating knowledge base...$(NC)"
	$(PYTHON) scripts/validate_knowledge_base.py

kb-version: ## 显示知识库版本
	@cat knowledge_base/assembly/VERSION

# 安全检查
security-check: ## 安全扫描
	@echo "$(GREEN)Running security scan...$(NC)"
	bandit -r $(SRC_DIR) -f json -o security_report.json
	safety check --json

# 性能测试
perf-test: ## 性能基准测试
	@echo "$(GREEN)Running performance benchmarks...$(NC)"
	$(PYTHON) benchmarks/assembly_benchmark.py


# 完整检查
check-all: ## 运行所有检查
	@echo "$(GREEN)Running all checks...$(NC)"
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test
	$(MAKE) test-baseline
	$(MAKE) security-check
	@echo "$(GREEN)All checks passed!$(NC)"

# 快速开始
quickstart: ## 快速开始指南
	@echo "$(GREEN)CAD ML Platform - Quick Start$(NC)"
	@echo "----------------------------------------"
	@echo "1. Setup: make dev"
	@echo "2. Test: make test"
	@echo "3. Run: make run"
	@echo "4. Demo: make run-demo"
	@echo "----------------------------------------"
	@echo "Visit http://localhost:8000/docs for API documentation"
eval-trend: ## 生成评测趋势图（reports/eval_history/plots）
	@echo "$(GREEN)Generating evaluation trends...$(NC)"
	python3 scripts/eval_trend.py --out reports/eval_history/plots

eval-validate: ## 校验评测历史文件的 schema 合规性
	@echo "$(GREEN)Validating evaluation history files...$(NC)"
	$(PYTHON) scripts/validate_eval_history.py --dir reports/eval_history

eval-migrate: ## 迁移旧版评测历史到 v1.0.0 schema
	@echo "$(YELLOW)Migrating legacy evaluation history files...$(NC)"
	@echo "This will create .bak backup files for all migrated files."
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	$(PYTHON) scripts/validate_eval_history.py --dir reports/eval_history --migrate
	@echo "$(GREEN)Migration complete! Run 'make eval-validate' to verify.$(NC)"

eval-retention: ## 查看数据保留策略状态（5层：7d全量/30d日快照/90d周快照/365d月快照/永久季度快照）
	@echo "$(GREEN)Checking retention policy (5-tier: 7d/30d/90d/365d/forever)...$(NC)"
	$(PYTHON) scripts/manage_eval_retention.py --dry-run

eval-retention-apply: ## 应用5层数据保留策略（删除冗余历史，需要确认）
	@echo "$(YELLOW)Applying retention policy will DELETE old files...$(NC)"
	@read -p "Archive files before deletion? (y/N): " archive; \
	if [ "$$archive" = "y" ]; then \
		$(PYTHON) scripts/manage_eval_retention.py --execute --archive; \
	else \
		read -p "Proceed without archiving? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1; \
		$(PYTHON) scripts/manage_eval_retention.py --execute; \
	fi
	@echo "$(GREEN)Retention policy applied!$(NC)"

eval-report-v2: ## 生成增强版 HTML 报告（交互式图表）
	@echo "$(GREEN)Generating enhanced evaluation report...$(NC)"
	$(PYTHON) scripts/generate_eval_report_v2.py --use-cdn
	@echo "$(GREEN)Enhanced report generated!$(NC)"
	@echo "Open: file://$(PWD)/reports/eval_history/report/index.html"

integrity-check: ## 检查关键依赖文件完整性（使用 config/eval_frontend.json）
	@echo "$(GREEN)Checking file integrity...$(NC)"
	$(PYTHON) scripts/check_integrity.py --verbose

integrity-check-strict: ## 严格完整性检查（失败时退出代码1）
	@echo "$(YELLOW)Running strict integrity check...$(NC)"
	$(PYTHON) scripts/check_integrity.py --strict --verbose

eval-validate-schema: ## 使用 JSON Schema 验证历史文件
	@echo "$(GREEN)Validating with JSON Schema...$(NC)"
	$(PYTHON) scripts/validate_eval_history.py --schema docs/eval_history.schema.json --summary

# ============================================================================
# Pre-commit and Developer Tools
# ============================================================================

eval-validate-soft: ## 软验证（用于本地开发，非阻塞）
	@echo "$(BLUE)Running soft validation for pre-commit check...$(NC)"
	@echo "================================================"
	@echo "Step 1/3: Checking file integrity (non-blocking)..."
	-@$(PYTHON) scripts/check_integrity.py --verbose 2>&1 | grep -E "PASS|WARNING|ERROR" || true
	@echo ""
	@echo "Step 2/3: Validating JSON schema (non-blocking)..."
	-@$(PYTHON) scripts/validate_eval_history.py --dir reports/eval_history --summary 2>&1 | grep -E "Valid|Invalid|WARNING" || true
	@echo ""
	@echo "Step 3/3: Running quick health check..."
	-@$(MAKE) health-check 2>&1 | tail -5 || true
	@echo "================================================"
	@echo "$(GREEN)✓ Soft validation complete (check output above)$(NC)"
	@echo "$(YELLOW)Note: This is non-blocking. Fix any issues before pushing.$(NC)"

pre-commit: eval-validate-soft ## 运行所有预提交检查
	@echo "$(GREEN)Pre-commit checks complete!$(NC)"

# ============================================================================
# End-to-End Workflows
# ============================================================================

eval-e2e: ## 完整端到端评估流程
	@echo "$(BLUE)Starting end-to-end evaluation workflow...$(NC)"
	@echo "Step 1/4: Running combined evaluation..."
	@$(MAKE) eval-combined-save
	@echo ""
	@echo "Step 2/4: Generating trend charts..."
	@$(MAKE) eval-trend || echo "$(YELLOW)Trend generation skipped$(NC)"
	@echo ""
	@echo "Step 3/4: Generating interactive report..."
	@$(MAKE) eval-report-v2
	@echo ""
	@echo "Step 4/4: Running validation..."
	@$(MAKE) eval-validate
	@echo "$(GREEN)✓ End-to-end workflow complete!$(NC)"

eval-full: eval-e2e ## 别名：完整评估流程
	@echo "$(GREEN)Full evaluation complete!$(NC)"

# ============================================================================
# Advanced Analytics and Security
# ============================================================================

eval-insights: ## 生成 LLM 洞察和异常检测报告 (Markdown)
	@echo "$(BLUE)Analyzing evaluation insights...$(NC)"
	$(PYTHON) scripts/analyze_eval_insights.py --days 30 --output reports/insights_$(shell date +%Y%m%d).md
	@echo "$(GREEN)Insights report generated!$(NC)"

eval-insights-json: ## 生成机器可解析的 JSON 洞察报告
	@echo "$(BLUE)Generating JSON insights report...$(NC)"
	@$(PYTHON) scripts/analyze_eval_insights.py --days 30 --output reports/insights/latest.json
	@echo "$(GREEN)JSON insights saved to: reports/insights/latest.json$(NC)"

eval-anomalies: ## 检测评估指标异常
	@echo "$(YELLOW)Detecting anomalies...$(NC)"
	$(PYTHON) scripts/analyze_eval_insights.py --days 7 --threshold 0.1 --narrative-only

metrics-export: ## 导出指标到 Prometheus 格式
	@echo "$(GREEN)Exporting metrics...$(NC)"
	$(PYTHON) scripts/export_eval_metrics.py --format prometheus

metrics-serve: ## 启动指标服务器 (端口 8000)
	@echo "$(GREEN)Starting metrics server on port 8000...$(NC)"
	$(PYTHON) scripts/export_eval_metrics.py --serve --port 8000

metrics-push: ## 推送指标到 Prometheus Gateway
	@echo "$(GREEN)Pushing metrics to Prometheus Gateway...$(NC)"
	$(PYTHON) scripts/export_eval_metrics.py --push-gateway http://localhost:9091

security-audit: ## 运行安全审计
	@echo "$(YELLOW)Running security audit...$(NC)"
	$(PYTHON) scripts/security_audit.py --severity medium
	@echo "$(GREEN)Security audit complete!$(NC)"

security-critical: ## 仅检查关键安全问题
	@echo "$(RED)Checking critical security issues...$(NC)"
	$(PYTHON) scripts/security_audit.py --severity critical --fail-on-high

eval-with-security: eval-combined-save security-audit ## 评估 + 安全扫描
	@echo "$(GREEN)Evaluation with security audit complete!$(NC)"

# ============================================================================
# Phase 6: Complete Advanced Workflow
# ============================================================================

eval-phase6: ## Phase 6 完整流程 (评估+洞察+指标+安全)
	@echo "$(BLUE)Running Phase 6 Advanced Workflow...$(NC)"
	@echo "Step 1/5: Running evaluation..."
	@$(MAKE) eval-combined-save
	@echo ""
	@echo "Step 2/5: Generating insights..."
	@$(MAKE) eval-insights
	@echo ""
	@echo "Step 3/5: Checking for anomalies..."
	@$(MAKE) eval-anomalies || true
	@echo ""
	@echo "Step 4/5: Exporting metrics..."
	@$(MAKE) metrics-export
	@echo ""
	@echo "Step 5/5: Running security audit..."
	@$(MAKE) security-audit
	@echo "$(GREEN)✓ Phase 6 workflow complete!$(NC)"

# ============================================================================
# Baseline Management
# ============================================================================

baseline-update: ## 更新异常检测基线
	@echo "🔄 Updating anomaly baseline from history..."
	@python3 scripts/anomaly_baseline.py --update

baseline-snapshot: ## 创建季度基线快照
	@echo "📸 Creating quarterly baseline snapshot..."
	@python3 scripts/snapshot_baseline.py

baseline-list: ## 列出所有基线快照
	@echo "📋 Listing baseline snapshots..."
	@python3 scripts/snapshot_baseline.py --list

baseline-compare: ## 比较两个基线快照 (用法: make baseline-compare SNAP1=2025_Q1 SNAP2=2025_Q2)
	@echo "📊 Comparing baseline snapshots..."
	@python3 scripts/snapshot_baseline.py --compare baseline_$(SNAP1).json baseline_$(SNAP2).json

# ========================================
# 录制规则版本管理
# ========================================

.PHONY: rules-init rules-commit rules-list rules-diff rules-rollback rules-validate rules-deploy

rules-init: ## 初始化录制规则版本管理
	@echo "Initializing recording rules versioning..."
	$(PYTHON) scripts/recording_rules_versioning.py init

rules-commit: ## 提交录制规则版本
	@echo "Creating new rules version..."
	$(PYTHON) scripts/recording_rules_versioning.py commit -m "$(MSG)" -a "$(USER)"

rules-list: ## 列出所有规则版本
	@echo "Listing rule versions..."
	$(PYTHON) scripts/recording_rules_versioning.py list -n 20

rules-diff: ## 比较规则版本差异
	@echo "Comparing rule versions..."
	$(PYTHON) scripts/recording_rules_versioning.py diff $(V1) $(V2)

rules-rollback: ## 回滚到指定版本
	@echo "Rolling back to version $(VERSION)..."
	$(PYTHON) scripts/recording_rules_versioning.py rollback $(VERSION)

rules-validate: ## 验证录制规则
	@echo "Validating recording rules..."
	@bash scripts/rules_ci_integration.sh validate

rules-deploy: ## 部署规则到 Prometheus
	@echo "Deploying rules to Prometheus..."
	@bash scripts/rules_ci_integration.sh deploy $(PROMETHEUS_URL)

rules-ci: ## 运行规则 CI 流程
	@echo "Running rules CI pipeline..."
	@bash scripts/rules_ci_integration.sh ci

rules-cd: ## 运行规则 CD 流程
	@echo "Running rules CD pipeline..."
	@bash scripts/rules_ci_integration.sh cd $(PROMETHEUS_URL)

rules-report: ## 生成规则版本报告
	@echo "Generating rules version report..."
	$(PYTHON) scripts/recording_rules_versioning.py report --format markdown
