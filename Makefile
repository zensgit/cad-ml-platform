# CAD ML Platform - Makefile
# 统一的开发工作流

.PHONY: help install dev test test-dedupcad-vision lint format type-check clean run docs docker eval-history health-check eval-trend \
	observability-up observability-down observability-status self-check metrics-validate prom-validate \
	dashboard-import security-audit metrics-audit cardinality-check verify-metrics test-targeted e2e-smoke \
	dedup2d-secure-smoke chrome-devtools cdp-console-demo cdp-network-demo cdp-perf-demo cdp-response-demo \
	cdp-screenshot-demo cdp-trace-demo playwright-console-demo playwright-trace-demo playwright-install \
		uvnet-checkpoint-inspect graph2d-freeze-baseline worktree-bootstrap validate-iso286 validate-tolerance \
		validate-openapi \
		graph2d-review-summary validate-core-fast test-provider-core test-provider-contract \
		validate-graph2d-seed-gate validate-graph2d-seed-gate-strict \
		validate-graph2d-seed-gate-regression validate-graph2d-seed-gate-strict-regression \
		validate-graph2d-seed-gate-context-drift-warn \
		validate-graph2d-seed-gate-baseline-health \
		update-graph2d-seed-gate-baseline \
	audit-pydantic-v2 audit-pydantic-v2-regression \
	audit-pydantic-style audit-pydantic-style-regression \
	openapi-snapshot-update \
	archive-experiments archive-workflow-dry-run-gh archive-workflow-apply-gh \
	validate-archive-workflow-dispatcher \
	watch-commit-workflows validate-watch-commit-workflows \
	validate-ci-watchers clean-ci-watch-summaries \
	check-gh-actions-ready validate-check-gh-actions-ready \
	watch-commit-workflows-safe clean-gh-readiness-summaries \
	clean-ci-watch-artifacts
.PHONY: test-unit test-contract-local test-e2e-local test-all-local test-tolerance test-service-mesh test-provider-core test-provider-contract validate-openapi

# 默认目标
.DEFAULT_GOAL := help

# 变量定义
# Prefer project venv or newer Python when available to match 3.10+ requirement.
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	elif command -v python3.11 >/dev/null 2>&1; then command -v python3.11; \
	elif command -v python3.10 >/dev/null 2>&1; then command -v python3.10; \
	elif command -v python3 >/dev/null 2>&1; then command -v python3; \
	else echo python3; fi)
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
MYPY := $(PYTHON) -m mypy
FLAKE8 := $(PYTHON) -m flake8
PROMETHEUS_URL ?= http://localhost:9091
UVNET_CHECKPOINT ?= models/uvnet_v1.pth
ARCHIVE_EXPERIMENTS_ROOT ?= reports/experiments
ARCHIVE_EXPERIMENTS_OUT ?= $(HOME)/Downloads/cad-ml-platform-experiment-archives
ARCHIVE_EXPERIMENTS_KEEP_DAYS ?= 7
ARCHIVE_EXPERIMENTS_TODAY ?=
ARCHIVE_EXPERIMENTS_MANIFEST ?= reports/archive_experiments_manifest.json
ARCHIVE_EXPERIMENTS_EXTRA_ARGS ?= --dry-run
ARCHIVE_WORKFLOW_REF ?= main
ARCHIVE_WORKFLOW_EXPERIMENTS_ROOT ?= reports/experiments
ARCHIVE_WORKFLOW_ARCHIVE_ROOT ?=
ARCHIVE_WORKFLOW_KEEP_DAYS ?= 7
ARCHIVE_WORKFLOW_TODAY ?=
ARCHIVE_WORKFLOW_DIRS_CSV ?=
ARCHIVE_WORKFLOW_REQUIRE_EXISTS ?= true
ARCHIVE_WORKFLOW_WATCH ?= 0
ARCHIVE_WORKFLOW_PRINT_ONLY ?= 0
ARCHIVE_WORKFLOW_WAIT_TIMEOUT ?= 120
ARCHIVE_WORKFLOW_POLL_INTERVAL ?= 3
CI_WATCH_SHA ?= HEAD
CI_WATCH_EVENTS ?= push
CI_WATCH_REQUIRED_WORKFLOWS ?= CI,CI Enhanced,CI Tiered Tests,Code Quality,Multi-Architecture Docker Build,Security Audit,Observability Checks,Self-Check,GHCR Publish,Evaluation Report
CI_WATCH_TIMEOUT ?= 1800
CI_WATCH_POLL_INTERVAL ?= 20
CI_WATCH_HEARTBEAT_INTERVAL ?= 120
CI_WATCH_LIST_LIMIT ?= 100
CI_WATCH_MAX_LIST_FAILURES ?= 3
CI_WATCH_MISSING_REQUIRED_MODE ?= fail-fast
CI_WATCH_FAILURE_MODE ?= fail-fast
CI_WATCH_SUCCESS_CONCLUSIONS ?= success,skipped
CI_WATCH_SUMMARY_JSON ?=
CI_WATCH_SUMMARY_DIR ?= reports/ci
CI_WATCH_PRINT_ONLY ?= 0
CI_WATCH_PRECHECK_STRICT ?= 1
GH_READY_JSON ?= reports/ci/gh_readiness_latest.json
GH_READY_SKIP_ACTIONS_API ?= 0

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

test-unit: ## 运行 unit 分层测试（快速质量门）
	@echo "$(GREEN)Running unit tests...$(NC)"
	bash scripts/test_with_local_api.sh --suite unit

test-contract-local: ## 自动起停本地 API 后运行 contract 测试
	@echo "$(GREEN)Running contract tests with local API...$(NC)"
	bash scripts/test_with_local_api.sh --suite contract

test-e2e-local: ## 自动起停本地 API 后运行 e2e 测试
	@echo "$(GREEN)Running e2e tests with local API...$(NC)"
	bash scripts/test_with_local_api.sh --suite e2e

test-all-local: ## 自动起停本地 API 后运行全量 tests
	@echo "$(GREEN)Running full tests with local API...$(NC)"
	bash scripts/test_with_local_api.sh --suite all

test-knowledge: ## 运行知识库相关测试
	@echo "$(GREEN)Running knowledge tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/unit/knowledge -v --junitxml=reports/junit-knowledge.xml

test-tolerance: ## 运行公差知识相关测试（unit + integration）
	@echo "$(GREEN)Running tolerance tests...$(NC)"
	$(PYTEST) \
		$(TEST_DIR)/unit/knowledge/test_tolerance.py \
		$(TEST_DIR)/unit/test_tolerance_fundamental_deviation.py \
		$(TEST_DIR)/unit/test_tolerance_limit_deviations.py \
		$(TEST_DIR)/unit/test_tolerance_api_normalization.py \
		$(TEST_DIR)/integration/test_tolerance_api_errors.py \
		$(TEST_DIR)/integration/test_tolerance_api.py -v

test-service-mesh: ## 运行 service-mesh 关键回归测试
	@echo "$(GREEN)Running service-mesh tests...$(NC)"
	$(PYTEST) \
		$(TEST_DIR)/unit/test_load_balancer_coverage.py \
		$(TEST_DIR)/unit/test_service_discovery_coverage.py -v

test-provider-core: ## 运行 provider 框架核心回归测试
	@echo "$(GREEN)Running provider core tests...$(NC)"
	$(PYTEST) \
		$(TEST_DIR)/unit/test_provider_registry_plugins.py \
		$(TEST_DIR)/unit/test_provider_plugin_metrics_exposed.py \
		$(TEST_DIR)/unit/test_bootstrap_coverage.py \
		$(TEST_DIR)/unit/test_provider_plugin_example_classifier.py \
		$(TEST_DIR)/unit/test_provider_registry_bootstrap.py \
		$(TEST_DIR)/unit/test_provider_knowledge_providers.py \
		$(TEST_DIR)/unit/test_provider_framework.py \
		$(TEST_DIR)/unit/test_provider_readiness.py \
		$(TEST_DIR)/unit/test_health_utils_coverage.py \
		$(TEST_DIR)/unit/test_health_hybrid_config.py \
		$(TEST_DIR)/unit/test_provider_health_endpoint.py \
		$(TEST_DIR)/unit/test_provider_check_metrics_exposed.py -v

test-provider-contract: ## 运行 provider 相关 API 契约回归测试
	@echo "$(GREEN)Running provider contract tests...$(NC)"
	$(PYTEST) \
		$(TEST_DIR)/contract/test_api_contract.py \
		-k "provider_health_endpoint_response_shape or health_payload_core_provider_plugin_summary_shape or provider_health_openapi_schema_contains_plugin_diagnostics or health_openapi_schema_contains_core_provider_plugin_summary" -v

validate-iso286: ## 验证 ISO286/GB-T 1800 偏差表数据（快速）
	@echo "$(GREEN)Validating ISO286 deviation tables...$(NC)"
	$(PYTHON) scripts/validate_iso286_deviations.py --spot-check
	$(PYTHON) scripts/validate_iso286_hole_deviations.py

validate-tolerance: ## 一键校验公差知识（数据 + API/模块测试）
	@echo "$(GREEN)Validating tolerance knowledge stack...$(NC)"
	$(MAKE) validate-iso286
	$(MAKE) test-tolerance

validate-openapi: ## 校验 OpenAPI operationId 唯一性
	@echo "$(GREEN)Validating OpenAPI operation IDs...$(NC)"
	$(PYTEST) \
		$(TEST_DIR)/contract/test_openapi_operation_ids.py \
		$(TEST_DIR)/contract/test_openapi_schema_snapshot.py \
		$(TEST_DIR)/unit/test_api_route_uniqueness.py -q

openapi-snapshot-update: ## 更新 OpenAPI 快照基线
	@echo "$(GREEN)Updating OpenAPI schema snapshot baseline...$(NC)"
	$(PYTHON) scripts/ci/generate_openapi_schema_snapshot.py --output config/openapi_schema_snapshot.json

archive-experiments: ## 归档 reports/experiments 日期目录（默认 dry-run）
	@echo "$(GREEN)Archiving experiment directories...$(NC)"
	$(PYTHON) scripts/ci/archive_experiment_dirs.py \
		--experiments-root "$(ARCHIVE_EXPERIMENTS_ROOT)" \
		--archive-root "$(ARCHIVE_EXPERIMENTS_OUT)" \
		--keep-latest-days "$(ARCHIVE_EXPERIMENTS_KEEP_DAYS)" \
		--today "$(ARCHIVE_EXPERIMENTS_TODAY)" \
		--manifest-json "$(ARCHIVE_EXPERIMENTS_MANIFEST)" \
		$(ARCHIVE_EXPERIMENTS_EXTRA_ARGS)
	@echo "$(GREEN)Archive manifest: $(ARCHIVE_EXPERIMENTS_MANIFEST)$(NC)"

archive-workflow-dry-run-gh: ## 通过 workflow_dispatch 触发 Experiment Archive Dry Run
	@echo "$(GREEN)Dispatching experiment archive dry-run workflow...$(NC)"
	@watch_flag=""; \
	if [ "$(ARCHIVE_WORKFLOW_WATCH)" = "1" ]; then watch_flag="--watch"; fi; \
	print_only_flag=""; \
	if [ "$(ARCHIVE_WORKFLOW_PRINT_ONLY)" = "1" ]; then print_only_flag="--print-only"; fi; \
	$(PYTHON) scripts/ci/dispatch_experiment_archive_workflow.py \
		--mode dry-run \
		--ref "$(ARCHIVE_WORKFLOW_REF)" \
		--experiments-root "$(ARCHIVE_WORKFLOW_EXPERIMENTS_ROOT)" \
		--archive-root "$(ARCHIVE_WORKFLOW_ARCHIVE_ROOT)" \
		--keep-latest-days "$(ARCHIVE_WORKFLOW_KEEP_DAYS)" \
		--today "$(ARCHIVE_WORKFLOW_TODAY)" \
		--wait-timeout-seconds "$(ARCHIVE_WORKFLOW_WAIT_TIMEOUT)" \
		--poll-interval-seconds "$(ARCHIVE_WORKFLOW_POLL_INTERVAL)" \
		$$watch_flag $$print_only_flag

archive-workflow-apply-gh: ## 通过 workflow_dispatch 触发 Experiment Archive Apply（需审批短语）
	@echo "$(GREEN)Dispatching experiment archive apply workflow...$(NC)"
	@test -n "$${ARCHIVE_APPROVAL_PHRASE:-}" || (echo "$(RED)ARCHIVE_APPROVAL_PHRASE is required$(NC)"; exit 1)
	@watch_flag=""; \
	if [ "$(ARCHIVE_WORKFLOW_WATCH)" = "1" ]; then watch_flag="--watch"; fi; \
	print_only_flag=""; \
	if [ "$(ARCHIVE_WORKFLOW_PRINT_ONLY)" = "1" ]; then print_only_flag="--print-only"; fi; \
	$(PYTHON) scripts/ci/dispatch_experiment_archive_workflow.py \
		--mode apply \
		--ref "$(ARCHIVE_WORKFLOW_REF)" \
		--experiments-root "$(ARCHIVE_WORKFLOW_EXPERIMENTS_ROOT)" \
		--archive-root "$(ARCHIVE_WORKFLOW_ARCHIVE_ROOT)" \
		--keep-latest-days "$(ARCHIVE_WORKFLOW_KEEP_DAYS)" \
		--today "$(ARCHIVE_WORKFLOW_TODAY)" \
		--approval-phrase "$${ARCHIVE_APPROVAL_PHRASE}" \
		--dirs-csv "$(ARCHIVE_WORKFLOW_DIRS_CSV)" \
		--require-exists "$(ARCHIVE_WORKFLOW_REQUIRE_EXISTS)" \
		--wait-timeout-seconds "$(ARCHIVE_WORKFLOW_WAIT_TIMEOUT)" \
		--poll-interval-seconds "$(ARCHIVE_WORKFLOW_POLL_INTERVAL)" \
		$$watch_flag $$print_only_flag

validate-archive-workflow-dispatcher: ## 一键校验 archive workflow dispatcher（脚本/工作流/Make 参数透传）
	@echo "$(GREEN)Validating archive workflow dispatcher...$(NC)"
	$(PYTEST) \
		$(TEST_DIR)/unit/test_dispatch_experiment_archive_workflow.py \
		$(TEST_DIR)/unit/test_experiment_archive_workflows.py \
		$(TEST_DIR)/unit/test_archive_experiment_dirs.py \
		$(TEST_DIR)/unit/test_archive_workflow_make_targets.py -q

watch-commit-workflows: ## 监控指定提交 SHA 的 CI 工作流并等待完成
	@echo "$(GREEN)Watching commit workflows...$(NC)"
	@print_only_flag=""; \
	if [ "$(CI_WATCH_PRINT_ONLY)" = "1" ]; then print_only_flag="--print-only"; fi; \
	$(PYTHON) scripts/ci/watch_commit_workflows.py \
		--sha "$(CI_WATCH_SHA)" \
		--events-csv "$(CI_WATCH_EVENTS)" \
		--require-workflows-csv "$(CI_WATCH_REQUIRED_WORKFLOWS)" \
		--wait-timeout-seconds "$(CI_WATCH_TIMEOUT)" \
		--poll-interval-seconds "$(CI_WATCH_POLL_INTERVAL)" \
		--heartbeat-interval-seconds "$(CI_WATCH_HEARTBEAT_INTERVAL)" \
		--list-limit "$(CI_WATCH_LIST_LIMIT)" \
		--max-list-failures "$(CI_WATCH_MAX_LIST_FAILURES)" \
		--missing-required-mode "$(CI_WATCH_MISSING_REQUIRED_MODE)" \
		--failure-mode "$(CI_WATCH_FAILURE_MODE)" \
		--success-conclusions-csv "$(CI_WATCH_SUCCESS_CONCLUSIONS)" \
		--summary-json-out "$(CI_WATCH_SUMMARY_JSON)" \
		$$print_only_flag

watch-commit-workflows-safe: ## 先做 gh readiness 预检，再执行 commit workflow watcher
	@if [ "$(CI_WATCH_PRECHECK_STRICT)" = "1" ]; then \
		$(MAKE) check-gh-actions-ready; \
	else \
		echo "$(YELLOW)[warn] CI_WATCH_PRECHECK_STRICT=0: precheck failures will be ignored$(NC)"; \
		$(MAKE) check-gh-actions-ready-soft; \
	fi
	@$(MAKE) watch-commit-workflows

validate-watch-commit-workflows: ## 校验 commit workflow watcher（脚本 + Make 参数透传）
	@echo "$(GREEN)Validating commit workflow watcher...$(NC)"
	$(PYTEST) \
		$(TEST_DIR)/unit/test_watch_commit_workflows.py \
		$(TEST_DIR)/unit/test_watch_commit_workflows_make_target.py -q

check-gh-actions-ready: ## 检查 gh CLI / 认证 / Actions API 可用性
	@echo "$(GREEN)Checking gh Actions readiness...$(NC)"
	@skip_actions_flag=""; \
	if [ "$(GH_READY_SKIP_ACTIONS_API)" = "1" ]; then skip_actions_flag="--skip-actions-api"; fi; \
	$(PYTHON) scripts/ci/check_gh_actions_ready.py \
		--json-out "$(GH_READY_JSON)" \
		$$skip_actions_flag

check-gh-actions-ready-soft: ## 检查 gh readiness（软模式：失败不返回非零）
	@echo "$(GREEN)Checking gh Actions readiness (soft mode)...$(NC)"
	@skip_actions_flag=""; \
	if [ "$(GH_READY_SKIP_ACTIONS_API)" = "1" ]; then skip_actions_flag="--skip-actions-api"; fi; \
	$(PYTHON) scripts/ci/check_gh_actions_ready.py \
		--json-out "$(GH_READY_JSON)" \
		$$skip_actions_flag \
		--allow-fail

validate-check-gh-actions-ready: ## 校验 gh readiness 检查脚本
	@echo "$(GREEN)Validating gh readiness checker...$(NC)"
	$(PYTEST) $(TEST_DIR)/unit/test_check_gh_actions_ready.py -q

validate-ci-watchers: ## 一键校验 CI watchers（commit + archive dispatcher）
	@echo "$(GREEN)Validating CI watcher stack...$(NC)"
	$(MAKE) validate-check-gh-actions-ready
	$(MAKE) validate-watch-commit-workflows
	$(MAKE) validate-archive-workflow-dispatcher

clean-ci-watch-summaries: ## 清理 watcher 运行时 summary JSON
	@echo "$(GREEN)Cleaning watcher summary artifacts...$(NC)"
	@mkdir -p "$(CI_WATCH_SUMMARY_DIR)"
	@rm -f "$(CI_WATCH_SUMMARY_DIR)"/watch_*_summary.json

clean-gh-readiness-summaries: ## 清理 gh readiness 运行时 JSON
	@echo "$(GREEN)Cleaning gh readiness artifacts...$(NC)"
	@mkdir -p "$(CI_WATCH_SUMMARY_DIR)"
	@rm -f "$(CI_WATCH_SUMMARY_DIR)"/gh_readiness*.json

clean-ci-watch-artifacts: ## 清理 watcher + readiness 全部运行时 JSON
	@$(MAKE) clean-ci-watch-summaries
	@$(MAKE) clean-gh-readiness-summaries

validate-core-fast: ## 一键执行当前稳定核心回归（tolerance + openapi + service-mesh + provider-core + provider-contract）
	@echo "$(GREEN)Running core fast validation...$(NC)"
	$(MAKE) validate-tolerance
	$(MAKE) validate-openapi
	$(MAKE) test-service-mesh
	$(MAKE) test-provider-core
	$(MAKE) test-provider-contract

validate-graph2d-seed-gate: ## Graph2D 多seed稳定性门禁（可用于 CI）
	@echo "$(GREEN)Running Graph2D seed stability gate...$(NC)"
	$(PYTHON) scripts/sweep_graph2d_profile_seeds.py \
		--config $${GRAPH2D_SEED_GATE_CONFIG:-config/graph2d_seed_gate.yaml} \
		--missing-dxf-dir-mode $${GRAPH2D_SEED_GATE_MISSING_DXF_DIR_MODE:-fail} \
		--work-root $${GRAPH2D_SEED_GATE_WORK_ROOT:-/tmp/graph2d-seed-gate}

validate-graph2d-seed-gate-strict: ## Graph2D 严格模式多seed稳定性门禁通道
	@echo "$(GREEN)Running Graph2D strict seed stability gate...$(NC)"
	$(PYTHON) scripts/sweep_graph2d_profile_seeds.py \
		--config $${GRAPH2D_SEED_GATE_STRICT_CONFIG:-config/graph2d_seed_gate_strict.yaml} \
		--missing-dxf-dir-mode $${GRAPH2D_SEED_GATE_STRICT_MISSING_DXF_DIR_MODE:-$${GRAPH2D_SEED_GATE_MISSING_DXF_DIR_MODE:-fail}} \
		--work-root $${GRAPH2D_SEED_GATE_STRICT_WORK_ROOT:-/tmp/graph2d-seed-gate-strict}

validate-graph2d-seed-gate-regression: ## Graph2D seed gate 基线回归检查（standard）
	@echo "$(GREEN)Checking Graph2D seed gate regression (standard)...$(NC)"
	$(PYTHON) scripts/ci/check_graph2d_seed_gate_regression.py \
		--summary-json $${GRAPH2D_SEED_GATE_SUMMARY_JSON:-/tmp/graph2d-seed-gate/seed_sweep_summary.json} \
		--baseline-json $${GRAPH2D_SEED_GATE_BASELINE_JSON:-config/graph2d_seed_gate_baseline.json} \
		--config $${GRAPH2D_SEED_GATE_REGRESSION_CONFIG:-config/graph2d_seed_gate_regression.yaml} \
		--channel standard

validate-graph2d-seed-gate-strict-regression: ## Graph2D seed gate 基线回归检查（strict）
	@echo "$(GREEN)Checking Graph2D seed gate regression (strict)...$(NC)"
	$(PYTHON) scripts/ci/check_graph2d_seed_gate_regression.py \
		--summary-json $${GRAPH2D_SEED_GATE_STRICT_SUMMARY_JSON:-/tmp/graph2d-seed-gate-strict/seed_sweep_summary.json} \
		--baseline-json $${GRAPH2D_SEED_GATE_BASELINE_JSON:-config/graph2d_seed_gate_baseline.json} \
		--config $${GRAPH2D_SEED_GATE_REGRESSION_CONFIG:-config/graph2d_seed_gate_regression.yaml} \
		--channel strict

validate-graph2d-seed-gate-context-drift-warn: ## Graph2D 上下文漂移观测（warn 通道，非阻塞）
	@echo "$(GREEN)Checking Graph2D seed gate context drift probe (warn mode)...$(NC)"
	$(PYTHON) scripts/ci/check_graph2d_seed_gate_regression.py \
		--summary-json $${GRAPH2D_SEED_GATE_SUMMARY_JSON:-/tmp/graph2d-seed-gate/seed_sweep_summary.json} \
		--baseline-json $${GRAPH2D_SEED_GATE_BASELINE_JSON:-config/graph2d_seed_gate_baseline.json} \
		--config $${GRAPH2D_SEED_GATE_REGRESSION_CONFIG:-config/graph2d_seed_gate_regression.yaml} \
		--channel strict \
		--context-mismatch-mode warn \
		--context-keys "$${GRAPH2D_CONTEXT_DRIFT_WARN_CONTEXT_KEYS:-manifest_label_mode,seeds,num_runs,max_samples,min_label_confidence,strict_low_conf_threshold}" \
		--max-accuracy-mean-drop $${GRAPH2D_CONTEXT_DRIFT_WARN_MAX_ACCURACY_MEAN_DROP:-1.0} \
		--max-accuracy-min-drop $${GRAPH2D_CONTEXT_DRIFT_WARN_MAX_ACCURACY_MIN_DROP:-1.0} \
		--max-top-pred-ratio-increase $${GRAPH2D_CONTEXT_DRIFT_WARN_MAX_TOP_PRED_RATIO_INCREASE:-1.0} \
		--max-low-conf-ratio-increase $${GRAPH2D_CONTEXT_DRIFT_WARN_MAX_LOW_CONF_RATIO_INCREASE:-1.0}

validate-graph2d-context-drift-pipeline: ## Graph2D 上下文漂移全链路（更新+渲染+告警+索引+校验+归档）
	@echo "$(GREEN)Running Graph2D context drift pipeline...$(NC)"
	$(PYTHON) scripts/ci/update_graph2d_context_drift_history.py \
		--config $${GRAPH2D_CONTEXT_DRIFT_CONFIG:-config/graph2d_context_drift_alerts.yaml} \
		--history-json $${GRAPH2D_CONTEXT_DRIFT_HISTORY_JSON:-/tmp/graph2d-context-drift-history-local.json} \
		--output-json $${GRAPH2D_CONTEXT_DRIFT_HISTORY_JSON:-/tmp/graph2d-context-drift-history-local.json} \
		--run-id $${GRAPH2D_CONTEXT_DRIFT_RUN_ID:-local} \
		--run-number $${GRAPH2D_CONTEXT_DRIFT_RUN_NUMBER:-local} \
		--ref-name $${GRAPH2D_CONTEXT_DRIFT_REF_NAME:-local} \
		--sha $${GRAPH2D_CONTEXT_DRIFT_SHA:-local} \
		--report-json $${GRAPH2D_CONTEXT_DRIFT_REGRESSION_REPORT_JSON:-/tmp/graph2d-seed-gate-regression.json} \
		--report-json $${GRAPH2D_CONTEXT_DRIFT_WARN_REPORT_JSON:-/tmp/graph2d-context-drift-warn.json}
	$(PYTHON) scripts/ci/render_graph2d_context_drift_key_counts.py \
		--config $${GRAPH2D_CONTEXT_DRIFT_CONFIG:-config/graph2d_context_drift_alerts.yaml} \
		--report-json $${GRAPH2D_CONTEXT_DRIFT_REGRESSION_REPORT_JSON:-/tmp/graph2d-seed-gate-regression.json} \
		--report-json $${GRAPH2D_CONTEXT_DRIFT_WARN_REPORT_JSON:-/tmp/graph2d-context-drift-warn.json} \
		--title "Graph2D Context Drift Key Counts (Local)" \
		--output-json $${GRAPH2D_CONTEXT_DRIFT_KEY_COUNTS_JSON:-/tmp/graph2d-context-drift-key-counts-local.json} \
		--output-md $${GRAPH2D_CONTEXT_DRIFT_KEY_COUNTS_MD:-/tmp/graph2d-context-drift-key-counts-local.md}
	$(PYTHON) scripts/ci/render_graph2d_context_drift_history.py \
		--config $${GRAPH2D_CONTEXT_DRIFT_CONFIG:-config/graph2d_context_drift_alerts.yaml} \
		--history-json $${GRAPH2D_CONTEXT_DRIFT_HISTORY_JSON:-/tmp/graph2d-context-drift-history-local.json} \
		--title "Graph2D Context Drift History (Local)" \
		--output-json $${GRAPH2D_CONTEXT_DRIFT_HISTORY_SUMMARY_JSON:-/tmp/graph2d-context-drift-history-summary-local.json} \
		--output-md $${GRAPH2D_CONTEXT_DRIFT_HISTORY_MD:-/tmp/graph2d-context-drift-history-local.md}
	$(PYTHON) scripts/ci/check_graph2d_context_drift_alerts.py \
		--config $${GRAPH2D_CONTEXT_DRIFT_CONFIG:-config/graph2d_context_drift_alerts.yaml} \
		--history-json $${GRAPH2D_CONTEXT_DRIFT_HISTORY_JSON:-/tmp/graph2d-context-drift-history-local.json} \
		--title "Graph2D Context Drift Alerts (Local)" \
		--output-json $${GRAPH2D_CONTEXT_DRIFT_ALERTS_JSON:-/tmp/graph2d-context-drift-alerts-local.json} \
		--output-md $${GRAPH2D_CONTEXT_DRIFT_ALERTS_MD:-/tmp/graph2d-context-drift-alerts-local.md}
	$(PYTHON) scripts/ci/index_graph2d_context_drift_artifacts.py \
		--alerts-json $${GRAPH2D_CONTEXT_DRIFT_ALERTS_JSON:-/tmp/graph2d-context-drift-alerts-local.json} \
		--history-summary-json $${GRAPH2D_CONTEXT_DRIFT_HISTORY_SUMMARY_JSON:-/tmp/graph2d-context-drift-history-summary-local.json} \
		--key-counts-summary-json $${GRAPH2D_CONTEXT_DRIFT_KEY_COUNTS_JSON:-/tmp/graph2d-context-drift-key-counts-local.json} \
		--history-json $${GRAPH2D_CONTEXT_DRIFT_HISTORY_JSON:-/tmp/graph2d-context-drift-history-local.json} \
		--output-json $${GRAPH2D_CONTEXT_DRIFT_INDEX_JSON:-/tmp/graph2d-context-drift-index-local.json}
	$(PYTHON) scripts/ci/summarize_graph2d_context_drift_index.py \
		--index-json $${GRAPH2D_CONTEXT_DRIFT_INDEX_JSON:-/tmp/graph2d-context-drift-index-local.json} \
		--title "Graph2D Context Drift Index (Local)" \
		> $${GRAPH2D_CONTEXT_DRIFT_INDEX_MD:-/tmp/graph2d-context-drift-index-local.md}
	$(PYTHON) scripts/ci/validate_graph2d_context_drift_index.py \
		--index-json $${GRAPH2D_CONTEXT_DRIFT_INDEX_JSON:-/tmp/graph2d-context-drift-index-local.json} \
		--schema-json $${GRAPH2D_CONTEXT_DRIFT_INDEX_SCHEMA_JSON:-config/graph2d_context_drift_index_schema.json}
	$(PYTHON) scripts/ci/check_graph2d_context_drift_index_policy.py \
		--index-json $${GRAPH2D_CONTEXT_DRIFT_INDEX_JSON:-/tmp/graph2d-context-drift-index-local.json} \
		--config $${GRAPH2D_CONTEXT_DRIFT_INDEX_POLICY_CONFIG:-config/graph2d_context_drift_index_policy.yaml} \
		--fail-on-breach $${GRAPH2D_CONTEXT_DRIFT_INDEX_POLICY_FAIL_ON_BREACH:-auto} \
		--title "Graph2D Context Drift Index Policy (Local)" \
		--output-json $${GRAPH2D_CONTEXT_DRIFT_INDEX_POLICY_JSON:-/tmp/graph2d-context-drift-index-policy-local.json} \
		--output-md $${GRAPH2D_CONTEXT_DRIFT_INDEX_POLICY_MD:-/tmp/graph2d-context-drift-index-policy-local.md}
	$(PYTHON) scripts/ci/archive_graph2d_context_drift_artifacts.py \
		--output-root $${GRAPH2D_CONTEXT_DRIFT_ARCHIVE_ROOT:-reports/experiments} \
		--bucket $${GRAPH2D_CONTEXT_DRIFT_ARCHIVE_BUCKET:-graph2d_context_drift_local} \
		--require-exists \
		--artifact $${GRAPH2D_CONTEXT_DRIFT_ALERTS_JSON:-/tmp/graph2d-context-drift-alerts-local.json} \
		--artifact $${GRAPH2D_CONTEXT_DRIFT_ALERTS_MD:-/tmp/graph2d-context-drift-alerts-local.md} \
		--artifact $${GRAPH2D_CONTEXT_DRIFT_KEY_COUNTS_JSON:-/tmp/graph2d-context-drift-key-counts-local.json} \
		--artifact $${GRAPH2D_CONTEXT_DRIFT_KEY_COUNTS_MD:-/tmp/graph2d-context-drift-key-counts-local.md} \
		--artifact $${GRAPH2D_CONTEXT_DRIFT_HISTORY_JSON:-/tmp/graph2d-context-drift-history-local.json} \
		--artifact $${GRAPH2D_CONTEXT_DRIFT_HISTORY_SUMMARY_JSON:-/tmp/graph2d-context-drift-history-summary-local.json} \
		--artifact $${GRAPH2D_CONTEXT_DRIFT_HISTORY_MD:-/tmp/graph2d-context-drift-history-local.md} \
		--artifact $${GRAPH2D_CONTEXT_DRIFT_INDEX_JSON:-/tmp/graph2d-context-drift-index-local.json} \
		--artifact $${GRAPH2D_CONTEXT_DRIFT_INDEX_MD:-/tmp/graph2d-context-drift-index-local.md} \
		--artifact $${GRAPH2D_CONTEXT_DRIFT_INDEX_POLICY_JSON:-/tmp/graph2d-context-drift-index-policy-local.json} \
		--artifact $${GRAPH2D_CONTEXT_DRIFT_INDEX_POLICY_MD:-/tmp/graph2d-context-drift-index-policy-local.md}

validate-graph2d-seed-gate-baseline-health: ## Graph2D 基线健康检查（不依赖当前 summary）
	@echo "$(GREEN)Checking Graph2D seed gate baseline health (standard + strict)...$(NC)"
	$(PYTHON) scripts/ci/check_graph2d_seed_gate_regression.py \
		--baseline-json $${GRAPH2D_SEED_GATE_BASELINE_JSON:-config/graph2d_seed_gate_baseline.json} \
		--config $${GRAPH2D_SEED_GATE_REGRESSION_CONFIG:-config/graph2d_seed_gate_regression.yaml} \
		--channel standard \
		--use-baseline-as-current
	$(PYTHON) scripts/ci/check_graph2d_seed_gate_regression.py \
		--baseline-json $${GRAPH2D_SEED_GATE_BASELINE_JSON:-config/graph2d_seed_gate_baseline.json} \
		--config $${GRAPH2D_SEED_GATE_REGRESSION_CONFIG:-config/graph2d_seed_gate_regression.yaml} \
		--channel strict \
		--use-baseline-as-current

update-graph2d-seed-gate-baseline: ## 用最新 seed gate summary 刷新稳定基线与日期快照
	@echo "$(GREEN)Updating Graph2D seed gate baseline...$(NC)"
	$(PYTHON) scripts/ci/update_graph2d_seed_gate_baseline.py \
		--standard-summary-json $${GRAPH2D_SEED_GATE_SUMMARY_JSON:-/tmp/graph2d-seed-gate/seed_sweep_summary.json} \
		--strict-summary-json $${GRAPH2D_SEED_GATE_STRICT_SUMMARY_JSON:-/tmp/graph2d-seed-gate-strict/seed_sweep_summary.json} \
		--output-baseline-json $${GRAPH2D_SEED_GATE_BASELINE_JSON:-config/graph2d_seed_gate_baseline.json}

audit-pydantic-v2: ## 审计 Pydantic v2 兼容性风险模式（输出现状）
	@echo "$(GREEN)Auditing pydantic v2 compatibility patterns...$(NC)"
	$(PYTHON) scripts/ci/audit_pydantic_v2.py --roots src

audit-pydantic-v2-regression: ## 基于 baseline 校验 Pydantic v2 兼容性模式不回退
	@echo "$(GREEN)Checking pydantic v2 compatibility regression...$(NC)"
	$(PYTHON) scripts/ci/audit_pydantic_v2.py \
		--roots src \
		--baseline config/pydantic_v2_audit_baseline.json \
		--check-regression

audit-pydantic-style: ## 审计 Pydantic 模型字段/配置风格（输出现状）
	@echo "$(GREEN)Auditing pydantic model style...$(NC)"
	$(PYTHON) scripts/ci/audit_pydantic_model_style.py --roots src

audit-pydantic-style-regression: ## 基于 baseline 校验 Pydantic 模型风格不回退
	@echo "$(GREEN)Checking pydantic model-style regression...$(NC)"
	$(PYTHON) scripts/ci/audit_pydantic_model_style.py \
		--roots src \
		--baseline config/pydantic_model_style_baseline.json \
		--check-regression

test-dedupcad-vision: ## 运行测试（依赖 DedupCAD Vision 已启动）
	@echo "$(GREEN)Running tests with DedupCAD Vision required...$(NC)"
	@echo "$(YELLOW)Ensure dedupcad-vision is running at $${DEDUPCAD_VISION_URL:-http://localhost:58001}$(NC)"
	DEDUPCAD_VISION_REQUIRED=1 \
	PYTHONPATH=$(PWD) \
	DEDUPCAD_VISION_URL=$${DEDUPCAD_VISION_URL:-http://localhost:58001} \
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

test-assembly: ## 运行装配模块测试
	@echo "$(GREEN)Running assembly module tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/assembly -v --cov=$(ASSEMBLY_MODULE)

test-baseline: ## 运行基线评测
	@echo "$(GREEN)Running baseline evaluation...$(NC)"
	$(PYTHON) scripts/run_baseline_evaluation.py

graph2d-freeze-baseline: ## 冻结当前 Graph2D 模型为可追踪基线包
	@echo "$(GREEN)Freezing Graph2D baseline...$(NC)"
	$(PYTHON) scripts/freeze_graph2d_baseline.py --checkpoint $${GRAPH2D_MODEL_PATH:-models/graph2d_merged_latest.pth}

worktree-bootstrap: ## 创建并初始化并行开发 worktree（示例：make worktree-bootstrap BRANCH=feat/x TARGET=../repo-x）
	@echo "$(GREEN)Bootstrapping worktree...$(NC)"
	@test -n "$(BRANCH)" || (echo "$(RED)BRANCH is required$(NC)"; exit 1)
	scripts/bootstrap_worktree.sh "$(BRANCH)" "$${TARGET:-}" "$${BASE:-main}"

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
	GRAPH2D_MIN_CONF=$${GRAPH2D_MIN_CONF:-0.6} uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

chrome-devtools: ## Launch Chrome with remote debugging enabled
	@scripts/chrome_devtools.sh

cdp-console-demo: ## Demo capturing console logs via CDP
	$(PYTHON) scripts/cdp_capture_console.py

cdp-network-demo: ## Demo capturing network events via CDP
	$(PYTHON) scripts/cdp_network_capture.py

cdp-perf-demo: ## Demo capturing performance metrics via CDP
	$(PYTHON) scripts/cdp_performance_capture.py

cdp-response-demo: ## Demo capturing response bodies via CDP
	$(PYTHON) scripts/cdp_response_capture.py

cdp-screenshot-demo: ## Demo capturing a screenshot via CDP
	$(PYTHON) scripts/cdp_screenshot_capture.py

cdp-trace-demo: ## Demo capturing a trace via CDP
	$(PYTHON) scripts/cdp_trace_capture.py

playwright-console-demo: ## Demo capturing console logs via Playwright (optional)
	$(PYTHON) scripts/playwright_console_capture.py

playwright-trace-demo: ## Demo capturing Playwright trace (optional)
	$(PYTHON) scripts/playwright_trace_capture.py

playwright-install: ## Install Playwright browser (requires network)
	$(PYTHON) -m playwright install chromium

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

uvnet-checkpoint-inspect: ## Inspect UV-Net checkpoint config and forward pass
	$(PYTHON) scripts/uvnet_checkpoint_inspect.py --path $(UVNET_CHECKPOINT)

self-check-enhanced: ## Run comprehensive self-check
	@echo "$(GREEN)Running enhanced self-check...$(NC)"
	$(PYTHON) scripts/self_check_enhanced.py

verify-metrics: ## Verify required metrics are exported
	@echo "$(GREEN)Verifying metrics export...$(NC)"
	$(PYTHON) scripts/check_metrics_consistency.py
	$(PYTHON) scripts/verify_metrics_export.py
	@echo "$(GREEN)Metrics export verification passed!$(NC)"

test-targeted: ## Run targeted tests (Faiss health/ETA scheduling)
	@echo "$(GREEN)Running targeted tests...$(NC)"
	$(PYTEST) tests/unit/test_faiss_eta_reset_on_recovery.py \
		tests/unit/test_faiss_health_response.py \
		tests/unit/test_faiss_eta_schedules_on_failed_recovery.py -q || true
	@echo "$(GREEN)Targeted tests completed (allowing skips).$(NC)"

e2e-smoke: ## Run E2E smoke tests against running services
	@echo "$(GREEN)Running E2E smoke tests...$(NC)"
	API_BASE_URL=$${API_BASE_URL:-http://localhost:8000} \
	DEDUPCAD_VISION_URL=$${DEDUPCAD_VISION_URL:-http://localhost:58001} \
	$(PYTEST) tests/integration/test_e2e_api_smoke.py \
		tests/integration/test_dedupcad_vision_contract.py -v -rs

dedup2d-secure-smoke: ## Run Dedup2D secure callback smoke test
	@echo "$(GREEN)Running Dedup2D secure callback smoke test...$(NC)"
	DEDUPCAD_VISION_START=$${DEDUPCAD_VISION_START:-0} \
	DEDUP2D_SECURE_SMOKE_CLEANUP=$${DEDUP2D_SECURE_SMOKE_CLEANUP:-1} \
		scripts/e2e_dedup2d_secure_callback.sh

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
	@docker run --rm --entrypoint promtool -v $(PWD)/docs/prometheus:/rules:ro \
		prom/prometheus:latest \
		check rules /rules/recording_rules.yml || echo "$(YELLOW)Promtool not available$(NC)"

promtool-validate-all: ## 使用 promtool 验证所有规则文件
	@echo "$(GREEN)Validating all Prometheus rules with promtool...$(NC)"
	bash scripts/validate_prometheus.sh

dashboard-import: ## 导入Grafana仪表板
	@echo "$(GREEN)Importing Grafana dashboard...$(NC)"
	@echo "Please ensure Grafana is running on http://localhost:3000"
	@echo "Login with admin/admin and import the dashboard from:"
	@echo "  docs/grafana/observability_dashboard.json"
	@open http://localhost:3000/dashboard/import || echo "Open http://localhost:3000/dashboard/import manually"

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
	$(PYTHON) scripts/cardinality_audit.py --prometheus-url $(PROMETHEUS_URL) --format markdown
	@echo "$(GREEN)Audit complete!$(NC)"

cardinality-check: ## 检查指标基数并生成报告
	@echo "$(GREEN)Checking metrics cardinality...$(NC)"
	$(PYTHON) scripts/cardinality_audit.py \
		--prometheus-url $(PROMETHEUS_URL) \
		--warning-threshold 100 \
		--critical-threshold 1000 \
		--format json \
		--output reports/cardinality_report.json
	@echo "$(GREEN)Report saved to reports/cardinality_report.json$(NC)"

metrics-audit-watch: ## 持续监控指标基数
	@echo "$(GREEN)Starting continuous cardinality monitoring...$(NC)"
	@while true; do \
		clear; \
		$(PYTHON) scripts/cardinality_audit.py --prometheus-url $(PROMETHEUS_URL) --format markdown; \
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

# Graph2D review summarization
GRAPH2D_REVIEW_TEMPLATE ?= reports/experiments/20260123/soft_override_calibrated_added_review_template_20260124.csv
GRAPH2D_REVIEW_OUT_DIR ?= reports/experiments/$$(date +%Y%m%d)

graph2d-review-summary: ## 汇总 Graph2D soft-override 复核模板（生成 summary + correct-label counts）
	@echo "$(GREEN)Summarizing Graph2D soft-override review...$(NC)"
	$(PYTHON) scripts/summarize_soft_override_review.py \
		--review-template "$(GRAPH2D_REVIEW_TEMPLATE)" \
		--summary-out "$(GRAPH2D_REVIEW_OUT_DIR)/soft_override_review_summary.csv" \
		--correct-labels-out "$(GRAPH2D_REVIEW_OUT_DIR)/soft_override_correct_label_counts.csv"

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
	$(PYTHON) scripts/export_eval_metrics.py --push-gateway $${PUSHGATEWAY_URL:-http://localhost:9091}

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
