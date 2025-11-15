# CAD ML Platform - Makefile
# 统一的开发工作流

.PHONY: help install dev test lint format type-check clean run docs docker

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

lint: ## 运行代码检查
	@echo "$(GREEN)Running linters...$(NC)"
	$(FLAKE8) $(SRC_DIR) --max-line-length=100 --ignore=E203,W503
	@echo "$(GREEN)Linting passed!$(NC)"

format: ## 格式化代码
	@echo "$(GREEN)Formatting code...$(NC)"
	$(BLACK) $(SRC_DIR) $(TEST_DIR) --line-length=100
	$(ISORT) $(SRC_DIR) $(TEST_DIR) --profile black --line-length=100
	@echo "$(GREEN)Code formatted!$(NC)"

type-check: ## 类型检查
	@echo "$(GREEN)Type checking...$(NC)"
	$(MYPY) $(SRC_DIR) --ignore-missing-imports --strict

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
	@echo "$(GREEN)All golden evaluations complete!$(NC)"

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
metrics-export: ## 导出Prometheus指标
	@echo "$(GREEN)Exporting metrics...$(NC)"
	$(PYTHON) scripts/export_metrics.py

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

# 预提交钩子
pre-commit: ## 运行预提交检查
	@echo "$(GREEN)Running pre-commit checks...$(NC)"
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test
	@echo "$(GREEN)Ready to commit!$(NC)"

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