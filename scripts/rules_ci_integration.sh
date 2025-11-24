#!/bin/bash
# Recording Rules CI/CD Integration Script
# 集成录制规则版本管理到 CI/CD 流程

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "Checking dependencies..."

    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi

    # 检查 promtool (可选)
    if command -v promtool &> /dev/null; then
        log_info "promtool found, will use for validation"
    else
        log_warning "promtool not found, using basic validation only"
    fi

    # 检查 Git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed"
        exit 1
    fi

    log_success "All required dependencies are available"
}

# 验证规则文件
validate_rules() {
    log_info "Validating recording rules..."

    cd "$PROJECT_ROOT"

    # 使用版本管理工具验证
    python3 scripts/recording_rules_versioning.py init --rules-dir prometheus/rules

    # 验证所有规则文件
    for rule_file in prometheus/rules/*.yml prometheus/rules/*.yaml; do
        if [ -f "$rule_file" ]; then
            log_info "Validating $(basename "$rule_file")..."

            # 基本 YAML 验证
            python3 -c "import yaml; yaml.safe_load(open('$rule_file'))" || {
                log_error "YAML validation failed for $rule_file"
                exit 1
            }

            # 如果有 promtool，使用它验证
            if command -v promtool &> /dev/null; then
                promtool check rules "$rule_file" || {
                    log_error "promtool validation failed for $rule_file"
                    exit 1
                }
            fi
        fi
    done

    log_success "All rules validated successfully"
}

# 创建版本
create_version() {
    local message="${1:-CI/CD automated version}"
    local author="${2:-CI/CD Bot}"

    log_info "Creating new version..."

    cd "$PROJECT_ROOT"

    # 获取 Git 信息
    if [ -d .git ]; then
        commit_hash=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        message="$message (commit: $commit_hash, branch: $branch)"
    fi

    # 创建版本
    output=$(python3 scripts/recording_rules_versioning.py commit \
        -m "$message" \
        -a "$author" 2>&1)

    if echo "$output" | grep -q "No changes to commit"; then
        log_info "No changes detected, skipping version creation"
        return 0
    elif echo "$output" | grep -q "Created version"; then
        version_id=$(echo "$output" | grep "Created version" | awk '{print $NF}')
        log_success "Created version: $version_id"
        echo "$version_id" > .last_rules_version
        return 0
    else
        log_error "Failed to create version: $output"
        return 1
    fi
}

# 检查变更
check_changes() {
    log_info "Checking for rule changes..."

    cd "$PROJECT_ROOT"

    # 生成差异报告
    if [ -f .last_rules_version ]; then
        last_version=$(cat .last_rules_version)
        current_version=$(python3 -c "
import json
with open('.rules-versions/metadata.json') as f:
    print(json.load(f).get('current_version', 'unknown'))
        " 2>/dev/null || echo "unknown")

        if [ "$last_version" != "$current_version" ]; then
            log_info "Generating diff report..."
            python3 scripts/recording_rules_versioning.py diff \
                "$last_version" "$current_version" > rules_diff.txt

            log_success "Diff report saved to rules_diff.txt"
            return 0
        fi
    fi

    log_info "No previous version found for comparison"
    return 1
}

# 部署规则
deploy_rules() {
    local prometheus_url="${1:-http://localhost:9090}"

    log_info "Deploying rules to Prometheus..."

    # 检查 Prometheus 是否可达
    if ! curl -s "${prometheus_url}/-/healthy" > /dev/null; then
        log_error "Prometheus is not reachable at $prometheus_url"
        return 1
    fi

    # 重新加载 Prometheus 配置
    curl -X POST "${prometheus_url}/-/reload" || {
        log_error "Failed to reload Prometheus configuration"
        return 1
    }

    log_success "Rules deployed and Prometheus reloaded"
}

# 回滚版本
rollback_version() {
    local version_id="${1}"

    if [ -z "$version_id" ]; then
        log_error "Version ID is required for rollback"
        exit 1
    fi

    log_info "Rolling back to version $version_id..."

    cd "$PROJECT_ROOT"

    python3 scripts/recording_rules_versioning.py rollback "$version_id" || {
        log_error "Rollback failed"
        exit 1
    }

    log_success "Successfully rolled back to version $version_id"
}

# 生成报告
generate_report() {
    log_info "Generating version management report..."

    cd "$PROJECT_ROOT"

    # 生成 Markdown 报告
    python3 scripts/recording_rules_versioning.py report \
        --format markdown > rules_version_report.md

    # 生成 JSON 报告
    python3 scripts/recording_rules_versioning.py report \
        --format json > rules_version_report.json

    log_success "Reports generated: rules_version_report.md and rules_version_report.json"
}

# CI 流程
run_ci_pipeline() {
    log_info "Starting CI pipeline for recording rules..."

    # 1. 检查依赖
    check_dependencies

    # 2. 验证规则
    validate_rules

    # 3. 创建版本
    create_version "CI automated version at $(date '+%Y-%m-%d %H:%M:%S')"

    # 4. 检查变更
    check_changes || true

    # 5. 生成报告
    generate_report

    log_success "CI pipeline completed successfully"
}

# CD 流程
run_cd_pipeline() {
    local prometheus_url="${1:-http://localhost:9090}"

    log_info "Starting CD pipeline for recording rules..."

    # 1. 验证规则
    validate_rules

    # 2. 部署规则
    deploy_rules "$prometheus_url"

    # 3. 更新版本记录
    create_version "CD deployment at $(date '+%Y-%m-%d %H:%M:%S')"

    log_success "CD pipeline completed successfully"
}

# 主函数
main() {
    case "${1:-}" in
        validate)
            validate_rules
            ;;
        version)
            create_version "${2:-Manual version}" "${3:-$USER}"
            ;;
        check)
            check_changes
            ;;
        deploy)
            deploy_rules "${2:-http://localhost:9090}"
            ;;
        rollback)
            rollback_version "$2"
            ;;
        report)
            generate_report
            ;;
        ci)
            run_ci_pipeline
            ;;
        cd)
            run_cd_pipeline "${2:-http://localhost:9090}"
            ;;
        *)
            echo "Usage: $0 {validate|version|check|deploy|rollback|report|ci|cd} [args...]"
            echo ""
            echo "Commands:"
            echo "  validate           Validate all rule files"
            echo "  version [msg] [author]  Create new version"
            echo "  check              Check for changes since last version"
            echo "  deploy [url]       Deploy rules to Prometheus"
            echo "  rollback <version> Rollback to specific version"
            echo "  report             Generate version report"
            echo "  ci                 Run CI pipeline"
            echo "  cd [url]          Run CD pipeline"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"