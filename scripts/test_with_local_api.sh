#!/usr/bin/env bash
set -euo pipefail

SUITE="all"
BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
API_KEY_VALUE="${API_KEY:-test-api-key}"
WAIT_SECONDS=60
LOG_PATH="/tmp/cad_ml_uvicorn.log"
PYTHON_BIN="${PYTHON_BIN:-}"
PYTEST_BIN="${PYTEST_BIN:-}"
SERVER_PID=""
STARTED_LOCAL_API=0
# Contract tests can run in-process (TestClient fallback) when local port
# binding is not permitted. In CI we keep strict behavior (no fallback).
CONTRACT_INPROCESS_FALLBACK="${CONTRACT_INPROCESS_FALLBACK:-auto}"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run tiered tests with optional local API auto-start.

Options:
    --suite SUITE       Test suite to run: unit, contract, e2e, all (default: all)
    --base-url URL      API base URL (default: http://127.0.0.1:8000)
    --api-key KEY       API key for authentication (default: test-api-key)
    --wait-seconds SEC  Max seconds to wait for server readiness (default: 60)
    --wait SEC          Backward-compatible alias for --wait-seconds
    --log-path PATH     Uvicorn log output path (default: /tmp/cad_ml_uvicorn.log)
    --python-bin PATH   Python executable used for local uvicorn startup
    --pytest-bin CMD    Pytest command (default: .venv/bin/pytest if present, else pytest)
    --help              Show this help message

Examples:
    $0 --suite unit
    $0 --suite contract --wait-seconds 90
    $0 --suite e2e --base-url http://127.0.0.1:8000

Environment:
    CONTRACT_INPROCESS_FALLBACK
        auto (default): allow contract-only fallback when not in CI
        true: always allow fallback for contract suite
        false: never allow fallback (fail if API can't start)
EOF
    exit 0
}

default_python_bin() {
    if [[ -n "${PYTHON_BIN}" ]]; then
        return
    fi
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON_BIN=".venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        PYTHON_BIN="python"
    fi
}

default_pytest_bin() {
    if [[ -n "${PYTEST_BIN}" ]]; then
        return
    fi
    # Prefer venv pytest so tests run with the same dependency set as the app.
    if [[ -x ".venv/bin/pytest" ]]; then
        PYTEST_BIN=".venv/bin/pytest"
    elif command -v pytest >/dev/null 2>&1; then
        PYTEST_BIN="$(command -v pytest)"
    else
        PYTEST_BIN="pytest"
    fi
}

is_local_base_url() {
    case "${BASE_URL}" in
        http://127.0.0.1:*|http://localhost:*|https://127.0.0.1:*|https://localhost:*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_server_healthy() {
    curl -fsS -H "X-API-Key: ${API_KEY_VALUE}" "${BASE_URL}/health" >/dev/null 2>&1
}

contract_inprocess_fallback_enabled() {
    if [[ "${SUITE}" != "contract" ]]; then
        return 1
    fi
    case "${CONTRACT_INPROCESS_FALLBACK}" in
        true) return 0 ;;
        false) return 1 ;;
        auto)
            # CI should fail fast if the API cannot start; local dev can fall back.
            [[ -z "${CI:-}" ]]
            return $?
            ;;
        *) return 1 ;;
    esac
}

should_contract_fallback_now() {
    if ! contract_inprocess_fallback_enabled; then
        return 1
    fi
    # Prefer falling back only for known "can't bind" environments.
    if [[ ! -s "${LOG_PATH}" ]]; then
        return 0
    fi
    grep -Eiq "operation not permitted|permission denied|not permitted" "${LOG_PATH}"
}

parse_host_port() {
    local without_scheme host_port host port
    without_scheme="${BASE_URL#http://}"
    without_scheme="${without_scheme#https://}"
    host_port="${without_scheme%%/*}"
    host="${host_port%%:*}"
    if [[ "${host_port}" == *":"* ]]; then
        port="${host_port##*:}"
    else
        port="8000"
    fi
    echo "${host}" "${port}"
}

cleanup() {
    if [[ "${STARTED_LOCAL_API}" -eq 1 && -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}

ensure_api() {
    if is_server_healthy; then
        echo "Using existing healthy API at ${BASE_URL}"
        return
    fi

    if ! is_local_base_url; then
        echo "ERROR: API at ${BASE_URL} is not healthy and is not a local URL."
        echo "Please start the remote API manually before running suite '${SUITE}'."
        exit 1
    fi

    default_python_bin
    read -r host port <<<"$(parse_host_port)"
    echo "Starting local API via uvicorn (${host}:${port})"
    "${PYTHON_BIN}" -m uvicorn src.main:app --host "${host}" --port "${port}" >"${LOG_PATH}" 2>&1 &
    SERVER_PID=$!
    STARTED_LOCAL_API=1

    for i in $(seq 1 "${WAIT_SECONDS}"); do
        if is_server_healthy; then
            echo "Local API is healthy."
            return
        fi
        if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
            echo "ERROR: Local API exited early. Last logs:"
            tail -n 80 "${LOG_PATH}" || true
            if should_contract_fallback_now; then
                echo "WARN: Falling back to in-process contract assertions (TestClient)."
                echo "WARN: Live-schema contract checks will be skipped without a running server."
                return
            fi
            exit 1
        fi
        sleep 1
    done

    echo "ERROR: API not ready after ${WAIT_SECONDS}s."
    tail -n 80 "${LOG_PATH}" || true
    if should_contract_fallback_now; then
        echo "WARN: Falling back to in-process contract assertions (TestClient)."
        echo "WARN: Live-schema contract checks will be skipped without a running server."
        return
    fi
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --suite) SUITE="$2"; shift 2 ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        --api-key) API_KEY_VALUE="$2"; shift 2 ;;
        --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
        --wait) WAIT_SECONDS="$2"; shift 2 ;;
        --log-path) LOG_PATH="$2"; shift 2 ;;
        --python-bin) PYTHON_BIN="$2"; shift 2 ;;
        --pytest-bin) PYTEST_BIN="$2"; shift 2 ;;
        --help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ ! "${WAIT_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --wait-seconds must be a positive integer"
    exit 1
fi
default_pytest_bin

trap cleanup EXIT

echo "=== Tiered Test Runner ==="
echo "Base URL: $BASE_URL"
echo "Suite: $SUITE"
echo "Log Path: $LOG_PATH"
echo ""

export API_BASE_URL="$BASE_URL"
export API_KEY="$API_KEY_VALUE"

case $SUITE in
    unit)
        "${PYTEST_BIN}" tests/unit -v --tb=short
        ;;
    contract)
        ensure_api
        "${PYTEST_BIN}" tests/contract -v --tb=short
        ;;
    e2e)
        ensure_api
        "${PYTEST_BIN}" tests/e2e -v --tb=short
        ;;
    all)
        ensure_api
        "${PYTEST_BIN}" tests -v --tb=short
        ;;
    *)
        echo "ERROR: Unknown suite '${SUITE}'"
        echo "Valid suites: unit, contract, e2e, all"
        exit 1
        ;;
esac

echo ""
echo "=== Tests completed ==="
