#!/bin/bash
# Daily checkpoint script for 6-day development plan
# Run at 4pm each day to track progress

set -e

DAY=${1:-1}
REPORT_DIR="reports/daily_checkpoints"
mkdir -p "$REPORT_DIR"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
REPORT_FILE="$REPORT_DIR/day${DAY}_checkpoint_${TIMESTAMP}.md"

echo "=== Day $DAY Checkpoint ($(date '+%Y-%m-%d %H:%M:%S')) ===" | tee "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

# 1. Task completion rate
echo "## 📋 Task Completion" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

# Parse IMPLEMENTATION_TODO.md for Day N tasks
if [ -f "IMPLEMENTATION_TODO.md" ]; then
    TODO_COUNT=$(grep -c "^\- \[ \]" IMPLEMENTATION_TODO.md | head -1 || echo "0")
    DONE_COUNT=$(grep -c "^\- \[x\]" IMPLEMENTATION_TODO.md | head -1 || echo "0")
    TOTAL=$((TODO_COUNT + DONE_COUNT))

    if [ "$TOTAL" -gt 0 ]; then
        COMPLETION=$((DONE_COUNT * 100 / TOTAL))
        echo "✅ Completed: $DONE_COUNT / $TOTAL ($COMPLETION%)" | tee -a "$REPORT_FILE"
    else
        echo "⚠️  No tasks found in IMPLEMENTATION_TODO.md" | tee -a "$REPORT_FILE"
    fi
else
    echo "⚠️  IMPLEMENTATION_TODO.md not found" | tee -a "$REPORT_FILE"
fi
echo "" | tee -a "$REPORT_FILE"

# 2. Test statistics
echo "## 🧪 Test Status" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

if command -v pytest &> /dev/null; then
    echo "Running test suite..." | tee -a "$REPORT_FILE"

    # Run tests with brief output
    TEST_OUTPUT=$(pytest -v --tb=no -q 2>&1 || true)
    PASSED=$(echo "$TEST_OUTPUT" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+" || echo "0")
    FAILED=$(echo "$TEST_OUTPUT" | grep -oE "[0-9]+ failed" | grep -oE "[0-9]+" || echo "0")
    SKIPPED=$(echo "$TEST_OUTPUT" | grep -oE "[0-9]+ skipped" | grep -oE "[0-9]+" || echo "0")

    echo "✅ Passed: $PASSED" | tee -a "$REPORT_FILE"
    echo "❌ Failed: $FAILED" | tee -a "$REPORT_FILE"
    echo "⏭️  Skipped: $SKIPPED" | tee -a "$REPORT_FILE"

    if [ "$FAILED" -gt 0 ]; then
        echo "⚠️  WARNING: $FAILED tests failing!" | tee -a "$REPORT_FILE"
    fi
else
    echo "⚠️  pytest not installed" | tee -a "$REPORT_FILE"
fi
echo "" | tee -a "$REPORT_FILE"

# 3. Code coverage
echo "## 📊 Code Coverage" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

if command -v pytest &> /dev/null && python3 -c "import pytest_cov" 2>/dev/null; then
    echo "Generating coverage report..." | tee -a "$REPORT_FILE"

    COVERAGE_OUTPUT=$(pytest --cov=src --cov-report=term-missing --tb=no -q 2>&1 || true)
    COVERAGE_PCT=$(echo "$COVERAGE_OUTPUT" | grep "^TOTAL" | awk '{print $NF}' || echo "N/A")

    echo "📈 Total Coverage: $COVERAGE_PCT" | tee -a "$REPORT_FILE"

    # Extract coverage percentage as number
    if [[ "$COVERAGE_PCT" =~ ([0-9]+)% ]]; then
        COV_NUM="${BASH_REMATCH[1]}"
        if [ "$COV_NUM" -lt 80 ]; then
            echo "⚠️  Coverage below 80% threshold" | tee -a "$REPORT_FILE"
        fi
    fi
else
    echo "⚠️  pytest-cov not installed" | tee -a "$REPORT_FILE"
fi
echo "" | tee -a "$REPORT_FILE"

# 4. New metrics count
echo "## 📏 Metrics Status" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

if [ -f "src/utils/analysis_metrics.py" ]; then
    COUNTER_COUNT=$(grep -c "= Counter(" src/utils/analysis_metrics.py || echo "0")
    HISTOGRAM_COUNT=$(grep -c "= Histogram(" src/utils/analysis_metrics.py || echo "0")
    GAUGE_COUNT=$(grep -c "= Gauge(" src/utils/analysis_metrics.py || echo "0")
    TOTAL_METRICS=$((COUNTER_COUNT + HISTOGRAM_COUNT + GAUGE_COUNT))

    echo "📊 Total Metrics: $TOTAL_METRICS" | tee -a "$REPORT_FILE"
    echo "  - Counters: $COUNTER_COUNT" | tee -a "$REPORT_FILE"
    echo "  - Histograms: $HISTOGRAM_COUNT" | tee -a "$REPORT_FILE"
    echo "  - Gauges: $GAUGE_COUNT" | tee -a "$REPORT_FILE"
else
    echo "⚠️  src/utils/analysis_metrics.py not found" | tee -a "$REPORT_FILE"
fi
echo "" | tee -a "$REPORT_FILE"

# 5. New endpoints count
echo "## 🌐 API Endpoints" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

if [ -d "src/api/v1" ]; then
    ENDPOINT_COUNT=$(grep -r "@router\." src/api/v1/*.py 2>/dev/null | wc -l | xargs)
    echo "🔗 Total Endpoints: $ENDPOINT_COUNT" | tee -a "$REPORT_FILE"

    # Breakdown by HTTP method
    GET_COUNT=$(grep -r "@router\.get" src/api/v1/*.py 2>/dev/null | wc -l | xargs)
    POST_COUNT=$(grep -r "@router\.post" src/api/v1/*.py 2>/dev/null | wc -l | xargs)
    PUT_COUNT=$(grep -r "@router\.put" src/api/v1/*.py 2>/dev/null | wc -l | xargs)
    DELETE_COUNT=$(grep -r "@router\.delete" src/api/v1/*.py 2>/dev/null | wc -l | xargs)

    echo "  - GET: $GET_COUNT" | tee -a "$REPORT_FILE"
    echo "  - POST: $POST_COUNT" | tee -a "$REPORT_FILE"
    echo "  - PUT: $PUT_COUNT" | tee -a "$REPORT_FILE"
    echo "  - DELETE: $DELETE_COUNT" | tee -a "$REPORT_FILE"
else
    echo "⚠️  src/api/v1 directory not found" | tee -a "$REPORT_FILE"
fi
echo "" | tee -a "$REPORT_FILE"

# 6. Performance comparison (if applicable)
echo "## ⚡ Performance" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

PERF_BASELINE="reports/performance_baseline_day0.json"
if [ -f "$PERF_BASELINE" ]; then
    echo "📊 Baseline exists, comparison available" | tee -a "$REPORT_FILE"
    # TODO: Implement performance comparison logic
else
    echo "⚠️  No baseline found at $PERF_BASELINE" | tee -a "$REPORT_FILE"
fi
echo "" | tee -a "$REPORT_FILE"

# 7. Blocking issues
echo "## 🚨 Blocking Issues" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

BLOCKING_FOUND=false

# Check for failing tests
if [ "$FAILED" -gt 0 ]; then
    echo "❌ $FAILED tests failing (blocking if P0 features)" | tee -a "$REPORT_FILE"
    BLOCKING_FOUND=true
fi

# Check for coverage below threshold
if [[ "$COV_NUM" -lt 80 ]] 2>/dev/null; then
    echo "⚠️  Coverage ${COV_NUM}% below 80% threshold (warning)" | tee -a "$REPORT_FILE"
fi

# Check for syntax errors
if ! python3 -m py_compile src/**/*.py 2>/dev/null; then
    echo "❌ Syntax errors detected (blocking)" | tee -a "$REPORT_FILE"
    BLOCKING_FOUND=true
fi

if [ "$BLOCKING_FOUND" = false ]; then
    echo "✅ No blocking issues detected" | tee -a "$REPORT_FILE"
fi
echo "" | tee -a "$REPORT_FILE"

# 8. Summary and recommendations
echo "## 💡 Recommendations" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

if [ "$FAILED" -gt 0 ]; then
    echo "1. Address failing tests before proceeding to next phase" | tee -a "$REPORT_FILE"
fi

if [[ "$COV_NUM" -lt 80 ]] 2>/dev/null; then
    echo "2. Add tests to improve coverage to ≥80%" | tee -a "$REPORT_FILE"
fi

if [ "$TODO_COUNT" -gt 10 ]; then
    echo "3. Many tasks remaining ($TODO_COUNT), consider extending timeline" | tee -a "$REPORT_FILE"
fi

echo "" | tee -a "$REPORT_FILE"
echo "---" | tee -a "$REPORT_FILE"
echo "Report saved to: $REPORT_FILE" | tee -a "$REPORT_FILE"
echo "Next checkpoint: Day $((DAY + 1)) at 4pm" | tee -a "$REPORT_FILE"

# Exit code reflects blocking status
if [ "$BLOCKING_FOUND" = true ]; then
    exit 1
else
    exit 0
fi
