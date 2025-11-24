#!/bin/bash
# Progress tracking script - shows quick stats
# Usage: ./scripts/track_progress.sh [day_number]

set -e

DAY=${1:-"current"}

echo "=== CAD ML Platform Progress Report (Day $DAY) ==="
echo ""

# 1. Test count
if command -v pytest &> /dev/null; then
    TEST_TOTAL=$(pytest --co -q 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo "0")
    echo "Tests: $TEST_TOTAL total"
else
    echo "Tests: pytest not available"
fi

# 2. Coverage
if command -v pytest &> /dev/null && python3 -c "import pytest_cov" 2>/dev/null; then
    COVERAGE=$(pytest --cov=src --cov-report=term-missing --tb=no -q 2>&1 | grep "^TOTAL" | awk '{print $NF}' || echo "N/A")
    echo "Coverage: $COVERAGE"
else
    echo "Coverage: pytest-cov not available"
fi

# 3. Metrics
if [ -f "src/utils/analysis_metrics.py" ]; then
    METRICS=$(grep -cE "= (Counter|Histogram|Gauge)\(" src/utils/analysis_metrics.py || echo "0")
    echo "Metrics: $METRICS defined"
else
    echo "Metrics: analysis_metrics.py not found"
fi

# 4. Endpoints
if [ -d "src/api/v1" ]; then
    ENDPOINTS=$(grep -r "@router\." src/api/v1/*.py 2>/dev/null | wc -l | xargs)
    echo "Endpoints: $ENDPOINTS total"
else
    echo "Endpoints: api/v1 not found"
fi

# 5. Files modified today
if [ -d ".git" ]; then
    TODAY=$(date +"%Y-%m-%d")
    MODIFIED=$(git diff --name-only HEAD | wc -l | xargs)
    echo "Modified: $MODIFIED files (uncommitted)"
else
    echo "Modified: not a git repo"
fi

echo ""
echo "==="

# Show feature flags status
if [ -f "config/feature_flags.py" ]; then
    echo ""
    echo "Feature Flags:"
    python3 << 'PYTHON'
import sys
sys.path.insert(0, '.')
try:
    from config.feature_flags import get_feature_flags
    flags = get_feature_flags()
    for key, value in flags.items():
        status = "✓" if value else "✗"
        if isinstance(value, str):
            print(f"  {status} {key}: {value}")
        elif isinstance(value, bool):
            print(f"  {status} {key}")
        else:
            print(f"  • {key}: {value}")
except Exception as e:
    print(f"  Error loading feature flags: {e}")
PYTHON
fi
