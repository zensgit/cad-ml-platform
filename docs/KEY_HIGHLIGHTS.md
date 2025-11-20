# CAD ML Platform - Key Highlights & Implementation Map

## 📋 Operational Cadences

### Weekly Retention (✅ Implemented)
- **Make Target**: [`make eval-retention`](../Makefile#L172)
- **Script**: [`scripts/manage_eval_retention.py`](../scripts/manage_eval_retention.py)
- **Automation**: [`.github/workflows/retention-management.yml`](../.github/workflows/retention-management.yml)
- **Status**: Fully automated via GitHub Actions (runs every Sunday)

### Monthly Badge Thresholds (⚠️ Partial)
- **Badge Generator**: [`scripts/generate_badge.py`](../scripts/generate_badge.py)
- **Config**: [`config/eval_frontend.json`](../config/eval_frontend.json) (contains thresholds)
- **Gap**: No monthly review workflow - manual process only
- **Action Required**: See [Monthly Badge Review Workflow](#planned-monthly-badge-workflow)

### Quarterly Baselines (❌ Not Implemented)
- **Current Storage**: [`reports/eval_history/`](../reports/eval_history/)
- **Baseline Script**: [`scripts/anomaly_baseline.py`](../scripts/anomaly_baseline.py)
- **Gap**: No `reports/baselines/` directory or snapshot script
- **Action Required**: See [Baseline Snapshot Implementation](#planned-baseline-snapshots)

## 🚦 CI Failure Routing

### Exit Code Mapping (✅ Implemented)
Source: [`scripts/security_audit.py:418-453`](../scripts/security_audit.py#L418)

| Exit Code | Issue Type | Severity | Owner Team | Response Time |
|-----------|------------|----------|------------|---------------|
| 0 | No issues | - | - | - |
| 1 | General/mixed | Variable | DevOps | Next sprint |
| 2 | Critical vulnerabilities | P0 | Security | Immediate |
| 3 | Exposed secrets | P0 | Security | Immediate |
| 4 | High severity deps | P1 | Platform | Same day |
| 5 | Docker/container | P2 | Infrastructure | This week |
| 6 | Code security | P2 | Development | This sprint |

**Gap**: No formal routing documentation
**Action Required**: See [CI Failure Routing Guide](#planned-ci-routing-guide)

## 🔮 Future Enhancement Status

### Predictive Modeling (❌ Not Implemented)
- **Planned**: ARIMA/Prophet for `combined_score` forecasting
- **Target Output**: `reports/forecast.json`
- **Dependencies**: `statsmodels`, `prophet`
- **Status**: Concept only in [`EVALUATION_MAINTENANCE_GUIDE.md`](EVALUATION_MAINTENANCE_GUIDE.md#predictive-modeling-planned)

### Adaptive Thresholds (✅ Partially Implemented)
- **OCR Implementation**: [`src/core/ocr/rolling_stats.py`](../src/core/ocr/rolling_stats.py)
- **Tests**: [`tests/ocr/test_dynamic_threshold.py`](../tests/ocr/test_dynamic_threshold.py)
- **Anomaly Baseline**: [`scripts/anomaly_baseline.py`](../scripts/anomaly_baseline.py)
- **Gap**: Not integrated with main evaluation pipeline
- **Next Step**: Wire adaptive thresholds into [`scripts/analyze_eval_insights.py`](../scripts/analyze_eval_insights.py)

### Multi-Repo Federation (❌ Not Implemented)
- **Planned Config**: `config/federation.yaml`
- **Planned Script**: `scripts/federate_metrics.py`
- **Status**: Concept only
- **Use Case**: Aggregate health from OCR service, Vision service, etc.

### Lightweight WebUI (⚠️ Partial)
- **Current Interactive HTML**: [`scripts/generate_eval_report_v2.py`](../scripts/generate_eval_report_v2.py)
- **Chart.js Integration**: ✅ Implemented
- **Gap**: No standalone SPA, requires Python generation
- **Next Step**: Create `webapp/` with live JSON consumption

## ✅ Current State Implementation Map

### Data Governance
| Feature | Implementation | Location |
|---------|---------------|----------|
| Schema versioning | ✅ | [`docs/eval_history.schema.json`](../docs/eval_history.schema.json) |
| Unified config | ✅ | [`config/eval_frontend.json`](../config/eval_frontend.json) |
| 5-tier retention | ✅ | [`scripts/manage_eval_retention.py`](../scripts/manage_eval_retention.py) |
| SHA-384 integrity | ✅ | [`scripts/check_integrity.py`](../scripts/check_integrity.py) |

### Observability
| Feature | Implementation | Location |
|---------|---------------|----------|
| Chart.js trends | ✅ | [`scripts/eval_trend.py`](../scripts/eval_trend.py) |
| Interactive reports | ✅ | [`scripts/generate_eval_report_v2.py`](../scripts/generate_eval_report_v2.py) |
| Prometheus export | ✅ | [`scripts/export_eval_metrics.py`](../scripts/export_eval_metrics.py) |
| Metrics endpoint | ✅ | [`src/main.py`](../src/main.py) (`/metrics`) |
| Health endpoints | ✅ | [`src/main.py`](../src/main.py) (`/health`, `/ready`) |
| Vision metrics parity | ✅ | [`src/utils/metrics.py`](../src/utils/metrics.py) (`vision_requests_total`, durations, errors) |
| Input rejection counters | ✅ | [`src/utils/metrics.py`](../src/utils/metrics.py) (`vision_input_rejected_total`, `ocr_input_rejected_total`) |
| Structured error responses | ✅ | Vision & OCR endpoints (JSON with code) |

### Intelligence
| Feature | Implementation | Location |
|---------|---------------|----------|
| Z-score anomalies | ✅ | [`scripts/analyze_eval_insights.py`](../scripts/analyze_eval_insights.py) |
| LLM narratives | ✅ | [`scripts/analyze_eval_insights.py`](../scripts/analyze_eval_insights.py) |
| Trend analysis | ✅ | [`scripts/eval_trend.py`](../scripts/eval_trend.py) |
| Baseline caching | ✅ | [`scripts/anomaly_baseline.py`](../scripts/anomaly_baseline.py) |
| Rolling statistics | ✅ | [`src/core/ocr/rolling_stats.py`](../src/core/ocr/rolling_stats.py) |

### Security
| Feature | Implementation | Location |
|---------|---------------|----------|
| Multi-tool audit | ✅ | [`scripts/security_audit.py`](../scripts/security_audit.py) |
| Chart.js updates | ✅ | [`scripts/check_chartjs_updates.py`](../scripts/check_chartjs_updates.py) |
| Environment verify | ✅ | [`scripts/verify_environment.py`](../scripts/verify_environment.py) |
| Granular exit codes | ✅ | [`scripts/security_audit.py:418`](../scripts/security_audit.py#L418) |
| Payload size guard | ✅ | `VISION_MAX_BASE64_BYTES` in [`src/core/config.py`](../src/core/config.py) |

### Developer Experience
| Feature | Implementation | Location |
|---------|---------------|----------|
| Soft/strict modes | ✅ | [`scripts/validate_eval_history.py`](../scripts/validate_eval_history.py) |
| Pre-commit hooks | ✅ | [`scripts/pre_commit_check.sh`](../scripts/pre_commit_check.sh) |
| GitHub setup | ✅ | [`GITHUB_SETUP.md`](../GITHUB_SETUP.md) |
| JSON outputs | ✅ | All evaluation scripts support `--json` |
| Markdown reports | ✅ | [`scripts/analyze_eval_insights.py`](../scripts/analyze_eval_insights.py) |
| Unified error model | ✅ | `src/api/v1/vision.py`, `src/api/v1/ocr.py` (HTTP 200 + `{success,false,code}`) |

## ⚠️ Known Issues

### API Router Prefix Duplication
- **Issue**: Double prefixing in routes (e.g., `/api/v1/v1/analyze`)
- **Location**: [`src/api/__init__.py`](../src/api/__init__.py) + child routers
- **Fix Required**: Normalize child routers to resource-only prefixes
- **Example**: `router.include_router(analyze.router, prefix="/analyze")` not `"/v1/analyze"`

## 📅 Planned Implementations

### Monthly Badge Workflow
```yaml
# .github/workflows/badge-review.yml (to be created)
name: Monthly Badge Review
on:
  schedule:
    - cron: '0 0 1 * *'  # First day of month
  workflow_dispatch:

jobs:
  review-badges:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Generate badge report
        run: |
          python3 scripts/generate_badge.py --format json > reports/badge_review_$(date +%Y%m).json
      - name: Create review issue
        uses: actions/github-script@v6
        with:
          script: |
            // Create monthly review issue
```

### Baseline Snapshots
```bash
# scripts/snapshot_baseline.py (to be created)
#!/usr/bin/env python3
"""Quarterly baseline snapshot utility."""

import json
import shutil
from datetime import datetime
from pathlib import Path

def snapshot_baseline():
    source = Path("reports/insights/baseline.json")
    if not source.exists():
        print("No baseline to snapshot")
        return

    # Create baselines directory
    baselines_dir = Path("reports/baselines")
    baselines_dir.mkdir(exist_ok=True)

    # Generate quarterly filename
    quarter = (datetime.now().month - 1) // 3 + 1
    year = datetime.now().year
    dest = baselines_dir / f"baseline_{year}_Q{quarter}.json"

    # Copy with metadata
    shutil.copy2(source, dest)
    print(f"Baseline snapshot saved to {dest}")
```

### CI Routing Guide
```markdown
# docs/CI_FAILURE_ROUTING.md (to be created)

## Security Audit Exit Code Routing

### Automated Routing Rules

| Exit Code | GitHub Team | Slack Channel | PagerDuty |
|-----------|-------------|---------------|-----------|
| 2 | @security-team | #security-critical | Yes |
| 3 | @security-team | #security-critical | Yes |
| 4 | @platform-team | #deps-updates | No |
| 5 | @infra-team | #docker-security | No |
| 6 | @dev-team | #code-quality | No |

### Implementation in CI
...
```

## 🎯 Quick Actions

```bash
# Verify all scripts are present
find scripts -name "*.py" | wc -l  # Should be 20+

# Check configuration consistency
python3 scripts/validate_eval_history.py --config config/eval_frontend.json

# Test exit code routing
python3 scripts/security_audit.py --severity critical; echo "Exit code: $?"

# Generate current badges
make badges

# Run full Phase 6 workflow
make eval-phase6
```

## 📊 Metrics

### Coverage Status
- **Operational Features**: 85% implemented
- **Future Enhancements**: 25% implemented
- **Documentation**: 95% complete
- **Tests**: 70% coverage

### Priority Matrix
| Priority | Feature | Status | ETA |
|----------|---------|--------|-----|
| P0 | CI routing guide | Planned | This week |
| P1 | Baseline snapshots | Planned | This week |
| P1 | Router prefix fix | Planned | This week |
| P2 | Monthly badge workflow | Planned | Next sprint |
| P3 | Predictive modeling | Concept | Q2 2025 |
| P3 | Multi-repo federation | Concept | Q2 2025 |
| P3 | WebUI SPA | Partial | Q3 2025 |

---

*Generated: 2025-11-19*
*Next Update: After implementing planned items*
### Vision Base64 Size Configuration Visibility
- **Current**: Limit hard-coded via settings (1MB) not exposed in `/health`
- **Action**: Add `vision_max_base64_bytes` to health response for ops clarity
