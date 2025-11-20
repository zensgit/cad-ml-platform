# CAD ML Platform - Complete Evaluation System Guide

**Version**: 3.0.0
**Last Updated**: 2025-11-19
**Status**: Production Ready

---

## 📦 Version and Compatibility

### Dependencies
- **Chart.js**: v4.4.0 (locked version for stability)
  - CDN: `https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js`
  - Local backup: `reports/eval_history/report/assets/chart.min.js`
  - SHA-384: `vxfkwNn5NGXgAZEPcjDEwOXrc6Q3gSYPgfn8R6J0R6qKgNW2pV4s3u8V5Zg7Xk6S`
- **Python**: 3.10+ (3.13 tested)
- **GitHub Actions**: v3/v4 compatible

### Breaking Change Policy
- Chart.js updates require testing interactive features
- Schema version changes trigger automatic migration
- Notification API changes require environment variable updates

---

## 🎯 Executive Overview

The CAD ML Platform now features a comprehensive evaluation and monitoring system that provides:
- **Automated evaluation** of Vision and OCR modules
- **Interactive reporting** with Chart.js visualizations
- **CI/CD integration** via GitHub Actions
- **Multi-channel notifications** (Slack, Email, GitHub)
- **Data retention policies** for long-term storage optimization
- **Advanced filtering** and trend analysis capabilities

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                        │
├────────────────┬────────────────┬────────────────────────────┤
│  HTML Reports  │  GitHub Pages  │  Slack/Email Notifications │
└────────────────┴────────────────┴────────────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Pipeline                        │
├────────────────┬────────────────┬────────────────────────────┤
│ Vision Golden  │  OCR Golden    │  Combined Score Calculator │
└────────────────┴────────────────┴────────────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────────┐
│                     Data Management                           │
├────────────────┬────────────────┬────────────────────────────┤
│ Schema v1.0.0  │ Retention Policy│  Historical Analysis      │
└────────────────┴────────────────┴────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Basic Evaluation

```bash
# Run combined evaluation
make eval-combined

# Generate interactive report
make eval-report-v2

# Check system health
make health-check
```

### 2. CI/CD Evaluation

```yaml
# Automatic on push/PR
git push origin main

# Manual trigger with custom thresholds
gh workflow run evaluation-report.yml \
  -f min_combined=0.85 \
  -f min_vision=0.70 \
  -f min_ocr=0.92
```

### 3. Notification Setup

```bash
# Set environment variables for notifications
export EVAL_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
export EVAL_SMTP_HOST="smtp.gmail.com"
export EVAL_EMAIL_TO="team@example.com"

# Test notifications
python3 scripts/notify_eval_results.py --dry-run
```

---

## 📈 Feature Details

### 1. Combined Evaluation System

**Script**: `scripts/evaluate_vision_ocr_combined.py`

The combined evaluation calculates a weighted score:
```
Combined Score = w_vision × vision_score + w_ocr × ocr_normalized
OCR Normalized = recall × (1 - brier_score)
```

**Features**:
- Configurable weights (default 50/50)
- Threshold validation
- Git metadata tracking
- Schema versioning (v1.0.0)

**Usage**:
```bash
# Basic evaluation
python3 scripts/evaluate_vision_ocr_combined.py

# With custom thresholds
MIN_COMBINED=0.85 MIN_VISION=0.7 MIN_OCR=0.92 make ci-combined-check

# Save to history
python3 scripts/evaluate_vision_ocr_combined.py --save-history
```

### 2. Interactive HTML Reports

**Script**: `scripts/generate_eval_report_v2.py`

Enhanced reporting with Chart.js integration:
- **Three-tier fallback**: CDN → Local → PNG
- **Interactive charts**: Zoom, pan, filter
- **Advanced filtering**: Branch, date range
- **Responsive design**: Mobile-friendly

**Usage**:
```bash
# Generate with CDN (recommended)
make eval-report-v2

# Generate with local Chart.js
python3 scripts/generate_eval_report_v2.py --local-chartjs assets/chart.min.js

# With privacy options
python3 scripts/generate_eval_report_v2.py --redact-branch --redact-commit
```

### 3. Data Retention Management

**Script**: `scripts/manage_eval_retention.py`

#### Tiered Retention Policy (5 Layers)

| Age Range | Retention Rule | Description | Example |
|-----------|---------------|-------------|---------|
| **0-7 days** | Keep ALL | Full resolution, every evaluation preserved | 100 evals → 100 kept |
| **8-30 days** | Daily snapshots | Keep latest eval per day | 50 evals → ~22 kept |
| **31-90 days** | Weekly snapshots | Keep latest eval per week | 100 evals → ~8 kept |
| **91-365 days** | Monthly snapshots | Keep latest eval per month | 300 evals → ~9 kept |
| **>365 days** | Quarterly snapshots | Keep latest eval per quarter | 1000 evals → ~4 kept |

#### Selection Algorithm
```python
# For each retention period:
1. Group files by time period (day/week/month/quarter)
2. Select the LATEST file from each group
3. Mark older files for deletion
4. Apply per-branch to preserve branch-specific history
```

#### Custom Configuration Example
```python
# In scripts/manage_eval_retention.py
TIERS = [
    {"name": "full", "days": 7, "keep": "all"},           # Layer 1
    {"name": "daily", "days": 30, "keep": "daily"},       # Layer 2
    {"name": "weekly", "days": 90, "keep": "weekly"},     # Layer 3
    {"name": "monthly", "days": 365, "keep": "monthly"},  # Layer 4
    {"name": "quarterly", "days": float('inf'), "keep": "quarterly"} # Layer 5
]
```

**Usage**:
```bash
# Check retention status (dry run)
make eval-retention

# Apply retention with archiving
make eval-retention-apply  # Prompts for confirmation

# Manual execution
python3 scripts/manage_eval_retention.py --execute --archive
```

### 4. Multi-Channel Notifications

**Script**: `scripts/notify_eval_results.py`

Supports multiple channels:
- **Slack**: Webhook integration with rich formatting
- **Email**: HTML/text emails via SMTP
- **GitHub**: Automatic issue creation on breaches

**Configuration**:
```bash
# Slack
export EVAL_SLACK_WEBHOOK="https://hooks.slack.com/..."

# Email
export EVAL_SMTP_HOST="smtp.gmail.com"
export EVAL_SMTP_PORT="587"
export EVAL_EMAIL_FROM="alerts@example.com"
export EVAL_EMAIL_TO="team@example.com,manager@example.com"
export EVAL_EMAIL_PASSWORD="app-specific-password"

# GitHub
export GITHUB_TOKEN="ghp_..."
export GITHUB_REPOSITORY="org/repo"
```

**Usage**:
```bash
# Send to all configured channels
python3 scripts/notify_eval_results.py --channel all

# Only on threshold breach
python3 scripts/notify_eval_results.py --threshold-breach-only

# Dry run (preview without sending)
python3 scripts/notify_eval_results.py --dry-run
```

### 5. CI/CD Integration

**Workflow**: `.github/workflows/evaluation-report.yml`

Features:
- **Multiple triggers**: push, PR, schedule, manual
- **Artifact upload**: Reports and history files
- **PR comments**: Automatic result tables
- **GitHub Pages**: Auto-deployment for main branch
- **Job summaries**: Quick overview in Actions UI

**Scheduled Runs**:
```yaml
schedule:
  - cron: '0 2 * * *'  # Daily at 2 AM UTC
```

**Manual Trigger**:
```bash
gh workflow run evaluation-report.yml \
  -f min_combined=0.80 \
  -f min_vision=0.65 \
  -f min_ocr=0.90
```

### 6. Schema Versioning & Migration

**Script**: `scripts/validate_eval_history.py`

Schema v1.0.0 includes:
- `schema_version`: Version identifier
- `run_context`: Environment metadata
  - runner: local/ci
  - machine: hostname
  - os: platform info
  - python: version
  - ci_job_id: GitHub Actions ID
  - ci_workflow: Workflow name

**Usage**:
```bash
# Validate all history files
make eval-validate

# Migrate legacy files
make eval-migrate  # Creates .bak backups

# Manual validation
python3 scripts/validate_eval_history.py --strict
```

---

## 📋 Complete Command Reference

### Makefile Targets

```bash
# Core Evaluation
make eval-combined           # Run combined evaluation
make eval-combined-save      # Save with history
make eval-vision-golden      # Vision module only
make eval-ocr-golden        # OCR module only
make eval-all-golden        # All golden evaluations

# Reporting
make eval-report            # Basic HTML report (v1)
make eval-report-v2         # Enhanced interactive report
make eval-trend             # Generate trend charts
make health-check          # Quick health summary

# Data Management
make eval-validate          # Validate schema compliance
make eval-migrate          # Migrate legacy files
make eval-retention        # Check retention policy
make eval-retention-apply  # Apply retention (with confirmation)

# CI/CD
make ci-combined-check     # CI quality gate
make ci-check-metrics     # Check metric thresholds
make ci-test              # Full CI test suite

# Testing
make test-map             # Update test documentation
make test-map-overwrite  # Force update TEST_MAP.md
```

### Environment Variables

```bash
# Evaluation Thresholds
MIN_COMBINED=0.80         # Minimum combined score
MIN_VISION=0.65          # Minimum vision score
MIN_OCR=0.90            # Minimum OCR score

# Notifications
EVAL_SLACK_WEBHOOK       # Slack webhook URL
EVAL_SMTP_HOST          # Email server host
EVAL_SMTP_PORT          # Email server port (587)
EVAL_EMAIL_FROM         # Sender email
EVAL_EMAIL_TO           # Recipients (comma-separated)
EVAL_EMAIL_PASSWORD     # Email authentication

# GitHub Integration
GITHUB_TOKEN            # Personal access token
GITHUB_REPOSITORY       # org/repo format
GITHUB_RUN_ID          # Actions run ID (auto-set)
GITHUB_WORKFLOW        # Workflow name (auto-set)

# CI Detection
CI                     # Set to "true" in CI environment
```

---

## 📉 Threshold Configuration

### Production Thresholds
```yaml
combined_score: ≥ 0.80
vision_score:   ≥ 0.65
ocr_score:      ≥ 0.90
```

### Development Thresholds
```yaml
combined_score: ≥ 0.70
vision_score:   ≥ 0.60
ocr_score:      ≥ 0.85
```

### Custom Thresholds
```bash
# Via environment variables
export MIN_COMBINED=0.85
export MIN_VISION=0.70
export MIN_OCR=0.92

# Via command line
python3 scripts/evaluate_vision_ocr_combined.py \
  --min-combined 0.85 \
  --min-vision 0.70 \
  --min-ocr 0.92
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Python Command Not Found
```bash
# macOS uses python3 by default
alias python=python3

# Or update scripts to use python3
sed -i '' 's/python /python3 /g' Makefile
```

#### 2. Deprecation Warnings
```python
# Old (deprecated)
datetime.utcnow()

# New (timezone-aware)
datetime.now(timezone.utc)
```

#### 3. Chart.js Not Loading
```bash
# Download Chart.js locally
curl -o reports/eval_history/report/assets/chart.min.js \
  https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js

# Use local version
python3 scripts/generate_eval_report_v2.py \
  --local-chartjs assets/chart.min.js
```

#### 4. Notifications Not Sending
```bash
# Check environment variables
env | grep EVAL_

# Test with dry run
python3 scripts/notify_eval_results.py --dry-run

# Check webhook URL format
curl -X POST $EVAL_SLACK_WEBHOOK -d '{"text":"test"}'
```

---

## 🎓 Best Practices

### 1. Regular Evaluation
- Run evaluations on every PR
- Schedule daily comprehensive checks
- Monitor trends weekly

### 2. Threshold Management
- Start with conservative thresholds
- Gradually increase as system improves
- Different thresholds for dev/prod

### 3. Data Retention
- Run retention weekly via cron
- Always archive before deletion
- Keep quarterly snapshots indefinitely

### 4. Notification Strategy
- Immediate alerts for breaches
- Daily summaries for trends
- Weekly reports for management

### 5. Report Access
- Public: GitHub Pages for main branch
- Private: Artifacts for PRs
- Offline: Downloaded HTML files

---

## 🚦 System Status Indicators

### Green (Healthy) ✅
- Combined score ≥ 0.85
- All thresholds passing
- Trend improving/stable
- Recent evaluations successful

### Yellow (Warning) ⚠️
- Combined score 0.70-0.85
- Minor threshold breaches
- Trend declining slightly
- Some evaluations failing

### Red (Critical) 🚨
- Combined score < 0.70
- Major threshold breaches
- Trend declining rapidly
- Multiple evaluation failures

---

## 📅 Maintenance Schedule

### Daily
- Automated evaluation (2 AM UTC)
- Threshold checks
- Notification dispatch

### Weekly
- Retention policy application
- Trend analysis review
- Report archive

### Monthly
- Threshold adjustment
- System performance review
- Documentation update

### Quarterly
- Major version upgrades
- Architecture review
- Capacity planning

---

## 🔮 Future Enhancements (Phase 6+)

### Near Term (Q1 2025)
- [ ] LLM-powered trend analysis
- [ ] Anomaly detection algorithms
- [ ] Predictive failure alerts
- [ ] Custom metric plugins

### Medium Term (Q2 2025)
- [ ] Real-time streaming dashboard
- [ ] A/B testing framework
- [ ] Multi-model comparison
- [ ] Performance benchmarking

### Long Term (2025+)
- [ ] AutoML integration
- [ ] Self-healing systems
- [ ] Distributed evaluation
- [ ] Cloud-native architecture

---

## 📚 Additional Resources

### Documentation
- [Design Document](./EVAL_REPORT_DESIGN.md)
- [Implementation Summary](./EVAL_SYSTEM_IMPLEMENTATION_SUMMARY.md)
- [TEST_MAP](./TEST_MAP.md)
- [OCR Guide](./OCR_GUIDE.md)

### Scripts
- [`evaluate_vision_ocr_combined.py`](../scripts/evaluate_vision_ocr_combined.py)
- [`generate_eval_report_v2.py`](../scripts/generate_eval_report_v2.py)
- [`manage_eval_retention.py`](../scripts/manage_eval_retention.py)
- [`notify_eval_results.py`](../scripts/notify_eval_results.py)
- [`validate_eval_history.py`](../scripts/validate_eval_history.py)

### CI/CD
- [GitHub Actions Workflow](../.github/workflows/evaluation-report.yml)
- [Makefile](../Makefile)

---

## 👥 Support

For issues or questions:
1. Check this documentation
2. Review troubleshooting section
3. Create GitHub issue with:
   - Error messages
   - Environment details
   - Steps to reproduce

---

*Last Updated: 2025-11-19 | Version: 3.0.0 | CAD ML Platform Team*