# CAD ML Platform - Complete Evaluation System Documentation

## System Overview

The CAD ML Platform evaluation system provides comprehensive monitoring, reporting, and quality assurance for Vision and OCR modules. The system has been designed with integrity, reliability, and maintainability as core principles.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                   Evaluation System                      │
├───────────────────┬──────────────┬────────────────────┤
│   Configuration   │   Execution   │     Validation     │
│  eval_frontend.json│               │                    │
│  - Chart.js config│  Combined Eval │  Schema v1.0.0    │
│  - Retention policy│  Vision + OCR │  Integrity Check  │
│  - Validation rules│  Scoring Logic│  JSON Validation  │
└───────────────────┴──────────────┴────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Report Generation                     │
├──────────────────┬────────────────┬───────────────────┤
│   Static HTML    │  Interactive    │    Notifications  │
│   Base64 Charts  │   Chart.js 4.4  │   Slack/Email/GH │
│   Privacy Mode   │   CDN/Local/PNG │   Threshold Alert │
└──────────────────┴────────────────┴───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Data Management                       │
├──────────────────┬────────────────┬───────────────────┤
│  5-Layer Retention│  Version Monitor│    CI/CD Pipeline │
│  7d→30d→90d→365d→∞│  Weekly Checks  │   GitHub Actions  │
│  Archive on Delete│  NPM Registry   │   Auto Reports    │
└──────────────────┴────────────────┴───────────────────┘
```

## Configuration

### Central Configuration (`config/eval_frontend.json`)

All system settings are centralized in a single configuration file:

```json
{
  "chartjs": {
    "version": "4.4.0",
    "cdn_url": "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js",
    "sha384": "vxfkwNn5NGXgAZEPcjDEwOXrc6Q3gSYPgfn8R6J0R6qKgNW2pV4s3u8V5Zg7Xk6S",
    "local_path": "reports/eval_history/report/assets/chart.min.js",
    "expected_size_bytes": 737280,
    "integrity_check_enabled": true,
    "warn_only": true
  },
  "schema_version": "1.0.0",
  "validation": {
    "strict_mode": false,
    "allow_additional_fields": true,
    "max_invalid_threshold": 0,
    "migration_enabled": true
  },
  "retention_policy": {
    "tiers": [
      {"name": "full", "days": 7, "keep": "all"},
      {"name": "daily", "days": 30, "keep": "daily"},
      {"name": "weekly", "days": 90, "keep": "weekly"},
      {"name": "monthly", "days": 365, "keep": "monthly"},
      {"name": "quarterly", "days": -1, "keep": "quarterly"}
    ],
    "archive_before_delete": true
  },
  "version_monitoring": {
    "enabled": true,
    "check_interval_days": 7,
    "npm_registry_url": "https://registry.npmjs.org/chart.js",
    "timeout_seconds": 2,
    "create_issue_on_major": false,
    "suppressed_versions": []
  }
}
```

## Core Scripts

### 1. Combined Evaluation (`scripts/evaluate_vision_ocr_combined.py`)
- **Purpose**: Execute Vision and OCR evaluations with weighted scoring
- **Formula**: `0.5 * vision_score + 0.5 * (ocr_recall * (1 - brier_score))`
- **Output**: JSON history with schema v1.0.0, console report
- **Features**: Git tracking, thresholds, historical persistence

### 2. Integrity Checker (`scripts/check_integrity.py`)
- **Purpose**: Validate file integrity using SHA-384 hashes
- **Features**:
  - Lightweight design (no heavy dependencies)
  - Warn-only mode by default
  - Strict mode available with `--strict` flag
  - SRI format compatibility

### 3. Schema Validator (`scripts/validate_eval_history.py`)
- **Purpose**: Ensure evaluation outputs conform to schema
- **Features**:
  - JSON Schema Draft-07 validation
  - Optional jsonschema library (graceful degradation)
  - Automatic migration support
  - Batch validation of history

### 4. Report Generators

#### Static Report (`scripts/generate_eval_report.py`)
- Base64 embedded charts
- No external dependencies
- Privacy mode support
- One-command generation

#### Interactive Report (`scripts/generate_eval_report_v2.py`)
- Chart.js 4.4.0 integration
- Three-tier fallback (CDN → Local → PNG)
- Branch/date filtering
- Performance trends

### 5. Retention Manager (`scripts/manage_eval_retention.py`)
- **5-Layer Policy**:
  1. **Full (7 days)**: Keep all evaluations
  2. **Daily (30 days)**: Keep one per day
  3. **Weekly (90 days)**: Keep one per week
  4. **Monthly (365 days)**: Keep one per month
  5. **Quarterly (Forever)**: Keep one per quarter
- Archive before delete
- Per-branch preservation

### 6. Version Monitor (`scripts/check_chartjs_updates.py`)
- Weekly automated checks
- NPM registry queries with timeout
- Major version alerts
- Suppression list support

### 7. Notification System (`scripts/notify_eval_results.py`)
- Multi-channel support (Slack, Email, GitHub)
- Threshold-based alerts
- Trend detection
- Batch notification capabilities

## Test Suite

### Unit Test Suite (`scripts/test_eval_system.py`)
Comprehensive testing of all components:
- Configuration integrity
- File integrity checks
- Schema validation
- Evaluation pipeline
- Report generation
- Retention policy
- Version monitoring
- Makefile targets

### Integration Test (`scripts/run_full_integration_test.py`)
End-to-end workflow validation:
1. Configuration integrity check
2. Run combined evaluation
3. Validate output JSON
4. Generate both report types
5. Check retention policy
6. Version monitoring
7. Health check

## Makefile Targets

```makefile
# Evaluation
make eval                    # Run Vision+OCR evaluation
make eval-history           # Show evaluation history
make health-check           # System health check

# Validation & Integrity
make integrity-check        # Check file integrity (warn mode)
make integrity-check-strict # Check file integrity (strict mode)
make eval-validate         # Validate all history JSONs
make eval-validate-schema  # Validate against schema

# Reporting
make eval-report           # Generate static HTML report
make eval-report-v2        # Generate interactive report
make eval-report-open      # Open latest report

# Maintenance
make eval-retention        # Apply retention policy
make eval-clean           # Clean evaluation outputs
make eval-setup           # First-time setup
```

## CI/CD Integration

### GitHub Actions Workflows

#### Evaluation Report Workflow (`.github/workflows/evaluation-report.yml`)
- Triggers: Push to main, PR, manual, schedule (daily)
- Generates reports automatically
- Uploads artifacts
- Posts PR comments
- Deploys to GitHub Pages

#### Version Monitoring (`.github/workflows/version-monitor.yml`)
- Weekly schedule (Monday 3 AM UTC)
- Checks for Chart.js updates
- Creates issues for major versions
- Non-blocking failures

## Security & Integrity

### SHA-384 Integrity Verification
- All JavaScript libraries verified before use
- SRI (Subresource Integrity) format support
- Fail-soft approach: warn by default, strict optional
- Automatic fallback on verification failure

### Schema Validation
- JSON Schema Draft-07 compliance
- Schema version tracking (currently v1.0.0)
- Backward compatibility through migrations
- Validation before processing

### Privacy Controls
- Configurable privacy mode for reports
- No external CDN dependencies in privacy mode
- Local-only operation support
- Secure credential handling for notifications

## Performance Characteristics

### Execution Times
- Combined evaluation: ~1-2 seconds
- Report generation: ~0.5 seconds
- Integrity check: ~0.1 seconds
- Schema validation: ~0.2 seconds
- Full integration test: ~3 seconds

### Resource Usage
- Minimal memory footprint
- No heavy dependencies
- Efficient JSON processing
- Optimized Chart.js bundle (737KB)

## Monitoring & Observability

### Health Checks
- Continuous monitoring of all components
- Automated testing in CI
- Performance metrics tracking
- Error rate monitoring

### Version Tracking
- Git branch and commit in all evaluations
- Schema version in outputs
- Dependency version monitoring
- Automated update notifications

## Data Retention & Archival

### Storage Strategy
- JSON history in `reports/eval_history/`
- HTML reports in `reports/eval_history/report/`
- Archives before deletion
- Branch-aware retention

### Backup & Recovery
- Git-tracked evaluation history
- CI artifact preservation
- Local archive support
- Manual recovery procedures

## Quick Start Guide

### Initial Setup
```bash
# 1. Install dependencies (optional but recommended)
pip install jsonschema==4.21.1

# 2. Run setup
make eval-setup

# 3. Verify installation
make health-check
```

### Daily Usage
```bash
# Run evaluation
make eval

# Generate report
make eval-report-v2

# View history
make eval-history
```

### Maintenance
```bash
# Weekly: Apply retention policy
make eval-retention

# Monthly: Check for updates
python3 scripts/check_chartjs_updates.py

# As needed: Validate data
make eval-validate
```

## Troubleshooting

### Common Issues

1. **Integrity Check Failures**
   - Check SHA-384 hash in config
   - Verify file download completed
   - Use `--warn-only` flag for non-critical

2. **Schema Validation Errors**
   - Install jsonschema: `pip install jsonschema==4.21.1`
   - Check schema version compatibility
   - Enable migration in config

3. **Report Generation Issues**
   - Verify Chart.js availability
   - Check fallback mechanisms
   - Use `--use-png` for compatibility

4. **Network Timeouts**
   - Version monitoring has 2-second timeout
   - CDN fallback to local files
   - Offline mode fully supported

## Best Practices

### Configuration Management
- Always use centralized config file
- Version control all configuration changes
- Document suppressed versions
- Test configuration changes locally first

### Evaluation Workflow
1. Run evaluation before commits
2. Generate reports for reviews
3. Monitor trends over time
4. Set appropriate thresholds
5. Investigate score drops immediately

### Report Distribution
- Use interactive reports for analysis
- Use static reports for archival
- Enable notifications for teams
- Privacy mode for sensitive data

### Maintenance Schedule
- **Daily**: Automated evaluations (CI)
- **Weekly**: Retention policy, version check
- **Monthly**: Full system validation
- **Quarterly**: Archive old data

## System Requirements

### Minimum Requirements
- Python 3.8+
- 100MB free disk space
- Git for version tracking
- Make for automation

### Recommended Setup
- Python 3.10+
- jsonschema library installed
- 500MB free disk space
- GitHub Actions enabled
- Notification webhooks configured

## Future Roadmap

### Planned Enhancements
1. Real-time monitoring dashboard
2. Machine learning trend analysis
3. Automated performance optimization
4. Multi-model comparison framework
5. Advanced anomaly detection

### Extension Points
- Custom evaluation metrics
- Additional notification channels
- Alternative visualization libraries
- External data integrations
- Performance benchmarking suite

## Support & Documentation

### Documentation
- This guide: `docs/EVALUATION_SYSTEM_COMPLETE.md`
- Schema reference: `docs/eval_history.schema.json`
- API documentation: `docs/api/`
- Troubleshooting: `docs/troubleshooting.md`

### Getting Help
1. Check documentation first
2. Review test suite output
3. Enable verbose logging
4. Create GitHub issue with logs

## License & Credits

Part of the CAD ML Platform project.
Evaluation system designed for reliability, maintainability, and extensibility.

---

*Last Updated: 2025-11-19*
*System Version: 1.0.0*
*Schema Version: 1.0.0*