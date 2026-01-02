# Security Audit Report

- Date: 2025-12-30
- Scope: bandit + pip-audit + security audit script (severity=medium)
- Code changes: MD5/SHA1 usage replaced with SHA256 in vision hashing; DWG converter command no longer uses shell.

## Commands
- `.venv/bin/bandit -r src -f json -o reports/security/bandit_20251229_post.json`
- `.venv/bin/pip-audit --format json > reports/security/pip_audit_20251229_post4.json`
- `PATH=".venv/bin:$PATH" .venv/bin/python scripts/security_audit.py --severity medium`

## Result
- FINDINGS (non-zero exit due to remaining vulnerabilities)

## Summary
- bandit: 325 issues (0 high, 10 medium, 315 low)
- pip-audit: 1 vulnerability (ecdsa 0.19.1, CVE-2024-23342, no fix available)
- security_audit.py: 11 medium, 0 high

## Outputs
- `reports/security/bandit_20251229_post.json`
- `reports/security/pip_audit_20251229_post4.json`
- `reports/security/security_audit_20251230_023343.json`
