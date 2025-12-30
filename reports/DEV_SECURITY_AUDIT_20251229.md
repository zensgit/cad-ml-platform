# Security Audit Report

- Date: 2025-12-29
- Scope: pip-audit + bandit + security audit script (severity=medium)

## Commands
- `.venv/bin/pip-audit --format json > reports/security/pip_audit_20251229.json`
- `.venv/bin/bandit -r src -f json -o reports/security/bandit_20251229.json`
- `PATH=".venv/bin:$PATH" .venv/bin/python scripts/security_audit.py --severity medium`

## Result
- FINDINGS (non-zero exit due to vulnerabilities)

## Summary
- pip-audit: 21 vulnerabilities in 12 packages
- bandit: 367 issues (43 high, 10 medium, 314 low)
- security_audit.py summary (medium+): 74 total (43 high, 31 medium)

## Packages flagged by pip-audit
- aiohttp
- black
- ecdsa
- fastapi
- filelock
- h11
- python-jose
- python-multipart
- requests
- scikit-learn
- starlette
- tqdm

## Outputs
- `reports/security/pip_audit_20251229.json`
- `reports/security/bandit_20251229.json`
- `reports/security/security_audit_20251229_012805.json`
