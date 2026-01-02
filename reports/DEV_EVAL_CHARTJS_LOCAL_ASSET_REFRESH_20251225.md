# Integrity Fix - Local Chart.js Asset Refresh

- Date: 2025-12-25
- Scope: reports/eval_history/report/assets/chart.min.js
- Goal: Resolve integrity hash mismatch for local Chart.js fallback

## Changes
- Downloaded Chart.js 4.4.0 from CDN to replace placeholder asset.

## Validation
- python3 scripts/check_integrity.py --verbose

## Result
- PASS (hash and size match config/eval_frontend.json)
