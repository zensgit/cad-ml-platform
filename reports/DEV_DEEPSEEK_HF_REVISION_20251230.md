# DeepSeek HF Revision Pinning Report

- Date: 2025-12-30
- Scope: DEEPSEEK_HF_REVISION/Model pinning updates and targeted provider tests.

## Changes
- Added `DEEPSEEK_HF_MODEL`, `DEEPSEEK_HF_REVISION`, `DEEPSEEK_HF_ALLOW_UNPINNED` to `.env.example`.
- DeepSeek HF provider now reads `DEEPSEEK_HF_MODEL` from env; revision remains required unless explicitly unpinned.

## Command
- `.venv/bin/python -m pytest tests/test_metrics_consistency.py::TestDeepSeekHfProviderMetrics -v`

## Result
- PASS

## Summary
- Tests: 2 passed
- Duration: 22.45s
