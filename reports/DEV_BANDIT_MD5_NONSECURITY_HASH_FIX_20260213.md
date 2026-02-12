# DEV_BANDIT_MD5_NONSECURITY_HASH_FIX_20260213

## Goal

Unblock the GitHub Actions **Security Audit** workflow by eliminating Bandit **B324 (md5)** findings that were
being counted as **HIGH** severity in CI, while preserving the existing cache-key behavior (same digest).

These `md5` uses are for deterministic cache keys / fingerprints, not cryptographic security.

## Changes

- `src/inference/classifier_api.py`
  - `LRUCache._hash_content()`: `hashlib.md5(content, usedforsecurity=False).hexdigest()`
  - `HybridCache._make_key()`: `hashlib.md5(content, usedforsecurity=False).hexdigest()`
- `src/ml/part_classifier.py`
  - `PartClassifierV16._get_file_cache_key()`: add `usedforsecurity=False` for both cache-key paths

Rationale: Bandit (and FIPS-enabled environments) treat plain `md5` usage as potentially security-relevant.
Passing `usedforsecurity=False` makes the intent explicit without changing the digest.

## Validation

- Core validation:
  - `make validate-core-fast` (passed)
- Bandit gate (mirrors `.github/workflows/security-audit.yml`):
  - `.venv/bin/bandit -r src/ --exclude src/core/vision,src/core/ocr -f json -o /tmp/bandit-report.json`
  - Result: `bandit_high = 0`

