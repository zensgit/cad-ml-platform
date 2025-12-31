# DedupCAD Vision Vector/Hash Compatibility Check (2026-01-01)

## Scope

- Verify feature vector ordering, rehydration, and compare fallback expectations.

## Findings

- `FeatureExtractor.flatten()` uses canonical order: base geometric (5) + semantic (2) + geometric extensions.
- `rehydrate()` and `upgrade_vector()` assume the same canonical ordering.
- `/api/compare` uses `candidate_hash` directly as vector id; requires dedupcad-vision to register vectors with `id == file_hash` (documented in contract/runbooks).

## Tests

- `pytest tests/unit/test_feature_vector_layout.py tests/unit/test_feature_extractor_v4.py tests/unit/test_feature_rehydrate.py tests/unit/test_compare_endpoint.py -v`

## Results

- OK: 14 passed.
