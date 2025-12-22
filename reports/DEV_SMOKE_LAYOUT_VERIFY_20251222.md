#!/usr/bin/env markdown
# Dev Smoke: Vector Layout + Similarity (2025-12-22)

## Scope
- Validate canonical vector layout (flatten/rehydrate).
- Smoke-test analyze + similarity endpoints in dev.

## Tests
- Command:
  `.venv/bin/python -m pytest tests/unit/test_feature_vector_layout.py tests/unit/test_feature_rehydrate.py tests/unit/test_feature_slots.py tests/unit/test_similarity_endpoint.py tests/unit/test_similarity_topk.py -q`
- Result: `9 passed in 3.50s`

## Verification
- Vector layout round-trips without slot mis-ordering.
- Similarity endpoints respond with expected payloads.
