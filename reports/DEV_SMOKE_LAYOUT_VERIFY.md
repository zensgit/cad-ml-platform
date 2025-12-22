#!/usr/bin/env markdown
# Dev Smoke: Vector Layout + Similarity

## Scope
- Validate canonical vector layout (flatten/rehydrate).
- Smoke-test analyze + similarity endpoints in dev.

## Environment Notes
- Unit test run used default env (no `VECTOR_STORE_BACKEND` override).
- See `reports/VECTOR_STORE_LAYOUT_AUDIT_DEV.md` for current Redis/FAISS state.

## Tests
- Command:
  `.venv/bin/python -m pytest tests/unit/test_feature_vector_layout.py tests/unit/test_feature_rehydrate.py tests/unit/test_feature_slots.py tests/unit/test_similarity_endpoint.py tests/unit/test_similarity_topk.py -q`
- Result: `9 passed in 4.62s`

## Verification
- Vector layout round-trips without slot mis-ordering.
- Similarity endpoints respond with expected payloads.
