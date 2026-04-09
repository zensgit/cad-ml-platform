# Qdrant Coarse Metadata Validation

## Scope
- Normalize vector payload metadata to stable coarse/fine label contracts
- Add Qdrant payload indexes for `part_type`, `fine_part_type`, `coarse_part_type`, `decision_source`
- Keep writes additive for register, batch register, and metadata update paths

## Changed Files
- `src/core/vector_stores/qdrant_store.py`
- `tests/unit/test_qdrant_vector_store.py`

## Validation
```bash
python3 -m py_compile src/core/vector_stores/qdrant_store.py \
  tests/unit/test_qdrant_vector_store.py

flake8 src/core/vector_stores/qdrant_store.py \
  tests/unit/test_qdrant_vector_store.py \
  --max-line-length=100

pytest -q tests/unit/test_qdrant_vector_store.py
```

## Result
- `py_compile`: pass
- `flake8`: pass
- `pytest`: `6 passed, 18 skipped`

## Verified Behavior
- `_ensure_collection()` creates payload indexes for coarse/fine contract fields
- `register_vector()` normalizes payload metadata before upsert
- `register_vectors_batch()` normalizes payload metadata before batch upsert
- `update_metadata()` normalizes payload metadata before `set_payload()`
- `decision_source` is mirrored into `final_decision_source` when missing

## Notes
- Skipped tests are expected in environments without live Qdrant integration dependencies
- This change is additive and does not remove existing payload fields
