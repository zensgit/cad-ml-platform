# Assistant Explainability Contract Validation

## Goal
- Extend `/assistant/query` with a stable explainability structure without
  breaking existing `answer/confidence/sources/evidence` fields.
- Reuse existing assistant evidence and retrieval metadata instead of inventing
  a separate reasoning path.

## Key Changes
- Added response models:
  - `QueryAlternative`
  - `QueryUncertainty`
  - `QueryExplainability`
- Added `explainability` to `QueryResponse`
- Added helper `_build_query_explainability()` to derive:
  - `summary`
  - `decision_path`
  - `source_contributions`
  - `alternative_labels`
  - `uncertainty`
- Error responses now also include a stable explainability payload.

## Contract Shape
- `summary`: compact readable explanation summary
- `decision_path`: stable assistant reasoning path tokens
- `source_contributions`: normalized contribution by retrieval source
- `alternative_labels`: lower-ranked source candidates
- `uncertainty`:
  - `score`
  - `reasons`

## Validation Commands
```bash
python3 -m py_compile \
  src/api/v1/assistant.py \
  tests/unit/assistant/test_llm_api.py

flake8 \
  src/api/v1/assistant.py \
  tests/unit/assistant/test_llm_api.py \
  --max-line-length=100

pytest -q tests/unit/assistant/test_llm_api.py
```

## Expected Result
- `/assistant/query` continues to return the old fields.
- New `explainability` field is always present on success and failure paths.
- Existing structured evidence remains unchanged and is now complemented by a
  stable assistant-facing explainability contract.
