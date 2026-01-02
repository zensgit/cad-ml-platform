# Doc Update Validation - Batch Similarity fallback field

- Date: 2025-12-25
- Scope: README.md batch similarity response description
- Goal: Validate docs update does not regress targeted similarity behavior

## Changes
- Documented `fallback` field meaning in batch similarity response.

## Test Command
- .venv/bin/python -m pytest tests/unit/test_batch_similarity_faiss_unavailable.py -v

## Result
- PASS (2 tests)

## Notes
- No code-path changes; doc-only update validated with targeted similarity tests.
