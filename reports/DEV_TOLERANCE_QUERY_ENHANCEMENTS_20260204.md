# DEV_TOLERANCE_QUERY_ENHANCEMENTS_20260204

## Summary
- Added fundamental deviation lookup for symbols like H7/g6 with nominal size.
- Updated assistant examples and query suggestions with GB/T 1804/1184 and deviation queries.
- Added tests for fundamental deviation retrieval in assistant flow.

## Changes
- `src/core/assistant/query_analyzer.py`
  - Added tolerance symbol extraction and intent pattern coverage.
  - Added suggestion for fundamental deviation query.
- `src/core/assistant/knowledge_retriever.py`
  - Added fundamental deviation retrieval logic for hole/shaft symbols.
- `src/core/assistant/assistant.py`
  - Expanded supported queries with GB/T 1804/1184 and deviation examples.
- `tests/unit/assistant/test_assistant.py`
  - Added tests for H7/g6 fundamental deviation lookups.

## Validation
- `pytest tests/unit/assistant/test_assistant.py -q`

## Notes
- Warning observed from `python_multipart` deprecation (Starlette); unrelated to tolerance logic.
