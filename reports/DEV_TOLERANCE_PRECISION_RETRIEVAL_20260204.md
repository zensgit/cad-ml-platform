# DEV_TOLERANCE_PRECISION_RETRIEVAL_20260204

## Summary
- Integrated precision seed rules (GB/T 1804/1184) into assistant tolerance retrieval.
- Expanded intent detection to route GB/T 1804/1184 queries to tolerance retrieval.
- Added tests to validate precision rule retrieval in assistant flow.

## Changes
- `src/core/assistant/knowledge_retriever.py`
  - Added `_retrieve_precision_rules()` and hooked into tolerance retrieval.
- `src/core/assistant/query_analyzer.py`
  - Added tolerance intent patterns for GB/T 1804/1184 and general tolerance phrases.
- `tests/unit/assistant/test_assistant.py`
  - Added precision rule retrieval tests for GB/T 1804 and GB/T 1184 queries.

## Validation
- `pytest tests/unit/assistant/test_assistant.py -q`

## Notes
- Warning from `python_multipart` deprecation (Starlette) observed; unrelated to tolerance logic.
