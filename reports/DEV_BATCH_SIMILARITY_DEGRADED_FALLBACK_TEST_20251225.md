# Batch similarity degraded fallback regression test

- Date: 2025-12-25
- Change: Added regression test to ensure degraded mode forces `fallback` true.
- Command: `.venv/bin/python -m pytest tests/unit/test_batch_similarity_faiss_unavailable.py -v`
- Result: PASS (2 passed in 2.30s)
