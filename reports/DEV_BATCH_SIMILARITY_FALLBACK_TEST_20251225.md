# Batch similarity fallback unit test

- Date: 2025-12-25
- Command: `.venv/bin/python -m pytest tests/unit/test_batch_similarity_faiss_unavailable.py -v`
- Result: PASS (1 passed in 2.22s)

Notes:
- Initial run with system Python 3.13 failed due to missing `jose`; reran with project `.venv` and passed.
