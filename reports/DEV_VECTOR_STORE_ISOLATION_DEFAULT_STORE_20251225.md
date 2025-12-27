# Vector store isolation default store reset

- Date: 2025-12-25
- Change: restore `similarity._DEFAULT_STORE` in `tests/conftest.py` vector_store_isolation.
- Command: `.venv/bin/python -m pytest tests/unit/test_batch_similarity_faiss_unavailable.py -v`
- Result: PASS (2 passed in 6.48s)
