# Faiss Batch Similarity Degraded Tests Design

## Overview
Strengthen batch similarity fallback coverage to verify the memory fallback
metric increments and the response flags fallback when Faiss is unavailable.

## Updates
- Ensure fallback metric checks run only when Prometheus counters are available.
- Validate fallback flag in batch similarity responses.

## Files
- `tests/unit/test_faiss_degraded_batch.py`
