# Batch Similarity Empty Results Metrics Tests Design

## Overview
Strengthen batch similarity empty-results coverage by asserting the
`batch_empty_results` rejection metric increments when Prometheus counters are
available.

## Tests
- `tests/unit/test_batch_similarity_empty_results.py`

## Validation
- `pytest tests/unit/test_batch_similarity_empty_results.py -v`
