# Fine-tune From Feedback Design

## Overview
The fine-tuning script consumes labeled samples from Active Learning and attempts
to retrieve corresponding feature vectors for model training. It now resolves
vectors through the similarity vector store, with a safe fallback to mock data
when vectors are unavailable.

## Data Flow
1. Export labeled samples from Active Learning (JSONL).
2. Parse `doc_id` and `true_type` pairs.
3. Resolve vectors with `src.core.similarity.get_vector(doc_id)`.
4. Train or fine-tune the classifier if vectors exist.
5. Fallback to mock data if no vectors can be resolved.

## Vector Retrieval
- Primary source: in-memory vector store (default backend).
- Optional source: Redis vector backend (`vector:{doc_id}` hash entry).
- Missing vectors are skipped with a warning.

## Output
- Trained model saved to `models/` with timestamped version.
- Optional model reload via `reload_model()` (existing behavior).

## Limitations
- Vector availability depends on the running service or Redis backend.
- No automatic re-extraction of vectors from CAD files.
