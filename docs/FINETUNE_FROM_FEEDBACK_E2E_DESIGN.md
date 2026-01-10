# Fine-tune From Feedback E2E Demo Design

## Overview
This demo script provides a minimal, reproducible end-to-end path for
fine-tuning from feedback. It seeds a labeled Active Learning sample,
registers a vector, exports training data, loads vectors, and optionally
trains a model.

## Script
- `scripts/finetune_from_feedback_e2e.py`

## Flow
1. Reset Active Learning and vector store state.
2. Create a labeled feedback sample.
3. Register a feature vector for the same `doc_id`.
4. Export training data (JSONL) to disk.
5. Load vectors from the similarity store.
6. Optionally train a model.

## Inputs
- `--dim`: vector dimension (default 32).
- `--label`: true label for the sample (default "bolt").
- `--skip-train`: skip model training (useful when sklearn is unavailable).

## Outputs
- Console summary showing exported sample count and vector count.
- Optional training summary when sklearn is available.

## Notes
- Uses in-memory stores by default; no Redis required.
- Training is best-effort and can be skipped safely.
