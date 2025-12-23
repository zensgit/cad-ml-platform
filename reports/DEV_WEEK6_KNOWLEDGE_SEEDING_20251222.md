# Week6 Step4 - Knowledge Seeding + Validation (2025-12-22)

## Scope
- Added optional dev seeding from `seed/knowledge/` into `data/knowledge/` when empty.
- Environment toggle: `KNOWLEDGE_AUTO_SEED=1` (seed dir override via `KNOWLEDGE_SEED_DIR`).

## Tests
- `pytest tests/unit/test_knowledge_seed.py -q`

## Results
- `1 passed in 9.73s`

## Notes
- Seeding is opt-in and does not run unless `KNOWLEDGE_AUTO_SEED=1` is set.
