# DEV Knowledge Match Audit (2025-12-22)

## Scope
- Runtime knowledge matching flow and data sources
- Matchable categories and rule formats
- Gaps between design assets and runtime wiring

## Findings
- Runtime matching uses `KnowledgeManager.get_part_hints()` (keyword, regex, geometry patterns).
- Dynamic rules load from `data/knowledge/<category>_rules.json` via `JSONKnowledgeStore`.
- Current on-disk rules: only `data/knowledge/geometry_rules.json` (active-learning, keyword-driven).
- `knowledge_base/assembly/*` YAML rules are not wired into runtime matching.

## Fix Applied
- Geometry patterns with empty `conditions` are now skipped in geometry matching to avoid match-all behavior.

## Documentation
- Added `docs/KNOWLEDGE_MATCHING.md` describing runtime matching and data sources.

## Tests
- `.venv/bin/python -m pytest tests/unit/test_dynamic_knowledge_matching.py tests/test_l3_fusion_flow.py -q`
- Result: `6 passed in 2.60s`
