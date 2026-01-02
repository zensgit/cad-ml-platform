# Knowledge Matching (Runtime)

This note describes how knowledge matching works in the current runtime and what data is actually used.

## Runtime Matching Path
- `FusionClassifier` calls `KnowledgeManager.get_part_hints(text, geometric_features, entity_counts)`.
- `KnowledgeManager` matches in three ways:
  1) keyword substring match (case-insensitive)
  2) regex pattern match (OCR patterns)
  3) geometry pattern match (conditions over geometric features/entity counts)
- Geometry patterns are only considered when `conditions` are present.

## Data Sources
- Dynamic rules are loaded from `data/knowledge/<category>_rules.json` via `JSONKnowledgeStore`.
- Current on-disk data:
  - `data/knowledge/geometry_rules.json` (active-learning rules from `scripts/learn_from_standards.py`).
- The YAML files under `knowledge_base/assembly/` are design assets and are **not** wired into runtime matching.

## Matchable Categories
Defined in `src/core/knowledge/dynamic/models.py`:
- `material`, `precision`, `standard`, `functional`, `assembly`, `manufacturing`, `geometry`, `part_type`

## Geometry Pattern Inputs
`GeometryPattern.matches()` supports:
- entity ratios: `circle_ratio`, `line_ratio`, `arc_ratio`, etc. (computed from entity_counts)
- entity counts: `circle_count`, `line_count`, etc.
- base/derived features: `sphericity`, `aspect_variance`, `curved_ratio`, etc. (from geometric_features)

## Operational Notes
- Knowledge status and reload endpoints live under `/api/v1/maintenance/knowledge/*`.
- Add or edit JSON files under `data/knowledge/` and call `/api/v1/maintenance/knowledge/reload` to apply.

## Seeding (Dev Only)
- Set `KNOWLEDGE_AUTO_SEED=1` to copy seed JSON files into `data/knowledge/` when empty.
- Seed source defaults to `seed/knowledge/` and can be overridden via `KNOWLEDGE_SEED_DIR`.
