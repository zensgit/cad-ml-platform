# Benchmark Surpass Development Status

## Purpose

This document records the current parallel delivery state aligned with the benchmark-surpass roadmap.

## Completed Foundations Already in Main

The main branch already includes:

- coarse/fine normalization across multiple AI and vector surfaces
- provider coarse contract support
- batch classify coarse contract support
- similarity / compare / vector search coarse contracts
- feedback statistics and finetune observability
- Qdrant-native query, mutation, compare, stats, maintenance, and migration primitives
- online `.h5` and STEP example validation helpers
- B-Rep smoke and batch validation on Apple Silicon + micromamba

## In-Flight Stack

The current in-flight migration operability stack is:

1. `#131` vector migration recommendations
2. `#132` vector migration plan
3. `#133` vector migration plan runbook hints
4. `#134` vector migration plan estimates
5. `#135` vector migration plan advisories
6. `#136` vector migration plan coverage summary
7. `#137` vector migration plan truncation summary

Intent of the stack:

- make migration planning readable without inspecting raw pending data
- surface first-run payloads directly
- show readiness and blocking reasons
- show plan coverage and truncation explicitly

## Current Validation Pattern

Every low-conflict stack item is validated with:

```bash
python3 -m py_compile <changed files>
flake8 <changed files> --max-line-length=100
pytest -q \
  tests/unit/test_vector_migration_plan.py \
  tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py \
  tests/unit/test_api_route_uniqueness.py
```

This ensures:

- syntax correctness
- lint correctness
- API contract integrity
- route uniqueness
- snapshot stability

## Delivery Principles

### Contract First

Every user-visible API surface must expose:

- stable semantics
- operator context
- safe defaults

### Stack Narrowly

Wide-surface changes are acceptable only when they all serve the same contract family. The migration stack follows this rule by limiting scope to planning and operator observability.

### Keep Execution Safe

Planning and observability can expand quickly. Mutation/write behavior should stay conservative:

- partial-scan blocks remain explicit
- write paths are separate from planning paths
- no hidden auto-execution

## Remaining Gaps

The following areas still need additional work to claim a stronger benchmark position:

- merge the migration operability stack into main
- complete real `.h5` history evaluation on production-like data
- deepen knowledge outputs into review and final-decision summaries
- add enterprise-facing operational runbooks for vector migrations and review policies
- continue provider/review/feedback harmonization where any surface still lacks semantic parity

## Recommended Next Technical Themes

After the migration stack lands, the next highest-ROI themes are:

1. knowledge-backed review summaries
2. history real-data evaluation hardening
3. operator runbooks and rollout docs for vector migrations
4. coarse/fine parity audits across any remaining edge endpoints

## Exit Criteria For This Phase

This phase should be considered complete when:

- the migration plan stack is merged
- all related APIs have replayable validation docs
- a maintainer can plan migration execution from API output alone
- benchmark-surpass strategy is documented and can guide the next merge batches
