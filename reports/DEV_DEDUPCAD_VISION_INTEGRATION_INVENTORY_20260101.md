# DedupCAD Vision Integration Inventory (2026-01-01)

## Scope

- Inventory cad-ml-platform ↔ dedupcad-vision integration points, configuration, and code paths.

## Findings

- Core client: `src/core/dedupcad_vision.py` (health/search/index add/rebuild; retry + circuit).
- API router: `src/api/v1/dedup.py` (health proxy, search, async jobs, index add, tenant config).
- Pipeline orchestration: `src/core/dedupcad_2d_pipeline.py` (vision recall + local L4 precision).
- Worker integration: `src/core/dedupcad_2d_worker.py` (async jobs, rendering fallback, webhook).
- Metrics: `src/utils/analysis_metrics.py` (dedupcad-vision request/latency/errors/retry/circuit).
- Contract docs: `docs/DEDUP2D_VISION_INTEGRATION_CONTRACT.md` and runbooks/checklists.
- E2E tooling: `scripts/e2e_dedup2d_secure_callback.sh`, `scripts/e2e_dedup2d_webhook*.py`.
- Configuration: `DEDUPCAD_VISION_URL`, `DEDUPCAD_VISION_TIMEOUT_SECONDS`, retry/circuit envs, docker compose defaults.

## Verification

- Code inspection only (no tests run for this step).
