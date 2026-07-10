# Phase 0 · A2b — Vision package prune (VERIFICATION)

**Date:** 2026-07-09 · **Design:** `PHASE0_A2B_VISION_PRUNE_DESIGN_20260709.md`
**Branch:** `phase0-a2b-vision-prune-20260709` · **Grounded on:** `origin/main` @ `0af2fa80`

## Result

| Metric | Before | After |
|---|---:|---:|
| `src/core/vision/*.py` files | 112 | 14 |
| `src/core/vision/__init__.py` lines | 3705 | 117 |
| `__init__` re-export statements | 465 | 5 |
| `__all__` exports | ~1780 | 28 |
| Orphaned phase tests | 25 | 0 |
| Net diff | — | **+123 / −103k** (125 files) |

## Kept set (14) — the live core

`__init__.py`, `base.py`, `factory.py`, `manager.py`, `resilience.py`,
`providers/{__init__, anthropic, deepseek, deepseek_stub, doubao, glm4v, openai, qwen_vl,
vllm_vision}.py`.

## Evidence (execute, don't just compile)

### 1. Runtime import-smoke — THE gate (Python 3.9 local; passed)
```
import src.core.vision                          → OK
from src.core.vision import (ResilientVisionProvider, VisionAnalyzeRequest,
    VisionAnalyzeResponse, VisionInputError, VisionManager, VisionProviderError,
    create_vision_provider, get_available_providers, VisionDescription)   → OK
create_vision_provider('stub')   → DeepSeekStubProvider
get_available_providers()        → 7 providers
```
This is the exact check that caught the 2026-07-08 unsafe delete-set (`ModuleNotFoundError`); it is
now green. `py_compile` alone is **not** relied upon (it missed that error).

### 2. No dangling references
AST scan of `src/`, `tests/`, `scripts/`: **zero** surviving imports of any deleted vision module.
Positive control: the same scan flags `metrics_dashboard`/`circuit_breaker` when a probe importing
them is added.

### 3. Import-closure soundness
- No kept (non-`__init__`) module imports a delete-set module (empirical AST check).
- Zero `importlib`/`__import__`/`import_module` in the kept set → static closure is complete.
- Live downstream `src.core.providers.vision` adapter imports cleanly post-prune.

### 4. prune-safety gate — GREEN + observed-RED
```
GREEN:  prune-safety: OK (105 pruned modules unreferenced, 9 live twins intact, 1922 files scanned)
RED #1: resurrect core.vision.metrics_dashboard  → ::error:: … FAIL   (gate fires)
RED #2: resurrect core.vision.circuit_breaker    → ::error:: … FAIL   (moved twin guarded)
GREEN:  restored after probe removal
```

### 5. Kept vision tests
`tests/test_contract_schema.py` + `tests/test_metrics_consistency.py` **collect cleanly** (17 tests,
no import errors) — `base`/`manager`/`deepseek_stub` coverage retained.

## Known local-env limitation (not an A2b regression)

Full `pytest` execution errors locally under **Python 3.9** inside `conftest.py`'s autouse fixture,
which imports `src/api/v1/vectors.py` → a FastAPI forward-ref that uses `X | None` runtime union
syntax (3.10+ only): `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`. **This
reproduces identically on pristine `main` with no A2b changes** and is entirely in the vectors/API
subsystem, not vision. CI runs Python 3.11, where it does not occur. CI is the final arbiter for the
full suite; local evidence above is A2b-specific and sufficient.

## Delete manifest (98 files)

`experimental/` subtree (12: `__init__`, audit_logger, automl_engine, compliance_checker,
data_lifecycle, encryption_manager, experiment_tracker, feature_store, intelligent_automation,
model_registry, pipeline_orchestrator, security_scanner) + 86 flat modules:
ab_testing, access_control, alert_manager, analytics, anomaly_detection, api_gateway, api_versioning,
apm_integration, audit, audit_logging, auto_scaling, batch, cache, chaos_engineering, circuit_breaker,
comparison, compliance, config_management, cost_tracker, data_masking, data_pipeline, data_validation,
deduplication, deployment, discovery, distributed_cache, distributed_lock, distributed_tracing,
documentation_generator, embedding, encryption, event_bus, event_sourcing, failover, feature_flags,
graceful_degradation, health, hot_reload, integration_hub, intelligent_routing, key_management,
knowledge_base, load_balancer, log_aggregator, logging_middleware, message_queue, metrics_dashboard,
metrics_exporter, middleware, ml_integration, multi_tenancy, multiregion, observability,
observability_hub, persistence, plugin_manager, plugin_system, predictive_analytics, preprocessing,
priority, privacy_compliance, profiles, prompts, provider_pool, quotas, rate_limiter,
recommendations, reporting, request_context, retry_policy, saga_pattern, sdk_generator, security_audit,
security_governance, self_healing, service_mesh, sla_monitor, stream_processing, streaming, tracing,
transformation, validation, versioning, webhook_handler, webhooks, workflow_engine.

Orphan tests deleted (25): `test_vision_phase{3..24}` (per-phase set),
`test_vision_advanced`, `test_vision_extended`, `test_vision_persistence`.

> **CI-caught addendum:** the first push deleted 21; CI (3.11) collection then flagged
> `test_vision_phase{18,19,23,24}` — they import scaffolding names via the *top-package*
> form `from src.core.vision import AccessController` rather than a submodule path, which the
> initial static orphan scan (submodule-imports only) missed. Re-scan now covers **both** import
> styles and is clean. Another instance of "execute (CI collection), don't just static-scan".

## Follow-ups

- CI (3.11) full-suite green is the merge gate; this PR is for-review only (never self-merged).
- A4 forward-guard mechanics (diff-scoped hard gate) remain a separate branch; arming as a required
  check is owner-only.
