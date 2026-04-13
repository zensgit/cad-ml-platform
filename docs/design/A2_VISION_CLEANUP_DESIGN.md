# A2: Vision Module Audit & Cleanup

**Date:** 2026-04-13
**Phase:** A2 - Engineering Cleanup
**Scope:** `src/core/vision/` (101 Python modules, ~80K LOC)

## Summary

Audited all 101 Python modules in `src/core/vision/` (excluding `__init__.py`
and `providers/`) to determine which are used in production, which are internal,
and which are unused prototype/experimental code. Moved 11 unused modules
(10,215 LOC) to `src/core/vision/experimental/` to reduce the production API
surface while preserving backward compatibility through re-exports.

## Audit Methodology

For each module, checked:
1. Whether it is imported by code **outside** `src/core/vision/` (in `src/api/`,
   `src/ml/`, `src/core/` excluding vision, `scripts/`, or `tests/`)
2. Whether it is imported by other modules **within** `src/core/vision/`
3. Whether it has dedicated test coverage in `tests/`

Category definitions:
- **PRODUCTION**: Imported by code outside `src/core/vision/`
- **INTERNAL**: Used only by other vision modules (not externally)
- **UNUSED**: Only referenced from `__init__.py` re-exports

## Full Audit Results

| Module | LOC | Category | Reason |
|--------|-----|----------|--------|
| ab_testing | 743 | PRODUCTION | Imported by test_vision_phase5 |
| access_control | 925 | PRODUCTION | Imported by test_vision_phase14, phase19 |
| alert_manager | 824 | PRODUCTION | Imported by test_vision_phase17 |
| analytics | 511 | PRODUCTION | Imported by test_vision_persistence |
| anomaly_detection | 741 | PRODUCTION | Imported by test_vision_phase12 |
| api_gateway | 1103 | PRODUCTION | Imported by test_vision_phase10 |
| api_versioning | 676 | PRODUCTION | Imported by test_vision_phase16 |
| apm_integration | 851 | PRODUCTION | Imported by test_vision_phase17 |
| audit | 633 | PRODUCTION | Imported by test_vision_phase5 |
| **audit_logger** | **666** | **UNUSED** | Duplicate of audit_logging; no external imports |
| audit_logging | 798 | PRODUCTION | Imported by test_vision_phase11 |
| auto_scaling | 732 | PRODUCTION | Imported by test_vision_phase15 |
| **automl_engine** | **893** | **UNUSED** | ML framework stub; no external imports |
| base | 365 | PRODUCTION | Core base classes; imported by providers |
| batch | 285 | PRODUCTION | Imported by test_vision_advanced |
| cache | 355 | PRODUCTION | Imported by test_vision_advanced |
| chaos_engineering | 740 | PRODUCTION | Imported by test_vision_phase9 |
| circuit_breaker | 546 | PRODUCTION | Imported by dedupcad_vision, tests |
| comparison | 304 | PRODUCTION | Imported by test_vision_advanced |
| compliance | 890 | PRODUCTION | Imported by test_vision_phase9 |
| **compliance_checker** | **828** | **UNUSED** | Duplicate of compliance; no external imports |
| config_management | 996 | PRODUCTION | Imported by test_vision_phase8 |
| cost_tracker | 601 | PRODUCTION | Imported by test_vision_extended |
| **data_lifecycle** | **1280** | **UNUSED** | Advanced data management stub; no external imports |
| data_masking | 681 | PRODUCTION | Imported by test_vision_phase14 |
| data_pipeline | 915 | PRODUCTION | Imported by test_vision_phase11 |
| data_validation | 967 | PRODUCTION | Imported by test_vision_phase11 |
| deduplication | 540 | PRODUCTION | Imported by test_vision_phase6 |
| deployment | 943 | PRODUCTION | Imported by test_vision_phase9 |
| discovery | 440 | PRODUCTION | Imported by test_vision_phase6 |
| distributed_cache | 627 | PRODUCTION | Imported by test_vision_phase13 |
| distributed_lock | 1057 | PRODUCTION | Imported by test_vision_phase10 |
| distributed_tracing | 1090 | PRODUCTION | Imported by test_vision_phase8 |
| documentation_generator | 1068 | PRODUCTION | Imported by test_vision_phase16 |
| embedding | 652 | PRODUCTION | Imported by test_vision_phase4 |
| encryption | 738 | PRODUCTION | Imported by test_vision_phase11 |
| **encryption_manager** | **636** | **UNUSED** | Duplicate of encryption; no external imports |
| event_bus | 623 | PRODUCTION | Imported by test_vision_phase13 |
| event_sourcing | 735 | PRODUCTION | Imported by test_vision_phase10 |
| **experiment_tracker** | **945** | **UNUSED** | ML experiment tracking stub; no external imports |
| factory | 338 | PRODUCTION | Core factory; imported by providers |
| failover | 462 | PRODUCTION | Imported by test_vision_persistence |
| feature_flags | 904 | PRODUCTION | Imported by test_vision_phase8 |
| **feature_store** | **873** | **UNUSED** | Feature store stub; only used by data_lifecycle (also unused) |
| graceful_degradation | 876 | PRODUCTION | Imported by test_vision_phase8 |
| health | 714 | PRODUCTION | Imported by test_vision_persistence |
| hot_reload | 520 | PRODUCTION | Imported by test_vision_phase6 |
| integration_hub | 1096 | PRODUCTION | Imported by test_vision_phase16 |
| **intelligent_automation** | **1534** | **UNUSED** | Automation framework stub; no external imports |
| intelligent_routing | 696 | PRODUCTION | Imported by test_vision_phase15 |
| key_management | 780 | PRODUCTION | Imported by test_vision_phase14 |
| knowledge_base | 704 | PRODUCTION | Imported by test_vision_phase15 |
| load_balancer | 551 | PRODUCTION | Imported by test_vision_phase3 |
| log_aggregator | 878 | PRODUCTION | Imported by test_vision_phase17 |
| logging_middleware | 703 | PRODUCTION | Imported by test_vision_phase3 |
| manager | 446 | PRODUCTION | Imported by test_metrics_consistency |
| message_queue | 675 | PRODUCTION | Imported by test_vision_phase13 |
| metrics_dashboard | 849 | PRODUCTION | Imported by test_vision_phase17 |
| metrics_exporter | 644 | PRODUCTION | Imported by test_vision_phase4 |
| middleware | 599 | PRODUCTION | Imported by test_vision_phase7 |
| ml_integration | 682 | PRODUCTION | Imported by test_vision_phase12 |
| **model_registry** | **943** | **UNUSED** | Model registry stub; no external imports |
| multi_tenancy | 877 | PRODUCTION | Imported by test_vision_phase9 |
| multiregion | 533 | PRODUCTION | Imported by test_vision_phase5 |
| observability | 1085 | PRODUCTION | Imported by test_vision_phase9 |
| observability_hub | 1203 | PRODUCTION | Imported by test_vision_phase21 |
| persistence | 497 | PRODUCTION | Imported by test_vision_persistence, analytics |
| **pipeline_orchestrator** | **886** | **UNUSED** | Pipeline orchestrator stub; no external imports |
| plugin_manager | 659 | PRODUCTION | Imported by test_vision_phase16 |
| plugin_system | 1150 | PRODUCTION | Imported by test_vision_phase10 |
| predictive_analytics | 725 | PRODUCTION | Imported by test_vision_phase15 |
| preprocessing | 735 | PRODUCTION | Imported by test_vision_phase3 |
| priority | 622 | PRODUCTION | Imported by test_vision_phase5 |
| privacy_compliance | 882 | PRODUCTION | Imported by test_vision_phase14 |
| profiles | 691 | PRODUCTION | Imported by test_vision_phase3 |
| prompts | 435 | PRODUCTION | Imported by test_vision_extended |
| provider_pool | 503 | PRODUCTION | Imported by test_vision_phase7 |
| quotas | 728 | PRODUCTION | Imported by test_vision_phase4 |
| rate_limiter | 302 | PRODUCTION | Imported by test_vision_advanced |
| recommendations | 769 | PRODUCTION | Imported by test_vision_phase12 |
| reporting | 851 | PRODUCTION | Imported by test_vision_phase12 |
| request_context | 553 | PRODUCTION | Imported by test_vision_phase7 |
| resilience | 334 | PRODUCTION | Imported by test_vision_resilience |
| retry_policy | 508 | PRODUCTION | Imported by test_vision_phase7 |
| saga_pattern | 648 | PRODUCTION | Imported by test_vision_phase13 |
| sdk_generator | 1541 | PRODUCTION | Imported by test_vision_phase16 |
| security_audit | 849 | PRODUCTION | Imported by test_vision_phase14 |
| security_governance | 1349 | PRODUCTION | Imported by test_vision_phase22 |
| **security_scanner** | **731** | **UNUSED** | Duplicate of security_audit; no external imports |
| self_healing | 795 | PRODUCTION | Imported by test_vision_phase15 |
| service_mesh | 1039 | PRODUCTION | Imported by test_vision_phase13 |
| sla_monitor | 916 | PRODUCTION | Imported by test_vision_phase17 |
| stream_processing | 843 | PRODUCTION | Imported by test_vision_phase11 |
| streaming | 485 | PRODUCTION | Imported by test_vision_extended |
| tracing | 663 | PRODUCTION | Imported by test_vision_phase4 |
| transformation | 552 | PRODUCTION | Imported by test_vision_phase6 |
| validation | 694 | PRODUCTION | Imported by test_vision_phase5 |
| versioning | 560 | PRODUCTION | Imported by test_vision_phase6 |
| webhook_handler | 859 | PRODUCTION | Imported by test_vision_phase20 |
| webhooks | 578 | PRODUCTION | Imported by test_vision_extended |
| workflow_engine | 1099 | PRODUCTION | Imported by test_vision_phase10 |

## Modules Moved to experimental/

| Module | LOC | Reason for classification |
|--------|-----|--------------------------|
| audit_logger | 666 | Duplicate of audit_logging (Phase 11) |
| automl_engine | 893 | ML AutoML framework stub, no production consumers |
| compliance_checker | 828 | Duplicate of compliance (Phase 9) |
| data_lifecycle | 1280 | Advanced data management stub, no production consumers |
| encryption_manager | 636 | Duplicate of encryption (Phase 11) |
| experiment_tracker | 945 | ML experiment tracking stub, no production consumers |
| feature_store | 873 | Feature store stub, only used by data_lifecycle (also unused) |
| intelligent_automation | 1534 | Automation framework stub, no production consumers |
| model_registry | 943 | Model registry stub, no production consumers |
| pipeline_orchestrator | 886 | Pipeline orchestrator stub, no production consumers |
| security_scanner | 731 | Duplicate of security_audit (Phase 14) |
| **TOTAL** | **10,215** | |

## LOC Impact

| Metric | Value |
|--------|-------|
| Total modules in `src/core/vision/` (before) | 101 |
| Modules moved to experimental | 11 |
| Production modules remaining | 90 |
| LOC moved to experimental | 10,215 |
| `__init__.py` LOC (before) | 4,021 |
| `__init__.py` LOC (after) | 3,700 |
| Net `__init__.py` reduction | 321 lines (re-exports now route through experimental/) |

## Import Dependency Graph (Production Modules)

Key production dependencies (modules importing from other vision modules):

```
base.py  <--  ab_testing, access_control, alert_manager, analytics, ... (all modules)
persistence.py  <--  analytics.py (ResultPersistence, get_persistence)
factory.py  <--  src/core/providers/vision.py (create_vision_provider)
base.py  <--  src/core/providers/vision.py (VisionDescription, VisionProvider)
circuit_breaker.py  <--  src/core/dedupcad_vision.py
```

Internal experimental dependency:
```
feature_store.py  <--  data_lifecycle.py (TransformationType)
```

## Backward Compatibility

All 11 moved modules are re-exported from `__init__.py` via
`from .experimental.<module> import ...` to maintain backward compatibility.
Existing code that imports `from src.core.vision import AutoMLEngine` (for
example) will continue to work unchanged.

## Verification Results

```
$ python3 -m pytest tests/unit/test_vision_phase*.py -x -q --timeout=30
1489 passed, 8 warnings in 9.95s

$ python3 -m pytest tests/unit/test_vision_*.py tests/unit/test_enterprise_*.py -x -q --timeout=30
2159 passed, 1 skipped, 8 warnings in 21.55s

$ python3 -c "import src.core.vision; print('OK')"
OK
```

All vision phase tests pass (1489). All vision + enterprise tests pass (2159).
No import errors.

Pre-existing failures (unrelated to this change):
- `tests/unit/test_analyze_graph2d_gate_helpers.py` - missing symbol in `src.api.v1.analyze`
- `tests/contract/test_openapi_schema_snapshot.py` - snapshot mismatch (from A1 changes)

## Recommendations for Future Cleanup

1. **Remove backward-compat re-exports**: Once downstream code is confirmed to
   not use the experimental symbols, remove the re-export block at the end of
   `__init__.py` (saves ~350 lines).

2. **Consolidate duplicates**: `audit_logger` vs `audit_logging`, `encryption_manager`
   vs `encryption`, `compliance_checker` vs `compliance`, `security_scanner` vs
   `security_audit` -- these are duplicate implementations. Consider merging the
   best parts into the production module and deleting the experimental duplicate.

3. **Evaluate ML stubs**: `automl_engine`, `experiment_tracker`, `feature_store`,
   `model_registry`, `pipeline_orchestrator` are ML pipeline stubs. If the project
   plans to build ML pipeline functionality, these could be promoted. Otherwise
   they should be deleted.

4. **Split large modules**: Several production modules exceed 1000 LOC
   (`api_gateway` 1103, `distributed_tracing` 1090, `distributed_lock` 1057,
   `documentation_generator` 1068, `observability` 1085, `observability_hub` 1203,
   `plugin_system` 1150, `security_governance` 1349, `sdk_generator` 1541,
   `workflow_engine` 1099). These are candidates for the same split treatment
   applied in A1.

5. **Reduce __init__.py further**: At 3700 lines, `__init__.py` is still very
   large. Consider a lazy-loading approach or splitting re-exports into
   sub-packages.
