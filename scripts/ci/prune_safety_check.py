#!/usr/bin/env python3
"""Prune-safety gate — Phase 0 de-bloat.

Mandated by docs/PLATFORM_POSITIONING_AND_ROADMAP_DESIGN_20260706.md (§3 轨A, §5):
Phase 0 deletions must be defended by a *blocking* CI check, not a manual grep,
because pervasive fail-soft CI would otherwise report green on a mis-delete.

Two invariants, both hard-fail:

  1. NO RESURRECTION — none of the pruned module paths may be imported again.
     (Also stops the agent fleet from re-inflating the scaffolding it deleted.)

  2. NO MIS-DELETE — the same-named *live* modules must still exist. Several
     pruned dirs had live twins (e.g. src/core/circuit_breaker/ was dead, but
     src/utils/circuit_breaker.py is live). A future prune that confuses them
     would break real imports; this asserts the twins survive.
     (src/core/vision/circuit_breaker.py was itself pruned in A2b — its live
     generic twin is src/core/resilience/advanced_circuit_breaker.py.)

Runtime import smoke is intentionally NOT duplicated here: the existing test
suite already imports the app (tests/test_routes_smoke.py), so a broken import
reds CI on its own.

Exit 0 = safe. Exit 1 = violation. No `|| true` anywhere.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Module paths removed in Phase 0 slice 1. Never import these again.
PRUNED_MODULES: tuple[str, ...] = (
    "core.circuit_breaker",
    "core.dead_letter_queue",
    "core.outbox",
    "core.message_bus",
    "core.idempotency",
    "core.api_versioning",
    "core.rate_limiter",
    "core.webhook",
    "core.caching",
    "core.batch_processing",
    "core.event_sourcing",
    "core.health_check",
    "core.notifications",
    "api.v2",
    "api.grpc",
    "api.v1.batch",
    "api.v1.websocket",
    # Phase 0 slice B1 (#502): orphan FeedbackLearningPipeline deleted; guard against resurrection.
    "ml.learning.feedback_loop",
    # Phase 0 slice A2b (#A2b): vision observability/scaffolding zoo deleted
    # (98 modules; package-aware import-closure prune). Guard the fleet from
    # re-inflating it. circuit_breaker moved here from LIVE_TWINS — its live
    # generic twin now lives at src/core/resilience/advanced_circuit_breaker.py (#501).
    "core.vision.ab_testing",
    "core.vision.access_control",
    "core.vision.alert_manager",
    "core.vision.analytics",
    "core.vision.anomaly_detection",
    "core.vision.api_gateway",
    "core.vision.api_versioning",
    "core.vision.apm_integration",
    "core.vision.audit",
    "core.vision.audit_logging",
    "core.vision.auto_scaling",
    "core.vision.batch",
    "core.vision.cache",
    "core.vision.chaos_engineering",
    "core.vision.circuit_breaker",
    "core.vision.comparison",
    "core.vision.compliance",
    "core.vision.config_management",
    "core.vision.cost_tracker",
    "core.vision.data_masking",
    "core.vision.data_pipeline",
    "core.vision.data_validation",
    "core.vision.deduplication",
    "core.vision.deployment",
    "core.vision.discovery",
    "core.vision.distributed_cache",
    "core.vision.distributed_lock",
    "core.vision.distributed_tracing",
    "core.vision.documentation_generator",
    "core.vision.embedding",
    "core.vision.encryption",
    "core.vision.event_bus",
    "core.vision.event_sourcing",
    "core.vision.failover",
    "core.vision.feature_flags",
    "core.vision.graceful_degradation",
    "core.vision.health",
    "core.vision.hot_reload",
    "core.vision.integration_hub",
    "core.vision.intelligent_routing",
    "core.vision.key_management",
    "core.vision.knowledge_base",
    "core.vision.load_balancer",
    "core.vision.log_aggregator",
    "core.vision.logging_middleware",
    "core.vision.message_queue",
    "core.vision.metrics_dashboard",
    "core.vision.metrics_exporter",
    "core.vision.middleware",
    "core.vision.ml_integration",
    "core.vision.multi_tenancy",
    "core.vision.multiregion",
    "core.vision.observability",
    "core.vision.observability_hub",
    "core.vision.persistence",
    "core.vision.plugin_manager",
    "core.vision.plugin_system",
    "core.vision.predictive_analytics",
    "core.vision.preprocessing",
    "core.vision.priority",
    "core.vision.privacy_compliance",
    "core.vision.profiles",
    "core.vision.prompts",
    "core.vision.provider_pool",
    "core.vision.quotas",
    "core.vision.rate_limiter",
    "core.vision.recommendations",
    "core.vision.reporting",
    "core.vision.request_context",
    "core.vision.retry_policy",
    "core.vision.saga_pattern",
    "core.vision.sdk_generator",
    "core.vision.security_audit",
    "core.vision.security_governance",
    "core.vision.self_healing",
    "core.vision.service_mesh",
    "core.vision.sla_monitor",
    "core.vision.stream_processing",
    "core.vision.streaming",
    "core.vision.tracing",
    "core.vision.transformation",
    "core.vision.validation",
    "core.vision.versioning",
    "core.vision.webhook_handler",
    "core.vision.webhooks",
    "core.vision.workflow_engine",
    "core.vision.experimental",  # whole subpackage
)

# Same-named LIVE modules. Deleting any of these is a mis-delete, not de-bloat.
LIVE_TWINS: tuple[str, ...] = (
    "src/utils/circuit_breaker.py",
    "src/utils/idempotency.py",
    "src/utils/rate_limiter.py",
    "src/core/assistant/caching.py",
    "src/core/resilience/circuit_breaker.py",
    "src/core/resilience_enhanced/circuit_breaker.py",
    "src/core/gateway/circuit_breaker.py",
    "src/core/resilience/rate_limiter.py",
    "src/core/gateway/rate_limiter.py",
)

SCAN_ROOTS: tuple[str, ...] = ("src", "tests", "scripts")


def _importers_of(module: str, files: list[Path]) -> list[str]:
    # Matches `import src.core.outbox`, `from core.outbox import X`, submodules too.
    pattern = re.compile(
        rf"^\s*(?:from|import)\s+(?:src\.)?{re.escape(module)}(?:[\s.]|$)",
        re.MULTILINE,
    )
    hits: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.match(line):
                hits.append(f"{path}:{lineno}: {line.strip()}")
    return hits


def main() -> int:
    failed = False

    py_files = [
        p
        for root in SCAN_ROOTS
        if Path(root).is_dir()
        for p in Path(root).rglob("*.py")
    ]

    # Invariant 1 — no resurrection of pruned modules.
    for module in PRUNED_MODULES:
        hits = _importers_of(module, py_files)
        if hits:
            failed = True
            print(f"::error::pruned module '{module}' is imported again:")
            for hit in hits[:10]:
                print(f"    {hit}")

    # Invariant 2 — live twins must survive.
    for twin in LIVE_TWINS:
        if not Path(twin).exists():
            failed = True
            print(
                f"::error::live module '{twin}' is missing — it shares a name with a "
                "pruned scaffold but is NOT dead code. This is a mis-delete."
            )

    if failed:
        print("\nprune-safety: FAIL")
        return 1

    print(
        f"prune-safety: OK "
        f"({len(PRUNED_MODULES)} pruned modules unreferenced, "
        f"{len(LIVE_TWINS)} live twins intact, {len(py_files)} files scanned)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
