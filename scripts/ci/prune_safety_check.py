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
     src/utils/circuit_breaker.py and src/core/vision/circuit_breaker.py are
     live). A future prune that confuses them would break real imports; this
     asserts the twins survive.

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
    "src/core/vision/circuit_breaker.py",
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
