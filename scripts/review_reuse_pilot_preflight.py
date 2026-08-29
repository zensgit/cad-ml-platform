#!/usr/bin/env python3
"""Advisory preflight for ReviewReuse pilot env posture.

Reads the process environment and prints a posture A/B/C-style summary.
Does **not** set or enable REVIEW_REUSE_DECISIONS_ENABLED.

Exit codes
----------
0  Advisory OK (including safe default-off posture).
2  Dangerous / inconsistent pilot combo detected (warn + fail).

Dangerous combos (exit 2)
-------------------------
* DECISIONS=true AND INTEGRATION_AUTH_MODE != required
* DECISIONS=true AND JWT secret/audience/issuer is incomplete

Examples::

  python scripts/review_reuse_pilot_preflight.py
  make review-reuse-preflight
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.review_reuse.store import (
    ReviewReuseStoreError,
    validated_filesystem_tenants,
)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FILESYSTEM_BACKENDS = frozenset({"fs", "file", "filesystem", "disk"})

ENV_DECISIONS = "REVIEW_REUSE_DECISIONS_ENABLED"
ENV_REQUIRE_VALIDATED = "REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER"
ENV_LIVE_DEDUP = "REVIEW_REUSE_LIVE_DEDUP"
ENV_STORE = "REVIEW_REUSE_STORE"
ENV_STORE_DIR = "REVIEW_REUSE_STORE_DIR"
ENV_AUTH_MODE = "INTEGRATION_AUTH_MODE"
ENV_JWT_SECRET = "INTEGRATION_JWT_SECRET"
ENV_JWT_AUDIENCE = "INTEGRATION_JWT_AUDIENCE"
ENV_JWT_ISSUER = "INTEGRATION_JWT_ISSUER"


def _truthy(raw: Optional[str]) -> bool:
    return (raw or "").strip().lower() in _TRUE


def _get(env: Mapping[str, str], key: str, default: str = "") -> str:
    return (env.get(key) if key in env else default) or default


def read_posture(env: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    """Snapshot pilot-relevant flags from *env* (defaults to os.environ)."""
    e = env if env is not None else os.environ
    decisions = _truthy(_get(e, ENV_DECISIONS))
    require_validated = _truthy(_get(e, ENV_REQUIRE_VALIDATED))
    live_dedup = _truthy(_get(e, ENV_LIVE_DEDUP))
    store = (_get(e, ENV_STORE, "memory") or "memory").strip().lower() or "memory"
    store_dir = (_get(e, ENV_STORE_DIR, "data/review_reuse_tasks") or "").strip()
    auth_mode = (
        _get(e, ENV_AUTH_MODE, "disabled") or "disabled"
    ).strip().lower() or "disabled"
    jwt_issuer = _get(e, ENV_JWT_ISSUER)
    jwt_identity_complete = all(
        _get(e, key).strip()
        for key in (ENV_JWT_SECRET, ENV_JWT_AUDIENCE, ENV_JWT_ISSUER)
    )
    jwt_issuer_exact = jwt_issuer == jwt_issuer.strip()
    store_integrity = "not_applicable"
    store_tenants = 0
    if store in _FILESYSTEM_BACKENDS:
        if not store_dir:
            store_integrity = "invalid:store_record_corrupt"
        else:
            try:
                store_tenants = len(validated_filesystem_tenants(Path(store_dir)))
                store_integrity = "valid"
            except ReviewReuseStoreError as exc:
                store_integrity = f"invalid:{exc.code}"

    if not decisions and not live_dedup:
        label = "A"
        label_name = "offline_exercise"
    elif not decisions and (live_dedup or store in _FILESYSTEM_BACKENDS):
        label = "B"
        label_name = "evidence_only_pilot"
    elif decisions:
        label = "C"
        label_name = "decision_window"
    else:
        label = "A"
        label_name = "offline_exercise"

    return {
        "posture": label,
        "posture_name": label_name,
        "decisions": decisions,
        "require_validated": require_validated,
        "live_dedup": live_dedup,
        "store": store,
        "store_dir": store_dir,
        "integration_auth_mode": auth_mode,
        "jwt_identity_complete": jwt_identity_complete,
        "jwt_issuer_exact": jwt_issuer_exact,
        "store_integrity": store_integrity,
        "store_tenants": store_tenants,
        "raw": {
            ENV_DECISIONS: e.get(ENV_DECISIONS),
            ENV_REQUIRE_VALIDATED: e.get(ENV_REQUIRE_VALIDATED),
            ENV_LIVE_DEDUP: e.get(ENV_LIVE_DEDUP),
            ENV_STORE: e.get(ENV_STORE),
            ENV_STORE_DIR: e.get(ENV_STORE_DIR),
            ENV_AUTH_MODE: e.get(ENV_AUTH_MODE),
            ENV_JWT_SECRET: "<configured>" if _get(e, ENV_JWT_SECRET).strip() else None,
            ENV_JWT_AUDIENCE: e.get(ENV_JWT_AUDIENCE),
            ENV_JWT_ISSUER: e.get(ENV_JWT_ISSUER),
        },
    }


def check_dangerous_combos(posture: Mapping[str, object]) -> List[str]:
    """Return human-readable warnings for unsafe pilot combinations."""
    warnings: List[str] = []
    decisions = bool(posture.get("decisions"))
    store_integrity = str(posture.get("store_integrity") or "not_applicable")
    if store_integrity.startswith("invalid:"):
        warnings.append(
            "DANGER: ReviewReuse filesystem store integrity check failed "
            f"({store_integrity.partition(':')[2]})."
        )
    if not decisions:
        return warnings

    auth_mode = str(posture.get("integration_auth_mode") or "disabled").lower()

    if auth_mode != "required":
        warnings.append(
            f"DANGER: REVIEW_REUSE_DECISIONS_ENABLED=true but "
            f"INTEGRATION_AUTH_MODE={auth_mode!r} (not 'required') — "
            "pilot risk: JWT-validated subjects not enforced at the edge. "
            "Set INTEGRATION_AUTH_MODE=required for decision windows."
        )
    elif not bool(posture.get("jwt_identity_complete")):
        warnings.append(
            "DANGER: REVIEW_REUSE_DECISIONS_ENABLED=true but JWT secret, audience, "
            "or issuer is missing; validated tenant/reviewer principals cannot be built."
        )
    elif not bool(posture.get("jwt_issuer_exact")):
        warnings.append(
            "DANGER: REVIEW_REUSE_DECISIONS_ENABLED=true but JWT issuer has "
            "surrounding whitespace; runtime identity validation will reject it."
        )
    return warnings


def format_summary(posture: Mapping[str, object], warnings: Sequence[str]) -> str:
    lines = [
        "ReviewReuse pilot preflight (advisory; does not enable decisions)",
        f"  posture:              {posture['posture']} ({posture['posture_name']})",
        f"  decisions:            {posture['decisions']}",
        f"  require_validated:    {posture['require_validated']}",
        f"  live_dedup:           {posture['live_dedup']}",
        f"  store:                {posture['store']}",
        f"  store_dir:            {posture['store_dir']}",
        f"  integration_auth_mode:{posture['integration_auth_mode']}",
        f"  jwt_identity_complete:{posture['jwt_identity_complete']}",
        f"  jwt_issuer_exact:      {posture['jwt_issuer_exact']}",
        f"  store_integrity:       {posture['store_integrity']}",
        f"  store_tenants:         {posture['store_tenants']}",
    ]
    if warnings:
        lines.append("  warnings:")
        for w in warnings:
            lines.append(f"    - {w}")
    else:
        lines.append("  warnings:             none")
    return "\n".join(lines)


def run_preflight(env: Optional[Mapping[str, str]] = None) -> Tuple[int, str]:
    """Evaluate env; return (exit_code, summary_text). Never mutates env."""
    posture = read_posture(env)
    warnings = check_dangerous_combos(posture)
    text = format_summary(posture, warnings)
    return (2 if warnings else 0, text)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    # Read-only: never set DECISIONS or other pilot flags.
    code, text = run_preflight(os.environ)
    print(text)
    if code != 0:
        print(
            "preflight: FAIL (exit 2) — fix dangerous combo before pilot decision window",
            file=sys.stderr,
        )
    else:
        print("preflight: OK (exit 0)", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
