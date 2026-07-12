#!/usr/bin/env python3
"""L3 fail-closed gate: the retraining / model-promotion path is DISABLED.

WHY THIS GATE IS UNCONDITIONAL — AND WHY IT DELIBERATELY HAS NO BYPASS
---------------------------------------------------------------------
`scripts/auto_retrain.sh` evaluates a candidate model against
`data/manifests/golden_val_set.csv` and stamps "Ready for deployment" at a 91.5%
threshold. That validation set carries **262/914 (28.7%)** rows whose *bytes* are
identical to training rows (PRODUCT_STRATEGY.md §5.2). Accuracy computed on it is not
release-grade, so no model may be promoted on it.

An earlier draft of this gate accepted a JSON "evaluation-integrity" artifact as a
"you may proceed" token. That shape is wrong for two independent reasons:

1. **The token was unbound.** It carried neither a digest of the validation manifest
   actually used nor the hash of the candidate model actually promoted, while
   `auto_retrain.sh` independently chooses `GOLDEN_VAL`. An artifact produced for
   dataset A could therefore green-light a model evaluated on dataset B — a confused
   deputy.

2. **The token did not even need to be forged.** The sanctioned producer emitted a
   *passing* artifact with `holdout_rows: 0`, all-zero metrics, and a hard-coded
   `reproducible: true`. A gate built to stop fake-green was itself fake-green.

You cannot offer a bypass whose validity you cannot bind. The narrowest correct v1
therefore has **no pass path at all**: this gate always blocks.

RE-ENABLEMENT IS A CODE CHANGE, NOT A FLAG
------------------------------------------
No argument, environment variable, or file opens this gate. Re-enabling retraining means
*replacing the body of* `check()` with the real two-phase Track E gate
(PRODUCT_STRATEGY.md §8.1):

  pre-training :  validated manifest + content/family/label digest + NON-EMPTY holdout
  post-training:  result bound to (candidate-model hash, split digest, evaluator version,
                  thresholds) before any "Ready for deployment" is emitted

This module is kept as the *seam*: future work replaces `check()`, it does not re-wire
the pipeline.

Stdlib only, so the fail-closed path runs without the ML stack installed.
"""

from __future__ import annotations

import sys
from typing import List, Optional

_STRATEGY_REF = (
    "See docs/PRODUCT_STRATEGY.md §5.2 (evaluation integrity is not release-grade) "
    "and §8.1 (Track E: evaluation-integrity-v2)."
)


class GateBlocked(RuntimeError):
    """Raised unconditionally. This gate has no success path."""


def check() -> None:
    """Always raises ``GateBlocked``.

    There is deliberately no parameter, environment variable, or artifact that makes
    this return. Replacing this body with the real two-phase gate is the ONLY way to
    re-enable retraining / model promotion.
    """
    raise GateBlocked(
        "the retraining / model-promotion path is fail-closed: the golden validation "
        "split carries 262/914 (28.7%) rows byte-identical to training rows, so no "
        "accuracy computed on it may promote a model"
    )


def main(argv: Optional[List[str]] = None) -> int:
    # argv is accepted and IGNORED on purpose: no flag may change the outcome. Accepting
    # `--artifact` / `--force` / anything else must not create the illusion of a bypass.
    del argv

    try:
        check()
    except GateBlocked as exc:
        sys.stderr.write(
            f"EVALUATION-INTEGRITY GATE (fail-closed): {exc}\n"
            "Retraining and model promotion are DISABLED. This gate has no bypass: "
            "no artifact, no environment toggle, no flag opens it.\n"
            "Re-enablement requires replacing eval_integrity_gate.check() with the "
            "two-phase Track E gate.\n"
            f"{_STRATEGY_REF}\n"
        )
        return 1

    # Unreachable by construction. Kept — and still non-zero — so a future edit that
    # accidentally makes check() return cannot silently open the gate without also
    # having to change this line.
    sys.stderr.write(
        "EVALUATION-INTEGRITY GATE: invariant breach — check() returned without "
        "raising. Refusing to allow promotion.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
