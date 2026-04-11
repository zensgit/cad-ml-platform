#!/usr/bin/env python3
"""CI/cron-friendly single refresh entry for the eval reporting stack.

Sequentially calls:
1. generate_eval_reporting_bundle
2. check_eval_reporting_bundle_health
3. generate_eval_reporting_index
4. generate_eval_reporting_landing_page

Fail-closed: any step failure stops the pipeline immediately.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci import generate_eval_reporting_bundle as bundle_mod
from scripts.ci import check_eval_reporting_bundle_health as health_mod
from scripts.ci import generate_eval_reporting_index as index_mod
from scripts import generate_eval_reporting_landing_page as landing_mod


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the full eval reporting stack (bundle + health + index)."
    )
    parser.add_argument(
        "--eval-history-dir",
        default="reports/eval_history",
        help="Directory containing eval-history artifacts.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Weekly summary rolling window in days.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    eval_history_dir = str(args.eval_history_dir)
    days_str = str(max(1, int(args.days)))

    # 1. materialize bundle
    print("=== Step 1/4: materialize eval reporting bundle ===")
    rc = bundle_mod.main([
        "--eval-history-dir", eval_history_dir,
        "--days", days_str,
    ])
    if rc != 0:
        print(f"eval reporting bundle failed with rc={rc}")
        return rc

    # 2. health check (fail-closed)
    print("=== Step 2/4: check eval reporting bundle health ===")
    health_rc = health_mod.main([
        "--eval-history-dir", eval_history_dir,
    ])
    if health_rc != 0:
        print(f"health check failed with rc={health_rc}")
        return health_rc

    # 3. generate index
    print("=== Step 3/4: generate eval reporting index ===")
    rc = index_mod.main([
        "--eval-history-dir", eval_history_dir,
    ])
    if rc != 0:
        print(f"eval reporting index failed with rc={rc}")
        return rc

    # 4. generate landing page
    print("=== Step 4/4: generate eval reporting landing page ===")
    rc = landing_mod.main([
        "--eval-history-dir", eval_history_dir,
    ])
    if rc != 0:
        print(f"eval reporting landing page failed with rc={rc}")
        return rc

    print("Eval reporting stack refresh complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
