#!/usr/bin/env python3
"""Run the AI quality evaluation suite and save a Markdown report.

Usage:
    python scripts/run_ai_evaluation.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path so that ``src.*`` imports work when
# invoked directly with ``python scripts/run_ai_evaluation.py``.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.ml.evaluation.ai_eval import AIEvaluationSuite  # noqa: E402


def main() -> None:
    """Run the full evaluation and print + save the report."""
    suite = AIEvaluationSuite()
    results = asyncio.run(suite.run_full_evaluation())
    report = suite.generate_report(results)

    # Print to stdout
    print(report)
    print()

    # Summary line
    verdict = results["verdict"]
    score = results["overall_score"]
    total = results["total_cases"]
    print(f"[{verdict}] overall={score:.1%}  cases={total}")

    # Save to reports/
    reports_dir = _project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"ai_evaluation_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to {report_path}")

    # Also write a stable-name symlink for CI consumption
    latest_path = reports_dir / "ai_evaluation.md"
    latest_path.write_text(report, encoding="utf-8")
    print(f"Latest report at  {latest_path}")

    # Exit non-zero when verdict is not PASS
    if verdict != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
