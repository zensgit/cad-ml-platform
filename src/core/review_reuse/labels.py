"""Closed-set pilot label reason codes (Day 61–90 review-workflow metrics).

These are recorded on ``HumanDecision.reason_codes`` when decisions are
owner-enabled. They do **not** enable decisions and are not Track E metrics.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

# Closed set — keep in sync with metrics aggregation.
REASON_FALSE_DUPLICATE = "false_duplicate"
REASON_MISSED_REUSE = "missed_reuse"
REASON_USEFULNESS_PREFIX = "usefulness:"  # usefulness:1 .. usefulness:5


def parse_usefulness(reason_codes: Iterable[str]) -> Optional[int]:
    """Return the last valid usefulness:1..5 label, if any."""
    found: Optional[int] = None
    for raw in reason_codes or []:
        code = str(raw).strip().lower()
        if not code.startswith(REASON_USEFULNESS_PREFIX):
            continue
        tail = code[len(REASON_USEFULNESS_PREFIX) :]
        if tail.isdigit():
            n = int(tail)
            if 1 <= n <= 5:
                found = n
    return found


def count_label(reason_codes: Iterable[str], label: str) -> int:
    target = str(label).strip().lower()
    return sum(1 for raw in reason_codes or [] if str(raw).strip().lower() == target)


def extract_pilot_labels(reason_codes: Iterable[str]) -> Tuple[bool, bool, Optional[int]]:
    codes: List[str] = [str(c) for c in reason_codes or []]
    return (
        count_label(codes, REASON_FALSE_DUPLICATE) > 0,
        count_label(codes, REASON_MISSED_REUSE) > 0,
        parse_usefulness(codes),
    )
