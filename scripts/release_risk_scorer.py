#!/usr/bin/env python3
"""
Release Risk Scorer
Scores a pending release/PR on 0–100 and classifies risk level.

Usage (typical in CI):
  python scripts/release_risk_scorer.py \
    --base-branch main \
    --output-format json \
    --output-file risk_report.json

It internally invokes scripts/release_data_collector.py to compute deltas.
Outputs either JSON or Markdown per --output-format.

Risk levels:
  - LOW (<40), MEDIUM (<60), HIGH (<85), CRITICAL (>=85)
Blocking policy is true when score >=85 (configurable via env RELEASE_RISK_BLOCK_THRESHOLD)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class ScoreWeights:
    # 8 dimensions, normalized to sum ~1.0
    changes: float = 0.18        # files/lines churn
    tests: float = 0.22          # failures/errors dominate
    deps: float = 0.12           # new/removed deps
    error_codes: float = 0.16    # new/removals in ErrorCode enums
    metrics: float = 0.14        # new metrics (cardinality risk)
    workflows: float = 0.08      # workflow changes
    scripts: float = 0.05        # operational scripts
    docs_signal: float = 0.05    # docs vs code balance (weak signal)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def ensure_collector() -> Path:
    path = Path("scripts/release_data_collector.py")
    if not path.exists():
        raise FileNotFoundError("scripts/release_data_collector.py not found")
    return path


def collect(base_branch: str) -> Dict[str, Any]:
    ensure_collector()
    tmp = Path("risk_data.json")
    cmd = [
        "python3",
        "scripts/release_data_collector.py",
        "--base-branch",
        base_branch,
        "--output",
        str(tmp),
    ]
    try:
        run(cmd)
        data = json.loads(tmp.read_text(encoding="utf-8"))
        return data
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _cap01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_changes(d: Dict[str, Any]) -> float:
    files = d["git"]["files_changed"]
    lines = d["git"]["lines_added"] + d["git"]["lines_deleted"]
    # Normalize: files 0→0, 150+ → 1; lines 0→0, 6k+ → 1
    sf = _cap01(files / 150.0)
    sl = _cap01(lines / 6000.0)
    return 0.4 * sf + 0.6 * sl


def score_tests(d: Dict[str, Any]) -> float:
    t = d.get("tests", {})
    total = max(1, int(t.get("total", 0)))
    failed = int(t.get("failed", 0)) + int(t.get("errors", 0))
    skipped = int(t.get("skipped", 0))
    # Failure ratio (errors weigh heavier)
    fr = _cap01((failed * 1.0) / total)
    # Low test volume (<20) increases uncertainty risk slightly
    volume_penalty = 0.15 if total < 20 else 0.0
    # Excessive skipped tests add risk
    skip_ratio = _cap01(skipped / total)
    return _cap01(fr * 0.75 + skip_ratio * 0.1 + volume_penalty)


def score_deps(d: Dict[str, Any]) -> float:
    added = int(d["deps"]["added"]) if d.get("deps") else 0
    removed = int(d["deps"]["removed"]) if d.get("deps") else 0
    return _cap01((added * 2 + removed) / 20.0)


def score_error_codes(d: Dict[str, Any]) -> float:
    e = d.get("errors", {})
    added = int(e.get("added", 0))
    removed = int(e.get("removed", 0))
    # New codes weigh more than removals (client compatibility)
    return _cap01((added * 3 + removed) / 30.0)


def score_metrics(d: Dict[str, Any]) -> float:
    m = d.get("metrics", {})
    added = int(m.get("added", 0))
    removed = int(m.get("removed", 0))
    return _cap01((added * 2 + removed) / 40.0)


def score_workflows(d: Dict[str, Any]) -> float:
    area = d["git"]["by_area"]
    wf_changes = int(area.get("workflows", 0))
    return _cap01(wf_changes / 6.0)


def score_scripts(d: Dict[str, Any]) -> float:
    script_changes = int(d["git"]["by_area"].get("scripts", 0))
    return _cap01(script_changes / 20.0)


def score_docs_signal(d: Dict[str, Any]) -> float:
    area = d["git"]["by_area"]
    docs = int(area.get("docs", 0))
    code = int(area.get("src", 0))
    # More docs relative to code reduces risk marginally
    if code <= 0:
        return 0.3  # unknown code change amount: small neutral risk
    ratio = docs / max(1, code)
    # If docs >= code, reduce risk signal; else slight risk
    return _cap01(0.6 - min(0.4, ratio * 0.2))


def aggregate_score(d: Dict[str, Any], weights: ScoreWeights) -> Tuple[float, Dict[str, float]]:
    parts = {
        "changes": score_changes(d),
        "tests": score_tests(d),
        "deps": score_deps(d),
        "error_codes": score_error_codes(d),
        "metrics": score_metrics(d),
        "workflows": score_workflows(d),
        "scripts": score_scripts(d),
        "docs_signal": score_docs_signal(d),
    }
    score01 = (
        parts["changes"] * weights.changes
        + parts["tests"] * weights.tests
        + parts["deps"] * weights.deps
        + parts["error_codes"] * weights.error_codes
        + parts["metrics"] * weights.metrics
        + parts["workflows"] * weights.workflows
        + parts["scripts"] * weights.scripts
        + parts["docs_signal"] * weights.docs_signal
    )
    return _cap01(score01) * 100.0, parts


def classify_level(score: float) -> str:
    if score >= 85:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def blocking_decision(score: float) -> bool:
    threshold = float(os.getenv("RELEASE_RISK_BLOCK_THRESHOLD", "85"))
    return score >= threshold


def suggestions(score: float, parts: Dict[str, float], d: Dict[str, Any]) -> list[str]:
    tips: list[str] = []
    if parts.get("tests", 0) > 0.4:
        tips.append("Increase test reliability and fix failing/errored tests.")
    if parts.get("changes", 0) > 0.5:
        tips.append("Split large PR into smaller, coherent changes.")
    if parts.get("deps", 0) > 0.3:
        tips.append("Review new dependencies; add pinned versions and SBOM checks.")
    if parts.get("error_codes", 0) > 0.3:
        tips.append("Validate ErrorCode additions with lifecycle governance and client impact.")
    if parts.get("metrics", 0) > 0.3:
        tips.append("Run label policy and cardinality checks for new metrics.")
    if parts.get("workflows", 0) > 0.2:
        tips.append("Manually validate workflow changes in a dry run environment.")
    if not tips:
        tips.append("Proceed with standard rollout; monitor SLOs and error budget.")
    return tips


def to_markdown(score: float, level: str, blocking: bool, parts: Dict[str, float], d: Dict[str, Any]) -> str:
    icon = {
        "LOW": "✅",
        "MEDIUM": "⚠️",
        "HIGH": "🟠",
        "CRITICAL": "🔴",
    }[level]
    md = []
    md.append(f"### {icon} Release Risk Assessment\n")
    md.append(f"- Score: **{score:.1f}/100** ({level})\n")
    md.append(f"- Blocking: **{str(blocking).lower()}**\n")
    md.append("\n**Dimensions**\n")
    md.append("- Changes: {:.0f}%".format(parts["changes"] * 100))
    md.append("- Tests: {:.0f}%".format(parts["tests"] * 100))
    md.append("- Dependencies: {:.0f}%".format(parts["deps"] * 100))
    md.append("- Error Codes: {:.0f}%".format(parts["error_codes"] * 100))
    md.append("- Metrics: {:.0f}%".format(parts["metrics"] * 100))
    md.append("- Workflows: {:.0f}%".format(parts["workflows"] * 100))
    md.append("- Scripts: {:.0f}%".format(parts["scripts"] * 100))
    md.append("- Docs signal: {:.0f}%\n".format(parts["docs_signal"] * 100))
    md.append("\n**Git Summary**\n")
    g = d["git"]
    md.append(f"- Files changed: {g['files_changed']}  (+{g['lines_added']}/-{g['lines_deleted']})")
    md.append(f"- Areas: {json.dumps(g['by_area'])}")
    md.append(f"- Compare: `{g['compare_range']}`\n")
    md.append("\n**Actionable Suggestions**\n")
    for tip in suggestions(score, parts, d):
        md.append(f"- {tip}")
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Release risk scorer")
    ap.add_argument("--base-branch", required=True)
    ap.add_argument("--output-format", choices=["json", "markdown"], default="json")
    ap.add_argument("--output-file", default="-")
    ap.add_argument("--weights", help="JSON for custom weights", default=None)
    ap.add_argument("--init", action="store_true", help="Print help and exit successfully")
    args = ap.parse_args()

    if args.init:
        print("Release Risk Scorer ready. Use --base-branch <branch> to score.")
        return

    data = collect(args.base_branch)
    weights = ScoreWeights(**json.loads(args.weights)) if args.weights else ScoreWeights()
    score, parts = aggregate_score(data, weights)
    level = classify_level(score)
    blocking = blocking_decision(score)

    payload = {
        "score": round(score, 1),
        "level": level,
        "blocking": blocking,
        "parts": {k: round(v, 4) for k, v in parts.items()},
        "data": data,
    }

    if args.output_format == "json":
        out = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        out = to_markdown(score, level, blocking, parts, data)

    if args.output_file == "-":
        print(out)
    else:
        Path(args.output_file).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()

