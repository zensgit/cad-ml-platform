#!/usr/bin/env python3
"""Quick health summary script.

Outputs:
- Git branch/commit/tag
- API endpoints status hints
- Latest eval history snapshot (if any)
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT / "reports" / "eval_history"


def git_info() -> dict:
    info = {"branch": "unknown", "commit": "unknown", "tag": None}
    try:
        info["branch"] = (
            subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, cwd=ROOT)
            .stdout.strip()
            or "unknown"
        )
        info["commit"] = (
            subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT)
            .stdout.strip()
            or "unknown"
        )
        tag = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"], capture_output=True, text=True, cwd=ROOT
        )
        if tag.returncode == 0:
            info["tag"] = tag.stdout.strip()
    except Exception:
        pass
    return info


def latest_eval() -> dict | None:
    if not HISTORY_DIR.exists():
        return None
    files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)
    if not files:
        return None
    try:
        with files[0].open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["_file"] = str(files[0].relative_to(ROOT))
        return data
    except Exception:
        return None


def latest_combined_eval() -> dict | None:
    """Find the latest combined evaluation history file."""
    if not HISTORY_DIR.exists():
        return None
    # Combined files are named *_combined.json
    files = sorted(HISTORY_DIR.glob("*_combined.json"), reverse=True)
    if not files:
        return None
    try:
        with files[0].open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["_file"] = str(files[0].relative_to(ROOT))
        return data
    except Exception:
        return None


def main() -> None:
    gi = git_info()
    le = latest_eval()
    lc = latest_combined_eval()

    print("=== Quick Health Summary ===")
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} (UTC)")
    print(f"Git: {gi['branch']} @ {gi['commit']}{' (' + gi['tag'] + ')' if gi.get('tag') else ''}")
    print()
    print("API Hints:")
    print("- Local: http://localhost:8000/health | /metrics | /docs")
    print("- Vision: GET /api/v1/vision/health")
    print()

    # Combined evaluation (end-to-end health)
    if lc:
        c = lc.get("combined", {})
        v_w = c.get("vision_weight", 0.5)
        o_w = c.get("ocr_weight", 0.5)
        print("Latest Combined Eval:")
        print(f"  COMBINED SCORE: {c.get('combined_score', 0):.3f} (weights: v={v_w:.2f}, o={o_w:.2f})")
        print(f"  Timestamp: {lc.get('timestamp', 'unknown')}")
        print(f"  File: {lc.get('_file')}")
        print()
    else:
        print("Latest Combined Eval: (none) -> run 'make eval-combined-save'")
        print()

    # OCR-only evaluation (backward compatibility)
    if le:
        m = le.get("metrics", {})
        if m:  # OCR-only eval structure
            print("Latest OCR Eval:")
            print(f"  file: {le.get('_file')}")
            print(f"  dimension_recall: {m.get('dimension_recall')}")
            print(f"  brier_score: {m.get('brier_score')}")
            print(f"  edge_f1: {m.get('edge_f1')}")
    else:
        print("Latest OCR Eval: (none) -> run 'make eval-history'")


if __name__ == "__main__":
    main()

