#!/usr/bin/env python
"""Quick demo script to emit latency + degraded/recovery metrics.

Requires environment:
  ENABLE_FORCE_DEGRADE=true
  VECTOR_STORE_BACKEND=faiss (or simulated)
  FEATURE_VECTOR_EXPECTED_DIM=24
  X-API-Key / X-Admin-Token set for dual auth.

Usage:
  python scripts/demo_metrics.py --base http://localhost:8000 --api-key KEY --admin-token ADMIN
"""
import argparse
import pathlib
import time
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000")
    p.add_argument("--api-key", required=True)
    p.add_argument("--admin-token", required=True)
    args = p.parse_args()

    headers = {"X-API-Key": args.api_key, "X-Admin-Token": args.admin_token}

    step_path = pathlib.Path("examples/sample_part.step")
    if not step_path.exists():
        print("Sample STEP file missing")
        return

    # Feature extraction (latency histogram)
    files = {"file": (step_path.name, step_path.read_bytes(), "application/step")}
    try:
        r = requests.post(f"{args.base}/api/v1/analyze/feature-extract", headers=headers, files=files, timeout=30)
        print("Feature extract status:", r.status_code)
    except Exception as e:
        print("Feature extract error:", e)

    # Force degrade
    r = requests.post(f"{args.base}/api/v1/similarity/force-degrade", headers=headers, json={"reason": "demo"}, timeout=10)
    print("Force degrade:", r.json())
    time.sleep(2)

    # Force recover
    r = requests.post(f"{args.base}/api/v1/similarity/force-recover", headers=headers, timeout=10)
    print("Force recover:", r.json())

    # Fetch metrics
    m = requests.get(f"{args.base}/metrics", timeout=10)
    print("Metrics sample (first 30 lines):")
    print("\n".join(m.text.splitlines()[:30]))


if __name__ == "__main__":
    main()

