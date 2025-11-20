#!/usr/bin/env python3
"""
Quarterly baseline snapshot utility.

Archives anomaly detection baselines for historical reference.

Usage:
    python3 scripts/snapshot_baseline.py [--force]
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def get_quarter_string(date: Optional[datetime] = None) -> str:
    """Generate quarter string (e.g., '2025_Q1')."""
    if date is None:
        date = datetime.now()

    quarter = (date.month - 1) // 3 + 1
    year = date.year

    return f"{year}_Q{quarter}"


def load_baseline(baseline_path: Path) -> Optional[Dict]:
    """Load baseline JSON file."""
    if not baseline_path.exists():
        return None

    try:
        with open(baseline_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading baseline: {e}")
        return None


def snapshot_baseline(force: bool = False) -> int:
    """Create quarterly snapshot of anomaly baseline."""

    # Source baseline
    source = Path("reports/insights/baseline.json")

    if not source.exists():
        print("❌ No baseline file found at reports/insights/baseline.json")
        print("   Run 'make baseline-update' to create baseline first")
        return 1

    # Load and validate baseline
    baseline = load_baseline(source)
    if not baseline:
        print("❌ Failed to load baseline file")
        return 1

    # Check if baseline has sufficient data
    sample_count = baseline.get("sample_count", 0)
    if sample_count < 10 and not force:
        print(f"⚠️  Baseline has only {sample_count} samples (minimum recommended: 10)")
        print("   Use --force to snapshot anyway")
        return 1

    # Create baselines directory
    baselines_dir = Path("reports/baselines")
    baselines_dir.mkdir(parents=True, exist_ok=True)

    # Generate quarterly filename
    quarter_str = get_quarter_string()
    dest = baselines_dir / f"baseline_{quarter_str}.json"

    # Check if snapshot already exists
    if dest.exists() and not force:
        print(f"⚠️  Snapshot already exists: {dest}")
        print("   Use --force to overwrite")
        return 1

    # Add metadata to baseline
    baseline["snapshot_metadata"] = {
        "snapshot_date": datetime.now().isoformat(),
        "quarter": quarter_str,
        "source_samples": sample_count,
        "source_updated": baseline.get("updated_at", "unknown")
    }

    # Save snapshot with metadata
    try:
        with open(dest, "w") as f:
            json.dump(baseline, f, indent=2)

        print(f"✅ Baseline snapshot saved to {dest}")
        print(f"   Quarter: {quarter_str}")
        print(f"   Samples: {sample_count}")

        # Also create a 'latest' symlink for easy access
        latest_link = baselines_dir / "latest_snapshot.json"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create relative symlink
        latest_link.symlink_to(dest.name)
        print(f"   Latest symlink updated")

        return 0

    except Exception as e:
        print(f"❌ Failed to save snapshot: {e}")
        return 1


def list_snapshots() -> None:
    """List all existing baseline snapshots."""
    baselines_dir = Path("reports/baselines")

    if not baselines_dir.exists():
        print("No baselines directory found")
        return

    snapshots = sorted(baselines_dir.glob("baseline_*.json"))

    if not snapshots:
        print("No baseline snapshots found")
        return

    print("Existing baseline snapshots:")
    print("-" * 60)

    for snapshot in snapshots:
        try:
            with open(snapshot, "r") as f:
                data = json.load(f)

            metadata = data.get("snapshot_metadata", {})
            sample_count = data.get("sample_count", 0)

            print(f"📁 {snapshot.name}")
            print(f"   Date: {metadata.get('snapshot_date', 'unknown')}")
            print(f"   Samples: {sample_count}")
            print(f"   Updated: {data.get('updated_at', 'unknown')}")

        except Exception as e:
            print(f"📁 {snapshot.name} (error reading: {e})")

    # Check for latest symlink
    latest = baselines_dir / "latest_snapshot.json"
    if latest.exists() and latest.is_symlink():
        print(f"\n🔗 Latest: {latest.readlink()}")


def compare_snapshots(snapshot1: str, snapshot2: str) -> None:
    """Compare two baseline snapshots."""
    baselines_dir = Path("reports/baselines")

    path1 = baselines_dir / snapshot1
    path2 = baselines_dir / snapshot2

    if not path1.exists():
        print(f"❌ Snapshot not found: {path1}")
        return

    if not path2.exists():
        print(f"❌ Snapshot not found: {path2}")
        return

    base1 = load_baseline(path1)
    base2 = load_baseline(path2)

    if not base1 or not base2:
        print("❌ Failed to load snapshots")
        return

    print(f"Comparing {snapshot1} vs {snapshot2}")
    print("=" * 60)

    # Compare metrics
    for metric in ["combined", "vision", "ocr"]:
        m1 = base1.get("metrics", {}).get(metric, {})
        m2 = base2.get("metrics", {}).get(metric, {})

        print(f"\n{metric.upper()}:")

        if m1.get("mean") and m2.get("mean"):
            mean_diff = m2["mean"] - m1["mean"]
            sign = "📈" if mean_diff > 0 else "📉" if mean_diff < 0 else "➡️"
            print(f"  Mean: {m1['mean']:.3f} → {m2['mean']:.3f} ({sign} {mean_diff:+.3f})")

        if m1.get("stdev") and m2.get("stdev"):
            stdev_diff = m2["stdev"] - m1["stdev"]
            print(f"  StDev: {m1['stdev']:.3f} → {m2['stdev']:.3f} ({stdev_diff:+.3f})")

        sample_diff = len(m2.get("history", [])) - len(m1.get("history", []))
        print(f"  Samples: {len(m1.get('history', []))} → {len(m2.get('history', []))} (+{sample_diff})")

    # Compare configuration
    print("\nConfiguration Changes:")
    config1 = base1.get("config", {})
    config2 = base2.get("config", {})

    for key in set(config1.keys()) | set(config2.keys()):
        v1 = config1.get(key)
        v2 = config2.get(key)
        if v1 != v2:
            print(f"  {key}: {v1} → {v2}")


def main():
    parser = argparse.ArgumentParser(description="Manage baseline snapshots")
    parser.add_argument("--force", action="store_true",
                        help="Force snapshot even if it exists or has few samples")
    parser.add_argument("--list", action="store_true",
                        help="List all existing snapshots")
    parser.add_argument("--compare", nargs=2, metavar=("SNAP1", "SNAP2"),
                        help="Compare two snapshots")

    args = parser.parse_args()

    if args.list:
        list_snapshots()
        return 0

    if args.compare:
        compare_snapshots(args.compare[0], args.compare[1])
        return 0

    # Default action: create snapshot
    return snapshot_baseline(force=args.force)


if __name__ == "__main__":
    sys.exit(main())