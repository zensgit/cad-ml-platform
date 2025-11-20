#!/usr/bin/env python3
"""
Anomaly baseline caching for stable detection.

Maintains historical statistics to avoid small sample bias.

Usage:
    python3 scripts/anomaly_baseline.py [--update|--show|--reset]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import statistics


class BaselineManager:
    """Manage anomaly detection baselines."""

    def __init__(self, cache_file: str = "reports/insights/baseline.json"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.baseline = self.load_baseline()

    def load_baseline(self) -> Dict:
        """Load existing baseline or create new."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load baseline: {e}")

        # Default baseline structure
        return {
            "version": "1.0",
            "updated_at": None,
            "sample_count": 0,
            "metrics": {
                "combined": {
                    "mean": None,
                    "stdev": None,
                    "min": None,
                    "max": None,
                    "history": []
                },
                "vision": {
                    "mean": None,
                    "stdev": None,
                    "min": None,
                    "max": None,
                    "history": []
                },
                "ocr": {
                    "mean": None,
                    "stdev": None,
                    "min": None,
                    "max": None,
                    "history": []
                }
            },
            "config": {
                "max_history_size": 100,
                "min_samples_for_baseline": 10,
                "outlier_threshold": 2.0  # Z-score threshold
            }
        }

    def save_baseline(self) -> None:
        """Save baseline to cache file."""
        with open(self.cache_file, "w") as f:
            json.dump(self.baseline, f, indent=2)

    def update_from_evaluation(self, eval_path: str) -> bool:
        """Update baseline from evaluation file."""
        try:
            with open(eval_path, "r") as f:
                data = json.load(f)

            # Extract scores (handle both formats)
            if "scores" in data:
                combined = data["scores"]["combined"]
                vision = data["scores"]["vision"]["score"]
                ocr = data["scores"]["ocr"]["normalized"]
            elif "combined" in data:
                combined = data["combined"].get("combined_score", 0)
                vision = data["combined"].get("vision_score", 0)
                ocr = data["combined"].get("ocr_score", 0)
            else:
                return False

            # Update history
            timestamp = data.get("timestamp", datetime.now(timezone.utc).isoformat())

            self._add_to_history("combined", combined, timestamp)
            self._add_to_history("vision", vision, timestamp)
            self._add_to_history("ocr", ocr, timestamp)

            # Recalculate statistics
            self._update_statistics()

            self.baseline["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.baseline["sample_count"] = len(self.baseline["metrics"]["combined"]["history"])

            self.save_baseline()
            return True

        except Exception as e:
            print(f"Error updating from evaluation: {e}")
            return False

    def _add_to_history(self, metric: str, value: float, timestamp: str) -> None:
        """Add value to metric history."""
        history = self.baseline["metrics"][metric]["history"]
        max_size = self.baseline["config"]["max_history_size"]

        # Check if this timestamp already exists
        existing_timestamps = [h["timestamp"] for h in history]
        if timestamp not in existing_timestamps:
            history.append({"value": value, "timestamp": timestamp})

            # Trim to max size (keep most recent)
            if len(history) > max_size:
                history.sort(key=lambda x: x["timestamp"])
                self.baseline["metrics"][metric]["history"] = history[-max_size:]

    def _update_statistics(self) -> None:
        """Recalculate statistics from history."""
        min_samples = self.baseline["config"]["min_samples_for_baseline"]

        for metric in ["combined", "vision", "ocr"]:
            history = self.baseline["metrics"][metric]["history"]

            if len(history) >= min_samples:
                values = [h["value"] for h in history]

                self.baseline["metrics"][metric]["mean"] = statistics.mean(values)
                self.baseline["metrics"][metric]["stdev"] = statistics.stdev(values) if len(values) > 1 else 0
                self.baseline["metrics"][metric]["min"] = min(values)
                self.baseline["metrics"][metric]["max"] = max(values)

    def update_from_directory(self, history_dir: str = "reports/eval_history") -> int:
        """Update baseline from all evaluation files."""
        json_files = sorted(Path(history_dir).glob("*_combined.json"))
        updated_count = 0

        for json_file in json_files:
            if self.update_from_evaluation(str(json_file)):
                updated_count += 1

        return updated_count

    def check_anomaly(self, metric: str, value: float) -> Optional[Dict]:
        """Check if a value is anomalous based on baseline."""
        if metric not in self.baseline["metrics"]:
            return None

        stats = self.baseline["metrics"][metric]
        min_samples = self.baseline["config"]["min_samples_for_baseline"]

        # Need enough samples for meaningful baseline
        if len(stats["history"]) < min_samples:
            return None

        mean = stats["mean"]
        stdev = stats["stdev"]

        if mean is None or stdev is None or stdev == 0:
            return None

        # Calculate Z-score
        z_score = abs(value - mean) / stdev

        threshold = self.baseline["config"]["outlier_threshold"]

        if z_score > threshold:
            return {
                "metric": metric,
                "value": value,
                "mean": mean,
                "stdev": stdev,
                "z_score": z_score,
                "severity": "high" if z_score > 3 else "medium",
                "baseline_samples": len(stats["history"])
            }

        return None

    def get_summary(self) -> Dict:
        """Get baseline summary."""
        summary = {
            "updated_at": self.baseline["updated_at"],
            "sample_count": self.baseline["sample_count"],
            "metrics": {}
        }

        for metric in ["combined", "vision", "ocr"]:
            stats = self.baseline["metrics"][metric]
            summary["metrics"][metric] = {
                "mean": stats["mean"],
                "stdev": stats["stdev"],
                "min": stats["min"],
                "max": stats["max"],
                "samples": len(stats["history"])
            }

        return summary

    def reset(self) -> None:
        """Reset baseline to empty state."""
        self.baseline = self.load_baseline()
        self.baseline["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save_baseline()


def main():
    parser = argparse.ArgumentParser(description="Manage anomaly detection baseline")
    parser.add_argument("--update", action="store_true",
                        help="Update baseline from evaluation history")
    parser.add_argument("--show", action="store_true",
                        help="Show current baseline statistics")
    parser.add_argument("--reset", action="store_true",
                        help="Reset baseline")
    parser.add_argument("--check", type=float, nargs=2,
                        metavar=("METRIC", "VALUE"),
                        help="Check if value is anomalous")
    parser.add_argument("--cache-file", default="reports/insights/baseline.json",
                        help="Baseline cache file path")

    args = parser.parse_args()

    manager = BaselineManager(cache_file=args.cache_file)

    if args.reset:
        manager.reset()
        print("Baseline reset successfully")
        return 0

    if args.update:
        count = manager.update_from_directory()
        print(f"Updated baseline from {count} evaluation files")
        manager.save_baseline()
        return 0

    if args.show:
        summary = manager.get_summary()
        print(json.dumps(summary, indent=2))
        return 0

    if args.check:
        metric_map = {"0": "combined", "1": "vision", "2": "ocr"}
        metric = metric_map.get(str(int(args.check[0])), "combined")
        value = args.check[1]

        anomaly = manager.check_anomaly(metric, value)
        if anomaly:
            print(f"ANOMALY DETECTED: {json.dumps(anomaly, indent=2)}")
            return 1
        else:
            print(f"Value {value} for {metric} is within normal range")
            return 0

    # Default: show summary
    summary = manager.get_summary()
    print("Anomaly Detection Baseline Summary")
    print("=" * 50)
    print(f"Updated: {summary['updated_at'] or 'Never'}")
    print(f"Samples: {summary['sample_count']}")
    print("\nMetrics:")
    for metric, stats in summary["metrics"].items():
        if stats["mean"] is not None:
            print(f"  {metric.upper()}:")
            print(f"    Mean: {stats['mean']:.3f} ± {stats['stdev']:.3f}")
            print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"    Samples: {stats['samples']}")
        else:
            print(f"  {metric.upper()}: No baseline (insufficient samples)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
