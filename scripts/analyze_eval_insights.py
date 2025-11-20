#!/usr/bin/env python3
"""
LLM-powered evaluation insights and anomaly detection.

Analyzes evaluation history to detect anomalies and generate narrative summaries.

Usage:
    python3 scripts/analyze_eval_insights.py [--days 30] [--threshold 0.1]
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics


class InsightsAnalyzer:
    """Analyze evaluation metrics for insights and anomalies."""

    def __init__(self, history_dir: str = "reports/eval_history"):
        self.history_dir = Path(history_dir)
        self.history = []
        self.anomalies = []
        self.insights = []

    def load_history(self, days: int = 30) -> None:
        """Load evaluation history for analysis."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        json_files = sorted(self.history_dir.glob("*_combined.json"))

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Parse timestamp
                timestamp_str = data.get("timestamp", "")
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    if timestamp >= cutoff:
                        self.history.append(data)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

        # Sort by timestamp
        self.history.sort(key=lambda x: x.get("timestamp", ""))
        print(f"Loaded {len(self.history)} evaluations from last {days} days")

    def detect_anomalies(self, threshold: float = 0.1) -> List[Dict]:
        """Detect anomalies in evaluation metrics."""
        if len(self.history) < 3:
            return []

        # Extract scores (handle both new and legacy formats)
        combined_scores = []
        vision_scores = []
        ocr_scores = []

        for h in self.history:
            if "scores" in h:
                # New format
                combined_scores.append(h["scores"]["combined"])
                vision_scores.append(h["scores"]["vision"]["score"])
                ocr_scores.append(h["scores"]["ocr"]["normalized"])
            elif "combined" in h:
                # Legacy format
                combined_scores.append(h["combined"].get("combined_score", 0))
                vision_scores.append(h["vision_metrics"].get("AVG_HIT_RATE", 0))
                ocr_scores.append(h["combined"].get("ocr_score", 0))
            else:
                continue

        # Calculate statistics
        def find_outliers(scores: List[float], name: str) -> List[Dict]:
            outliers = []
            if len(scores) < 3:
                return outliers

            mean = statistics.mean(scores)
            stdev = statistics.stdev(scores) if len(scores) > 1 else 0

            for i, score in enumerate(scores):
                z_score = abs(score - mean) / stdev if stdev > 0 else 0

                # Check for significant deviations
                if z_score > 2:  # More than 2 standard deviations
                    outliers.append({
                        "index": i,
                        "timestamp": self.history[i]["timestamp"],
                        "metric": name,
                        "value": score,
                        "mean": mean,
                        "stdev": stdev,
                        "z_score": z_score,
                        "severity": "high" if z_score > 3 else "medium"
                    })

                # Check for sudden changes
                if i > 0:
                    change = abs(score - scores[i-1])
                    if change > threshold:
                        outliers.append({
                            "index": i,
                            "timestamp": self.history[i]["timestamp"],
                            "metric": name,
                            "value": score,
                            "previous": scores[i-1],
                            "change": change,
                            "severity": "high" if change > threshold * 2 else "medium"
                        })

            return outliers

        # Find anomalies in each metric
        self.anomalies.extend(find_outliers(combined_scores, "combined"))
        self.anomalies.extend(find_outliers(vision_scores, "vision"))
        self.anomalies.extend(find_outliers(ocr_scores, "ocr"))

        # Remove duplicates
        seen = set()
        unique_anomalies = []
        for anomaly in self.anomalies:
            key = (anomaly["index"], anomaly["metric"])
            if key not in seen:
                seen.add(key)
                unique_anomalies.append(anomaly)

        self.anomalies = unique_anomalies
        return self.anomalies

    def analyze_trends(self) -> Dict:
        """Analyze trends in evaluation metrics."""
        if len(self.history) < 2:
            return {"status": "insufficient_data"}

        # Extract scores (handle both new and legacy formats)
        combined_scores = []
        vision_scores = []
        ocr_scores = []

        for h in self.history:
            if "scores" in h:
                # New format
                combined_scores.append(h["scores"]["combined"])
                vision_scores.append(h["scores"]["vision"]["score"])
                ocr_scores.append(h["scores"]["ocr"]["normalized"])
            elif "combined" in h:
                # Legacy format
                combined_scores.append(h["combined"].get("combined_score", 0))
                vision_scores.append(h["vision_metrics"].get("AVG_HIT_RATE", 0))
                ocr_scores.append(h["combined"].get("ocr_score", 0))
            else:
                continue

        def calculate_trend(scores: List[float]) -> str:
            if len(scores) < 2:
                return "stable"

            # Simple linear regression
            n = len(scores)
            x = list(range(n))
            x_mean = sum(x) / n
            y_mean = sum(scores) / n

            numerator = sum((x[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                return "stable"

            slope = numerator / denominator

            # Classify trend
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "degrading"
            else:
                return "stable"

        trends = {
            "combined": calculate_trend(combined_scores),
            "vision": calculate_trend(vision_scores),
            "ocr": calculate_trend(ocr_scores),
            "period": {
                "start": self.history[0]["timestamp"],
                "end": self.history[-1]["timestamp"],
                "count": len(self.history)
            }
        }

        return trends

    def generate_narrative(self) -> str:
        """Generate a narrative summary of the analysis."""
        if not self.history:
            return "No evaluation data available for analysis."

        # Get latest scores
        latest = self.history[-1]
        if "scores" in latest:
            combined = latest["scores"]["combined"]
            vision = latest["scores"]["vision"]["score"]
            ocr = latest["scores"]["ocr"]["normalized"]
        else:
            combined = latest["combined"].get("combined_score", 0)
            vision = latest["vision_metrics"].get("AVG_HIT_RATE", 0)
            ocr = latest["combined"].get("ocr_score", 0)

        # Analyze trends
        trends = self.analyze_trends()

        # Build narrative
        narrative = []
        narrative.append("# Evaluation Insights Report")
        narrative.append(f"\n*Generated: {datetime.now(timezone.utc).isoformat()}*")

        # Executive Summary
        narrative.append("\n## Executive Summary")
        narrative.append(f"\n**Latest Performance**: Combined score of **{combined:.3f}** "
                        f"(Vision: {vision:.3f}, OCR: {ocr:.3f})")

        # Trend Analysis
        narrative.append("\n## Trend Analysis")

        trend_emoji = {
            "improving": "📈",
            "stable": "➡️",
            "degrading": "📉"
        }

        narrative.append(f"\n- **Combined Score**: {trend_emoji[trends['combined']]} "
                        f"{trends['combined'].capitalize()}")
        narrative.append(f"- **Vision Module**: {trend_emoji[trends['vision']]} "
                        f"{trends['vision'].capitalize()}")
        narrative.append(f"- **OCR Module**: {trend_emoji[trends['ocr']]} "
                        f"{trends['ocr'].capitalize()}")

        # Statistical Summary
        if len(self.history) > 1:
            combined_scores = []
            for h in self.history:
                if "scores" in h:
                    combined_scores.append(h["scores"]["combined"])
                elif "combined" in h:
                    combined_scores.append(h["combined"].get("combined_score", 0))
            narrative.append("\n## Statistical Summary")
            narrative.append(f"\n- **Mean Score**: {statistics.mean(combined_scores):.3f}")
            narrative.append(f"- **Std Deviation**: {statistics.stdev(combined_scores):.3f}")
            narrative.append(f"- **Min Score**: {min(combined_scores):.3f}")
            narrative.append(f"- **Max Score**: {max(combined_scores):.3f}")
            narrative.append(f"- **Sample Size**: {len(self.history)} evaluations")

        # Anomaly Report
        if self.anomalies:
            narrative.append("\n## ⚠️ Anomalies Detected")
            narrative.append(f"\nFound **{len(self.anomalies)}** anomalies:")

            for anomaly in self.anomalies[:5]:  # Show top 5
                if "z_score" in anomaly:
                    narrative.append(f"\n- **{anomaly['metric'].upper()}** at "
                                   f"{anomaly['timestamp']}: Score {anomaly['value']:.3f} "
                                   f"(Z-score: {anomaly['z_score']:.2f})")
                elif "change" in anomaly:
                    narrative.append(f"\n- **{anomaly['metric'].upper()}** at "
                                   f"{anomaly['timestamp']}: Sudden change of "
                                   f"{anomaly['change']:.3f}")
        else:
            narrative.append("\n## ✅ No Anomalies Detected")
            narrative.append("\nAll metrics are within expected ranges.")

        # Recommendations
        narrative.append("\n## Recommendations")

        if trends["combined"] == "degrading":
            narrative.append("\n⚠️ **Action Required**: Combined score is showing a "
                           "degrading trend. Consider:")
            narrative.append("- Review recent model changes")
            narrative.append("- Check data quality")
            narrative.append("- Run diagnostic tests")
        elif trends["combined"] == "improving":
            narrative.append("\n✅ **Positive Trend**: Performance is improving. "
                           "Consider:")
            narrative.append("- Document successful changes")
            narrative.append("- Establish new baseline")
            narrative.append("- Share learnings with team")
        else:
            narrative.append("\n✅ **Stable Performance**: System is performing "
                           "consistently.")
            narrative.append("- Continue monitoring")
            narrative.append("- Consider optimization opportunities")

        # Risk Assessment
        narrative.append("\n## Risk Assessment")

        risk_level = "low"
        if self.anomalies:
            high_severity = sum(1 for a in self.anomalies if a.get("severity") == "high")
            if high_severity > 2:
                risk_level = "high"
            elif high_severity > 0:
                risk_level = "medium"

        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        narrative.append(f"\n**Overall Risk Level**: {risk_emoji[risk_level]} "
                        f"{risk_level.upper()}")

        if risk_level != "low":
            narrative.append("\n**Mitigation Steps**:")
            narrative.append("1. Review anomaly details")
            narrative.append("2. Check system logs")
            narrative.append("3. Validate data pipeline")
            narrative.append("4. Consider rollback if necessary")

        return "\n".join(narrative)

    def export_insights(self, output_file: Optional[str] = None) -> None:
        """Export insights to file."""
        insights = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_period": {
                "start": self.history[0]["timestamp"] if self.history else None,
                "end": self.history[-1]["timestamp"] if self.history else None,
                "count": len(self.history)
            },
            "trends": self.analyze_trends(),
            "anomalies": self.anomalies,
            "statistics": self._calculate_statistics(),
            "narrative": self.generate_narrative()
        }

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_file.endswith('.json'):
                with open(output_path, "w") as f:
                    json.dump(insights, f, indent=2)
            else:
                with open(output_path, "w") as f:
                    f.write(insights["narrative"])

            print(f"Insights saved to: {output_path}")
        else:
            print(json.dumps(insights, indent=2))

    def _calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistics."""
        if not self.history:
            return {}

        combined_scores = []
        for h in self.history:
            if "scores" in h:
                combined_scores.append(h["scores"]["combined"])
            elif "combined" in h:
                combined_scores.append(h["combined"].get("combined_score", 0))

        if not combined_scores:
            return {}

        stats = {
            "combined": {
                "mean": statistics.mean(combined_scores),
                "median": statistics.median(combined_scores),
                "stdev": statistics.stdev(combined_scores) if len(combined_scores) > 1 else 0,
                "min": min(combined_scores),
                "max": max(combined_scores),
                "range": max(combined_scores) - min(combined_scores)
            }
        }

        return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation insights")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of history to analyze")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Anomaly detection threshold")
    parser.add_argument("--output", help="Output file (json or md)")
    parser.add_argument("--narrative-only", action="store_true",
                        help="Output narrative only")

    args = parser.parse_args()

    # Run analysis
    analyzer = InsightsAnalyzer()
    analyzer.load_history(days=args.days)

    if len(analyzer.history) == 0:
        print("No evaluation history found")
        return 1

    # Detect anomalies
    anomalies = analyzer.detect_anomalies(threshold=args.threshold)

    # Generate and output insights
    if args.narrative_only:
        print(analyzer.generate_narrative())
    else:
        analyzer.export_insights(output_file=args.output)

    # Return status based on anomalies
    if anomalies:
        high_severity = sum(1 for a in anomalies if a.get("severity") == "high")
        return 2 if high_severity > 0 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())