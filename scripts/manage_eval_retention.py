#!/usr/bin/env python3
"""
Evaluation History Retention Manager.

Implements tiered data retention policy:
- 7 days: Keep all records (full resolution)
- 30 days: Keep daily snapshots (one per day)
- 90 days: Keep weekly snapshots (one per week)
- 1 year: Keep monthly snapshots (one per month)
- Beyond 1 year: Keep quarterly snapshots

Usage:
    python3 scripts/manage_eval_retention.py [--dry-run] [--verbose]
"""

import argparse
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class RetentionPolicy:
    """Define retention tiers and rules."""

    TIERS = [
        {"name": "full", "days": 7, "keep": "all"},
        {"name": "daily", "days": 30, "keep": "daily"},
        {"name": "weekly", "days": 90, "keep": "weekly"},
        {"name": "monthly", "days": 365, "keep": "monthly"},
        {"name": "quarterly", "days": float('inf'), "keep": "quarterly"}
    ]

    @staticmethod
    def get_tier(age_days: int) -> Dict:
        """Get the appropriate tier for a given age."""
        for tier in RetentionPolicy.TIERS:
            if age_days <= tier["days"]:
                return tier
        return RetentionPolicy.TIERS[-1]

    @staticmethod
    def should_keep(file_date: datetime, tier: Dict, all_dates: List[datetime]) -> bool:
        """Determine if a file should be kept based on tier policy."""
        if tier["keep"] == "all":
            return True

        # Find all files in the same period
        if tier["keep"] == "daily":
            period_key = file_date.date()
            period_files = [d for d in all_dates if d.date() == period_key]

        elif tier["keep"] == "weekly":
            # Get Monday of the week
            monday = file_date - timedelta(days=file_date.weekday())
            period_key = monday.date()
            period_files = [d for d in all_dates
                           if (d - timedelta(days=d.weekday())).date() == period_key]

        elif tier["keep"] == "monthly":
            period_key = (file_date.year, file_date.month)
            period_files = [d for d in all_dates
                           if (d.year, d.month) == period_key]

        elif tier["keep"] == "quarterly":
            quarter = (file_date.month - 1) // 3
            period_key = (file_date.year, quarter)
            period_files = [d for d in all_dates
                           if (d.year, (d.month - 1) // 3) == period_key]

        else:
            return False

        # Keep the latest file in the period
        if period_files:
            return file_date == max(period_files)

        return False


class EvaluationFile:
    """Represents an evaluation history file."""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.name

        # Parse timestamp from filename (format: YYYYMMDD_HHMMSS_*.json)
        try:
            parts = self.name.split("_")
            if len(parts) >= 2:
                date_str = parts[0]
                time_str = parts[1]
                self.timestamp = datetime.strptime(
                    f"{date_str}_{time_str}",
                    "%Y%m%d_%H%M%S"
                ).replace(tzinfo=timezone.utc)
            else:
                # Fallback to file modification time
                self.timestamp = datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=timezone.utc
                )
        except Exception:
            # Fallback to file modification time
            self.timestamp = datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc
            )

        # Load JSON data to get type and branch
        try:
            with open(path, "r") as f:
                data = json.load(f)
                self.type = data.get("type", "unknown")
                self.branch = data.get("branch", "unknown")
                self.commit = data.get("commit", "unknown")
        except Exception:
            self.type = "unknown"
            self.branch = "unknown"
            self.commit = "unknown"

        # Calculate age
        self.age_days = (datetime.now(timezone.utc) - self.timestamp).days

    def __repr__(self):
        return f"EvalFile({self.name}, {self.age_days}d old, {self.branch})"


def categorize_files_by_branch_and_type(files: List[EvaluationFile]) -> Dict:
    """Group files by branch and type for separate retention policies."""
    categorized = defaultdict(lambda: defaultdict(list))

    for file in files:
        categorized[file.branch][file.type].append(file)

    return categorized


def apply_retention_policy(
    files: List[EvaluationFile],
    dry_run: bool = True,
    verbose: bool = False
) -> Tuple[List[EvaluationFile], List[EvaluationFile]]:
    """Apply retention policy to a list of files."""

    # Sort files by timestamp
    files.sort(key=lambda f: f.timestamp)

    keep_files = []
    delete_files = []

    # Group files by age tier
    tier_groups = defaultdict(list)
    for file in files:
        tier = RetentionPolicy.get_tier(file.age_days)
        tier_groups[tier["name"]].append(file)

    # Process each tier
    for tier_name, tier_files in tier_groups.items():
        if not tier_files:
            continue

        tier = next(t for t in RetentionPolicy.TIERS if t["name"] == tier_name)

        if verbose:
            print(f"  Processing {tier_name} tier ({len(tier_files)} files):")

        # Get all timestamps for this tier
        all_timestamps = [f.timestamp for f in tier_files]

        for file in tier_files:
            should_keep = RetentionPolicy.should_keep(file.timestamp, tier, all_timestamps)

            if should_keep:
                keep_files.append(file)
                if verbose:
                    print(f"    KEEP: {file.name} ({file.age_days}d old)")
            else:
                delete_files.append(file)
                if verbose:
                    print(f"    DELETE: {file.name} ({file.age_days}d old)")

    return keep_files, delete_files


def archive_files(files: List[EvaluationFile], archive_dir: Path, dry_run: bool = True):
    """Archive files to a separate directory before deletion."""
    if not files:
        return

    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        archive_path = archive_dir / file.name
        if not dry_run:
            shutil.copy2(file.path, archive_path)
        print(f"  Archived: {file.name} -> {archive_path}")


def delete_files(files: List[EvaluationFile], dry_run: bool = True):
    """Delete files according to retention policy."""
    for file in files:
        if not dry_run:
            file.path.unlink()
        print(f"  Deleted: {file.name}")


def generate_retention_report(
    all_files: List[EvaluationFile],
    keep_files: List[EvaluationFile],
    delete_files: List[EvaluationFile]
) -> str:
    """Generate a summary report of retention actions."""

    report = []
    report.append("=" * 60)
    report.append("RETENTION POLICY REPORT")
    report.append("=" * 60)
    report.append("")

    # Statistics
    report.append("Statistics:")
    report.append(f"  Total files: {len(all_files)}")
    report.append(f"  Files to keep: {len(keep_files)}")
    report.append(f"  Files to delete: {len(delete_files)}")
    report.append(f"  Space to reclaim: {sum(f.path.stat().st_size for f in delete_files) / 1024:.1f} KB")
    report.append("")

    # Age distribution
    report.append("Age Distribution:")
    for tier in RetentionPolicy.TIERS:
        tier_files = [f for f in all_files
                     if RetentionPolicy.get_tier(f.age_days)["name"] == tier["name"]]
        if tier_files:
            report.append(f"  {tier['name']:10} ({tier['keep']:10}): {len(tier_files):3} files")

    report.append("")

    # Branch breakdown
    branches = defaultdict(lambda: {"keep": 0, "delete": 0})
    for file in keep_files:
        branches[file.branch]["keep"] += 1
    for file in delete_files:
        branches[file.branch]["delete"] += 1

    report.append("Branch Breakdown:")
    for branch, counts in sorted(branches.items()):
        report.append(f"  {branch:20} Keep: {counts['keep']:3}  Delete: {counts['delete']:3}")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Manage evaluation history retention")
    parser.add_argument("--dir", default="reports/eval_history",
                        help="Directory containing evaluation history files")
    parser.add_argument("--archive-dir", default="reports/eval_history/archive",
                        help="Directory for archived files before deletion")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Simulate actions without deleting files (default: True)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete files (overrides --dry-run)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--archive", action="store_true",
                        help="Archive files before deletion")

    args = parser.parse_args()

    # Override dry-run if execute is specified
    if args.execute:
        args.dry_run = False

    history_dir = Path(args.dir)
    archive_dir = Path(args.archive_dir)

    if not history_dir.exists():
        print(f"Directory not found: {history_dir}")
        return 1

    # Load all JSON files
    json_files = []
    for path in history_dir.glob("*.json"):
        # Skip backup files
        if path.name.endswith(".json.bak"):
            continue
        # Skip files in subdirectories
        if path.parent != history_dir:
            continue

        try:
            eval_file = EvaluationFile(path)
            json_files.append(eval_file)
        except Exception as e:
            print(f"Warning: Could not process {path}: {e}")

    if not json_files:
        print("No evaluation files found")
        return 0

    print(f"Found {len(json_files)} evaluation files")

    # Categorize by branch and type
    categorized = categorize_files_by_branch_and_type(json_files)

    all_keep_files = []
    all_delete_files = []

    # Apply retention policy per branch and type
    for branch, types in categorized.items():
        for file_type, files in types.items():
            if not files:
                continue

            print(f"\nProcessing {branch}/{file_type} ({len(files)} files):")

            keep, delete = apply_retention_policy(files, args.dry_run, args.verbose)

            all_keep_files.extend(keep)
            all_delete_files.extend(delete)

    # Generate report
    report = generate_retention_report(json_files, all_keep_files, all_delete_files)
    print("\n" + report)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files were actually deleted ***")
        print("*** Use --execute to apply changes ***")
    else:
        # Archive if requested
        if args.archive and all_delete_files:
            print(f"\nArchiving {len(all_delete_files)} files...")
            archive_files(all_delete_files, archive_dir, args.dry_run)

        # Delete files
        if all_delete_files:
            print(f"\nDeleting {len(all_delete_files)} files...")
            delete_files(all_delete_files, args.dry_run)
            print("Retention policy applied successfully")

    return 0


if __name__ == "__main__":
    exit(main())