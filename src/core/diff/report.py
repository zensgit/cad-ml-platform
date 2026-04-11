"""Report generation for drawing version diffs.

Produces Markdown-formatted diff summaries and Engineering Change Notices (ECN)
from DiffResult data.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.core.diff.models import DiffResult, EntityChange

logger = logging.getLogger(__name__)


class DiffReportGenerator:
    """Generate human-readable reports from a DiffResult."""

    # ------------------------------------------------------------------
    # Markdown diff report
    # ------------------------------------------------------------------

    def generate_markdown(
        self,
        diff: DiffResult,
        file_a: str,
        file_b: str,
    ) -> str:
        """Produce a Markdown diff report.

        Sections:
        1. Header with file names and timestamp
        2. Summary table (added / removed / modified counts)
        3. Detailed changes by type
        4. Change regions with bounding-box coordinates
        """
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines: List[str] = []

        # -- Header --
        lines.append("# Drawing Version Diff Report")
        lines.append("")
        lines.append(f"- **Baseline:** `{file_a}`")
        lines.append(f"- **Revised:** `{file_b}`")
        lines.append(f"- **Generated:** {now}")
        lines.append("")

        # -- Summary table --
        lines.append("## Summary")
        lines.append("")
        lines.append("| Change Type | Count |")
        lines.append("|-------------|-------|")
        lines.append(f"| Added       | {diff.summary.get('added', 0)} |")
        lines.append(f"| Removed     | {diff.summary.get('removed', 0)} |")
        lines.append(f"| Modified    | {diff.summary.get('modified', 0)} |")

        total = (
            diff.summary.get("added", 0)
            + diff.summary.get("removed", 0)
            + diff.summary.get("modified", 0)
        )
        lines.append(f"| **Total**   | **{total}** |")
        lines.append("")

        if diff.is_empty():
            lines.append("*No changes detected between the two revisions.*")
            lines.append("")
            return "\n".join(lines)

        # -- Detailed changes --
        lines.append("## Detailed Changes")
        lines.append("")

        if diff.added:
            lines.append("### Added Entities")
            lines.append("")
            lines.extend(self._format_change_list(diff.added))
            lines.append("")

        if diff.removed:
            lines.append("### Removed Entities")
            lines.append("")
            lines.extend(self._format_change_list(diff.removed))
            lines.append("")

        if diff.modified:
            lines.append("### Modified Entities")
            lines.append("")
            lines.extend(self._format_change_list(diff.modified))
            lines.append("")

        # -- Change regions --
        if diff.change_regions:
            lines.append("## Change Regions")
            lines.append("")
            lines.append("| # | Min X | Min Y | Max X | Max Y | Changes |")
            lines.append("|---|-------|-------|-------|-------|---------|")
            for i, region in enumerate(diff.change_regions, 1):
                lines.append(
                    f"| {i} "
                    f"| {region.get('min_x', 0):.2f} "
                    f"| {region.get('min_y', 0):.2f} "
                    f"| {region.get('max_x', 0):.2f} "
                    f"| {region.get('max_y', 0):.2f} "
                    f"| {region.get('change_count', 0)} |"
                )
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Engineering Change Notice (ECN)
    # ------------------------------------------------------------------

    def generate_ecn(
        self,
        diff: DiffResult,
        part_number: str,
        revision: str,
    ) -> str:
        """Produce an Engineering Change Notice in Markdown.

        Sections:
        1. ECN header (part number, revision, date)
        2. Change description (auto-generated from diff summary)
        3. Affected areas
        4. Reviewer sign-off template
        """
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        lines: List[str] = []

        # -- ECN header --
        lines.append("# Engineering Change Notice")
        lines.append("")
        lines.append(f"- **Part Number:** {part_number}")
        lines.append(f"- **Revision:** {revision}")
        lines.append(f"- **Date:** {now}")
        lines.append(f"- **Status:** Draft")
        lines.append("")

        # -- Change description --
        lines.append("## Change Description")
        lines.append("")

        added_count = diff.summary.get("added", 0)
        removed_count = diff.summary.get("removed", 0)
        modified_count = diff.summary.get("modified", 0)

        description_parts: List[str] = []
        if added_count:
            types_added = self._summarize_entity_types(diff.added)
            description_parts.append(
                f"{added_count} entit{'y' if added_count == 1 else 'ies'} added ({types_added})"
            )
        if removed_count:
            types_removed = self._summarize_entity_types(diff.removed)
            description_parts.append(
                f"{removed_count} entit{'y' if removed_count == 1 else 'ies'} removed ({types_removed})"
            )
        if modified_count:
            types_modified = self._summarize_entity_types(diff.modified)
            description_parts.append(
                f"{modified_count} entit{'y' if modified_count == 1 else 'ies'} modified ({types_modified})"
            )

        if description_parts:
            for part in description_parts:
                lines.append(f"- {part}")
        else:
            lines.append("No changes detected.")
        lines.append("")

        # -- Affected areas --
        lines.append("## Affected Areas")
        lines.append("")
        if diff.change_regions:
            for i, region in enumerate(diff.change_regions, 1):
                lines.append(
                    f"{i}. Region ({region.get('min_x', 0):.1f}, {region.get('min_y', 0):.1f}) "
                    f"to ({region.get('max_x', 0):.1f}, {region.get('max_y', 0):.1f}) "
                    f"-- {region.get('change_count', 0)} change(s)"
                )
        else:
            lines.append("No localized change regions identified.")
        lines.append("")

        # -- Reviewer sign-off --
        lines.append("## Reviewer Sign-off")
        lines.append("")
        lines.append("| Role | Name | Date | Signature |")
        lines.append("|------|------|------|-----------|")
        lines.append("| Design Engineer | | | |")
        lines.append("| Checker | | | |")
        lines.append("| Approver | | | |")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_change_list(changes: List[EntityChange]) -> List[str]:
        """Format a list of EntityChange objects as Markdown bullet points."""
        lines: List[str] = []
        for change in changes:
            loc = f"({change.location[0]:.2f}, {change.location[1]:.2f})"
            detail_str = ""
            if change.details:
                parts = [f"{k}={v}" for k, v in change.details.items()]
                detail_str = " -- " + ", ".join(parts)
            lines.append(f"- **{change.entity_type}** at {loc}{detail_str}")
        return lines

    @staticmethod
    def _summarize_entity_types(changes: List[EntityChange]) -> str:
        """Return a comma-separated summary of entity types in a change list."""
        counts: Dict[str, int] = {}
        for c in changes:
            counts[c.entity_type] = counts.get(c.entity_type, 0) + 1
        parts = [f"{count} {etype}" for etype, count in sorted(counts.items())]
        return ", ".join(parts) if parts else "none"


__all__ = ["DiffReportGenerator"]
