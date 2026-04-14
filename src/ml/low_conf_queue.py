"""Low-confidence prediction queue for active learning (B5.3).

When HybridClassifier produces a prediction with confidence below a
configurable threshold, the file hash + metadata is written to a CSV
review queue.  A human annotator reviews the queue, corrects labels,
and the confirmed samples are appended to the training manifest for the
next fine-tuning round (B5.4 active learning loop).

Design principles:
  - Append-only CSV — never overwrites existing entries
  - File-hash deduplication — same content is not queued twice in a session
  - Zero external dependencies (stdlib only)
  - Thread-safe via file-level append (GIL + OS atomic append on POSIX)

Usage::

    queue = LowConfidenceQueue()

    # inside HybridClassifier.classify():
    queue.maybe_enqueue(
        file_hash=file_hash_hex,
        filename=filename,
        predicted_class=result.label or "unknown",
        confidence=result.confidence,
    )
"""

from __future__ import annotations

import csv
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_FIELDNAMES = [
    "file_hash",
    "filename",
    "predicted_class",
    "confidence",
    "source",
    "timestamp",
    "reviewed_label",        # left blank; filled by human annotator
    "notes",                 # left blank; filled by human annotator
    "sample_source",         # legacy_low_conf_queue
    "label_source",          # filled when reviewed
    "human_verified",        # filled when reviewed
    "eligible_for_training", # filled when reviewed
]

# Default confidence threshold below which samples are enqueued
DEFAULT_THRESHOLD: float = 0.50


class LowConfidenceQueue:
    """Collect low-confidence predictions for human review.

    Args:
        queue_path: Path to the CSV review queue file.
        threshold: Predictions with confidence < threshold are enqueued.
        dedup_session: If True (default), deduplicate by file_hash within a
            single process lifetime to avoid flooding the queue with the same
            file during batch processing.
    """

    def __init__(
        self,
        queue_path: str = "data/review_queue/low_conf.csv",
        threshold: float = DEFAULT_THRESHOLD,
        dedup_session: bool = True,
    ) -> None:
        self.queue_path = Path(queue_path)
        self.threshold = threshold
        self.dedup_session = dedup_session
        self._seen_hashes: set[str] = set()
        self._ensure_header()

    # ── Public API ────────────────────────────────────────────────────────────

    def maybe_enqueue(
        self,
        file_hash: str,
        filename: str,
        predicted_class: str,
        confidence: float,
        source: str = "hybrid",
        threshold: Optional[float] = None,
    ) -> bool:
        """Enqueue a prediction record if confidence is below threshold.

        Args:
            file_hash: MD5 or SHA-256 hex digest of the DXF file content.
            filename: Original DXF filename (for human context).
            predicted_class: Top-1 predicted class label.
            confidence: Top-1 prediction confidence (0–1).
            source: Which classifier branch produced this result.
            threshold: Override instance threshold for this call.

        Returns:
            True if the record was enqueued, False otherwise.
        """
        effective_threshold = threshold if threshold is not None else self.threshold
        if confidence >= effective_threshold:
            return False

        if self.dedup_session and file_hash in self._seen_hashes:
            logger.debug(
                "low_conf_queue: skipping duplicate file_hash=%s (already queued this session)",
                file_hash,
            )
            return False

        row = {
            "file_hash": file_hash,
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.4f}",
            "source": source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reviewed_label": "",
            "notes": "",
            "sample_source": "legacy_low_conf_queue",
            "label_source": "",
            "human_verified": "",
            "eligible_for_training": "",
        }
        self._append_row(row)

        if self.dedup_session:
            self._seen_hashes.add(file_hash)

        logger.debug(
            "low_conf_queue: enqueued file_hash=%s  class=%s  conf=%.3f",
            file_hash,
            predicted_class,
            confidence,
        )
        return True

    def size(self) -> int:
        """Return the number of queued entries (reads the file each call)."""
        if not self.queue_path.exists():
            return 0
        with open(self.queue_path, "r", encoding="utf-8") as f:
            return sum(1 for row in csv.DictReader(f))

    def pending_review(self) -> int:
        """Return the number of entries that have no reviewed_label yet."""
        if not self.queue_path.exists():
            return 0
        count = 0
        with open(self.queue_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("reviewed_label", "").strip():
                    count += 1
        return count

    def reviewed_entries(self) -> list[dict]:
        """Return all entries that have a reviewed_label (ready for training).

        An entry is considered reviewed when it has a reviewed_label. The
        human_verified column is also checked when present so that callers can
        distinguish fully verified samples from those with a label only.
        """
        if not self.queue_path.exists():
            return []
        result = []
        with open(self.queue_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("reviewed_label", "").strip():
                    result.append(row)
        return result

    def human_verified_entries(self) -> list[dict]:
        """Return entries where both reviewed_label and human_verified are set."""
        if not self.queue_path.exists():
            return []
        result = []
        with open(self.queue_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                has_label = row.get("reviewed_label", "").strip()
                human_verified = str(row.get("human_verified", "")).strip().lower()
                if has_label and human_verified in ("true", "1", "yes"):
                    result.append(row)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ensure_header(self) -> None:
        """Create the CSV file with a header row if it doesn't exist."""
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.queue_path.exists():
            with open(self.queue_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                writer.writeheader()
            logger.info("low_conf_queue: created %s", self.queue_path)

    def _append_row(self, row: dict) -> None:
        """Append a single row to the CSV queue (atomic POSIX append)."""
        with open(self.queue_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writerow(row)


# ── Convenience helper ────────────────────────────────────────────────────────

def dxf_file_hash(dxf_bytes: bytes, length: int = 12) -> str:
    """Return a short SHA-256 hex digest of DXF file bytes for queue keys."""
    return hashlib.sha256(dxf_bytes).hexdigest()[:length]
