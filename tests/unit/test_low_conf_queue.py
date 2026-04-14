"""Unit tests for src/ml/low_conf_queue.py (B5.3 LowConfidenceQueue).

Covers:
- maybe_enqueue: enqueues when confidence < threshold
- maybe_enqueue: does not enqueue when confidence >= threshold
- Session deduplication: same file_hash not queued twice
- size() and pending_review() counts
- reviewed_entries() returns only annotated rows
- Header written once (no duplicate headers on multiple runs)
- dxf_file_hash utility
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import pytest

from src.ml.low_conf_queue import LowConfidenceQueue, dxf_file_hash


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_queue(tmp_path):
    """A LowConfidenceQueue backed by a temp file."""
    q = LowConfidenceQueue(
        queue_path=str(tmp_path / "queue.csv"),
        threshold=0.50,
        dedup_session=True,
    )
    return q


# ── maybe_enqueue ─────────────────────────────────────────────────────────────

class TestMaybeEnqueue:
    def test_enqueues_below_threshold(self, tmp_queue):
        result = tmp_queue.maybe_enqueue(
            file_hash="abc123", filename="part.dxf",
            predicted_class="法兰", confidence=0.30,
        )
        assert result is True
        assert tmp_queue.size() == 1

    def test_does_not_enqueue_at_threshold(self, tmp_queue):
        result = tmp_queue.maybe_enqueue(
            file_hash="abc123", filename="part.dxf",
            predicted_class="法兰", confidence=0.50,  # == threshold
        )
        assert result is False
        assert tmp_queue.size() == 0

    def test_does_not_enqueue_above_threshold(self, tmp_queue):
        result = tmp_queue.maybe_enqueue(
            file_hash="abc123", filename="part.dxf",
            predicted_class="法兰", confidence=0.90,
        )
        assert result is False
        assert tmp_queue.size() == 0

    def test_returns_false_on_high_confidence(self, tmp_queue):
        assert tmp_queue.maybe_enqueue("h1", "a.dxf", "轴类", 0.99) is False

    def test_custom_threshold_override(self, tmp_queue):
        # Instance threshold=0.50, but call-level override=0.80
        result = tmp_queue.maybe_enqueue(
            file_hash="h2", filename="b.dxf",
            predicted_class="箱体", confidence=0.70,
            threshold=0.80,
        )
        assert result is True

    def test_multiple_different_files_enqueued(self, tmp_queue):
        for i in range(5):
            tmp_queue.maybe_enqueue(f"hash{i}", f"part{i}.dxf", "法兰", 0.1)
        assert tmp_queue.size() == 5


# ── Session deduplication ─────────────────────────────────────────────────────

class TestDeduplication:
    def test_same_hash_not_enqueued_twice(self, tmp_queue):
        tmp_queue.maybe_enqueue("dup_hash", "part.dxf", "法兰", 0.2)
        tmp_queue.maybe_enqueue("dup_hash", "part.dxf", "法兰", 0.2)
        assert tmp_queue.size() == 1

    def test_different_hashes_both_enqueued(self, tmp_queue):
        tmp_queue.maybe_enqueue("hash_a", "a.dxf", "法兰", 0.2)
        tmp_queue.maybe_enqueue("hash_b", "b.dxf", "法兰", 0.2)
        assert tmp_queue.size() == 2

    def test_dedup_disabled_allows_duplicates(self, tmp_path):
        q = LowConfidenceQueue(
            queue_path=str(tmp_path / "queue_no_dedup.csv"),
            threshold=0.50,
            dedup_session=False,
        )
        q.maybe_enqueue("dup_hash", "part.dxf", "法兰", 0.2)
        q.maybe_enqueue("dup_hash", "part.dxf", "法兰", 0.2)
        assert q.size() == 2


# ── size() and pending_review() ───────────────────────────────────────────────

class TestSizeAndPending:
    def test_size_empty(self, tmp_queue):
        assert tmp_queue.size() == 0

    def test_pending_review_empty(self, tmp_queue):
        assert tmp_queue.pending_review() == 0

    def test_pending_review_all_pending(self, tmp_queue):
        for i in range(3):
            tmp_queue.maybe_enqueue(f"h{i}", f"p{i}.dxf", "法兰", 0.1)
        assert tmp_queue.pending_review() == 3

    def test_pending_review_excludes_reviewed(self, tmp_path):
        q = LowConfidenceQueue(str(tmp_path / "q.csv"), threshold=0.5)
        q.maybe_enqueue("h1", "a.dxf", "法兰", 0.1)
        q.maybe_enqueue("h2", "b.dxf", "轴类", 0.1)
        # Manually mark one as reviewed
        rows = []
        with open(q.queue_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows[0]["reviewed_label"] = "换热器"
        with open(q.queue_path, "w", newline="", encoding="utf-8") as f:
            from src.ml.low_conf_queue import _FIELDNAMES
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        assert q.pending_review() == 1


# ── reviewed_entries() ────────────────────────────────────────────────────────

class TestReviewedEntries:
    def test_empty_when_no_reviews(self, tmp_queue):
        tmp_queue.maybe_enqueue("h1", "a.dxf", "法兰", 0.1)
        assert tmp_queue.reviewed_entries() == []

    def test_returns_reviewed_row(self, tmp_path):
        q = LowConfidenceQueue(str(tmp_path / "q.csv"), threshold=0.5)
        q.maybe_enqueue("h1", "a.dxf", "法兰", 0.1)
        # Simulate human annotation
        rows = []
        with open(q.queue_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows[0]["reviewed_label"] = "换热器"
        from src.ml.low_conf_queue import _FIELDNAMES
        with open(q.queue_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        reviewed = q.reviewed_entries()
        assert len(reviewed) == 1
        assert reviewed[0]["reviewed_label"] == "换热器"
        assert reviewed[0]["file_hash"] == "h1"


# ── CSV integrity ─────────────────────────────────────────────────────────────

class TestCsvIntegrity:
    def test_header_present_in_file(self, tmp_queue):
        tmp_queue.maybe_enqueue("h1", "a.dxf", "法兰", 0.1)
        with open(tmp_queue.queue_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        assert "file_hash" in first_line

    def test_no_duplicate_header(self, tmp_path):
        """Creating two queue instances on the same file should not duplicate header."""
        path = str(tmp_path / "q.csv")
        q1 = LowConfidenceQueue(path, threshold=0.5)
        q1.maybe_enqueue("h1", "a.dxf", "法兰", 0.1)
        # Second instance on same file
        q2 = LowConfidenceQueue(path, threshold=0.5)
        q2.maybe_enqueue("h2", "b.dxf", "轴类", 0.1)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Count header occurrences
        assert content.count("file_hash") == 1  # header line only

    def test_fields_correct_in_row(self, tmp_queue):
        tmp_queue.maybe_enqueue(
            file_hash="abc", filename="test.dxf",
            predicted_class="换热器", confidence=0.25, source="graph2d",
        )
        with open(tmp_queue.queue_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        assert row["file_hash"] == "abc"
        assert row["filename"] == "test.dxf"
        assert row["predicted_class"] == "换热器"
        assert float(row["confidence"]) == pytest.approx(0.25, abs=0.0001)
        assert row["source"] == "graph2d"
        assert row["reviewed_label"] == ""


# ── dxf_file_hash utility ────────────────────────────────────────────────────

class TestDxfFileHash:
    def test_returns_hex_string(self):
        h = dxf_file_hash(b"some dxf content")
        assert isinstance(h, str)
        assert all(c in "0123456789abcdef" for c in h)

    def test_default_length_12(self):
        h = dxf_file_hash(b"content")
        assert len(h) == 12

    def test_custom_length(self):
        h = dxf_file_hash(b"content", length=8)
        assert len(h) == 8

    def test_deterministic(self):
        h1 = dxf_file_hash(b"same content")
        h2 = dxf_file_hash(b"same content")
        assert h1 == h2

    def test_different_bytes_different_hash(self):
        h1 = dxf_file_hash(b"content A")
        h2 = dxf_file_hash(b"content B")
        assert h1 != h2
