"""Tests for drawing version diff module.

Creates temporary DXF files using ezdxf to exercise GeometryDiff,
AnnotationDiff, and DiffReportGenerator end-to-end.
"""

from __future__ import annotations

import os
import tempfile

import pytest

ezdxf = pytest.importorskip("ezdxf")

from src.core.diff.annotation_diff import AnnotationDiff
from src.core.diff.geometry_diff import GeometryDiff
from src.core.diff.models import DiffResult, EntityChange
from src.core.diff.report import DiffReportGenerator


# ------------------------------------------------------------------
# Helpers: create temporary DXF files with known content
# ------------------------------------------------------------------


def _write_dxf(draw_fn) -> str:
    """Create a temp DXF file, call *draw_fn(msp)* to populate it, return path."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    draw_fn(msp)
    fd, path = tempfile.mkstemp(suffix=".dxf")
    os.close(fd)
    doc.saveas(path)
    return path


def _base_drawing(msp):
    """A simple baseline drawing with a few geometric entities."""
    msp.add_line((0, 0), (10, 0))
    msp.add_line((10, 0), (10, 10))
    msp.add_circle((5, 5), radius=2)
    msp.add_text("TITLE", dxfattribs={"insert": (1, 12), "height": 0.5})


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestIdenticalFiles:
    def test_identical_files_no_changes(self):
        path_a = _write_dxf(_base_drawing)
        path_b = _write_dxf(_base_drawing)
        try:
            diff = GeometryDiff().compare(path_a, path_b)
            assert diff.summary["added"] == 0
            assert diff.summary["removed"] == 0
            assert diff.summary["modified"] == 0
            assert diff.is_empty()
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestAddedEntities:
    def test_added_entities(self):
        def drawing_with_extra(msp):
            _base_drawing(msp)
            msp.add_circle((20, 20), radius=3)
            msp.add_line((30, 0), (30, 10))

        path_a = _write_dxf(_base_drawing)
        path_b = _write_dxf(drawing_with_extra)
        try:
            diff = GeometryDiff().compare(path_a, path_b)
            assert diff.summary["added"] == 2
            added_types = {c.entity_type for c in diff.added}
            assert "CIRCLE" in added_types
            assert "LINE" in added_types
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestRemovedEntities:
    def test_removed_entities(self):
        def drawing_fewer(msp):
            msp.add_line((0, 0), (10, 0))
            # Only 1 line instead of 2 lines + circle + text

        path_a = _write_dxf(_base_drawing)
        path_b = _write_dxf(drawing_fewer)
        try:
            diff = GeometryDiff().compare(path_a, path_b)
            assert diff.summary["removed"] > 0
            removed_types = {c.entity_type for c in diff.removed}
            assert len(removed_types) >= 1
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestModifiedEntityLayer:
    def test_modified_entity_layer(self):
        def drawing_layer_changed(msp):
            msp.add_line((0, 0), (10, 0), dxfattribs={"layer": "MODIFIED"})
            msp.add_line((10, 0), (10, 10))
            msp.add_circle((5, 5), radius=2)
            msp.add_text("TITLE", dxfattribs={"insert": (1, 12), "height": 0.5})

        path_a = _write_dxf(_base_drawing)
        path_b = _write_dxf(drawing_layer_changed)
        try:
            diff = GeometryDiff().compare(path_a, path_b)
            assert diff.summary["modified"] >= 1
            layer_changes = [
                c
                for c in diff.modified
                if "attribute_changes" in c.details
                and any("layer" in d for d in c.details["attribute_changes"])
            ]
            assert len(layer_changes) >= 1
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestChangeRegions:
    def test_change_regions_computed(self):
        def drawing_with_additions(msp):
            _base_drawing(msp)
            # Add entities in a distinct spatial cluster
            msp.add_circle((100, 100), radius=5)
            msp.add_line((102, 100), (108, 100))

        path_a = _write_dxf(_base_drawing)
        path_b = _write_dxf(drawing_with_additions)
        try:
            diff = GeometryDiff().compare(path_a, path_b)
            assert len(diff.change_regions) >= 1
            for region in diff.change_regions:
                assert "min_x" in region
                assert "min_y" in region
                assert "max_x" in region
                assert "max_y" in region
                assert "change_count" in region
                assert region["change_count"] >= 1
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestSummaryCounts:
    def test_summary_counts(self):
        def drawing_mixed(msp):
            # Keep first line at same position (match)
            msp.add_line((0, 0), (10, 0))
            # Remove second line (not present here)
            # Modify circle (different layer)
            msp.add_circle((5, 5), radius=2, dxfattribs={"layer": "NEW_LAYER"})
            # Keep text
            msp.add_text("TITLE", dxfattribs={"insert": (1, 12), "height": 0.5})
            # Add new entity
            msp.add_line((50, 50), (60, 50))

        path_a = _write_dxf(_base_drawing)
        path_b = _write_dxf(drawing_mixed)
        try:
            diff = GeometryDiff().compare(path_a, path_b)
            total = diff.summary["added"] + diff.summary["removed"] + diff.summary["modified"]
            assert total > 0
            # Verify the summary dict keys exist
            assert "added" in diff.summary
            assert "removed" in diff.summary
            assert "modified" in diff.summary
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestDiffReportMarkdown:
    def test_diff_report_markdown(self):
        def drawing_with_extra(msp):
            _base_drawing(msp)
            msp.add_circle((20, 20), radius=3)

        path_a = _write_dxf(_base_drawing)
        path_b = _write_dxf(drawing_with_extra)
        try:
            diff = GeometryDiff().compare(path_a, path_b)
            generator = DiffReportGenerator()
            report = generator.generate_markdown(diff, "rev_A.dxf", "rev_B.dxf")
            assert "# Drawing Version Diff Report" in report
            assert "rev_A.dxf" in report
            assert "rev_B.dxf" in report
            assert "## Summary" in report
            assert "Added" in report
            assert "Removed" in report
            assert "Modified" in report
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestEcnGeneration:
    def test_ecn_generation(self):
        def drawing_with_extra(msp):
            _base_drawing(msp)
            msp.add_line((40, 40), (50, 50))

        path_a = _write_dxf(_base_drawing)
        path_b = _write_dxf(drawing_with_extra)
        try:
            diff = GeometryDiff().compare(path_a, path_b)
            generator = DiffReportGenerator()
            ecn = generator.generate_ecn(diff, part_number="PN-12345", revision="B")
            assert "# Engineering Change Notice" in ecn
            assert "PN-12345" in ecn
            assert "Revision" in ecn
            assert "B" in ecn
            assert "Change Description" in ecn
            assert "Affected Areas" in ecn
            assert "Reviewer Sign-off" in ecn
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestAnnotationDiff:
    def test_annotation_text_change(self):
        def drawing_a(msp):
            msp.add_text("OLD TITLE", dxfattribs={"insert": (5, 5), "height": 0.5})
            msp.add_text("NOTE A", dxfattribs={"insert": (5, 10), "height": 0.3})

        def drawing_b(msp):
            msp.add_text("NEW TITLE", dxfattribs={"insert": (5, 5), "height": 0.5})
            msp.add_text("NOTE A", dxfattribs={"insert": (5, 10), "height": 0.3})

        path_a = _write_dxf(drawing_a)
        path_b = _write_dxf(drawing_b)
        try:
            diff = AnnotationDiff().compare(path_a, path_b)
            assert diff.summary["modified"] >= 1
            text_mods = [c for c in diff.modified if "text_old" in c.details]
            assert len(text_mods) >= 1
            assert text_mods[0].details["text_old"] == "OLD TITLE"
            assert text_mods[0].details["text_new"] == "NEW TITLE"
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestEmptyDiffResult:
    def test_empty_diff_result(self):
        result = DiffResult(summary={"added": 0, "removed": 0, "modified": 0})
        assert result.is_empty()
        generator = DiffReportGenerator()
        md = generator.generate_markdown(result, "a.dxf", "b.dxf")
        assert "No changes detected" in md
