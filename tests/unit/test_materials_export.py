"""Tests for materials export module."""

from __future__ import annotations

import csv
import io
import os
import tempfile

import pytest

from src.core.materials.export import export_equivalence_csv, export_materials_csv


class TestExportMaterialsCsv:
    """Tests for export_materials_csv()."""

    def test_returns_csv_string(self):
        """Should return a non-empty CSV string."""
        result = export_materials_csv()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_csv_has_header(self):
        """CSV should start with expected header row."""
        result = export_materials_csv()
        reader = csv.reader(io.StringIO(result))
        header = next(reader)
        assert "牌号" in header
        assert "名称" in header

    def test_csv_has_data_rows(self):
        """CSV should have data rows after header."""
        result = export_materials_csv()
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) > 1  # header + at least one data row

    def test_csv_parseable(self):
        """Entire CSV output should be parseable."""
        result = export_materials_csv()
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        # All rows should have same number of columns as header
        header_len = len(rows[0])
        for i, row in enumerate(rows[1:], start=2):
            assert len(row) == header_len, f"Row {i} has {len(row)} cols, expected {header_len}"

    def test_write_to_file(self):
        """Should write CSV to file when filepath provided."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            filepath = f.name

        try:
            result = export_materials_csv(filepath=filepath)
            assert os.path.exists(filepath)
            with open(filepath, encoding="utf-8-sig") as f:
                content = f.read()
            assert len(content) > 0
            # Verify file contains valid CSV (don't compare exact bytes due to BOM/newline diffs)
            assert "牌号" in content
        finally:
            os.unlink(filepath)

    def test_no_file_without_filepath(self):
        """Without filepath, should only return string (no side effects)."""
        result = export_materials_csv()
        assert isinstance(result, str)


class TestExportEquivalenceCsv:
    """Tests for export_equivalence_csv()."""

    def test_returns_csv_string(self):
        """Should return a non-empty CSV string."""
        result = export_equivalence_csv()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_csv_has_header(self):
        """CSV should have equivalence header."""
        result = export_equivalence_csv()
        reader = csv.reader(io.StringIO(result))
        header = next(reader)
        assert "牌号" in header
        assert "UNS" in header or "美国(US)" in header

    def test_csv_has_data_rows(self):
        """CSV should have data rows."""
        result = export_equivalence_csv()
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) > 1

    def test_write_to_file(self):
        """Should write CSV to file when filepath provided."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            filepath = f.name

        try:
            export_equivalence_csv(filepath=filepath)
            assert os.path.exists(filepath)
            with open(filepath, encoding="utf-8-sig") as f:
                content = f.read()
            assert len(content) > 0
        finally:
            os.unlink(filepath)
