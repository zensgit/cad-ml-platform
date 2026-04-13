"""Tests for materials equivalence module."""

from __future__ import annotations

import pytest

from src.core.materials.equivalence import (
    find_equivalent_material,
    get_material_equivalence,
    list_material_standards,
)


class TestGetMaterialEquivalence:
    """Tests for get_material_equivalence()."""

    def test_known_material_s30408(self):
        """S30408 (304) should have equivalence data."""
        equiv = get_material_equivalence("S30408")
        assert equiv is not None
        assert "CN" in equiv
        assert "US" in equiv

    def test_equivalence_has_us_standard(self):
        """S30408 US equivalent should be 304."""
        equiv = get_material_equivalence("S30408")
        assert equiv is not None
        assert equiv["US"] == "304"

    def test_reverse_lookup(self):
        """Should find equivalence by foreign standard name (reverse lookup)."""
        equiv = get_material_equivalence("304")
        assert equiv is not None
        assert "CN" in equiv

    def test_unknown_material(self):
        """Unknown material should return None."""
        assert get_material_equivalence("NONEXISTENT_XYZ") is None

    def test_316l_equivalence(self):
        """S31603 (316L) should have JP standard."""
        equiv = get_material_equivalence("S31603")
        assert equiv is not None
        assert "JP" in equiv
        assert equiv["JP"] == "SUS316L"


class TestFindEquivalentMaterial:
    """Tests for find_equivalent_material()."""

    def test_find_us_equivalent(self):
        """Should find US equivalent for Chinese grade."""
        result = find_equivalent_material("S30408", "US")
        assert result == "304"

    def test_find_jp_equivalent(self):
        """Should find JP equivalent."""
        result = find_equivalent_material("S30408", "JP")
        assert result == "SUS304"

    def test_find_de_equivalent(self):
        """Should find DE equivalent."""
        result = find_equivalent_material("S30408", "DE")
        assert result == "1.4301"

    def test_unknown_target_standard(self):
        """Unknown target standard should return None."""
        result = find_equivalent_material("S30408", "XX")
        assert result is None

    def test_unknown_material(self):
        """Unknown material should return None."""
        assert find_equivalent_material("NONEXISTENT", "US") is None


class TestListMaterialStandards:
    """Tests for list_material_standards()."""

    def test_known_material_returns_list(self):
        """Should return list of (standard, grade) tuples."""
        standards = list_material_standards("S30408")
        assert len(standards) >= 2
        std_names = [s[0] for s in standards]
        assert "CN" in std_names
        assert "US" in std_names

    def test_excludes_name_field(self):
        """Should not include 'name' as a standard."""
        standards = list_material_standards("S30408")
        std_names = [s[0] for s in standards]
        assert "name" not in std_names

    def test_unknown_material_returns_empty(self):
        """Unknown material should return empty list."""
        assert list_material_standards("NONEXISTENT") == []

    def test_tuple_format(self):
        """Each entry should be a (str, str) tuple."""
        standards = list_material_standards("S30408")
        for std, val in standards:
            assert isinstance(std, str)
            assert isinstance(val, str)
