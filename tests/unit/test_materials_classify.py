"""Tests for materials classification and search module."""

from __future__ import annotations

import pytest

from src.core.materials.classify import (
    _calculate_similarity,
    classify_material_detailed,
    classify_material_simple,
    search_materials,
)
from src.core.materials.data_models import MaterialCategory, MaterialGroup


class TestClassifyMaterialDetailed:
    """Tests for classify_material_detailed()."""

    def test_known_material_q235(self):
        """Q235B should be classified as carbon steel."""
        info = classify_material_detailed("Q235B")
        assert info is not None
        assert info.grade == "Q235B"
        assert info.group == MaterialGroup.CARBON_STEEL
        assert info.category == MaterialCategory.METAL

    def test_known_material_304ss(self):
        """304 stainless steel (S30408) should be classified correctly."""
        info = classify_material_detailed("S30408")
        assert info is not None
        assert info.group == MaterialGroup.STAINLESS_STEEL

    def test_known_material_al6061(self):
        """6061 aluminum alloy should be classified as aluminum."""
        info = classify_material_detailed("6061")
        assert info is not None
        assert info.group == MaterialGroup.ALUMINUM
        assert info.category == MaterialCategory.METAL

    def test_case_insensitive_match(self):
        """Classification should be case-insensitive."""
        info_upper = classify_material_detailed("Q235B")
        info_lower = classify_material_detailed("q235b")
        assert info_upper is not None
        assert info_lower is not None
        assert info_upper.grade == info_lower.grade

    def test_empty_string_returns_none(self):
        """Empty string input should return None."""
        assert classify_material_detailed("") is None

    def test_none_input_returns_none(self):
        """None input should return None."""
        assert classify_material_detailed(None) is None

    def test_unknown_material_returns_none(self):
        """Unknown material name should return None."""
        assert classify_material_detailed("NONEXISTENT_MATERIAL_XYZ") is None

    def test_whitespace_only_returns_none(self):
        """Whitespace-only input should return None."""
        assert classify_material_detailed("   ") is None

    def test_stainless_steel_keyword_match(self):
        """Input containing '不锈钢' should match stainless steel."""
        info = classify_material_detailed("某型不锈钢")
        assert info is not None
        assert info.group == MaterialGroup.STAINLESS_STEEL

    def test_alias_match(self):
        """Materials should be findable by alias."""
        # 304 is a common alias for S30408
        info = classify_material_detailed("304")
        assert info is not None
        assert info.group == MaterialGroup.STAINLESS_STEEL

    def test_returns_material_info_type(self):
        """Return value should be a MaterialInfo dataclass."""
        info = classify_material_detailed("45")
        assert info is not None
        assert hasattr(info, "grade")
        assert hasattr(info, "name")
        assert hasattr(info, "category")
        assert hasattr(info, "group")


class TestClassifyMaterialSimple:
    """Tests for classify_material_simple()."""

    def test_returns_group_string(self):
        """Should return the material group value string."""
        result = classify_material_simple("Q235B")
        assert result is not None
        assert result == "carbon_steel"

    def test_stainless_steel_group(self):
        """Stainless steel should return correct group."""
        result = classify_material_simple("S30408")
        assert result == "stainless_steel"

    def test_none_for_empty(self):
        """Empty input returns None."""
        assert classify_material_simple("") is None

    def test_none_for_unknown(self):
        """Unknown material returns None."""
        assert classify_material_simple("NONEXISTENT_XYZ") is None


class TestSearchMaterials:
    """Tests for search_materials()."""

    def test_exact_match_returns_single(self):
        """Exact match should return a single result with score 1.0."""
        results = search_materials("Q235B")
        assert len(results) == 1
        assert results[0]["grade"] == "Q235B"
        assert results[0]["score"] == 1.0
        assert results[0]["match_type"] == "exact"

    def test_fuzzy_search_returns_results(self):
        """Fuzzy search should return relevant results."""
        results = search_materials("304", limit=5)
        assert len(results) >= 1
        # Should find S30408 (304 stainless)
        grades = [r["grade"] for r in results]
        assert any("S304" in g or "304" in g for g in grades)

    def test_empty_query_returns_empty(self):
        """Empty query should return empty list."""
        assert search_materials("") == []
        assert search_materials("   ") == []

    def test_limit_respected(self):
        """Limit parameter should cap results."""
        results = search_materials("钢", limit=3)
        assert len(results) <= 3

    def test_category_filter(self):
        """Category filter should restrict results."""
        results = search_materials("钢", category="metal", limit=50)
        for r in results:
            assert r["category"] == "metal"

    def test_min_score_filter(self):
        """Results should respect min_score threshold."""
        results = search_materials("铝", min_score=0.5)
        for r in results:
            assert r["score"] >= 0.5

    def test_result_structure(self):
        """Each result should have required keys."""
        results = search_materials("Q235B")
        assert len(results) >= 1
        r = results[0]
        assert "grade" in r
        assert "name" in r
        assert "category" in r
        assert "group" in r
        assert "score" in r
        assert "match_type" in r

    def test_pinyin_search(self):
        """Pinyin search should find materials."""
        results = search_materials("buxiugang", limit=5)
        # Should find stainless steel materials
        assert len(results) >= 1


class TestCalculateSimilarity:
    """Tests for _calculate_similarity()."""

    def test_exact_match_is_one(self):
        assert _calculate_similarity("Q235B", "Q235B") == 1.0

    def test_case_insensitive_match(self):
        assert _calculate_similarity("q235b", "Q235B") == 1.0

    def test_substring_match(self):
        score = _calculate_similarity("304", "S30408")
        assert score > 0.0

    def test_completely_different(self):
        score = _calculate_similarity("abc", "xyz")
        assert score < 0.5

    def test_empty_strings(self):
        score = _calculate_similarity("", "")
        assert score == 1.0
