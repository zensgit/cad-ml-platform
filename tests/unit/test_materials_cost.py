"""Tests for materials cost module."""

from __future__ import annotations

import pytest

from src.core.materials.cost import (
    compare_material_costs,
    get_cost_tier_info,
    get_material_cost,
    search_by_cost,
)


class TestGetMaterialCost:
    """Tests for get_material_cost()."""

    def test_known_material_q235(self):
        """Q235B (baseline material) should have tier 1 and cost_index 1.0."""
        result = get_material_cost("Q235B")
        assert result is not None
        assert result["tier"] == 1
        assert result["cost_index"] == 1.0
        assert result["grade"] == "Q235B"

    def test_stainless_steel_cost(self):
        """Stainless steel should be more expensive than carbon steel."""
        ss = get_material_cost("S30408")
        cs = get_material_cost("Q235B")
        assert ss is not None and cs is not None
        assert ss["cost_index"] > cs["cost_index"]

    def test_titanium_high_cost(self):
        """Titanium should be a high-cost material."""
        result = get_material_cost("TC4")
        assert result is not None
        assert result["tier"] >= 3

    def test_unknown_material_returns_none(self):
        """Unknown material returns None."""
        assert get_material_cost("NONEXISTENT_XYZ") is None

    def test_result_structure(self):
        """Result should have all expected keys."""
        result = get_material_cost("Q235B")
        assert result is not None
        expected_keys = {"grade", "name", "tier", "tier_name", "tier_description",
                         "cost_index", "price_range", "group"}
        assert expected_keys.issubset(result.keys())

    def test_group_default_estimation(self):
        """Materials not in cost table should get estimated cost from group defaults."""
        # 6061 aluminum should still get a cost even if not in direct table
        result = get_material_cost("6061")
        assert result is not None
        assert result["tier"] >= 1


class TestCompareMaterialCosts:
    """Tests for compare_material_costs()."""

    def test_compare_multiple(self):
        """Should return sorted results for multiple materials."""
        results = compare_material_costs(["Q235B", "S30408", "TC4"])
        assert len(results) >= 2
        # Should be sorted by cost_index ascending
        costs = [r["cost_index"] for r in results]
        assert costs == sorted(costs)

    def test_relative_to_cheapest(self):
        """Each result should have relative_to_cheapest field."""
        results = compare_material_costs(["Q235B", "S30408"])
        assert len(results) >= 1
        assert results[0]["relative_to_cheapest"] == 1.0

    def test_include_missing(self):
        """With include_missing=True, should return tuple with missing list."""
        results, missing = compare_material_costs(
            ["Q235B", "NONEXISTENT_XYZ"], include_missing=True
        )
        assert "NONEXISTENT_XYZ" in missing

    def test_empty_list(self):
        """Empty input should return empty results."""
        results = compare_material_costs([])
        assert results == []


class TestSearchByCost:
    """Tests for search_by_cost()."""

    def test_max_tier_filter(self):
        """Should only return materials at or below specified tier."""
        results = search_by_cost(max_tier=1, limit=50)
        for r in results:
            assert r["tier"] <= 1

    def test_max_cost_index_filter(self):
        """Should only return materials at or below specified cost index."""
        results = search_by_cost(max_cost_index=2.0, limit=50)
        for r in results:
            assert r["cost_index"] <= 2.0

    def test_limit_respected(self):
        """Limit parameter should cap results."""
        results = search_by_cost(limit=5)
        assert len(results) <= 5

    def test_results_sorted_by_cost(self):
        """Results should be sorted by cost_index ascending."""
        results = search_by_cost(limit=20)
        costs = [r["cost_index"] for r in results]
        assert costs == sorted(costs)

    def test_result_structure(self):
        """Each result should have required keys."""
        results = search_by_cost(limit=1)
        if results:
            r = results[0]
            assert "grade" in r
            assert "tier" in r
            assert "cost_index" in r


class TestGetCostTierInfo:
    """Tests for get_cost_tier_info()."""

    def test_returns_five_tiers(self):
        """Should return 5 cost tiers."""
        tiers = get_cost_tier_info()
        assert len(tiers) == 5

    def test_tier_structure(self):
        """Each tier should have required fields."""
        tiers = get_cost_tier_info()
        for t in tiers:
            assert "tier" in t
            assert "name" in t
            assert "description" in t
            assert "price_range" in t

    def test_tiers_ordered(self):
        """Tiers should be ordered 1-5."""
        tiers = get_cost_tier_info()
        tier_nums = [t["tier"] for t in tiers]
        assert tier_nums == [1, 2, 3, 4, 5]
