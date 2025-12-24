"""Tests for src/core/ocr/parsing/bbox_mapper.py to improve coverage.

Covers:
- _normalize_text function
- _similarity function
- _best_line_for_value function
- _type_hint_boost function
- assign_bboxes function for dimensions
- assign_bboxes function for symbols
- polygon_to_bbox function
"""

from __future__ import annotations

from src.core.ocr.base import DimensionInfo, DimensionType, SymbolInfo, SymbolType


class TestNormalizeText:
    """Tests for _normalize_text function."""

    def test_strips_whitespace(self):
        """Test strips leading and trailing whitespace."""
        from src.core.ocr.parsing.bbox_mapper import _normalize_text

        result = _normalize_text("  hello world  ")
        assert result == "hello world"

    def test_lowercases_text(self):
        """Test converts to lowercase."""
        from src.core.ocr.parsing.bbox_mapper import _normalize_text

        result = _normalize_text("HELLO WORLD")
        assert result == "hello world"

    def test_collapses_whitespace(self):
        """Test collapses multiple whitespace to single space."""
        from src.core.ocr.parsing.bbox_mapper import _normalize_text

        result = _normalize_text("hello   world\t\ntest")
        assert result == "hello world test"

    def test_handles_empty_string(self):
        """Test handles empty string."""
        from src.core.ocr.parsing.bbox_mapper import _normalize_text

        result = _normalize_text("")
        assert result == ""

    def test_converts_non_string(self):
        """Test converts non-string input to string."""
        from src.core.ocr.parsing.bbox_mapper import _normalize_text

        result = _normalize_text(123)
        assert result == "123"


class TestSimilarity:
    """Tests for _similarity function."""

    def test_identical_strings(self):
        """Test identical strings return 1.0."""
        from src.core.ocr.parsing.bbox_mapper import _similarity

        result = _similarity("hello", "hello")
        assert result == 1.0

    def test_completely_different(self):
        """Test completely different strings return low score."""
        from src.core.ocr.parsing.bbox_mapper import _similarity

        result = _similarity("abc", "xyz")
        assert result < 0.5

    def test_empty_string_a(self):
        """Test empty first string returns 0."""
        from src.core.ocr.parsing.bbox_mapper import _similarity

        result = _similarity("", "hello")
        assert result == 0.0

    def test_empty_string_b(self):
        """Test empty second string returns 0."""
        from src.core.ocr.parsing.bbox_mapper import _similarity

        result = _similarity("hello", "")
        assert result == 0.0

    def test_partial_match(self):
        """Test partial match returns intermediate score."""
        from src.core.ocr.parsing.bbox_mapper import _similarity

        result = _similarity("hello", "hallo")
        assert 0.5 < result < 1.0


class TestBestLineForValue:
    """Tests for _best_line_for_value function."""

    def test_exact_match(self):
        """Test exact numeric match returns high score."""
        from src.core.ocr.parsing.bbox_mapper import _best_line_for_value

        result = _best_line_for_value("20.00mm", 20.0)
        assert result > 0.5

    def test_close_match(self):
        """Test close numeric match returns good score."""
        from src.core.ocr.parsing.bbox_mapper import _best_line_for_value

        result = _best_line_for_value("20.05mm", 20.0)
        assert result > 0.3

    def test_no_numbers_in_text(self):
        """Test returns 0 when no numbers in text."""
        from src.core.ocr.parsing.bbox_mapper import _best_line_for_value

        result = _best_line_for_value("hello world", 20.0)
        assert result == 0.0

    def test_multiple_numbers(self):
        """Test finds closest match among multiple numbers."""
        from src.core.ocr.parsing.bbox_mapper import _best_line_for_value

        result = _best_line_for_value("10 20 30", 20.0)
        assert result > 0.9

    def test_far_value(self):
        """Test far value returns low score."""
        from src.core.ocr.parsing.bbox_mapper import _best_line_for_value

        result = _best_line_for_value("100mm", 10.0)
        assert result < 0.2

    def test_decimal_numbers(self):
        """Test handles decimal numbers."""
        from src.core.ocr.parsing.bbox_mapper import _best_line_for_value

        result = _best_line_for_value("15.5mm", 15.5)
        assert result > 0.9


class TestTypeHintBoost:
    """Tests for _type_hint_boost function."""

    def test_diameter_with_phi(self):
        """Test diameter boost with Φ symbol."""
        from src.core.ocr.parsing.bbox_mapper import _type_hint_boost

        result = _type_hint_boost("Φ20", "diameter")
        assert result == 0.1

    def test_diameter_with_empty_set_symbol(self):
        """Test diameter boost with ⌀ symbol."""
        from src.core.ocr.parsing.bbox_mapper import _type_hint_boost

        result = _type_hint_boost("⌀20", "diameter")
        assert result == 0.1

    def test_diameter_with_diameter_symbol(self):
        """Test diameter boost with ∅ symbol."""
        from src.core.ocr.parsing.bbox_mapper import _type_hint_boost

        result = _type_hint_boost("∅20", "diameter")
        assert result == 0.1

    def test_radius_with_r(self):
        """Test radius boost with R character."""
        from src.core.ocr.parsing.bbox_mapper import _type_hint_boost

        result = _type_hint_boost("R10", "radius")
        assert result == 0.1

    def test_thread_with_m(self):
        """Test thread boost with M character."""
        from src.core.ocr.parsing.bbox_mapper import _type_hint_boost

        result = _type_hint_boost("M10x1.5", "thread")
        assert result == 0.1

    def test_no_boost_for_other_type(self):
        """Test no boost for unmatched type."""
        from src.core.ocr.parsing.bbox_mapper import _type_hint_boost

        result = _type_hint_boost("20mm", "length")
        assert result == 0.0

    def test_no_boost_for_none_type(self):
        """Test returns 0 for None dim_type."""
        from src.core.ocr.parsing.bbox_mapper import _type_hint_boost

        result = _type_hint_boost("Φ20", None)
        assert result == 0.0


class TestAssignBboxesDimensions:
    """Tests for assign_bboxes with dimensions."""

    def test_exact_raw_match(self):
        """Test assigns bbox on exact raw text match."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=20.0, raw="20mm")
        ocr_lines = [{"text": "20mm", "bbox": [10, 20, 30, 40]}]

        assign_bboxes([dim], [], ocr_lines)

        assert dim.bbox == [10, 20, 30, 40]

    def test_skips_if_bbox_already_set(self):
        """Test skips dimension if bbox already set."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=20.0, raw="20mm", bbox=[1, 2, 3, 4])
        ocr_lines = [{"text": "20mm", "bbox": [10, 20, 30, 40]}]

        assign_bboxes([dim], [], ocr_lines)

        assert dim.bbox == [1, 2, 3, 4]  # Unchanged

    def test_value_substring_match(self):
        """Test matches on value substring when raw doesn't match."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=20.0, raw="missing")
        ocr_lines = [{"text": "Dimension: 20.0", "bbox": [10, 20, 30, 40]}]

        assign_bboxes([dim], [], ocr_lines)

        assert dim.bbox == [10, 20, 30, 40]

    def test_assigns_confidence_from_score(self):
        """Test assigns confidence from line score."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=20.0, raw="20mm")
        ocr_lines = [{"text": "20mm", "bbox": [10, 20, 30, 40], "score": 0.95}]

        assign_bboxes([dim], [], ocr_lines)

        assert dim.confidence == 0.95

    def test_no_match_returns_no_bbox(self):
        """Test no bbox assigned when no match found."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=999.0, raw="xyz")
        ocr_lines = [{"text": "completely different", "bbox": [10, 20, 30, 40]}]

        assign_bboxes([dim], [], ocr_lines)

        assert dim.bbox is None

    def test_skips_line_without_bbox(self):
        """Test skips OCR lines without bbox."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=20.0, raw="20mm")
        ocr_lines = [
            {"text": "20mm", "bbox": None},
            {"text": "20mm", "bbox": [10, 20, 30, 40]},
        ]

        assign_bboxes([dim], [], ocr_lines)

        assert dim.bbox == [10, 20, 30, 40]


class TestAssignBboxesSymbols:
    """Tests for assign_bboxes with symbols."""

    def test_symbol_exact_match(self):
        """Test assigns bbox on exact symbol match."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        sym = SymbolInfo(type=SymbolType.surface_roughness, value="Ra 3.2")
        ocr_lines = [{"text": "Ra 3.2", "bbox": [10, 20, 30, 40]}]

        assign_bboxes([], [sym], ocr_lines)

        assert sym.bbox == [10, 20, 30, 40]

    def test_symbol_skips_if_bbox_set(self):
        """Test skips symbol if bbox already set."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        sym = SymbolInfo(type=SymbolType.surface_roughness, value="Ra 3.2", bbox=[1, 2, 3, 4])
        ocr_lines = [{"text": "Ra 3.2", "bbox": [10, 20, 30, 40]}]

        assign_bboxes([], [sym], ocr_lines)

        assert sym.bbox == [1, 2, 3, 4]  # Unchanged

    def test_symbol_assigns_confidence(self):
        """Test assigns confidence from line score for symbol."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        sym = SymbolInfo(type=SymbolType.surface_roughness, value="Ra 3.2")
        ocr_lines = [{"text": "Ra 3.2", "bbox": [10, 20, 30, 40], "score": 0.88}]

        assign_bboxes([], [sym], ocr_lines)

        assert sym.confidence == 0.88

    def test_symbol_similarity_fallback(self):
        """Test symbol uses similarity fallback when no exact match."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        sym = SymbolInfo(type=SymbolType.surface_roughness, value="Ra 3.2")
        ocr_lines = [{"text": "Ra 3.3", "bbox": [10, 20, 30, 40], "score": 0.9}]

        assign_bboxes([], [sym], ocr_lines)

        # Should find via similarity-based scoring
        assert sym.bbox is not None


class TestPolygonToBbox:
    """Tests for polygon_to_bbox function."""

    def test_empty_polygon(self):
        """Test returns empty list for empty polygon."""
        from src.core.ocr.parsing.bbox_mapper import polygon_to_bbox

        result = polygon_to_bbox([])
        assert result == []

    def test_bbox_like_list(self):
        """Test converts bbox-like flat list."""
        from src.core.ocr.parsing.bbox_mapper import polygon_to_bbox

        result = polygon_to_bbox([10.5, 20.5, 30.5, 40.5])
        assert result == [10, 20, 30, 40]

    def test_polygon_points_list(self):
        """Test converts polygon points list."""
        from src.core.ocr.parsing.bbox_mapper import polygon_to_bbox

        poly = [[10, 20], [50, 20], [50, 60], [10, 60]]
        result = polygon_to_bbox(poly)

        # x=10, y=20, w=50-10=40, h=60-20=40
        assert result == [10, 20, 40, 40]

    def test_polygon_with_floats(self):
        """Test handles polygon with float coordinates."""
        from src.core.ocr.parsing.bbox_mapper import polygon_to_bbox

        poly = [[10.5, 20.5], [50.5, 20.5], [50.5, 60.5], [10.5, 60.5]]
        result = polygon_to_bbox(poly)

        assert result == [10, 20, 40, 40]

    def test_irregular_polygon(self):
        """Test converts irregular polygon to bbox."""
        from src.core.ocr.parsing.bbox_mapper import polygon_to_bbox

        # Triangle
        poly = [[0, 0], [100, 50], [50, 100]]
        result = polygon_to_bbox(poly)

        # x=0, y=0, w=100-0=100, h=100-0=100
        assert result == [0, 0, 100, 100]


class TestAssignBboxesHeuristic:
    """Tests for heuristic scoring in assign_bboxes."""

    def test_heuristic_scoring_with_type_boost(self):
        """Test heuristic scoring with type boost."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.diameter, value=20.0, raw="Φ20")
        ocr_lines = [
            {"text": "Test 20", "bbox": [5, 5, 5, 5], "score": 0.5},
            {"text": "Φ20mm", "bbox": [10, 20, 30, 40], "score": 0.9},
        ]

        assign_bboxes([dim], [], ocr_lines)

        # Should prefer the one with Φ symbol due to type hint boost
        assert dim.bbox is not None

    def test_heuristic_below_threshold(self):
        """Test no bbox assigned when heuristic score below threshold."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=999.0, raw="")
        ocr_lines = [{"text": "completely unrelated text", "bbox": [10, 20, 30, 40]}]

        assign_bboxes([dim], [], ocr_lines)

        # Score should be below 0.6 threshold
        assert dim.bbox is None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_ocr_lines(self):
        """Test handles empty OCR lines."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=20.0, raw="20mm")
        assign_bboxes([dim], [], [])

        assert dim.bbox is None

    def test_empty_dimensions_and_symbols(self):
        """Test handles empty dimensions and symbols."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        ocr_lines = [{"text": "20mm", "bbox": [10, 20, 30, 40]}]
        # Should not raise
        assign_bboxes([], [], ocr_lines)

    def test_line_without_text(self):
        """Test handles OCR line without text key."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=20.0, raw="20mm")
        ocr_lines = [{"bbox": [10, 20, 30, 40]}]  # No text

        assign_bboxes([dim], [], ocr_lines)

        # Should not crash, text defaults to ""

    def test_dimension_without_raw(self):
        """Test dimension without raw text."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=20.0)
        ocr_lines = [{"text": "20.0", "bbox": [10, 20, 30, 40]}]

        assign_bboxes([dim], [], ocr_lines)

        # Should match on value
        assert dim.bbox == [10, 20, 30, 40]

    def test_integer_score_conversion(self):
        """Test handles integer score conversion."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=20.0, raw="20mm")
        ocr_lines = [{"text": "20mm", "bbox": [10, 20, 30, 40], "score": 1}]

        assign_bboxes([dim], [], ocr_lines)

        assert dim.confidence == 1.0

    def test_case_insensitive_match(self):
        """Test matching is case insensitive."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        dim = DimensionInfo(type=DimensionType.length, value=20.0, raw="20MM")
        ocr_lines = [{"text": "20mm", "bbox": [10, 20, 30, 40]}]

        assign_bboxes([dim], [], ocr_lines)

        assert dim.bbox == [10, 20, 30, 40]


class TestImports:
    """Tests for module imports."""

    def test_assign_bboxes_import(self):
        """Test assign_bboxes can be imported."""
        from src.core.ocr.parsing.bbox_mapper import assign_bboxes

        assert callable(assign_bboxes)

    def test_polygon_to_bbox_import(self):
        """Test polygon_to_bbox can be imported."""
        from src.core.ocr.parsing.bbox_mapper import polygon_to_bbox

        assert callable(polygon_to_bbox)
