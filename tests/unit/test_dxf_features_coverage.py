"""Additional tests for dxf_features to improve coverage.

Targets uncovered code paths in src/utils/dxf_features.py:
- Line 35: ezdxf_doc parameter usage
- Lines 70-97: ARC, TEXT/MTEXT, LWPOLYLINE/POLYLINE, INSERT, DIMENSION, HATCH entity handling
- Lines 136, 148, 151, 172, 206, 210: empty data branches
- Lines 232-234: exception handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils.dxf_features import extract_features_v6


# --- Mock Entity Classes ---


def _make_mock_entity(etype: str, **attrs):
    """Create a mock DXF entity."""
    entity = MagicMock()
    entity.dxftype.return_value = etype
    entity.dxf = MagicMock()
    for key, value in attrs.items():
        setattr(entity.dxf, key, value)
    return entity


def _make_point(x: float, y: float, z: float = 0.0):
    """Create a mock point."""
    point = MagicMock()
    point.x = x
    point.y = y
    point.z = z
    return point


# --- Tests for ezdxf_doc parameter ---


class TestEzdxfDocParameter:
    """Tests for ezdxf_doc parameter usage."""

    def test_extract_with_preloaded_doc(self):
        """Line 35: Use pre-loaded ezdxf document."""
        mock_entity = _make_mock_entity(
            "LINE",
            start=_make_point(0, 0),
            end=_make_point(100, 0),
            layer="0",
        )

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_entity]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        # Pass pre-loaded doc - should not call ezdxf.readfile
        with patch("ezdxf.readfile") as mock_readfile:
            result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
            mock_readfile.assert_not_called()

        assert result is not None
        assert len(result) == 48


# --- Tests for entity type handling ---


class TestEntityTypeHandling:
    """Tests for different entity type handling."""

    def test_arc_entity(self):
        """Lines 70-77: ARC entity handling."""
        mock_arc = _make_mock_entity(
            "ARC",
            center=_make_point(50, 50),
            radius=25.0,
            start_angle=0.0,
            end_angle=90.0,
            layer="0",
        )

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_arc]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None
        assert len(result) == 48

    def test_arc_entity_large_angle(self):
        """Line 75-76: ARC with angle > 180."""
        mock_arc = _make_mock_entity(
            "ARC",
            center=_make_point(50, 50),
            radius=25.0,
            start_angle=0.0,
            end_angle=270.0,  # 270 > 180, so angle = 360 - 270 = 90
            layer="0",
        )

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_arc]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None

    def test_text_entity(self):
        """Lines 78-80: TEXT entity handling."""
        mock_text = _make_mock_entity(
            "TEXT",
            insert=_make_point(10, 20),
            layer="TEXT_LAYER",
        )

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_text]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None

    def test_mtext_entity(self):
        """Lines 78-80: MTEXT entity handling."""
        mock_mtext = _make_mock_entity(
            "MTEXT",
            insert=_make_point(10, 20),
            layer="0",
        )

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_mtext]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None

    def test_lwpolyline_entity(self):
        """Lines 81-86: LWPOLYLINE entity handling."""
        mock_polyline = _make_mock_entity("LWPOLYLINE", layer="0")
        mock_polyline.get_points.return_value = [
            (0, 0),
            (10, 0),
            (10, 10),
            (0, 10),
        ]

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_polyline]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None

    def test_insert_entity(self):
        """Lines 87-91: INSERT entity (block reference) handling."""
        mock_insert = _make_mock_entity(
            "INSERT",
            insert=_make_point(100, 100),
            name="BLOCK_NAME",
            layer="0",
        )

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_insert]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None

    def test_dimension_entity(self):
        """Lines 92-93: DIMENSION entity handling."""
        mock_dim = _make_mock_entity("DIMENSION", layer="DIM_LAYER")

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_dim]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None

    def test_hatch_entity(self):
        """Lines 94-95: HATCH entity handling."""
        mock_hatch = _make_mock_entity("HATCH", layer="0")

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_hatch]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None

    def test_entity_parse_exception_handled(self):
        """Lines 96-97: Entity parse exception is handled gracefully."""
        # Create a valid LINE entity first, then one that will fail
        good_entity = _make_mock_entity(
            "LINE",
            start=_make_point(0, 0),
            end=_make_point(10, 0),
            layer="0",
        )

        # Create an entity that will raise during parsing
        bad_entity = MagicMock()
        bad_entity.dxftype.return_value = "LINE"
        bad_entity.dxf = MagicMock()
        bad_entity.dxf.layer = "0"
        # Make accessing start raise an exception
        type(bad_entity.dxf).start = property(lambda self: (_ for _ in ()).throw(ValueError("Parse error")))

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([good_entity, bad_entity]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        # Should not raise, exception is caught and logged
        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        # Result should still be valid because we have one good entity
        assert result is not None
        assert len(result) == 48


# --- Tests for empty data branches ---


class TestEmptyDataBranches:
    """Tests for empty data branches."""

    def test_empty_modelspace(self):
        """Lines 136, 148, 172, 206: Empty modelspace returns zeros."""
        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None
        assert len(result) == 48
        # Should have zeros for empty features
        assert result[13] == 0  # log1p(0)/10 = 0

    def test_no_circles_no_arcs(self):
        """Lines 148, 151: No circles or arcs."""
        mock_line = _make_mock_entity(
            "LINE",
            start=_make_point(0, 0),
            end=_make_point(100, 0),
            layer="0",
        )

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_line]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None
        # Circle features should be 0
        assert result[16] == 0  # mean circle radius
        assert result[17] == 0  # std circle radius
        # Arc features should be 0
        assert result[19] == 0  # mean arc radius

    def test_no_polylines(self):
        """Line 210: No polylines."""
        mock_circle = _make_mock_entity(
            "CIRCLE",
            center=_make_point(50, 50),
            radius=25.0,
            layer="0",
        )

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter([mock_circle]))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None
        # Polyline features should be 0
        assert result[40] == 0  # mean vertex count
        assert result[41] == 0  # max vertex count


# --- Tests for exception handling ---


class TestExceptionHandling:
    """Tests for exception handling."""

    def test_ezdxf_read_failure(self):
        """Lines 232-234: ezdxf read failure returns None."""
        with patch("ezdxf.readfile", side_effect=Exception("File read error")):
            result = extract_features_v6("nonexistent.dxf")

        assert result is None

    def test_modelspace_failure(self):
        """Lines 232-234: modelspace() failure returns None."""
        mock_doc = MagicMock()
        mock_doc.modelspace.side_effect = Exception("Modelspace error")

        with patch("ezdxf.readfile", return_value=mock_doc):
            result = extract_features_v6("test.dxf")

        assert result is None


# --- Tests for complex drawings ---


class TestComplexDrawings:
    """Tests for complex drawings with multiple entity types."""

    def test_mixed_entities(self):
        """Test with multiple entity types."""
        entities = [
            _make_mock_entity("LINE", start=_make_point(0, 0), end=_make_point(100, 0), layer="0"),
            _make_mock_entity("LINE", start=_make_point(100, 0), end=_make_point(100, 50), layer="0"),
            _make_mock_entity("CIRCLE", center=_make_point(50, 25), radius=10.0, layer="CIRCLES"),
            _make_mock_entity("ARC", center=_make_point(75, 25), radius=5.0, start_angle=0, end_angle=90, layer="0"),
            _make_mock_entity("TEXT", insert=_make_point(10, 40), layer="TEXT"),
            _make_mock_entity("DIMENSION", layer="DIM"),
        ]

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter(entities))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None
        assert len(result) == 48
        # Should have non-zero values for various features
        assert result[0] > 0  # LINE ratio
        assert result[1] > 0  # CIRCLE ratio
        assert result[2] > 0  # ARC ratio

    def test_multiple_circles_for_std(self):
        """Test with multiple circles to cover std calculation."""
        entities = [
            _make_mock_entity("CIRCLE", center=_make_point(10, 10), radius=5.0, layer="0"),
            _make_mock_entity("CIRCLE", center=_make_point(30, 30), radius=10.0, layer="0"),
            _make_mock_entity("CIRCLE", center=_make_point(50, 50), radius=15.0, layer="0"),
        ]

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter(entities))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None
        # std of circle radii should be non-zero
        assert result[17] > 0

    def test_multiple_lines_for_std(self):
        """Test with multiple lines to cover std calculation."""
        entities = [
            _make_mock_entity("LINE", start=_make_point(0, 0), end=_make_point(10, 0), layer="0"),
            _make_mock_entity("LINE", start=_make_point(0, 10), end=_make_point(50, 10), layer="0"),
            _make_mock_entity("LINE", start=_make_point(0, 20), end=_make_point(100, 20), layer="0"),
        ]

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter(entities))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None
        # std of line lengths should be non-zero
        assert result[23] > 0


# --- Tests for layer features ---


class TestLayerFeatures:
    """Tests for layer-based features."""

    def test_special_layer_names(self):
        """Lines 178-182: Special layer name detection."""
        entities = [
            _make_mock_entity("LINE", start=_make_point(0, 0), end=_make_point(10, 0), layer="DIM_LAYER"),
            _make_mock_entity("TEXT", insert=_make_point(5, 5), layer="TEXT_ANNOTATIONS"),
            _make_mock_entity("LINE", start=_make_point(0, 5), end=_make_point(10, 5), layer="CENTER_LINE"),
            _make_mock_entity("LINE", start=_make_point(0, 10), end=_make_point(10, 10), layer="HIDDEN"),
            _make_mock_entity("LINE", start=_make_point(0, 15), end=_make_point(10, 15), layer="SECTION_CUT"),
        ]

        mock_msp = MagicMock()
        mock_msp.__iter__ = MagicMock(return_value=iter(entities))

        mock_doc = MagicMock()
        mock_doc.modelspace.return_value = mock_msp

        result = extract_features_v6("dummy.dxf", ezdxf_doc=mock_doc)
        assert result is not None
        # Layer indicators should be 1.0
        assert result[27] == 1.0  # has "dim" layer
        assert result[28] == 1.0  # has "text" layer
        assert result[29] == 1.0  # has "center" layer
        assert result[30] == 1.0  # has "hidden" layer
        assert result[31] == 1.0  # has "section" layer
