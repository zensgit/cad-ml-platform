from __future__ import annotations

import io

import pytest


def test_guess_dxf_encoding_prefers_utf8_for_r2007_plus() -> None:
    pytest.importorskip("ezdxf")
    from src.utils.dxf_io import guess_dxf_encoding

    payload = (
        "  0\n"
        "SECTION\n"
        "  2\n"
        "HEADER\n"
        "  9\n"
        "$ACADVER\n"
        "  1\n"
        "AC1032\n"
        "  9\n"
        "$DWGCODEPAGE\n"
        "  3\n"
        "ANSI_936\n"
        "  0\n"
        "ENDSEC\n"
        "  0\n"
        "EOF\n"
    ).encode("latin1")
    assert guess_dxf_encoding(payload) == "utf-8"


def test_guess_dxf_encoding_uses_codepage_for_pre_r2007() -> None:
    pytest.importorskip("ezdxf")
    from src.utils.dxf_io import guess_dxf_encoding

    payload = (
        "  0\n"
        "SECTION\n"
        "  2\n"
        "HEADER\n"
        "  9\n"
        "$ACADVER\n"
        "  1\n"
        "AC1018\n"
        "  9\n"
        "$DWGCODEPAGE\n"
        "  3\n"
        "ANSI_936\n"
        "  0\n"
        "ENDSEC\n"
        "  0\n"
        "EOF\n"
    ).encode("latin1")
    assert guess_dxf_encoding(payload) == "gbk"


def test_read_dxf_document_from_bytes_roundtrip() -> None:
    ezdxf = pytest.importorskip("ezdxf")
    from src.utils.dxf_io import read_dxf_document_from_bytes

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 0))
    msp.add_line((0, 0), (0, 100))
    msp.add_text("名称: 人孔", dxfattribs={"height": 2, "insert": (80, 10)})

    buf = io.StringIO()
    doc.write(buf)
    data = buf.getvalue().encode("utf-8")

    parsed = read_dxf_document_from_bytes(data)
    texts = [e.dxf.text for e in parsed.modelspace() if e.dxftype() == "TEXT"]
    assert any("人孔" in text for text in texts)


def test_guess_dxf_encoding_empty_data() -> None:
    """Test guess_dxf_encoding returns utf-8 for empty data."""
    pytest.importorskip("ezdxf")
    from src.utils.dxf_io import guess_dxf_encoding

    assert guess_dxf_encoding(b"") == "utf-8"


def test_guess_dxf_encoding_no_acadver() -> None:
    """Test guess_dxf_encoding returns utf-8 when no $ACADVER found."""
    pytest.importorskip("ezdxf")
    from src.utils.dxf_io import guess_dxf_encoding

    payload = (
        "  0\n"
        "SECTION\n"
        "  2\n"
        "HEADER\n"
        "  0\n"
        "ENDSEC\n"
    ).encode("latin1")
    assert guess_dxf_encoding(payload) == "utf-8"


def test_guess_dxf_encoding_acadver_at_end_no_value() -> None:
    """Test _find_header_value handles $ACADVER at end of lines (no value idx)."""
    pytest.importorskip("ezdxf")
    from src.utils.dxf_io import guess_dxf_encoding

    # $ACADVER is at the very end with no following lines
    payload = (
        "  9\n"
        "$ACADVER\n"
        "  1\n"
    ).encode("latin1")
    # value_idx = idx+2 would be out of bounds
    assert guess_dxf_encoding(payload) == "utf-8"


def test_guess_dxf_encoding_codepage_exception() -> None:
    """Test guess_dxf_encoding handles codepage conversion exception."""
    pytest.importorskip("ezdxf")
    from unittest.mock import patch

    from src.utils.dxf_io import guess_dxf_encoding

    payload = (
        "  0\n"
        "SECTION\n"
        "  2\n"
        "HEADER\n"
        "  9\n"
        "$ACADVER\n"
        "  1\n"
        "AC1018\n"
        "  9\n"
        "$DWGCODEPAGE\n"
        "  3\n"
        "INVALID_CODEPAGE\n"
        "  0\n"
        "ENDSEC\n"
    ).encode("latin1")

    # Mock ezdxf.tools.codepage to raise exception
    with patch("ezdxf.tools.codepage.toencoding") as mock_toencoding:
        mock_toencoding.side_effect = Exception("Unknown codepage")
        result = guess_dxf_encoding(payload)

    assert result == "utf-8"


def test_guess_dxf_encoding_codepage_returns_none() -> None:
    """Test guess_dxf_encoding handles codepage returning None."""
    pytest.importorskip("ezdxf")
    from unittest.mock import patch

    from src.utils.dxf_io import guess_dxf_encoding

    payload = (
        "  0\n"
        "SECTION\n"
        "  2\n"
        "HEADER\n"
        "  9\n"
        "$ACADVER\n"
        "  1\n"
        "AC1018\n"
        "  9\n"
        "$DWGCODEPAGE\n"
        "  3\n"
        "UNKNOWN\n"
        "  0\n"
        "ENDSEC\n"
    ).encode("latin1")

    with patch("ezdxf.tools.codepage.toencoding", return_value=None):
        result = guess_dxf_encoding(payload)

    assert result == "utf-8"


def test_read_dxf_document_from_bytes_empty_raises() -> None:
    """Test read_dxf_document_from_bytes raises on empty data."""
    pytest.importorskip("ezdxf")
    from src.utils.dxf_io import read_dxf_document_from_bytes

    with pytest.raises(ValueError, match="empty"):
        read_dxf_document_from_bytes(b"")


def test_read_dxf_entities_from_bytes() -> None:
    """Test read_dxf_entities_from_bytes returns list of entities."""
    ezdxf = pytest.importorskip("ezdxf")
    from src.utils.dxf_io import read_dxf_entities_from_bytes

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 0))
    msp.add_line((0, 0), (0, 100))

    buf = io.StringIO()
    doc.write(buf)
    data = buf.getvalue().encode("utf-8")

    entities = read_dxf_entities_from_bytes(data)
    assert isinstance(entities, list)
    assert len(entities) >= 2


def test_strip_dxf_text_entities_removes_modelspace_text() -> None:
    ezdxf = pytest.importorskip("ezdxf")
    from src.utils.dxf_io import read_dxf_document_from_bytes, strip_dxf_text_entities_from_bytes

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 0))
    msp.add_text("名称: 人孔", dxfattribs={"height": 2, "insert": (80, 10)})

    buf = io.StringIO()
    doc.write(buf)
    payload = buf.getvalue().encode("utf-8")

    stripped = strip_dxf_text_entities_from_bytes(payload)
    parsed = read_dxf_document_from_bytes(stripped)

    types = [e.dxftype() for e in parsed.modelspace()]
    assert "TEXT" not in types
    assert "LINE" in types


def test_strip_dxf_text_entities_removes_block_text() -> None:
    ezdxf = pytest.importorskip("ezdxf")
    from src.utils.dxf_io import read_dxf_document_from_bytes, strip_dxf_text_entities_from_bytes

    doc = ezdxf.new(setup=True)
    block = doc.blocks.new(name="TITLEBLOCK")
    block.add_text("人孔", dxfattribs={"height": 2, "insert": (0, 0)})

    msp = doc.modelspace()
    msp.add_blockref("TITLEBLOCK", insert=(0, 0))

    buf = io.StringIO()
    doc.write(buf)
    payload = buf.getvalue().encode("utf-8")

    stripped = strip_dxf_text_entities_from_bytes(payload, strip_blocks=True)
    parsed = read_dxf_document_from_bytes(stripped)

    parsed_block = parsed.blocks.get("TITLEBLOCK")
    block_types = [e.dxftype() for e in parsed_block]
    assert "TEXT" not in block_types


def test_find_header_value_empty_value() -> None:
    """Test _find_header_value returns None for empty value."""
    pytest.importorskip("ezdxf")
    from src.utils.dxf_io import _find_header_value

    lines = [
        "  9",
        "$ACADVER",
        "  1",
        "",  # Empty value
    ]
    result = _find_header_value(lines, "$ACADVER")
    assert result is None


def test_find_header_value_not_found() -> None:
    """Test _find_header_value returns None when key not found."""
    pytest.importorskip("ezdxf")
    from src.utils.dxf_io import _find_header_value

    lines = ["  0", "SECTION", "  2", "HEADER"]
    result = _find_header_value(lines, "$ACADVER")
    assert result is None
