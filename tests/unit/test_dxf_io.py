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

