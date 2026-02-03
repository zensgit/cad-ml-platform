"""Coverage tests for ISO 286 hole deviation overrides."""
import json
from pathlib import Path


def test_iso286_hole_symbol_coverage() -> None:
    path = Path("data/knowledge/iso286_hole_deviations.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    deviations = payload.get("deviations", {})
    expected = {
        "A",
        "B",
        "C",
        "CD",
        "D",
        "E",
        "EF",
        "F",
        "FG",
        "G",
        "H",
        "JS",
        "J",
        "K",
        "M",
        "N",
        "P",
        "R",
        "S",
        "T",
        "U",
        "V",
        "X",
        "Y",
        "Z",
        "ZA",
        "ZB",
        "ZC",
    }
    assert expected.issubset(set(deviations.keys()))


def test_iso286_hole_symbol_min_rows() -> None:
    path = Path("data/knowledge/iso286_hole_deviations.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    deviations = payload.get("deviations", {})
    min_rows = 10
    relaxed_symbols = {"CD", "EF", "FG"}
    short = {
        symbol: len(rows)
        for symbol, rows in deviations.items()
        if symbol not in relaxed_symbols and len(rows) < min_rows
    }
    assert not short, f"Symbols with <{min_rows} rows: {short}"
    for symbol in relaxed_symbols:
        if symbol in deviations:
            assert len(deviations[symbol]) >= 3
