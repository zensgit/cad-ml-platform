from __future__ import annotations


def test_parse_seeds_comma_delimited() -> None:
    from scripts.sweep_graph2d_profile_seeds import _parse_seeds

    assert _parse_seeds("7,21,42") == [7, 21, 42]


def test_parse_seeds_strips_whitespace() -> None:
    from scripts.sweep_graph2d_profile_seeds import _parse_seeds

    assert _parse_seeds(" 7,  13 ,42 ") == [7, 13, 42]
