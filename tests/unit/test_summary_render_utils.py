from __future__ import annotations

import pytest


def test_read_json_object_returns_dict(tmp_path) -> None:
    from scripts.ci.summary_render_utils import read_json_object

    path = tmp_path / "payload.json"
    path.write_text('{"ok": true}', encoding="utf-8")

    payload = read_json_object(path, "dispatch")

    assert payload == {"ok": True}


def test_read_json_object_rejects_invalid_json(tmp_path) -> None:
    from scripts.ci.summary_render_utils import read_json_object

    path = tmp_path / "invalid.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed to parse summary json"):
        read_json_object(path, "summary")


def test_read_json_object_rejects_non_object_payload(tmp_path) -> None:
    from scripts.ci.summary_render_utils import read_json_object

    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="validation json must be an object"):
        read_json_object(path, "validation")


def test_boolish_and_is_zeroish_handle_string_forms() -> None:
    from scripts.ci.summary_render_utils import boolish, is_zeroish

    assert boolish(True) is True
    assert boolish("YES") is True
    assert boolish("0") is False
    assert is_zeroish(0) is True
    assert is_zeroish("0") is True
    assert is_zeroish("00") is False


def test_top_nonempty_filters_and_limits() -> None:
    from scripts.ci.summary_render_utils import top_nonempty

    rows = top_nonempty(["", "  ", "alpha", None, "beta", "gamma", "delta"], limit=3)

    assert rows == ["alpha", "beta", "gamma"]
