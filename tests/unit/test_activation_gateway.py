"""Phase-A C3+C4 activation-gateway + baseline-manifest discriminators.

Runnable in isolation (the repo-global conftest imports FastAPI routers that
fail to collect under the local Python)::

    cd /private/tmp/cadml-c2c6
    PYTHONPATH=. python3.11 -m pytest --noconftest tests/unit/test_activation_gateway.py -q

Proves the ratified fail-closed / fail-loud posture:
  * default NO-PIN (unconfigured, and configured-empty) → degrade to ``None``,
  * a real pinned file → SUCCESS (exact bytes), not merely a refusal,
  * tampered bytes → digest-mismatch degrade (``None``, never the bytes),
  * malformed manifest under a configured store → LOUD ``ValueError``,
  * unconfigured store + present manifest → degrade (``None``), never raises,
  * degraded logs are path-safe (reason emitted, no filesystem path leaked).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List

import pytest

from src.core.model_activation import activation_gateway as gw
from src.core.model_activation.baseline_manifest import load_baseline_pins


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_LID = "family.classifier"
_AID = "weights"
_RELPATH = "model.bin"


@pytest.fixture(autouse=True)
def _clean_gateway(monkeypatch: pytest.MonkeyPatch):
    """Isolate every test: no inherited env, fresh gateway before and after."""
    monkeypatch.delenv(gw.ENV_STORE_ROOT, raising=False)
    monkeypatch.delenv(gw.ENV_FREEZE_PARENT, raising=False)
    monkeypatch.delenv("MODEL_ACTIVATION_BASELINE_MANIFEST", raising=False)
    gw.reset_gateway_for_tests()
    yield
    gw.reset_gateway_for_tests()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(path: Path, entries: List[dict]) -> Path:
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _single_file_entry(digest: str, relpath: str = _RELPATH) -> dict:
    return {
        "logical_activation_id": _LID,
        "artifact_id": _AID,
        "kind": "single_file",
        "digest": digest,
        "store_relpath": relpath,
    }


def _configure_pinned_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    data: bytes,
) -> Path:
    """Create store_root with a pinned file + manifest; wire env; reset gateway.

    Returns the store_root path (its absolute string is the sensitive value the
    log-safety test must never see leaked).
    """
    store_root = tmp_path / "store"
    store_root.mkdir()
    (store_root / _RELPATH).write_bytes(data)

    manifest = _write_manifest(
        tmp_path / "baseline.json", [_single_file_entry(_sha256_hex(data))]
    )
    monkeypatch.setenv(gw.ENV_STORE_ROOT, str(store_root))
    monkeypatch.setenv("MODEL_ACTIVATION_BASELINE_MANIFEST", str(manifest))
    gw.reset_gateway_for_tests()
    return store_root


# ---------------------------------------------------------------------------
# Default NO-PIN posture (degrade, never raise)
# ---------------------------------------------------------------------------

def test_default_no_pin_unconfigured_degrades_to_none() -> None:
    """No store root, no manifest → activate_file degrades to None, no raise."""
    assert gw.activate_file(_LID, _AID) is None


def test_configured_store_empty_manifest_degrades_via_pin_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A CONFIGURED store with no manifest has ZERO pins → PIN_ABSENT → None.

    This is the ratified NO-PIN production posture on a real store (not merely
    the store-less unconfigured shortcut).
    """
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv(gw.ENV_STORE_ROOT, str(store_root))
    # No MODEL_ACTIVATION_BASELINE_MANIFEST set → zero pins.
    gw.reset_gateway_for_tests()
    assert gw.activate_file(_LID, _AID) is None


# ---------------------------------------------------------------------------
# SUCCESS path (exact bytes) — proves activation works, not just refuses
# ---------------------------------------------------------------------------

def test_pinned_file_activates_exact_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = b"phase-a c4 pinned model bytes \x00\x01\x02payload"
    _configure_pinned_file(monkeypatch, tmp_path, data)
    result = gw.activate_file(_LID, _AID)
    assert result == data


# ---------------------------------------------------------------------------
# Digest mismatch (tamper after pinning) → degrade, never the bytes
# ---------------------------------------------------------------------------

def test_digest_mismatch_degrades_not_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = b"the authentic pinned artifact"
    store_root = _configure_pinned_file(monkeypatch, tmp_path, original)
    # Tamper the file on disk AFTER it was pinned (digest still targets original).
    tampered = b"malicious swapped-in bytes!!!"
    assert tampered != original
    (store_root / _RELPATH).write_bytes(tampered)

    result = gw.activate_file(_LID, _AID)
    assert result is None  # degraded; never returns the tampered bytes
    assert result != tampered


# ---------------------------------------------------------------------------
# Malformed manifest under a configured store → LOUD ValueError
# ---------------------------------------------------------------------------

def test_load_baseline_pins_bad_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json,,,", encoding="utf-8")
    with pytest.raises(ValueError):
        load_baseline_pins(source=str(bad))


def test_load_baseline_pins_bad_digest_raises(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "m.json", [_single_file_entry("not-a-valid-64-hex-digest")]
    )
    with pytest.raises(ValueError):
        load_baseline_pins(source=str(manifest))


def test_load_baseline_pins_duplicate_key_raises(tmp_path: Path) -> None:
    digest = _sha256_hex(b"x")
    manifest = _write_manifest(
        tmp_path / "m.json",
        [_single_file_entry(digest), _single_file_entry(digest, relpath="other.bin")],
    )
    with pytest.raises(ValueError):
        load_baseline_pins(source=str(manifest))


def test_load_baseline_pins_not_a_list_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"logical_activation_id": _LID}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_baseline_pins(source=str(manifest))


def test_load_baseline_pins_traversal_relpath_raises(tmp_path: Path) -> None:
    """Attacker-influenceable traversal store_relpath fails loud at load time."""
    manifest = _write_manifest(
        tmp_path / "m.json",
        [_single_file_entry(_sha256_hex(b"x"), relpath="../escape.bin")],
    )
    with pytest.raises(ValueError):
        load_baseline_pins(source=str(manifest))


def test_bootstrap_malformed_manifest_fails_loud_through_activate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A configured store + malformed manifest must raise THROUGH activate_file.

    The bootstrap ValueError is NOT swallowed into a silent None degrade.
    """
    store_root = tmp_path / "store"
    store_root.mkdir()
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text("[ this is not json", encoding="utf-8")
    monkeypatch.setenv(gw.ENV_STORE_ROOT, str(store_root))
    monkeypatch.setenv("MODEL_ACTIVATION_BASELINE_MANIFEST", str(bad_manifest))
    gw.reset_gateway_for_tests()
    with pytest.raises(ValueError):
        gw.activate_file(_LID, _AID)


# ---------------------------------------------------------------------------
# Absent / empty manifest → silent empty (NOT loud) — default posture
# ---------------------------------------------------------------------------

def test_load_baseline_pins_unset_returns_empty() -> None:
    assert load_baseline_pins() == ()


def test_load_baseline_pins_absent_file_returns_empty(tmp_path: Path) -> None:
    assert load_baseline_pins(source=str(tmp_path / "does-not-exist.json")) == ()


# ---------------------------------------------------------------------------
# Unconfigured store + present manifest → degrade (None), never raises
# ---------------------------------------------------------------------------

def test_unconfigured_store_with_manifest_degrades(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _write_manifest(
        tmp_path / "baseline.json", [_single_file_entry(_sha256_hex(b"data"))]
    )
    # Manifest present and VALID, but NO store root → unconfigured → degrade.
    monkeypatch.delenv(gw.ENV_STORE_ROOT, raising=False)
    monkeypatch.setenv("MODEL_ACTIVATION_BASELINE_MANIFEST", str(manifest))
    gw.reset_gateway_for_tests()
    assert gw.activate_file(_LID, _AID) is None  # degraded, did not raise


# ---------------------------------------------------------------------------
# Log safety — reason emitted (non-vacuous) AND no filesystem path leaked
# ---------------------------------------------------------------------------

def test_degraded_log_is_path_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    original = b"authentic bytes for log-safety check"
    store_root = _configure_pinned_file(monkeypatch, tmp_path, original)
    (store_root / _RELPATH).write_bytes(b"tampered bytes for log-safety check!")

    with caplog.at_level(logging.WARNING, logger="src.core.model_activation.activation_gateway"):
        assert gw.activate_file(_LID, _AID) is None

    # Non-vacuous: a degraded line WAS emitted with the path-safe reason value.
    assert "digest_mismatch" in caplog.text
    assert _LID in caplog.text
    # And it leaked NO filesystem path: neither the absolute store root nor the
    # store_relpath appears in the captured logs.
    assert str(store_root) not in caplog.text
    assert str(tmp_path) not in caplog.text
    assert os.sep + _RELPATH not in caplog.text


def test_unconfigured_degraded_log_reason_sentinel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="src.core.model_activation.activation_gateway"):
        assert gw.activate_file(_LID, _AID) is None
    assert "store_unconfigured" in caplog.text
    assert _LID in caplog.text


# ---------------------------------------------------------------------------
# Bundle degrade contract (refusal → None) without a full freeze
# ---------------------------------------------------------------------------

def test_activate_bundle_pin_absent_degrades(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bundle activation with no matching pin degrades to None, not raise."""
    data = b"single-file-only store"
    _configure_pinned_file(monkeypatch, tmp_path, data)
    # Ask for a BUNDLE id that isn't pinned → PIN_ABSENT → None.
    assert gw.activate_bundle("no.such.bundle", "artifact") is None
