"""Phase-A C2 discriminator: part/v6 and part/v16-v6pt wired through the
activation gateway.

Runnable in isolation (the repo-global conftest fails to collect under some
local Python setups here)::

    cd /private/tmp/cadml-c2c6
    PYTHONPATH=. python3.11 -m pytest --noconftest tests/unit/test_c2_part.py -q

NOTE: this sandbox's ``python3.11`` has no ``torch`` installed, so under it
every test here SKIPS via ``pytest.importorskip("torch")`` rather than
actually running. The real (non-skipped) run was done with the box's
``/usr/bin/python3`` (3.9.6), which does have torch 2.8.0 installed::

    cd /private/tmp/cadml-c2c6
    PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest tests/unit/test_c2_part.py -q

Covers TWO activations (this unit owns both files):

* ``part/v6`` (artifact "main") -> ``PartClassifier._load_model``
  (src/ml/part_classifier.py) gated live by
  ``src/core/analyzer.py:_get_ml_classifier``.
* ``part/v16-v6pt`` (TWO artifacts, "v6pt" and "v14ens", sharing one logical
  id per owner decision 1) -> ``PartClassifierV16._load_models``
  (src/ml/part_classifier.py).

Each activation is proven:
  * pin-absent (unconfigured store)     -> degrades (raises), no raw load.
  * fixture pin (real store + manifest) -> success, reconstructed from the
    EXACT pinned bytes (verified via a distinguishing bias value baked into
    the checkpoint's state dict).
  * digest-tamper (bytes on disk changed after the manifest pins the
    original digest) -> degrades, never uses the tampered bytes.

V16 additionally proves the "all-or-nothing" contract: either component
missing degrades the WHOLE family (never a partial v6-only V16), matching
the pre-wiring behavior where analyzer only attempted V16 construction when
BOTH files were present on disk.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.core.model_activation import activation_gateway as gw  # noqa: E402
from src.ml.part_classifier import (  # noqa: E402
    PartClassifier,
    PartClassifierV16,
    _FusionModelV14,
)

V6_LID = "part/v6"
V6_AID = "main"
V6_RELPATH = "part_v6_main.pt"

V16_LID = "part/v16-v6pt"
V6PT_AID = "v6pt"
V6PT_RELPATH = "part_v16_v6pt.pt"
V14ENS_AID = "v14ens"
V14ENS_RELPATH = "part_v16_v14ens.pt"


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


def _entry(lid: str, aid: str, relpath: str, digest: str) -> dict:
    return {
        "logical_activation_id": lid,
        "artifact_id": aid,
        "kind": "single_file",
        "digest": digest,
        "store_relpath": relpath,
    }


def _configure_pins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files: List[Tuple[str, str, str, bytes]],
) -> Path:
    """Write a store + manifest for a list of (lid, aid, relpath, data) pins."""
    store_root = tmp_path / "store"
    store_root.mkdir(exist_ok=True)
    entries = []
    for lid, aid, relpath, data in files:
        (store_root / relpath).write_bytes(data)
        entries.append(_entry(lid, aid, relpath, _sha256_hex(data)))

    manifest = tmp_path / "baseline.json"
    manifest.write_text(json.dumps(entries), encoding="utf-8")

    monkeypatch.setenv(gw.ENV_STORE_ROOT, str(store_root))
    monkeypatch.setenv("MODEL_ACTIVATION_BASELINE_MANIFEST", str(manifest))
    gw.reset_gateway_for_tests()
    return store_root


# ---------------------------------------------------------------------------
# Fixture checkpoint builders — real architectures, real torch.save bytes.
# ---------------------------------------------------------------------------

class _V2Replica(nn.Module):
    """Exact op-sequence copy of PartClassifier._build_v2_model's nested class.

    Same layer order -> identical state_dict keys -> load_state_dict succeeds
    against the REAL nested class defined inside the source module.
    """

    def __init__(self, in_dim, hid_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hid_dim, hid_dim // 2),
            nn.BatchNorm1d(hid_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class _V6Replica(nn.Module):
    """Exact op-sequence copy of PartClassifierV16._load_models's nested
    ImprovedClassifierV6, at the SAME hardcoded dims (48, 256, 5, 0.5) the
    source uses."""

    def __init__(self, in_dim=48, hid_dim=256, n_classes=5, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim // 2),
            nn.BatchNorm1d(hid_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hid_dim // 2, hid_dim // 4),
            nn.BatchNorm1d(hid_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hid_dim // 4, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _make_v6_checkpoint_bytes(bias_value: float) -> bytes:
    """A minimal, real PartClassifier (v2) checkpoint."""
    input_dim, hidden_dim, num_classes = 6, 8, 2
    model = _V2Replica(input_dim, hidden_dim, num_classes)
    with torch.no_grad():
        model.net[-1].bias.fill_(bias_value)

    checkpoint = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "version": "v2",
        "id_to_label": {"0": "cat_a", "1": "cat_b"},
        "model_state_dict": model.state_dict(),
    }
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    return buf.getvalue()


def _make_v16_v6pt_checkpoint_bytes(bias_value: float) -> bytes:
    """A minimal, real PartClassifierV16 "v6pt" component checkpoint."""
    model = _V6Replica(48, 256, 5, 0.5)
    with torch.no_grad():
        model.net[-1].bias.fill_(bias_value)
    checkpoint = {"model_state_dict": model.state_dict()}
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    return buf.getvalue()


def _make_v16_v14ens_checkpoint_bytes(bias_value: float) -> bytes:
    """A minimal, real PartClassifierV16 "v14ens" component checkpoint (1 fold)."""
    model = _FusionModelV14(48, 5)
    with torch.no_grad():
        model.classifier[-1].bias.fill_(bias_value)
    checkpoint = {"fold_states": [model.state_dict()]}
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# part/v6 (single artifact "main")
# ---------------------------------------------------------------------------

def test_v6_pin_absent_degrades_no_raw_load(tmp_path: Path):
    """Unconfigured store (no pin at all) -> family degrades, no raw load."""
    # A real-looking checkpoint sitting at model_path, to prove it is NEVER
    # read when the gateway is unconfigured (no raw-load-by-path fallback).
    bogus_path = tmp_path / "cad_classifier_v6.pt"
    bogus_path.write_bytes(_make_v6_checkpoint_bytes(bias_value=999.0))

    with pytest.raises(RuntimeError, match="part/v6"):
        PartClassifier(model_path=str(bogus_path))


def test_v6_fixture_pin_success_uses_exact_pinned_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A correctly pinned checkpoint -> success path loads from those exact bytes."""
    data = _make_v6_checkpoint_bytes(bias_value=42.0)
    _configure_pins(monkeypatch, tmp_path, [(V6_LID, V6_AID, V6_RELPATH, data)])

    # model_path points somewhere that does NOT contain these bytes, proving
    # the load came from the activation gateway, not the filesystem path.
    missing_path = tmp_path / "does_not_exist_on_disk.pt"
    clf = PartClassifier(model_path=str(missing_path))

    assert clf.model is not None
    assert clf.version == "v2"
    assert torch.allclose(clf.model.net[-1].bias.cpu(), torch.full((2,), 42.0))
    assert clf.id_to_label == {0: "cat_a", 1: "cat_b"}


def test_v6_digest_tamper_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Bytes on disk changed after the manifest pinned the original digest -> degrade."""
    original = _make_v6_checkpoint_bytes(bias_value=7.0)
    store_root = _configure_pins(
        monkeypatch, tmp_path, [(V6_LID, V6_AID, V6_RELPATH, original)]
    )

    tampered = _make_v6_checkpoint_bytes(bias_value=13.0)
    assert tampered != original
    (store_root / V6_RELPATH).write_bytes(tampered)

    with pytest.raises(RuntimeError, match="part/v6"):
        PartClassifier(model_path=str(tmp_path / "irrelevant.pt"))


# ---------------------------------------------------------------------------
# part/v16-v6pt (TWO artifacts: "v6pt" + "v14ens", all-or-nothing)
# ---------------------------------------------------------------------------

def test_v16_pin_absent_degrades_no_raw_load(tmp_path: Path):
    """Unconfigured store -> V16 unavailable (raises), never a raw load."""
    classifier = PartClassifierV16(model_dir=str(tmp_path), use_jit=False)

    with pytest.raises(RuntimeError, match="part/v16-v6pt"):
        classifier._load_models()

    assert classifier.loaded is False
    assert classifier.v6_model is None
    assert classifier.v14_models == []


def test_v16_fixture_pins_success_uses_exact_bytes_for_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Both pins present -> success, each component reconstructed from its
    OWN exact pinned bytes (proves the two pins are wired independently)."""
    v6pt_data = _make_v16_v6pt_checkpoint_bytes(bias_value=11.0)
    v14ens_data = _make_v16_v14ens_checkpoint_bytes(bias_value=22.0)
    _configure_pins(
        monkeypatch,
        tmp_path,
        [
            (V16_LID, V6PT_AID, V6PT_RELPATH, v6pt_data),
            (V16_LID, V14ENS_AID, V14ENS_RELPATH, v14ens_data),
        ],
    )

    classifier = PartClassifierV16(
        model_dir=str(tmp_path / "unused-model-dir"), use_jit=False
    )
    classifier._load_models()

    assert classifier.loaded is True
    assert classifier.v6_model is not None
    assert torch.allclose(classifier.v6_model.net[-1].bias.cpu(), torch.full((5,), 11.0))

    assert len(classifier.v14_models) == 1
    assert torch.allclose(
        classifier.v14_models[0].classifier[-1].bias.cpu(), torch.full((5,), 22.0)
    )


def test_v16_v14ens_missing_degrades_whole_family_all_or_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """v6pt pinned but v14ens absent -> the WHOLE V16 family degrades (raises),
    never a partial v6-only fallback inside V16 (owner decision 1: all-or-nothing)."""
    v6pt_data = _make_v16_v6pt_checkpoint_bytes(bias_value=5.0)
    _configure_pins(monkeypatch, tmp_path, [(V16_LID, V6PT_AID, V6PT_RELPATH, v6pt_data)])

    classifier = PartClassifierV16(model_dir=str(tmp_path), use_jit=False)

    with pytest.raises(RuntimeError, match="v14ens"):
        classifier._load_models()

    # loaded is only flipped True at the very end of _load_models, so a raise
    # partway through (after v6 succeeded but before v14 did) must leave it False.
    assert classifier.loaded is False


def test_v16_v6pt_digest_tamper_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """v6pt tampered after pinning -> degrades before v14ens is even attempted."""
    original = _make_v16_v6pt_checkpoint_bytes(bias_value=1.0)
    v14ens_data = _make_v16_v14ens_checkpoint_bytes(bias_value=2.0)
    store_root = _configure_pins(
        monkeypatch,
        tmp_path,
        [
            (V16_LID, V6PT_AID, V6PT_RELPATH, original),
            (V16_LID, V14ENS_AID, V14ENS_RELPATH, v14ens_data),
        ],
    )

    tampered = _make_v16_v6pt_checkpoint_bytes(bias_value=999.0)
    assert tampered != original
    (store_root / V6PT_RELPATH).write_bytes(tampered)

    classifier = PartClassifierV16(model_dir=str(tmp_path), use_jit=False)
    with pytest.raises(RuntimeError, match="v6pt"):
        classifier._load_models()

    assert classifier.v6_model is None
    assert classifier.v14_models == []
