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
* ``part/v16-v6pt`` (THREE artifacts, "v6pt", "v14ens", and the F5
  ensemble-config "v16config", all sharing one logical id — owner decision 1's
  two-pin shape extended to three and owner-ratified this review round) ->
  ``PartClassifierV16._load_models`` (src/ml/part_classifier.py).

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
# F5: the ensemble config is the THIRD single-file pin under the same logical id.
# store_relpath matches the baseline-manifest documented path (models/…json).
V16CONFIG_AID = "v16config"
V16CONFIG_RELPATH = "models/cad_classifier_v16_config.json"


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
        target = store_root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
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


def _make_v16_config_bytes(v6_weight: float, v14_weight: float) -> bytes:
    """A minimal, real V16 ensemble config JSON (the F5 third pinned artifact).

    Shape matches what PartClassifierV16._load_models reads:
    config['components']['v6']['weight'] and ['v14_ensemble']['weight'].
    """
    config = {
        "components": {
            "v6": {"weight": v6_weight},
            "v14_ensemble": {"weight": v14_weight},
        }
    }
    return json.dumps(config).encode("utf-8")


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
    # Distinguishing weights (NOT the __init__ defaults 0.3/0.7) so we can prove
    # the weights came from the pinned config, not the fallback defaults.
    config_data = _make_v16_config_bytes(v6_weight=0.25, v14_weight=0.75)
    _configure_pins(
        monkeypatch,
        tmp_path,
        [
            (V16_LID, V6PT_AID, V6PT_RELPATH, v6pt_data),
            (V16_LID, V14ENS_AID, V14ENS_RELPATH, v14ens_data),
            (V16_LID, V16CONFIG_AID, V16CONFIG_RELPATH, config_data),
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
    # F5: weights came from the pinned config, proving it was read via the gateway.
    assert classifier.v6_weight == 0.25
    assert classifier.v14_weight == 0.75


def test_v16_v14ens_missing_degrades_whole_family_all_or_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """v6pt pinned but v14ens absent -> the WHOLE V16 family degrades (raises),
    never a partial v6-only fallback inside V16 (owner decision 1: all-or-nothing)."""
    v6pt_data = _make_v16_v6pt_checkpoint_bytes(bias_value=5.0)
    config_data = _make_v16_config_bytes(v6_weight=0.3, v14_weight=0.7)
    # config + v6pt pinned, v14ens deliberately absent: we must progress PAST the
    # config and v6pt steps to prove the all-or-nothing failure is the v14ens gap.
    _configure_pins(
        monkeypatch,
        tmp_path,
        [
            (V16_LID, V16CONFIG_AID, V16CONFIG_RELPATH, config_data),
            (V16_LID, V6PT_AID, V6PT_RELPATH, v6pt_data),
        ],
    )

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
    config_data = _make_v16_config_bytes(v6_weight=0.3, v14_weight=0.7)
    store_root = _configure_pins(
        monkeypatch,
        tmp_path,
        [
            (V16_LID, V16CONFIG_AID, V16CONFIG_RELPATH, config_data),
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


# ---------------------------------------------------------------------------
# part/v16-v6pt "v16config" — the F5 THIRD single-file pin (ensemble weights).
#
# The config json sets self.v6_weight / self.v14_weight, which combine the V6
# and V14 predictions. Even with both checkpoints pinned, editing this json
# changes outputs -> it is an unpinned weight-bearing artifact unless pinned.
# These prove: absent config -> whole V16 degrades (no silent default-weight
# fallback); tampered config -> degrades; valid config -> weights loaded.
# ---------------------------------------------------------------------------

def test_v16_config_pin_absent_degrades_whole_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Both checkpoints pinned but the config pin absent -> the WHOLE V16 family
    degrades (raises), NEVER a silent proceed with the default __init__ weights.

    This is the F5 discriminator: without the fix, a missing config silently
    used self.v6_weight=0.3 / self.v14_weight=0.7 and loaded the models anyway;
    with the fix it is all-or-nothing, so the config gap alone degrades V16."""
    v6pt_data = _make_v16_v6pt_checkpoint_bytes(bias_value=3.0)
    v14ens_data = _make_v16_v14ens_checkpoint_bytes(bias_value=4.0)
    # Both checkpoints pinned; v16config deliberately NOT pinned.
    _configure_pins(
        monkeypatch,
        tmp_path,
        [
            (V16_LID, V6PT_AID, V6PT_RELPATH, v6pt_data),
            (V16_LID, V14ENS_AID, V14ENS_RELPATH, v14ens_data),
        ],
    )

    classifier = PartClassifierV16(model_dir=str(tmp_path), use_jit=False)
    with pytest.raises(RuntimeError, match="v16config"):
        classifier._load_models()

    # Config is read FIRST, so degrade happens before any checkpoint is loaded.
    assert classifier.loaded is False
    assert classifier.v6_model is None
    assert classifier.v14_models == []
    # Weights remain the untouched __init__ defaults — never silently applied.
    assert classifier.v6_weight == 0.3
    assert classifier.v14_weight == 0.7


def test_v16_config_digest_tamper_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Config bytes on disk changed after the manifest pinned the original
    digest -> the gateway refuses (digest mismatch) -> V16 degrades, never
    using the tampered (attacker-chosen) weights."""
    original_config = _make_v16_config_bytes(v6_weight=0.25, v14_weight=0.75)
    v6pt_data = _make_v16_v6pt_checkpoint_bytes(bias_value=1.0)
    v14ens_data = _make_v16_v14ens_checkpoint_bytes(bias_value=2.0)
    store_root = _configure_pins(
        monkeypatch,
        tmp_path,
        [
            (V16_LID, V16CONFIG_AID, V16CONFIG_RELPATH, original_config),
            (V16_LID, V6PT_AID, V6PT_RELPATH, v6pt_data),
            (V16_LID, V14ENS_AID, V14ENS_RELPATH, v14ens_data),
        ],
    )

    # Attacker swaps the config to skew the ensemble weights.
    tampered = _make_v16_config_bytes(v6_weight=0.99, v14_weight=0.01)
    assert tampered != original_config
    (store_root / V16CONFIG_RELPATH).write_bytes(tampered)

    classifier = PartClassifierV16(model_dir=str(tmp_path), use_jit=False)
    with pytest.raises(RuntimeError, match="v16config"):
        classifier._load_models()

    assert classifier.loaded is False
    assert classifier.v6_model is None
    # Tampered weights were NEVER applied — defaults remain untouched.
    assert classifier.v6_weight == 0.3
    assert classifier.v14_weight == 0.7


def test_v16_valid_config_pin_loads_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A correctly pinned config -> its weights are applied (read via the
    gateway from the exact pinned bytes), and the full V16 family activates."""
    config_data = _make_v16_config_bytes(v6_weight=0.4, v14_weight=0.6)
    v6pt_data = _make_v16_v6pt_checkpoint_bytes(bias_value=8.0)
    v14ens_data = _make_v16_v14ens_checkpoint_bytes(bias_value=9.0)
    _configure_pins(
        monkeypatch,
        tmp_path,
        [
            (V16_LID, V16CONFIG_AID, V16CONFIG_RELPATH, config_data),
            (V16_LID, V6PT_AID, V6PT_RELPATH, v6pt_data),
            (V16_LID, V14ENS_AID, V14ENS_RELPATH, v14ens_data),
        ],
    )

    classifier = PartClassifierV16(model_dir=str(tmp_path), use_jit=False)
    classifier._load_models()

    assert classifier.loaded is True
    assert classifier.v6_weight == 0.4
    assert classifier.v14_weight == 0.6
