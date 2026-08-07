"""Phase-A C6 golden matrix — EXECUTED, at the WIRING level.

This is the Phase-A §5 golden matrix proven as running evidence one layer ABOVE
the C1 store: it drives a real wired family (``PartClassifier`` / ``part/v6``)
through its own ``__init__`` -> ``_load_model`` -> ``activate_file`` path, so it
proves the *family* activates verified bytes and degrades on refusal — not just
that the store layer refuses. The C1-internal RED cases (symlink/escape,
same-fd TOCTOU, size/count/depth bounds, wrong-kind at the store API, digest
mismatch) already live in ``test_model_activation_c1_core.py`` and the C5
structural discriminator lives in ``test_activation_surface_enumerator.py`` —
this matrix REFERENCES those with executable pointers (see the two
``*_documented`` tests below) rather than duplicating them.

Matrix executed here (representative family = ``part/v6``, artifact ``main``):

    row              | mechanism                              | outcome
    -----------------+----------------------------------------+-----------------
    GREEN fixed-pin  | valid checkpoint pinned + digest-locked| loads verified
                     |                                        | bytes, produces
                     |                                        | REAL inference
    RED pin-absent   | store configured, no pin for part/v6   | degrade (raise)
    RED store-unconf | no MODEL_ACTIVATION_STORE_ROOT         | degrade (raise)
    RED digest-tamper| bytes swapped on disk after pinning    | degrade (raise)
    RED wrong-kind   | pinned as BUNDLE, asked as file        | degrade (raise)

Every RED places a REAL, loadable checkpoint at ``model_path`` and proves the
family still RAISES rather than reading it — i.e. the refusal came from the
gateway, never a raw path load (Phase-A decision #3).

Interpreter note (honest): torch is installed only under the box's
``/usr/bin/python3`` (3.9.6); the sandbox ``python3.11`` has NO torch, so under
it this whole module SKIPS via ``pytest.importorskip("torch")``. The real
(non-skipped) run is::

    cd /private/tmp/cadml-c2c6
    PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest tests/unit/test_c2_golden_matrix.py -q

The repo-global conftest fails to collect under the local Python, so
``--noconftest`` is used. Local is NOT CI — CI-on-Linux is the authority.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.core.model_activation import activation_gateway as gw  # noqa: E402
from src.ml.part_classifier import PartClassifier  # noqa: E402

_V6_LID = "part/v6"
_V6_AID = "main"
_V6_RELPATH = "part_v6_main.pt"

_HERE = Path(__file__).resolve().parent
_C1_SUITE = _HERE / "test_model_activation_c1_core.py"
_C5_SUITE = _HERE / "test_activation_surface_enumerator.py"


@pytest.fixture(autouse=True)
def _clean_gateway(monkeypatch: pytest.MonkeyPatch):
    """Isolate every test: no inherited env, fresh process-wide gateway."""
    monkeypatch.delenv(gw.ENV_STORE_ROOT, raising=False)
    monkeypatch.delenv(gw.ENV_FREEZE_PARENT, raising=False)
    monkeypatch.delenv("MODEL_ACTIVATION_BASELINE_MANIFEST", raising=False)
    gw.reset_gateway_for_tests()
    yield
    gw.reset_gateway_for_tests()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _V2Replica(nn.Module):
    """Exact op-sequence copy of ``PartClassifier._build_v2_model``'s nested
    class so the state_dict keys line up and ``load_state_dict`` succeeds
    against the REAL nested class defined inside the source module."""

    def __init__(self, in_dim: int, hid_dim: int, n_classes: int) -> None:
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

    def forward(self, x):  # noqa: ANN001, ANN201
        return self.net(x)


def _make_v6_checkpoint_bytes(bias_value: float) -> bytes:
    """A minimal, real ``PartClassifier`` (v2) checkpoint; ``bias_value`` is
    baked into the head bias so a test can prove WHICH bytes were loaded."""
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


def _make_v6_deterministic_checkpoint_bytes() -> bytes:
    """A real v2 checkpoint wired to predict class 1 ("cat_b") for ANY input.

    The final Linear weight is zeroed and its bias set to ``[0, 5]`` so the
    logits equal the bias regardless of the feature vector -> softmax argmax is
    class 1. This makes the GREEN inference output deterministic AND provably
    derived from THESE pinned bytes (change the bias -> change the prediction).
    """
    input_dim, hidden_dim, num_classes = 6, 8, 2
    model = _V2Replica(input_dim, hidden_dim, num_classes)
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias.copy_(torch.tensor([0.0, 5.0]))
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


def _configure_pin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    data: bytes,
    *,
    kind: str = "single_file",
    lid: str = _V6_LID,
    aid: str = _V6_AID,
    relpath: str = _V6_RELPATH,
) -> Path:
    """Write a store + manifest pinning ``data`` and wire the gateway env.

    ``kind`` defaults to ``single_file``; the wrong-kind RED passes ``bundle``.
    Returns the store root (the sensitive path that must never be raw-read).
    """
    store_root = tmp_path / "store"
    store_root.mkdir(exist_ok=True)
    (store_root / relpath).write_bytes(data)
    manifest = tmp_path / "baseline.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "logical_activation_id": lid,
                    "artifact_id": aid,
                    "kind": kind,
                    "digest": _sha256_hex(data),
                    "store_relpath": relpath,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(gw.ENV_STORE_ROOT, str(store_root))
    monkeypatch.setenv("MODEL_ACTIVATION_BASELINE_MANIFEST", str(manifest))
    gw.reset_gateway_for_tests()
    return store_root


def _decoy_checkpoint_at(tmp_path: Path, name: str = "local_decoy.pt") -> Path:
    """A REAL, loadable checkpoint on disk at model_path. Its only purpose is to
    prove the family never raw-loads it: if it were read, the constructor would
    SUCCEED instead of raising, so every RED that places it and still raises
    proves the refusal came from the gateway, not a corrupt/absent path."""
    p = tmp_path / name
    p.write_bytes(_make_v6_checkpoint_bytes(bias_value=999.0))
    return p


# ---------------------------------------------------------------------------
# GREEN — fixed pin SUCCESS, proven end-to-end THROUGH activate_file
# ---------------------------------------------------------------------------

def test_green_fixed_pin_loads_and_produces_real_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A correctly pinned checkpoint -> the family loads the verified bytes AND
    produces its real inference output (not merely a store-layer byte return)."""
    data = _make_v6_deterministic_checkpoint_bytes()
    _configure_pin(monkeypatch, tmp_path, data)

    # model_path points at a file that does NOT exist, proving the bytes came
    # from the gateway (activate_file), not the filesystem path.
    clf = PartClassifier(model_path=str(tmp_path / "does_not_exist_on_disk.pt"))

    # Loaded from the EXACT pinned bytes, through the real family __init__.
    assert clf.model is not None
    assert clf.version == "v2"
    assert clf.id_to_label == {0: "cat_a", 1: "cat_b"}

    # Produce the family's REAL output: the same forward -> softmax -> argmax
    # -> label math ``predict()`` runs, fed a synthetic feature vector to
    # isolate the ACTIVATION path from DXF/ezdxf I/O. Deterministic because the
    # pinned head predicts class 1 for any input.
    x = torch.zeros(1, clf.input_dim, dtype=torch.float32, device=clf.device)
    with torch.inference_mode():
        probs = torch.softmax(clf.model(x), dim=1)[0].float().cpu()

    assert abs(float(probs.sum().item()) - 1.0) < 1e-5  # a real distribution
    pred_id = int(torch.argmax(probs).item())
    assert clf.id_to_label[pred_id] == "cat_b"
    assert float(probs[pred_id].item()) > 0.9


# ---------------------------------------------------------------------------
# RED — each degrades (raise), never a raw path load
# ---------------------------------------------------------------------------

def test_red_pin_absent_degrades_never_raw_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Store configured but NO pin for part/v6 -> PIN_ABSENT -> degrade."""
    # Configure a store that pins a DIFFERENT activation, so part/v6 is absent.
    _configure_pin(
        monkeypatch,
        tmp_path,
        _make_v6_checkpoint_bytes(bias_value=1.0),
        lid="some.other/activation",
        aid="main",
    )
    assert gw.activate_file(_V6_LID, _V6_AID) is None  # PIN_ABSENT degrade signal

    decoy = _decoy_checkpoint_at(tmp_path)
    with pytest.raises(RuntimeError, match="part/v6"):
        PartClassifier(model_path=str(decoy))


def test_red_store_unconfigured_degrades_never_raw_loads(tmp_path: Path) -> None:
    """No MODEL_ACTIVATION_STORE_ROOT -> unconfigured -> degrade (default posture)."""
    assert gw.activate_file(_V6_LID, _V6_AID) is None  # store_unconfigured signal

    decoy = _decoy_checkpoint_at(tmp_path)
    with pytest.raises(RuntimeError, match="part/v6"):
        PartClassifier(model_path=str(decoy))


def test_red_digest_tamper_degrades_never_raw_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bytes swapped on disk AFTER the manifest pinned the original digest ->
    DIGEST_MISMATCH -> degrade; the tampered bytes are never returned."""
    original = _make_v6_checkpoint_bytes(bias_value=7.0)
    store_root = _configure_pin(monkeypatch, tmp_path, original)

    tampered = _make_v6_checkpoint_bytes(bias_value=13.0)
    assert tampered != original
    (store_root / _V6_RELPATH).write_bytes(tampered)

    assert gw.activate_file(_V6_LID, _V6_AID) is None  # digest_mismatch degrade signal

    decoy = _decoy_checkpoint_at(tmp_path)
    with pytest.raises(RuntimeError, match="part/v6"):
        PartClassifier(model_path=str(decoy))


def test_red_wrong_kind_bundle_asked_as_file_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """part/v6 pinned as a BUNDLE but activated as a SINGLE_FILE -> KIND_MISMATCH
    -> degrade (the store refuses before reading a byte)."""
    data = _make_v6_checkpoint_bytes(bias_value=3.0)
    _configure_pin(monkeypatch, tmp_path, data, kind="bundle")

    assert gw.activate_file(_V6_LID, _V6_AID) is None  # kind_mismatch degrade signal

    decoy = _decoy_checkpoint_at(tmp_path)
    with pytest.raises(RuntimeError, match="part/v6"):
        PartClassifier(model_path=str(decoy))


# ---------------------------------------------------------------------------
# Coverage pointers — DOCUMENT (do not re-execute) the C1 + C5 RED coverage.
# These fail if a referenced sibling test is renamed/removed, keeping the
# matrix's documented full-coverage claim honest.
# ---------------------------------------------------------------------------

def _defined_test_names(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }


def test_referenced_c1_reds_are_documented() -> None:
    """The C1-internal escape / TOCTOU / bounds / wrong-kind / digest REDs are
    owned by the C1 core suite; this matrix references them by name so the
    §5 coverage claim is anchored to real tests, not prose."""
    c1 = _defined_test_names(_C1_SUITE)
    referenced = {
        # escape / symlink / path-swap
        "test_intermediate_symlink_refused",
        "test_leaf_symlink_refused_even_inside_store",
        "test_parent_swap_to_symlink_mid_walk_refused",
        "test_bundle_symlink_member_red_pass1_no_freeze_created",
        # same-fd TOCTOU
        "test_same_fd_toctou_returns_hashed_bytes_not_reread",
        "test_growing_file_refused_on_same_fd",
        "test_inode_swap_after_open_does_not_affect_same_fd_read",
        "test_freeze_inplace_mutation_red",
        "test_freeze_path_redirect_red",
        # bounds / bombs
        "test_oversized_single_file_red",
        "test_bundle_directory_bomb_red",
        "test_bundle_dirent_bomb_red",
        "test_bundle_depth_bomb_red",
        "test_bundle_relpath_bomb_red",
        "test_bundle_file_count_red",
        "test_bundle_per_file_bytes_red",
        # wrong-kind (both API directions) + digest
        "test_wrong_kind_single_file_api_on_bundle_pin",
        "test_wrong_kind_bundle_api_on_single_file_pin",
        "test_single_file_digest_mismatch_red",
        "test_bundle_digest_mismatch_red",
    }
    missing = referenced - c1
    assert not missing, f"referenced C1 RED tests missing from {_C1_SUITE.name}: {sorted(missing)}"


def test_referenced_c5_discriminators_are_documented() -> None:
    """The C5 enumerator discriminators (new un-annotated loader -> CI RED, the
    remove-the-wrapper structural RED under enforce, and the real-tree-passes-
    enforce control) are owned by the enumerator suite; referenced here."""
    c5 = _defined_test_names(_C5_SUITE)
    referenced = {
        "test_new_unclassified_load_site_reds",
        "test_remove_the_wrapper_is_observed_RED_under_enforce",
        "test_remove_the_wrapper_is_advisory_only_by_default",
        "test_real_tree_is_structurally_consistent_under_enforce",
    }
    missing = referenced - c5
    assert not missing, f"referenced C5 discriminators missing from {_C5_SUITE.name}: {sorted(missing)}"
