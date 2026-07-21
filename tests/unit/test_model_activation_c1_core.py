"""Phase-A C1 model-activation core — positive controls + observed-RED discriminators.

Scope: reusable core only. No family wiring, no production pins, no reload.
"""

from __future__ import annotations

import ast
import ctypes
import errno
import hashlib
import os
import resource
import stat
import threading
import time
import traceback
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from unittest import mock

import pytest

from src.core.model_activation import (
    ActivationRefusal,
    ArtifactKind,
    BoundPolicy,
    ControlledStore,
    PinRecord,
    RefusalReason,
    ResolverImpl,
    last_open_impl,
    openat2_available,
    sha256_hex,
    tree_digest_v1_from_file_bytes,
    validate_raw_pin,
)
from src.core.model_activation.digest import read_fd_bounded_once
from src.core.model_activation.fd_dir import list_dir_fd, scandir_dir_fd
from src.core.model_activation.resolver import open_pinned, open_store_root
from src.core.model_activation.store import FreezeResourceLease
from src.core.model_activation.types import TerminalKind
import src.core.model_activation.digest as digest_mod
import src.core.model_activation.fd_dir as fd_dir_mod
import src.core.model_activation.resolver as resolver_mod
import src.core.model_activation.store as store_mod
import src.core.model_activation.types as types_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _trusted_freeze_parent(path: Path) -> Path:
    """Create a service-private freeze parent: absolute, owned, mode 0700."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def _store(
    root: Path,
    pins: Iterable[PinRecord],
    *,
    bounds: Optional[BoundPolicy] = None,
    impl: Optional[ResolverImpl] = None,
    freeze_parent: Optional[Path] = None,
) -> ControlledStore:
    fp: Optional[str] = None
    if freeze_parent is not None:
        fp = str(_trusted_freeze_parent(freeze_parent))
    return ControlledStore(
        str(root),
        list(pins),
        bounds=bounds,
        freeze_parent=fp,
        resolver_impl=impl or ResolverImpl.COMPONENT,
    )


def _sf_pin(lid: str, aid: str, rel: str, data: bytes) -> PinRecord:
    return PinRecord(
        logical_activation_id=lid,
        artifact_id=aid,
        kind=ArtifactKind.SINGLE_FILE,
        digest=_hex(data),
        store_relpath=rel,
    )


def _bundle_pin(
    lid: str,
    aid: str,
    rel: str,
    files: List[Tuple[str, bytes]],
) -> PinRecord:
    return PinRecord(
        logical_activation_id=lid,
        artifact_id=aid,
        kind=ArtifactKind.BUNDLE,
        digest=tree_digest_v1_from_file_bytes(files),
        store_relpath=rel,
    )


def _loose_bounds(**kwargs: int) -> BoundPolicy:
    base = dict(
        max_single_file_bytes=1024 * 1024,
        max_bundle_file_count=10_000,
        max_bundle_per_file_bytes=1024 * 1024,
        max_bundle_aggregate_bytes=10 * 1024 * 1024,
        max_total_dirents=50_000,
        max_directories=10_000,
        max_depth=64,
        max_relpath_utf8_bytes=4096,
    )
    base.update(kwargs)
    return BoundPolicy(**base)


# ---------------------------------------------------------------------------
# Raw pin domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "a//b",
        "a/./b",
        "a/../b",
        "family/model//weights.pt",
        "family/./model.pt",
        "family/../secret.pt",
        "/etc/model.pt",
        "abs/trailing/",
        "",
        ".",
        "..",
        "a/b/",
        "/leading",
        "has\x00nul",
        "\x00",
        "//double",
    ],
)
def test_raw_pin_domain_rejects(raw: str) -> None:
    with pytest.raises(ActivationRefusal) as ei:
        validate_raw_pin(raw)
    assert ei.value.reason is RefusalReason.RAW_PIN_INVALID


@pytest.mark.parametrize(
    "raw",
    [
        "weights.pt",
        "family/model.pt",
        "a/b/c/d.pt",
        "ocr/deepseek/bundle_root",
    ],
)
def test_raw_pin_domain_accepts(raw: str) -> None:
    parts = validate_raw_pin(raw)
    assert parts == tuple(raw.split("/"))


def test_raw_pin_rejects_path_object_type() -> None:
    with pytest.raises(ActivationRefusal) as ei:
        validate_raw_pin(Path("a/b"))  # type: ignore[arg-type]
    assert ei.value.reason is RefusalReason.RAW_PIN_INVALID


# ---------------------------------------------------------------------------
# Single-file
# ---------------------------------------------------------------------------


def test_single_file_exact_hash_green(tmp_path: Path) -> None:
    root = tmp_path / "store"
    data = b"model-bytes-v1"
    rel = "family/model.pt"
    _write(root / rel, data)
    pin = _sf_pin("act.graph2d", "art.g2d.v1", rel, data)
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        out = store.assert_fixed_hash("act.graph2d", "art.g2d.v1")
        assert out == data
        assert store.load_pinned_file("act.graph2d", "art.g2d.v1") == data
    finally:
        store.close()


def test_single_file_digest_mismatch_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    data = b"model-bytes-v1"
    rel = "family/model.pt"
    _write(root / rel, data)
    pin = PinRecord(
        logical_activation_id="act.x",
        artifact_id="art.x",
        kind=ArtifactKind.SINGLE_FILE,
        digest=_hex(b"other"),
        store_relpath=rel,
    )
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.x", "art.x")
        assert ei.value.reason is RefusalReason.DIGEST_MISMATCH
    finally:
        store.close()


def test_pin_absent_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    store = _store(root, [])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("missing", "missing")
        assert ei.value.reason is RefusalReason.PIN_ABSENT
    finally:
        store.close()


def test_wrong_kind_single_file_api_on_bundle_pin(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    _write(bdir / "a.bin", b"a")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("a.bin", b"a")])
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.b", "art.b")
        assert ei.value.reason is RefusalReason.KIND_MISMATCH
    finally:
        store.close()


def test_wrong_kind_bundle_api_on_single_file_pin(tmp_path: Path) -> None:
    root = tmp_path / "store"
    data = b"x"
    rel = "m.pt"
    _write(root / rel, data)
    pin = _sf_pin("act.s", "art.s", rel, data)
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.s", "art.s")
        assert ei.value.reason is RefusalReason.KIND_MISMATCH
    finally:
        store.close()


def test_missing_artifact_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    pin = _sf_pin("act.m", "art.m", "gone.pt", b"x")
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.m", "art.m")
        assert ei.value.reason in (
            RefusalReason.ARTIFACT_MISSING,
            RefusalReason.CONTAINMENT,
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Raw domain via both resolvers
# ---------------------------------------------------------------------------


_RAW_INVALID = [
    "a//b",
    "a/./b",
    "a/../b",
    "has\x00nul",
    "/absolute.pt",
    "trail/",
]


@pytest.mark.parametrize("raw", _RAW_INVALID)
def test_component_resolver_rejects_raw_domain(tmp_path: Path, raw: str) -> None:
    root = tmp_path / "store"
    root.mkdir()
    fd, _ = open_store_root(str(root))
    try:
        with pytest.raises(ActivationRefusal) as ei:
            open_pinned(
                fd,
                raw,
                TerminalKind.REGULAR_FILE,
                impl=ResolverImpl.COMPONENT,
            )
        assert ei.value.reason is RefusalReason.RAW_PIN_INVALID
    finally:
        os.close(fd)


@pytest.mark.parametrize("raw", _RAW_INVALID)
def test_openat2_resolver_rejects_raw_domain_identically(
    tmp_path: Path, raw: str
) -> None:
    if not openat2_available():
        pytest.skip("openat2 syscall unavailable on this kernel/platform")
    root = tmp_path / "store"
    root.mkdir()
    fd, _ = open_store_root(str(root))
    try:
        with pytest.raises(ActivationRefusal) as ei:
            open_pinned(
                fd,
                raw,
                TerminalKind.REGULAR_FILE,
                impl=ResolverImpl.OPENAT2,
            )
        assert ei.value.reason is RefusalReason.RAW_PIN_INVALID
    finally:
        os.close(fd)


def test_resolver_impl_parity_raw_domain_matrix(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    fd, _ = open_store_root(str(root))
    try:
        for raw in _RAW_INVALID:
            reasons = []
            with pytest.raises(ActivationRefusal) as ei:
                open_pinned(
                    fd, raw, TerminalKind.REGULAR_FILE, impl=ResolverImpl.COMPONENT
                )
            reasons.append(ei.value.reason)
            if openat2_available():
                with pytest.raises(ActivationRefusal) as ei:
                    open_pinned(
                        fd,
                        raw,
                        TerminalKind.REGULAR_FILE,
                        impl=ResolverImpl.OPENAT2,
                    )
                reasons.append(ei.value.reason)
            assert all(r is RefusalReason.RAW_PIN_INVALID for r in reasons)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Containment / special files / TOCTOU
# ---------------------------------------------------------------------------


def test_intermediate_symlink_refused(tmp_path: Path) -> None:
    root = tmp_path / "store"
    outside = tmp_path / "outside"
    outside.mkdir()
    _write(outside / "evil.pt", b"evil")
    family = root / "family"
    family.parent.mkdir(parents=True, exist_ok=True)
    family.symlink_to(outside, target_is_directory=True)
    pin = _sf_pin("act.s", "art.s", "family/evil.pt", b"evil")
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.s", "art.s")
        assert ei.value.reason in (
            RefusalReason.SYMLINK_REJECTED,
            RefusalReason.NOT_DIRECTORY,
            RefusalReason.CONTAINMENT,
            RefusalReason.ARTIFACT_MISSING,
        )
    finally:
        store.close()


def test_leaf_symlink_refused_even_inside_store(tmp_path: Path) -> None:
    root = tmp_path / "store"
    real = root / "real.pt"
    _write(real, b"real-bytes")
    link = root / "link.pt"
    link.symlink_to(real)
    pin = _sf_pin("act.s", "art.s", "link.pt", b"real-bytes")
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.s", "art.s")
        assert ei.value.reason in (
            RefusalReason.SYMLINK_REJECTED,
            RefusalReason.SPECIAL_FILE,
            RefusalReason.NOT_REGULAR_FILE,
            RefusalReason.CONTAINMENT,
        )
    finally:
        store.close()


def test_parent_swap_to_symlink_mid_walk_refused(tmp_path: Path) -> None:
    root = tmp_path / "store"
    target = root / "family" / "leaf.pt"
    _write(target, b"ok")
    outside = tmp_path / "out"
    outside.mkdir()
    _write(outside / "leaf.pt", b"evil")

    fd, _ = open_store_root(str(root))
    try:
        leaf_fd = open_pinned(
            fd,
            "family/leaf.pt",
            TerminalKind.REGULAR_FILE,
            impl=ResolverImpl.COMPONENT,
        )
        os.close(leaf_fd)

        family = root / "family"
        family_real = root / "family_real"
        family.rename(family_real)
        family.symlink_to(outside, target_is_directory=True)

        with pytest.raises(ActivationRefusal) as ei:
            open_pinned(
                fd,
                "family/leaf.pt",
                TerminalKind.REGULAR_FILE,
                impl=ResolverImpl.COMPONENT,
            )
        assert ei.value.reason in (
            RefusalReason.SYMLINK_REJECTED,
            RefusalReason.NOT_DIRECTORY,
            RefusalReason.CONTAINMENT,
        )
    finally:
        os.close(fd)


def test_fifo_leaf_does_not_block(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    os.mkfifo(root / "pipe.pt")
    pin = _sf_pin("act.f", "art.f", "pipe.pt", b"x")
    store = _store(root, [pin])
    try:
        t0 = time.monotonic()
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.f", "art.f")
        assert time.monotonic() - t0 < 2.0
        assert ei.value.reason in (
            RefusalReason.SPECIAL_FILE,
            RefusalReason.NOT_REGULAR_FILE,
            RefusalReason.SYMLINK_REJECTED,
            RefusalReason.CONTAINMENT,
            RefusalReason.UNREADABLE,
        )
    finally:
        store.close()


def test_oversized_single_file_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    data = b"x" * 100
    rel = "big.pt"
    _write(root / rel, data)
    pin = _sf_pin("act.o", "art.o", rel, data)
    store = _store(root, [pin], bounds=BoundPolicy(max_single_file_bytes=50))
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.o", "art.o")
        assert ei.value.reason is RefusalReason.OVERSIZE
    finally:
        store.close()


def test_growing_file_refused_on_same_fd(tmp_path: Path) -> None:
    path = tmp_path / "grow.bin"
    path.write_bytes(b"abcd")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        st = os.fstat(fd)
        with open(path, "ab") as w:
            w.write(b"EFGH")
        with pytest.raises(ActivationRefusal) as ei:
            read_fd_bounded_once(fd, max_bytes=1024, expect_size=st.st_size)
        assert ei.value.reason is RefusalReason.GROWING_FILE
    finally:
        os.close(fd)


def test_same_fd_toctou_returns_hashed_bytes_not_reread(tmp_path: Path) -> None:
    root = tmp_path / "store"
    rel = "m.pt"
    original = b"original-model-bytes"
    _write(root / rel, original)
    pin = _sf_pin("act.t", "art.t", rel, original)
    store = _store(root, [pin])
    try:
        out = store.assert_fixed_hash("act.t", "art.t")
        assert out == original
        (root / rel).write_bytes(b"swapped-after-hash!!!!!!")
        assert out == original
        assert sha256_hex(out) == pin.digest
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.t", "art.t")
        assert ei.value.reason is RefusalReason.DIGEST_MISMATCH
    finally:
        store.close()


def test_inode_swap_after_open_does_not_affect_same_fd_read(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    rel = "leaf.pt"
    path = root / rel
    _write(path, b"inode-A")
    fd, _ = open_store_root(str(root))
    try:
        leaf = open_pinned(
            fd, rel, TerminalKind.REGULAR_FILE, impl=ResolverImpl.COMPONENT
        )
        try:
            path.unlink()
            path.write_bytes(b"inode-B-swapped")
            data = read_fd_bounded_once(leaf, 1024, expect_size=7)
            assert data == b"inode-A"
        finally:
            os.close(leaf)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Bundle: digest from freeze, DFS, preflight, sealed handle
# ---------------------------------------------------------------------------


def test_bundle_exact_digest_and_determinism(tmp_path: Path) -> None:
    root = tmp_path / "store"
    files = [
        ("a.txt", b"aaa"),
        ("sub/b.txt", b"bbb"),
        ("sub/c.txt", b""),
    ]
    for rel, data in files:
        _write(root / "bundle" / rel, data)
    pin = _bundle_pin("act.emb", "art.emb", "bundle", files)
    assert pin.digest == tree_digest_v1_from_file_bytes(list(reversed(files)))

    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        frozen = store.assert_bundle_digest("act.emb", "art.emb")
        try:
            assert frozen.digest == pin.digest
            assert frozen.file_count == 3
            assert frozen.read_member("a.txt") == b"aaa"
            assert frozen.read_member("sub/b.txt") == b"bbb"
            assert frozen.read_member("sub/c.txt") == b""
            # Source mutation GREEN for freeze (reads via held fd).
            (root / "bundle" / "a.txt").write_bytes(b"mutated")
            assert frozen.read_member("a.txt") == b"aaa"
        finally:
            frozen.cleanup()
    finally:
        store.close()


def test_bundle_digest_mismatch_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    freeze_parent = tmp_path / "freeze"
    _write(root / "bundle" / "a.txt", b"aaa")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="ab" * 32,
        store_relpath="bundle",
    )
    store = _store(root, [pin], freeze_parent=freeze_parent)
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.DIGEST_MISMATCH
        # Load-bearing: this path creates+seals a real freeze before the
        # mismatch, so a leftover here means the failure-path destroy regressed.
        assert list(freeze_parent.glob("cadml-freeze-*")) == []
    finally:
        store.close()


def test_bundle_digest_computed_from_freeze_not_source_buffer(
    tmp_path: Path,
) -> None:
    """If freeze write is corrupted, digest-from-freeze must RED (not accept pin)."""
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")

    real_write = os.write

    def evil_write(fd: int, data: bytes | memoryview) -> int:  # type: ignore[override]
        # Corrupt freeze payload while reporting the same length.
        if isinstance(data, memoryview):
            raw = data.tobytes()
        else:
            raw = bytes(data)
        if raw == b"good":
            return real_write(fd, b"evil")
        return real_write(fd, data)

    try:
        with mock.patch.object(store_mod.os, "write", side_effect=evil_write):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
            assert ei.value.reason is RefusalReason.DIGEST_MISMATCH
    finally:
        store.close()


def test_bundle_symlink_member_red_pass1_no_freeze_created(
    tmp_path: Path,
) -> None:
    """Pass-1 refusal happens before bind_root: the freeze root is NEVER made."""
    root = tmp_path / "store"
    freeze_parent = tmp_path / "freeze"
    freeze_parent.mkdir()
    _write(root / "bundle" / "good.txt", b"good")
    (root / "bundle" / "link").symlink_to("good.txt")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("good.txt", b"good")])
    store = _store(root, [pin], freeze_parent=freeze_parent)
    real_mkdir = os.mkdir
    mkdir_calls = {"n": 0}

    def spy_mkdir(*args: object, **kwargs: object) -> None:
        mkdir_calls["n"] += 1
        real_mkdir(*args, **kwargs)  # type: ignore[arg-type]

    try:
        with mock.patch.object(store_mod.os, "mkdir", side_effect=spy_mkdir):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.SYMLINK_REJECTED
        assert mkdir_calls["n"] == 0, "Pass-1 refusal must precede any freeze mkdir"
        leftovers = [
            p
            for p in freeze_parent.iterdir()
            if p.is_dir() and p.name.startswith("cadml-freeze-")
        ]
        assert leftovers == []
    finally:
        store.close()


def test_bundle_fifo_member_red_no_block(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    _write(bdir / "good.txt", b"good")
    os.mkfifo(bdir / "pipe")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("good.txt", b"good")])
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        t0 = time.monotonic()
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert time.monotonic() - t0 < 2.0
        assert ei.value.reason is RefusalReason.SPECIAL_FILE
    finally:
        store.close()


def test_preflight_metadata_cap_zero_destination_writes(tmp_path: Path) -> None:
    """Metadata-detectable max_directories=1 must refuse before any freeze write."""
    root = tmp_path / "store"
    bdir = root / "bundle"
    _write(bdir / "good.txt", b"goodxx")  # 6 bytes — spy length check
    (bdir / "d1").mkdir(parents=True)
    (bdir / "d2").mkdir(parents=True)
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    bounds = _loose_bounds(max_directories=1)
    store = _store(
        root, [pin], bounds=bounds, freeze_parent=tmp_path / "freeze"
    )
    writes: list[int] = []
    real_write = os.write

    def spy_write(fd: int, data: bytes | memoryview) -> int:  # type: ignore[override]
        n = real_write(fd, data)
        writes.append(n)
        return n

    try:
        with mock.patch.object(store_mod.os, "write", side_effect=spy_write):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
            assert ei.value.reason is RefusalReason.BUNDLE_DIRECTORY_COUNT
            assert writes == [], f"expected zero freeze writes, got {writes}"
    finally:
        store.close()


def test_preflight_file_count_zero_writes(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    for i in range(5):
        _write(bdir / f"f{i}.bin", b"x")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_bundle_file_count=2),
        freeze_parent=tmp_path / "freeze",
    )
    writes: list[int] = []
    real_write = os.write

    def spy_write(fd: int, data: bytes | memoryview) -> int:  # type: ignore[override]
        n = real_write(fd, data)
        writes.append(n)
        return n

    try:
        with mock.patch.object(store_mod.os, "write", side_effect=spy_write):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
            assert ei.value.reason is RefusalReason.BUNDLE_FILE_COUNT
            assert writes == []
    finally:
        store.close()


def test_preflight_aggregate_zero_writes(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    _write(bdir / "a.bin", b"a" * 30)
    _write(bdir / "b.bin", b"b" * 30)
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_bundle_aggregate_bytes=40),
        freeze_parent=tmp_path / "freeze",
    )
    writes: list[int] = []
    real_write = os.write

    def spy_write(fd: int, data: bytes | memoryview) -> int:  # type: ignore[override]
        n = real_write(fd, data)
        writes.append(n)
        return n

    try:
        with mock.patch.object(store_mod.os, "write", side_effect=spy_write):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
            assert ei.value.reason is RefusalReason.BUNDLE_AGGREGATE_BYTES
            assert writes == []
    finally:
        store.close()


def test_bundle_directory_bomb_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    for i in range(20):
        (bdir / f"d{i}").mkdir()
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_directories=5),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.BUNDLE_DIRECTORY_COUNT
    finally:
        store.close()


def test_bundle_dirent_bomb_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    for i in range(30):
        (bdir / f"f{i}.txt").write_bytes(b"")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_total_dirents=10),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason in (
            RefusalReason.BUNDLE_TOTAL_DIRENTS,
            RefusalReason.BUNDLE_FILE_COUNT,
        )
    finally:
        store.close()


def test_bundle_depth_bomb_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    cur = root / "bundle"
    for i in range(10):
        cur = cur / f"d{i}"
        cur.mkdir(parents=True, exist_ok=True)
    (cur / "leaf.txt").write_bytes(b"x")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_depth=3),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.BUNDLE_DEPTH
    finally:
        store.close()


def test_bundle_relpath_bomb_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    long_name = "n" * 80
    (bdir / long_name).write_bytes(b"x")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_relpath_utf8_bytes=40),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.BUNDLE_RELPATH_BYTES
    finally:
        store.close()


def test_bundle_file_count_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    for i in range(5):
        _write(bdir / f"f{i}.bin", b"x")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_bundle_file_count=2),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.BUNDLE_FILE_COUNT
    finally:
        store.close()


def test_bundle_per_file_bytes_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    _write(bdir / "big.bin", b"x" * 200)
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_bundle_per_file_bytes=100),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.BUNDLE_PER_FILE_BYTES
    finally:
        store.close()


def test_partial_freeze_cleanup_on_mid_walk_failure(tmp_path: Path) -> None:
    """Genuine mid-Pass-2 failure: freeze root exists, members partly written,
    then the copy fails — the partial freeze must be destroyed."""
    root = tmp_path / "store"
    freeze_parent = tmp_path / "freeze"
    freeze_parent.mkdir()
    bdir = root / "bundle"
    _write(bdir / "a.bin", b"first")
    _write(bdir / "b.bin", b"second")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(root, [pin], freeze_parent=freeze_parent)
    real_write = os.write
    real_mkdir = os.mkdir
    calls = {"write": 0, "mkdir": 0}

    def failing_write(fd: int, data: bytes | memoryview) -> int:
        calls["write"] += 1
        if calls["write"] >= 2:
            raise OSError(errno.EIO, "disk error mid-freeze")
        return real_write(fd, data)

    def spy_mkdir(*args: object, **kwargs: object) -> None:
        calls["mkdir"] += 1
        real_mkdir(*args, **kwargs)  # type: ignore[arg-type]

    try:
        with mock.patch.object(store_mod.os, "mkdir", side_effect=spy_mkdir), mock.patch.object(
            store_mod.os, "write", side_effect=failing_write
        ):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.FREEZE_FAILED
        # Non-vacuous: the freeze root was really created and partly populated.
        assert calls["mkdir"] >= 1
        assert calls["write"] >= 2
        assert list(freeze_parent.glob("cadml-freeze-*")) == []
    finally:
        store.close()


def test_bundle_source_mutation_after_freeze_green(tmp_path: Path) -> None:
    root = tmp_path / "store"
    files = [("a.txt", b"stable")]
    _write(root / "bundle" / "a.txt", b"stable")
    pin = _bundle_pin("act.b", "art.b", "bundle", files)
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        frozen = store.load_pinned_bundle("act.b", "art.b")
        try:
            (root / "bundle" / "a.txt").write_bytes(b"mutated-source")
            assert frozen.read_member("a.txt") == b"stable"
        finally:
            frozen.cleanup()
    finally:
        store.close()


def test_frozen_bundle_has_no_public_path(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        frozen = store.assert_bundle_digest("act.b", "art.b")
        try:
            assert not hasattr(frozen, "path") or "path" not in getattr(
                frozen, "__dict__", {}
            )
            with pytest.raises(AttributeError):
                _ = frozen.path  # type: ignore[attr-defined]
            assert frozen.read_member("m.bin") == b"good"
        finally:
            frozen.cleanup()
    finally:
        store.close()


def test_freeze_inplace_mutation_red(tmp_path: Path) -> None:
    """Sealed freeze: mode 0400 always; write refusal when DAC applies (non-root)."""
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    freeze_parent = tmp_path / "freeze"
    store = _store(root, [pin], freeze_parent=freeze_parent)
    try:
        frozen = store.assert_bundle_digest("act.b", "art.b")
        try:
            # Discover private backing via process (not public API).
            backing = frozen._backing_path  # noqa: SLF001 — test only
            target = Path(backing) / "m.bin"
            # Privilege-honest: seal always sets 0400; root bypasses DAC write refusal.
            assert (target.stat().st_mode & 0o777) == 0o400
            # Write-refusal portion: skip only when geteuid()==0 — root bypasses
            # 0400 mode bits (CAP_DAC_OVERRIDE / traditional root write privilege).
            if os.geteuid() != 0:
                with pytest.raises(OSError):
                    target.write_bytes(b"evil")
            # Held-fd read still sees original sealed bytes (runs under root too).
            assert frozen.read_member("m.bin") == b"good"
        finally:
            frozen.cleanup()
    finally:
        store.close()


def test_freeze_path_redirect_red(tmp_path: Path) -> None:
    """Renaming/repointing the backing path must not affect dir-fd reads."""
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    freeze_parent = tmp_path / "freeze"
    store = _store(root, [pin], freeze_parent=freeze_parent)
    try:
        frozen = store.assert_bundle_digest("act.b", "art.b")
        try:
            backing = Path(frozen._backing_path)  # noqa: SLF001
            moved = freeze_parent / "moved-away"
            backing.rename(moved)
            # Attacker recreates path with evil content.
            backing.mkdir()
            (backing / "m.bin").write_bytes(b"evil")
            # Dir-fd still points at original freeze inode.
            assert frozen.read_member("m.bin") == b"good"
        finally:
            # Restore for cleanup: put tree back if needed.
            try:
                if moved.exists() and not Path(frozen._backing_path).exists():  # noqa: SLF001
                    moved.rename(frozen._backing_path)  # noqa: SLF001
            except Exception:
                pass
            frozen.cleanup()
    finally:
        store.close()


def test_dfs_sibling_dirs_under_low_rlimit(tmp_path: Path) -> None:
    """200 sibling dirs must succeed with RLIMIT_NOFILE=128 (O(depth) fds)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = 128
    if hard < new_soft and hard != resource.RLIM_INFINITY:
        pytest.skip(f"hard RLIMIT_NOFILE={hard} < 128")
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    n_dirs = 200
    for i in range(n_dirs):
        (bdir / f"d{i:04d}").mkdir()
    # Empty dirs only → empty tree digest.
    pin = _bundle_pin("act.b", "art.b", "bundle", [])
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_directories=500, max_total_dirents=1000),
        freeze_parent=tmp_path / "freeze",
    )
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        try:
            frozen = store.assert_bundle_digest("act.b", "art.b")
            try:
                assert frozen.file_count == 0
                assert frozen.digest == pin.digest
            finally:
                frozen.cleanup()
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    finally:
        store.close()


def test_bundle_non_utf8_entry_red(tmp_path: Path) -> None:
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    bad_name = b"bad-\xff-name"
    bdir_fd = os.open(str(bdir), os.O_RDONLY)
    try:
        try:
            fd = os.open(
                bad_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=bdir_fd,
            )
            os.write(fd, b"x")
            os.close(fd)
        except OSError as exc:
            pytest.skip(f"filesystem rejects non-UTF-8 names: {exc}")
    finally:
        os.close(bdir_fd)

    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason in (
            RefusalReason.NON_UTF8_ENTRY,
            RefusalReason.MALFORMED_ENTRY,
            RefusalReason.UNREADABLE,
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# fd_dir / openat2 / policy surface
# ---------------------------------------------------------------------------


def test_list_dir_fd_oserror_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)

    def boom(_dir_fd: int) -> list[str]:
        raise OSError(5, "simulated readdir failure")

    try:
        with mock.patch.object(fd_dir_mod.os, "listdir", side_effect=boom):
            with pytest.raises(ActivationRefusal) as ei:
                list_dir_fd(fd)
            assert ei.value.reason is RefusalReason.UNREADABLE
    finally:
        os.close(fd)


def test_list_dir_fd_happy(tmp_path: Path) -> None:
    d = tmp_path / "d"
    d.mkdir()
    (d / "a").write_bytes(b"x")
    fd = os.open(str(d), os.O_RDONLY | os.O_DIRECTORY)
    try:
        names = set(list_dir_fd(fd))
        assert "a" in names
    finally:
        os.close(fd)


def test_no_repin_api_and_immutable_pins(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    store = _store(root, [])
    try:
        assert not hasattr(store, "add_pin")
        assert not hasattr(store, "update_pin")
        assert not hasattr(store, "set_pin")
        assert not hasattr(store, "reload_pins")
        with pytest.raises(TypeError):
            store._pins[("a", "b")] = None  # type: ignore[index]
    finally:
        store.close()


def test_refusal_messages_are_path_safe(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    secret = "/tmp/should-not-appear-in-refusal"
    pin = _sf_pin("act.x", "art.x", "missing.pt", b"x")
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.x", "art.x")
        text = str(ei.value)
        assert secret not in text
        assert str(root) not in text
        assert "missing.pt" not in text
    finally:
        store.close()


def test_openat2_green_path_records_impl(tmp_path: Path) -> None:
    if not openat2_available():
        pytest.skip(
            "openat2 syscall unavailable — component-walk is the tested fallback"
        )
    root = tmp_path / "store"
    data = b"openat2-bytes"
    rel = "f/model.pt"
    _write(root / rel, data)
    pin = _sf_pin("act.o2", "art.o2", rel, data)
    store = _store(root, [pin], impl=ResolverImpl.OPENAT2)
    try:
        assert store.assert_fixed_hash("act.o2", "art.o2") == data
        assert last_open_impl() is ResolverImpl.OPENAT2
    finally:
        store.close()


def test_openat2_bundle_green_path_records_impl(tmp_path: Path) -> None:
    """Linux: BUNDLE root open must record OPENAT2 and yield a readable freeze."""
    if not openat2_available():
        pytest.skip(
            "openat2 syscall unavailable — component-walk is the tested fallback"
        )
    root = tmp_path / "store"
    member = b"openat2-bundle-member"
    _write(root / "bundle" / "m.bin", member)
    pin = _bundle_pin("act.o2b", "art.o2b", "bundle", [("m.bin", member)])
    store = _store(
        root,
        [pin],
        impl=ResolverImpl.OPENAT2,
        freeze_parent=tmp_path / "freeze",
    )
    try:
        frozen = store.assert_bundle_digest("act.o2b", "art.o2b")
        try:
            # last_open_impl must prove OPENAT2 (not a silent COMPONENT fallback).
            assert last_open_impl() is ResolverImpl.OPENAT2
            assert frozen.read_member("m.bin") == member
        finally:
            frozen.cleanup()
    finally:
        store.close()


def test_openat2_bundle_intermediate_symlink_refused(tmp_path: Path) -> None:
    """Linux: BUNDLE pin through intermediate symlink rejected via OPENAT2."""
    if not openat2_available():
        pytest.skip(
            "openat2 syscall unavailable — component-walk is the tested fallback"
        )
    import src.core.model_activation.resolver as resolver_mod

    root = tmp_path / "store"
    outside = tmp_path / "outside"
    # Real terminal bundle root lives outside the store; only the intermediate
    # pin component (family) is the symlink.
    _write(outside / "bundle" / "m.bin", b"evil")
    family = root / "family"
    family.parent.mkdir(parents=True, exist_ok=True)
    family.symlink_to(outside, target_is_directory=True)
    pin = _bundle_pin(
        "act.o2s", "art.o2s", "family/bundle", [("m.bin", b"evil")]
    )
    # freeze_parent required since the no-default-temp contract; the refusal
    # under test must come from the OPENAT2 resolver, not the freeze pre-gate.
    store = _store(
        root,
        [pin],
        impl=ResolverImpl.OPENAT2,
        freeze_parent=tmp_path / "freeze",
    )
    try:
        # last_open_impl is set only on successful open; on refusal, prove the
        # OPENAT2 path ran (and COMPONENT did not) via spy — equivalent guard
        # against a silent component fallback passing this RED case.
        with mock.patch.object(
            resolver_mod,
            "_component_walk",
            side_effect=AssertionError(
                "silent COMPONENT fallback — OPENAT2 path required"
            ),
        ), mock.patch.object(
            resolver_mod,
            "_openat2_syscall",
            wraps=resolver_mod._openat2_syscall,
        ) as openat2_spy:
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.o2s", "art.o2s")
        assert openat2_spy.called, "OPENAT2 syscall must be attempted"
        assert ei.value.reason in (
            RefusalReason.SYMLINK_REJECTED,
            RefusalReason.NOT_DIRECTORY,
            RefusalReason.CONTAINMENT,
            RefusalReason.ARTIFACT_MISSING,
        )
    finally:
        store.close()


def test_component_and_openat2_agree_on_green_when_both(
    tmp_path: Path,
) -> None:
    if not openat2_available():
        pytest.skip("openat2 syscall unavailable")
    root = tmp_path / "store"
    data = b"parity-green"
    rel = "g/m.pt"
    _write(root / rel, data)
    pin = _sf_pin("act.p", "art.p", rel, data)
    for impl in (ResolverImpl.COMPONENT, ResolverImpl.OPENAT2):
        store = _store(root, [pin], impl=impl)
        try:
            assert store.assert_fixed_hash("act.p", "art.p") == data
            assert last_open_impl() is impl
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Authority-fix: path-safe refusal chaining + AST completeness guard
# ---------------------------------------------------------------------------

_SCOPE_PY = [
    Path("src/core/model_activation/digest.py"),
    Path("src/core/model_activation/fd_dir.py"),
    Path("src/core/model_activation/resolver.py"),
    Path("src/core/model_activation/store.py"),
    Path("src/core/model_activation/types.py"),
]

def _assert_path_safe_refusal(exc: BaseException, hostile_path: str) -> None:
    assert isinstance(exc, ActivationRefusal)
    assert exc.__cause__ is None, f"__cause__ leaked: {exc.__cause__!r}"
    assert exc.__context__ is None, f"__context__ leaked: {exc.__context__!r}"
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert hostile_path not in tb
    assert hostile_path not in str(exc)
    assert hostile_path not in repr(exc)


def test_refusal_no_oserror_context_from_missing_open(tmp_path: Path) -> None:
    """Mapped OSError must not survive as __context__/__cause__ with path."""
    root = tmp_path / "store"
    root.mkdir()
    hostile = str(root / "missing.pt")
    pin = _sf_pin("act.x", "art.x", "missing.pt", b"x")
    store = _store(root, [pin])
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_fixed_hash("act.x", "art.x")
        _assert_path_safe_refusal(ei.value, hostile)
        _assert_path_safe_refusal(ei.value, str(root))
    finally:
        store.close()


def test_list_dir_fd_oserror_no_context(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)

    def boom(_dir_fd: int) -> list[str]:
        raise OSError(5, "simulated readdir failure", "/secret/hostile/path")

    try:
        with mock.patch.object(fd_dir_mod.os, "listdir", side_effect=boom):
            with pytest.raises(ActivationRefusal) as ei:
                list_dir_fd(fd)
        _assert_path_safe_refusal(ei.value, "/secret/hostile/path")
    finally:
        os.close(fd)


def test_read_fd_oserror_no_context(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abcd")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)

    def boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise OSError(5, "read boom", str(path))

    try:
        with mock.patch.object(digest_mod.os, "read", side_effect=boom):
            with pytest.raises(ActivationRefusal) as ei:
                read_fd_bounded_once(fd, 1024, expect_size=4)
        _assert_path_safe_refusal(ei.value, str(path))
    finally:
        os.close(fd)


def test_ast_refusal_mapping_completeness_guard() -> None:
    """Static guard: OSError handlers must not raise ActivationRefusal with chain.

    Catches:
    - ``raise ... from <exc>`` inside except OSError handlers
    - mapper calls whose result is raised with ``from`` in the same handler
    - annotated / walrus assignments of mappers that are still raised-with-from
    """
    violations: list[str] = []

    for path in _SCOPE_PY:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            # Only care about OSError (and bare Exception used for OS mapping).
            type_name = ""
            if isinstance(node.type, ast.Name):
                type_name = node.type.id
            elif isinstance(node.type, ast.Tuple):
                type_name = ",".join(
                    elt.id for elt in node.type.elts if isinstance(elt, ast.Name)
                )
            if type_name and "OSError" not in type_name and type_name != "Exception":
                # Keep scanning Exception handlers that might wrap OSError mappers.
                if "OSError" not in type_name:
                    pass
            for child in ast.walk(node):
                if not isinstance(child, ast.Raise):
                    continue
                if child.cause is None:
                    continue
                # raise X from Y inside except — forbidden for path-safe mapping
                # (even from None still sets __context__ to the OSError).
                violations.append(
                    f"{path}:{child.lineno}: raise-with-cause inside except "
                    f"({type_name or 'bare'})"
                )

    # Also: any `raise _map_*Error(...)` or `raise ActivationRefusal(...) from`
    # at module level patterns already covered. Mapper *calls* must exist for
    # open/member paths.
    store_src = Path("src/core/model_activation/store.py").read_text(encoding="utf-8")
    resolver_src = Path("src/core/model_activation/resolver.py").read_text(
        encoding="utf-8"
    )
    assert "_map_open_error" in resolver_src
    assert "_map_member_open_error" in store_src
    assert "from exc" not in store_src  # no raise ... from exc
    assert "from exc" not in resolver_src
    assert "from exc" not in Path("src/core/model_activation/digest.py").read_text(
        encoding="utf-8"
    )
    assert "from exc" not in Path("src/core/model_activation/fd_dir.py").read_text(
        encoding="utf-8"
    )
    assert "from exc" not in Path("src/core/model_activation/types.py").read_text(
        encoding="utf-8"
    )

    # Walrus / annotated assignment of mapper results inside except must not
    # be raised with cause in that same handler (covered by raise-with-cause).
    assert violations == [], "path-safe raise violations:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Freeze lease: parent validation, inventory, fstat-after-O_CREAT, dest race
# ---------------------------------------------------------------------------


def test_freeze_parent_required_no_default_temp(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"x")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"x")])
    store = ControlledStore(str(root), [pin], resolver_impl=ResolverImpl.COMPONENT)
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.FREEZE_FAILED
        assert "parent" in ei.value.detail
    finally:
        store.close()


def test_freeze_parent_mode_and_owner_validated(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"x")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"x")])
    bad = tmp_path / "world-readable"
    bad.mkdir()
    os.chmod(bad, 0o755)  # group/other bits set → refuse
    store = ControlledStore(
        str(root),
        [pin],
        freeze_parent=str(bad),
        resolver_impl=ResolverImpl.COMPONENT,
    )
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.FREEZE_FAILED
    finally:
        store.close()


def test_bind_root_path_swap_stays_on_opened_parent_fd(tmp_path: Path) -> None:
    """Parent pathname swap after open must not redirect freeze creation.

    After O_NOFOLLOW open of the trusted parent, rename it away and place an
    attacker-controlled directory at the original path. mkdir/openat via the
    held parent fd must create under the original inode (or refuse) — never
    under the replacement path. No model bytes under the attacker directory.
    """
    trusted = tmp_path / "trusted-parent"
    trusted.mkdir()
    os.chmod(trusted, 0o700)
    trusted_st = os.stat(trusted)
    trusted_path = str(trusted)
    real_open = os.open
    swapped = {"n": 0}

    def swap_after_parent_open(  # type: ignore[no-untyped-def]
        path, flags, mode=0o777, *, dir_fd=None
    ):
        # Only the initial parent open (no dir_fd) of the trusted pathname.
        if (
            dir_fd is None
            and path == trusted_path
            and (flags & os.O_DIRECTORY)
            and swapped["n"] == 0
        ):
            fd = real_open(path, flags, mode)
            swapped["n"] += 1
            # Path swap: original inode moves; attacker occupies the name.
            trusted.rename(tmp_path / "trusted-moved")
            attacker = tmp_path / "attacker-tmp"
            attacker.mkdir()
            os.chmod(attacker, 0o700)
            attacker.rename(trusted)  # original pathname now attacker-owned
            return fd
        return real_open(path, flags, mode, dir_fd=dir_fd)

    lease = FreezeResourceLease()
    try:
        with mock.patch.object(store_mod.os, "open", side_effect=swap_after_parent_open):
            lease.bind_root(trusted_path)
        assert swapped["n"] == 1
        # Parent fd identity is the original trusted inode.
        pst = os.fstat(lease._parent_fd)  # noqa: SLF001
        assert (pst.st_dev, pst.st_ino) == (trusted_st.st_dev, trusted_st.st_ino)
        # Freeze root lives under the moved trusted directory (opened inode).
        moved = tmp_path / "trusted-moved"
        freezes = [p for p in moved.iterdir() if p.name.startswith("cadml-freeze-")]
        assert len(freezes) == 1
        # Replacement at original pathname has no freeze / no model bytes.
        attacker_now = Path(trusted_path)
        assert attacker_now.is_dir()
        assert list(attacker_now.glob("cadml-freeze-*")) == []
        for f in attacker_now.rglob("*"):
            if f.is_file() and f.stat().st_size > 0:
                pytest.fail(f"model/content under swapped parent path: {f}")
    finally:
        lease.release()


def test_finalize_fstat_failure_after_ocreat_leaves_no_model_byte(
    tmp_path: Path,
) -> None:
    """P1: if fstat fails after O_CREAT, no cadml-freeze-*/m.bin may remain.

    Would fail on the old implementation that closed the create fd without
    adopting it into a lease before fstat.
    """
    root = tmp_path / "store"
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    _write(root / "bundle" / "m.bin", b"model-bytes")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"model-bytes")])
    store = _store(root, [pin], freeze_parent=freeze_parent)

    real_fstat = os.fstat
    create_fds: set[int] = set()
    real_open = os.open

    def spy_open(path, flags, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        fd = real_open(path, flags, mode, dir_fd=dir_fd)
        if flags & os.O_CREAT:
            create_fds.add(fd)
        return fd

    fstat_calls_on_create = {"n": 0}

    def evil_fstat(fd: int):  # type: ignore[no-untyped-def]
        if fd in create_fds:
            fstat_calls_on_create["n"] += 1
            if fstat_calls_on_create["n"] == 1:
                raise OSError(errno.EIO, "simulated fstat fail", "/secret/m.bin")
        return real_fstat(fd)

    try:
        with mock.patch.object(store_mod.os, "open", side_effect=spy_open), mock.patch.object(
            store_mod.os, "fstat", side_effect=evil_fstat
        ):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
        _assert_path_safe_refusal(ei.value, "/secret/m.bin")
        # No model-byte residual under freeze parent.
        leftovers = list(freeze_parent.glob("cadml-freeze-*"))
        for d in leftovers:
            mbin = list(d.rglob("m.bin"))
            assert mbin == [], f"model byte residual: {mbin}"
            # Empty shell residual is allowed only when non-atomic; bytes must
            # not remain.
            for f in d.rglob("*"):
                if f.is_file():
                    assert f.stat().st_size == 0 or f.name != "m.bin"
                    # Any regular file residual with content is a failure.
                    if f.is_file() and f.stat().st_size > 0:
                        pytest.fail(f"non-empty residual {f}")
    finally:
        store.close()


def test_inventory_max_1_leaves_no_model_byte(tmp_path: Path) -> None:
    """P1: with inventory max=1, reserve before create → no m.bin residual."""
    root = tmp_path / "store"
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    _write(root / "bundle" / "m.bin", b"model-bytes")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"model-bytes")])
    store = _store(root, [pin], freeze_parent=freeze_parent)

    # Root bind consumes the only inventory slot; file create must refuse
    # before O_CREAT.
    try:
        with mock.patch.object(store_mod, "_MAX_FREEZE_IDENTITY_INVENTORY", 1):
            # Also patch the default used by new leases.
            original_init = FreezeResourceLease.__init__

            def limited_init(self, *a, **k):  # type: ignore[no-untyped-def]
                original_init(self, *a, **k)
                self._max_inventory = 1

            with mock.patch.object(FreezeResourceLease, "__init__", limited_init):
                with pytest.raises(ActivationRefusal) as ei:
                    store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.FREEZE_FAILED
        leftovers = list(freeze_parent.glob("cadml-freeze-*"))
        for d in leftovers:
            assert list(d.rglob("m.bin")) == []
            for f in d.rglob("*"):
                if f.is_file() and f.stat().st_size > 0:
                    pytest.fail(f"model byte residual {f}")
    finally:
        store.close()


def test_dest_dir_open_failure_race_preserves_foreign_no_mbin(
    tmp_path: Path,
) -> None:
    """mkdir nested dest succeeds; open(O_DIRECTORY) of 'sub' races.

    Intercept first open of 'sub' as directory under freeze: rename owned
    freeze/sub → owned-moved, create **empty** foreign freeze/sub, raise EACCES.
    Must cancel reservation only (no name-stat/commit of foreign). Record the
    foreign empty directory's (dev, ino) and assert the **same object** remains
    at freeze/sub after refusal/cleanup. Marker stays **outside** the target so
    blind rmdir of an empty foreign dir cannot be masked. Zero model bytes; no
    handle/digest/load. Would go RED if name-stat/adoption of the replacement
    were reintroduced.
    """
    root = tmp_path / "store"
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    _write(root / "bundle" / "sub" / "m.bin", b"payload")
    pin = _bundle_pin(
        "act.b", "art.b", "bundle", [("sub/m.bin", b"payload")]
    )
    store = _store(root, [pin], freeze_parent=freeze_parent)

    real_open = os.open
    real_mkdir = os.mkdir
    mkdir_freeze_sub = {"n": 0}
    seen = {"n": 0}
    # Marker outside the replacement directory (must not make rmdir fail).
    foreign_marker = freeze_parent / "_foreign_alive"
    foreign_id: dict[str, object] = {}

    def spy_mkdir(path, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        real_mkdir(path, mode, dir_fd=dir_fd)
        name = path if isinstance(path, str) else path
        # Freeze-side mkdir of nested dest (after source walk already saw "sub").
        if name == "sub" and dir_fd is not None:
            mkdir_freeze_sub["n"] += 1

    def race_open(path, flags, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        name = path if isinstance(path, str) else path
        is_dir_open = bool(flags & os.O_DIRECTORY)
        # Only the freeze-side open after mkdir_owned — never the source open.
        if (
            is_dir_open
            and name == "sub"
            and dir_fd is not None
            and mkdir_freeze_sub["n"] > 0
            and seen["n"] == 0
        ):
            seen["n"] += 1
            freeze_root = None
            for cand in freeze_parent.iterdir():
                if cand.name.startswith("cadml-freeze-") and cand.is_dir():
                    freeze_root = cand
                    break
            if freeze_root is not None and (freeze_root / "sub").exists():
                owned_moved = freeze_parent / "owned-moved"
                (freeze_root / "sub").rename(owned_moved)
                # Empty foreign replacement — no interior marker files.
                (freeze_root / "sub").mkdir()
                st = os.lstat(freeze_root / "sub")
                foreign_id["path"] = freeze_root / "sub"
                foreign_id["dev_ino"] = (st.st_dev, st.st_ino)
                foreign_id["freeze_root"] = freeze_root
                foreign_marker.write_text("foreign-sub-at-freeze")
            raise OSError(errno.EACCES, "raced open", "sub")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    try:
        with mock.patch.object(store_mod.os, "mkdir", side_effect=spy_mkdir), mock.patch.object(
            store_mod.os, "open", side_effect=race_open
        ):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.FREEZE_FAILED
        assert seen["n"] == 1
        assert "dev_ino" in foreign_id
        foreign_path = Path(foreign_id["path"])  # type: ignore[arg-type]
        # Same empty foreign object must remain at freeze/sub.
        assert foreign_path.is_dir()
        assert list(foreign_path.iterdir()) == [], "foreign replacement must stay empty"
        st_after = os.lstat(foreign_path)
        assert (st_after.st_dev, st_after.st_ino) == foreign_id["dev_ino"]
        # Outside marker only.
        assert foreign_marker.exists()
        # Zero model bytes anywhere under freeze parent.
        for f in freeze_parent.rglob("m.bin"):
            pytest.fail(f"model byte residual {f}")
        # Renamed owned empty shell may remain (honest non-atomic residual).
        assert (freeze_parent / "owned-moved").exists()
    finally:
        store.close()


def test_ordinary_dest_dir_open_failure_zero_model_bytes(tmp_path: Path) -> None:
    root = tmp_path / "store"
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    _write(root / "bundle" / "sub" / "m.bin", b"payload")
    pin = _bundle_pin(
        "act.b", "art.b", "bundle", [("sub/m.bin", b"payload")]
    )
    store = _store(root, [pin], freeze_parent=freeze_parent)
    real_open = os.open
    real_mkdir = os.mkdir
    mkdir_freeze_sub = {"n": 0}
    seen = {"n": 0}

    def spy_mkdir(path, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        real_mkdir(path, mode, dir_fd=dir_fd)
        name = path if isinstance(path, str) else path
        if name == "sub" and dir_fd is not None:
            mkdir_freeze_sub["n"] += 1

    def fail_open(path, flags, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        name = path if isinstance(path, str) else path
        if (
            (flags & os.O_DIRECTORY)
            and name == "sub"
            and mkdir_freeze_sub["n"] > 0
            and seen["n"] == 0
        ):
            seen["n"] += 1
            raise OSError(errno.EACCES, "open fail", "sub")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    try:
        with mock.patch.object(store_mod.os, "mkdir", side_effect=spy_mkdir), mock.patch.object(
            store_mod.os, "open", side_effect=fail_open
        ):
            with pytest.raises(ActivationRefusal):
                store.assert_bundle_digest("act.b", "art.b")
        for f in freeze_parent.rglob("m.bin"):
            pytest.fail(f"model byte residual {f}")
    finally:
        store.close()


def test_reconcile_foreign_survives_and_refuses(tmp_path: Path) -> None:
    """observed − ledger → foreign: refuse without deleting foreign."""
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()
    # Plant foreign file under freeze root by path (adversary).
    backing = Path(lease.backing_path)
    foreign = backing / "evil.bin"
    foreign.write_bytes(b"evil-bytes")
    with pytest.raises(ActivationRefusal) as ei:
        lease.reconcile_observed_against_owned_ledger()
    assert ei.value.reason is RefusalReason.FREEZE_MUTATED
    assert ei.value.__cause__ is None
    assert ei.value.__context__ is None
    assert foreign.exists()
    assert foreign.read_bytes() == b"evil-bytes"
    lease.release()


def test_reconcile_foreign_dir_refuses_before_descent_no_recursion(
    tmp_path: Path,
) -> None:
    """Foreign directory under freeze root must FREEZE_MUTATED before descent.

    With a lowered recursion limit, a deep foreign chain would RecursionError on
    the old walk that descended before ledger membership checks. New code
    refuses at the first foreign identity — typed/path-safe, no RecursionError.
    """
    import sys

    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    backing = Path(lease.backing_path)
    # Depth above a lowered recursion limit; short names + modest depth for FS.
    depth = 80
    parent_fd = os.open(str(backing), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for _ in range(depth):
            os.mkdir("x", 0o700, dir_fd=parent_fd)
            child_fd = os.open(
                "x",
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
            os.close(parent_fd)
            parent_fd = child_fd
    finally:
        try:
            os.close(parent_fd)
        except OSError:
            pass

    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(50)  # depth 80 would blow if foreign descent ran
        with pytest.raises(ActivationRefusal) as ei:
            lease.reconcile_observed_against_owned_ledger()
        assert ei.value.reason is RefusalReason.FREEZE_MUTATED
        assert ei.value.__cause__ is None
        assert ei.value.__context__ is None
        assert (backing / "x").is_dir()
    finally:
        sys.setrecursionlimit(old_limit)
        # Iterative teardown so pytest tmpdir rmtree stays shallow.
        stack: list[int] = []
        try:
            stack.append(
                os.open(str(backing), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            )
            while True:
                try:
                    child = os.open(
                        "x",
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=stack[-1],
                    )
                except OSError:
                    break
                stack.append(child)
            while len(stack) > 1:
                child = stack.pop()
                try:
                    os.close(child)
                except OSError:
                    pass
                try:
                    os.rmdir("x", dir_fd=stack[-1])
                except OSError:
                    pass
        finally:
            while stack:
                try:
                    os.close(stack.pop())
                except OSError:
                    pass
        lease.release()


def test_reconcile_owned_missing_refuses(tmp_path: Path) -> None:
    """ledger − observed → owned missing/moved: refuse."""
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    # Create owned file then rename it away.
    root_fd = lease.root_dir_fd()
    fd = lease.create_file_owned(root_fd, "m.bin")
    os.write(fd, b"x")
    os.close(fd)
    backing = Path(lease.backing_path)
    (backing / "m.bin").rename(tmp_path / "moved-owned.bin")
    with pytest.raises(ActivationRefusal) as ei:
        lease.reconcile_observed_against_owned_ledger()
    assert ei.value.reason is RefusalReason.FREEZE_MUTATED
    lease.release()


def test_dup_dir_fd_caller_owned_and_cleanup_concurrent(tmp_path: Path) -> None:
    """Concurrent cleanup cannot expose recycled unrelated fds to dup/open.

    Deterministic barrier protocol:
      1. Pre-return a caller-owned duplicate (must remain valid after cleanup).
      2. Reader validates under lock / bumps in_flight, then pauses (one-shot).
      3. Cleanup thread runs cleanup() and blocks until in_flight drains.
      4. Reader proceeds: in-flight dup still targets the live freeze root.
      5. After drain, cleanup closes the root; main forces old fd-number reuse
         with a poison file; further dup/open refuse (never poison).
      6. Pre-returned duplicate identity remains the freeze inode.
    """
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    pre_dup = -1
    try:
        frozen = store.assert_bundle_digest("act.b", "art.b")
        try:
            assert not hasattr(type(frozen), "dir_fd") or not isinstance(
                getattr(type(frozen), "dir_fd", None), property
            )

            # Already-returned duplicate must survive cleanup (inode held open).
            pre_dup = frozen.dup_dir_fd()
            pre_st = os.fstat(pre_dup)
            assert "m.bin" in set(os.listdir(pre_dup))

            old_fd_num = int(frozen._dir_fd)  # noqa: SLF001 — race fixture only
            validated = threading.Event()
            proceed = threading.Event()
            cleanup_done = threading.Event()
            results: dict[str, object] = {}

            def after_validate() -> None:
                # One-shot so open_member after the barrier is not re-paused.
                object.__setattr__(frozen, "_after_validate_hook", None)
                validated.set()
                assert proceed.wait(timeout=5.0), "proceed never signaled"

            object.__setattr__(frozen, "_after_validate_hook", after_validate)

            def reader() -> None:
                try:
                    d = frozen.dup_dir_fd()
                    try:
                        st = os.fstat(d)
                        results["dup_st"] = (
                            st.st_dev,
                            st.st_ino,
                            stat.S_ISDIR(st.st_mode),
                        )
                        results["dup_names"] = set(os.listdir(d))
                    finally:
                        os.close(d)
                    # Still inside/after protected window: freeze bytes if scrub
                    # has not yet unlinked members; else path-safe closed.
                    try:
                        mfd = frozen.open_member("m.bin")
                        try:
                            results["member"] = os.read(mfd, 64)
                        finally:
                            os.close(mfd)
                    except ActivationRefusal as ar:
                        results["member_refused"] = ar.reason
                except Exception as exc:  # noqa: BLE001 — capture for main
                    results["error"] = exc

            def cleanup_worker() -> None:
                try:
                    frozen.cleanup()
                finally:
                    cleanup_done.set()

            t_reader = threading.Thread(target=reader)
            t_reader.start()
            assert validated.wait(timeout=5.0), "reader never reached validate barrier"

            # Cleanup blocks on in_flight until reader finishes the protected dup.
            t_clean = threading.Thread(target=cleanup_worker)
            t_clean.start()
            # Allow cleanup to enter the in_flight wait.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if int(frozen._in_flight) > 0 and frozen._closed:  # noqa: SLF001
                    break
                time.sleep(0.01)

            proceed.set()
            t_reader.join(timeout=5.0)
            assert not t_reader.is_alive()
            assert cleanup_done.wait(timeout=5.0), "cleanup did not finish"
            t_clean.join(timeout=2.0)

            assert "error" not in results, f"reader failed: {results.get('error')!r}"
            dup_st = results.get("dup_st")
            assert dup_st is not None
            dev, ino, is_dir = dup_st  # type: ignore[misc]
            assert is_dir is True
            # In-flight dup must match freeze identity (pre_dup), never poison.
            assert (dev, ino) == (pre_st.st_dev, pre_st.st_ino)
            assert "m.bin" in results.get("dup_names", set())  # type: ignore[operator]
            if "member" in results:
                assert results["member"] == b"good"
            else:
                assert results.get("member_refused") in (
                    RefusalReason.FREEZE_FAILED,
                    RefusalReason.UNREADABLE,
                )

            # Force the old fd number to be reused by an unrelated open.
            poison_path = tmp_path / "poison.bin"
            poison_path.write_bytes(b"POISON-UNRELATED-FD")
            holders: list[int] = []
            reused = False
            try:
                for _ in range(512):
                    fd = os.open(poison_path, os.O_RDONLY | os.O_CLOEXEC)
                    holders.append(fd)
                    if fd == old_fd_num:
                        reused = True
                        break
            finally:
                # Post-cleanup + reuse pressure: must refuse, not attach poison.
                with pytest.raises(ActivationRefusal) as ei:
                    frozen.dup_dir_fd()
                assert ei.value.reason is RefusalReason.FREEZE_FAILED
                with pytest.raises(ActivationRefusal) as ei2:
                    frozen.open_member("m.bin")
                assert ei2.value.reason is RefusalReason.FREEZE_FAILED
                for fd in holders:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            # Best-effort: we tried to land the old number; refusal is required
            # regardless. Record reuse when the kernel gave us the same slot.
            _ = reused

            # Pre-returned duplicate remains the freeze inode after cleanup.
            post_st = os.fstat(pre_dup)
            assert (post_st.st_dev, post_st.st_ino) == (pre_st.st_dev, pre_st.st_ino)
            assert stat.S_ISDIR(post_st.st_mode)
        finally:
            if pre_dup >= 0:
                try:
                    os.close(pre_dup)
                except OSError:
                    pass
            frozen.cleanup()
    finally:
        store.close()


def test_successful_bundle_lifecycle_scrubs_lease_root_and_tree(
    tmp_path: Path,
) -> None:
    """Successful activation + cleanup must scrub lease-owned root, not just the dup.

    Would fail on the detach_root_fd bug: closing only the bundle dup and
    zeroing lease._root_fd without close left the original fd open, skipped
    _scrub_tree, reported complete with a nonempty ledger, and left the freeze
    tree on disk.
    """
    root = tmp_path / "store"
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    _write(root / "bundle" / "m.bin", b"lifecycle-bytes")
    pin = _bundle_pin(
        "act.b", "art.b", "bundle", [("m.bin", b"lifecycle-bytes")]
    )
    store = _store(root, [pin], freeze_parent=freeze_parent)
    try:
        frozen = store.assert_bundle_digest("act.b", "art.b")
        lease = frozen._lease  # noqa: SLF001
        assert lease is not None
        lease_root_fd = int(lease.root_dir_fd())
        assert lease_root_fd >= 0
        # Prove lease root is live before cleanup.
        os.fstat(lease_root_fd)
        backing = Path(lease.backing_path)
        assert backing.is_dir()
        assert (backing / "m.bin").is_file()
        assert len(lease._ledger) >= 1  # noqa: SLF001

        assert frozen.cleanup() is True

        # Original lease root fd closed (not merely forgotten).
        with pytest.raises(OSError) as ei:
            os.fstat(lease_root_fd)
        assert ei.value.errno == errno.EBADF
        assert int(lease._root_fd) < 0  # noqa: SLF001
        assert lease._ledger == {}  # noqa: SLF001
        assert lease.cleanup_complete is True
        # Backing freeze tree destroyed (no residual freeze dirs under parent).
        leftovers = [
            p
            for p in freeze_parent.iterdir()
            if p.is_dir() and p.name.startswith("cadml-freeze-")
        ]
        assert leftovers == [], f"freeze tree residual: {leftovers}"
        assert not backing.exists() or not any(backing.rglob("*"))
        # Idempotent repeated cleanup.
        assert frozen.cleanup() is True
        assert frozen.cleanup() is True
    finally:
        store.close()


def test_cleanup_retries_same_lease_until_release_succeeds(tmp_path: Path) -> None:
    """Incomplete lease.release() must not drop the lease; next cleanup retries it."""
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        frozen = store.assert_bundle_digest("act.b", "art.b")
        lease = frozen._lease  # noqa: SLF001
        assert lease is not None
        lease_id = id(lease)
        real_release = lease.release
        calls: list[int] = []

        def release_then_ok() -> bool:
            calls.append(id(lease))
            if len(calls) == 1:
                # Simulate incomplete scrub (retain ownership).
                return False
            return bool(real_release())

        with mock.patch.object(lease, "release", side_effect=release_then_ok):
            assert frozen.cleanup() is False
            # Lease retained for retry — same object.
            assert frozen._lease is lease  # noqa: SLF001
            assert id(frozen._lease) == lease_id  # noqa: SLF001
            assert frozen.cleanup() is True
            assert frozen._lease is None  # noqa: SLF001
        assert calls == [lease_id, lease_id]
        # Idempotent after success.
        assert frozen.cleanup() is True
    finally:
        store.close()


def test_concurrent_cleanup_serializes_release_no_premature_true(
    tmp_path: Path,
) -> None:
    """Two concurrent cleanup() callers must not run release concurrently.

    Neither returns True while a release is still in flight; only one release
    runs at a time.
    """
    root = tmp_path / "store"
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        frozen = store.assert_bundle_digest("act.b", "art.b")
        lease = frozen._lease  # noqa: SLF001
        assert lease is not None
        real_release = lease.release
        active = {"n": 0}
        max_active = {"n": 0}
        gate = threading.Lock()
        started = threading.Barrier(2)
        results: list[bool] = []
        results_lock = threading.Lock()

        def slow_release() -> bool:
            with gate:
                active["n"] += 1
                if active["n"] > max_active["n"]:
                    max_active["n"] = active["n"]
            try:
                # Hold the exclusive release slot long enough for the peer to
                # observe serialization.
                time.sleep(0.15)
                return bool(real_release())
            finally:
                with gate:
                    active["n"] -= 1

        with mock.patch.object(lease, "release", side_effect=slow_release):

            def worker() -> None:
                started.wait(timeout=5.0)
                ok = frozen.cleanup()
                with results_lock:
                    results.append(ok)

            t1 = threading.Thread(target=worker)
            t2 = threading.Thread(target=worker)
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)
            assert not t1.is_alive() and not t2.is_alive()

        assert max_active["n"] == 1, (
            f"release ran concurrently (max_active={max_active['n']})"
        )
        assert results == [True, True] or set(results) == {True}
        assert all(results), f"premature False/True mix: {results}"
        assert frozen._lease is None  # noqa: SLF001
        # No concurrent True while releasing: both finish after exclusive release.
        assert frozen.cleanup() is True
    finally:
        store.close()


def test_cleanup_scandir_lazy_cap(tmp_path: Path) -> None:
    """Cleanup must use lazy os.scandir; pure listdir materialization fails."""
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    # Create a child so scrub has something to walk.
    root_fd = lease.root_dir_fd()
    child = lease.mkdir_owned(root_fd, "sub")
    os.close(child)

    calls = {"scandir": 0}
    real_scandir = os.scandir

    def spy_scandir(path):  # type: ignore[no-untyped-def]
        calls["scandir"] += 1
        return real_scandir(path)

    def forbid_listdir(*_a, **_k):  # type: ignore[no-untyped-def]
        raise AssertionError(
            "cleanup must not fall back to unbounded os.listdir materialization"
        )

    with mock.patch.object(fd_dir_mod.os, "scandir", side_effect=spy_scandir), mock.patch.object(
        fd_dir_mod.os, "listdir", side_effect=forbid_listdir
    ), mock.patch.object(store_mod.os, "listdir", side_effect=forbid_listdir):
        assert lease.release() is True

    # Discriminating: mutation that skips scandir (or only uses listdir) fails.
    assert calls["scandir"] >= 1, (
        "expected os.scandir during descriptor-relative cleanup scrub"
    )


def test_scandir_unavailable_no_listdir_retains_ownership(tmp_path: Path) -> None:
    """If native fd-scandir and /dev/fd|/proc/self/fd fail, refuse without listdir.

    Cleanup must not materialise the directory; ownership (root fd + ledger)
    is retained for retry (release returns False / cleanup incomplete).
    """
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()
    # Owned child + file so incomplete scrub is observable.
    child = lease.mkdir_owned(root_fd, "sub")
    ffd = lease.create_file_owned(child, "m.bin")
    os.write(ffd, b"model-bytes")
    os.close(ffd)
    os.close(child)

    ledger_before = dict(lease._ledger)  # noqa: SLF001
    assert len(ledger_before) >= 2
    root_fd_before = int(lease.root_dir_fd())

    listdir_calls: list[object] = []

    def fail_scandir(_path):  # type: ignore[no-untyped-def]
        # Make both native int fd and /dev/fd|/proc/self/fd path forms fail.
        raise OSError(errno.ENOTSUP, "scandir unavailable")

    def spy_listdir(*_a, **_k):  # type: ignore[no-untyped-def]
        listdir_calls.append((_a, _k))
        raise AssertionError("os.listdir must never be called during cleanup scan")

    with mock.patch.object(fd_dir_mod.os, "scandir", side_effect=fail_scandir), mock.patch.object(
        fd_dir_mod.os, "listdir", side_effect=spy_listdir
    ), mock.patch.object(store_mod.os, "listdir", side_effect=spy_listdir):
        done = lease.release()
        # Path-safe direct helper refusal under the same unavailable binding.
        with pytest.raises(ActivationRefusal) as ei:
            scandir_dir_fd(root_fd_before)
        assert ei.value.reason is RefusalReason.UNREADABLE
        assert ei.value.__cause__ is None
        assert ei.value.__context__ is None
        assert "dev/fd" not in str(ei.value)
        assert "/proc/" not in str(ei.value)

    assert done is False, "cleanup must refuse complete when scandir cannot bind"
    assert listdir_calls == [], "listdir must not be called (no materialisation)"
    # Ownership retained: live root fd and creation-time ledger still present.
    assert int(lease._root_fd) == root_fd_before  # noqa: SLF001
    assert int(lease._root_fd) >= 0  # noqa: SLF001
    assert lease._ledger == ledger_before  # noqa: SLF001
    assert lease.cleanup_complete is False

    # Real scandir available again — retry can finish without materialising.
    assert lease.release() is True


def test_cleanup_scandir_cap_plus_one_stops_without_exhausting(
    tmp_path: Path,
) -> None:
    """Cap+1 discrimination stops iteration without exhausting a large generator."""
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))

    next_calls = {"n": 0}
    closed = {"n": 0}

    class _Entry:
        __slots__ = ("name",)

        def __init__(self, name: str) -> None:
            self.name = name

    class _CountingIt:
        def __iter__(self):  # type: ignore[no-untyped-def]
            return self

        def __next__(self) -> _Entry:
            next_calls["n"] += 1
            # Unbounded generator: would hang/assert if scrub exhausted it.
            if next_calls["n"] > 10_000:
                raise AssertionError("iterator exhausted past cap — not lazy-capped")
            return _Entry(f"e{next_calls['n']:05d}")

        def close(self) -> None:
            closed["n"] += 1

    def fake_scandir(_dir_fd: int):  # type: ignore[no-untyped-def]
        return _CountingIt()

    # Small cap: process until count > cap+1 then break.
    with mock.patch.object(store_mod, "_CLEANUP_SCANDIR_CAP", 2), mock.patch.object(
        store_mod, "_scandir_dir_fd", side_effect=fake_scandir
    ), mock.patch.object(
        fd_dir_mod.os,
        "listdir",
        side_effect=AssertionError("listdir must not run"),
    ), mock.patch.object(
        store_mod.os,
        "listdir",
        side_effect=AssertionError("listdir must not run"),
    ):
        done = lease.release()

    # With cap=2, stop when count > 3 → fourth __next__ then break.
    assert next_calls["n"] == 4, f"expected cap+1 stop at 4 nexts, got {next_calls['n']}"
    assert next_calls["n"] < 100, "must not exhaust unbounded generator"
    assert closed["n"] >= 1, "iterator must be closed on cap stop"
    assert done is False  # incomplete scrub under synthetic entries

    # Finish with real scandir (empty freeze root).
    assert lease.release() is True


def test_preflight_dirent_cap_lazy_no_listdir_exhaustion(tmp_path: Path) -> None:
    """max_total_dirents refusal must not materialise via listdir or exhaust generator.

    Proves source preflight uses shared lazy scandir: cap+1 stop without reading
    the rest of an unbounded iterator, and any os.listdir path is forbidden.
    """
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    # Real empty seed so open_pinned of the bundle root succeeds; names come from mock.
    (bdir / "seed.txt").write_bytes(b"")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(
        root,
        [pin],
        bounds=_loose_bounds(max_total_dirents=3),
        freeze_parent=tmp_path / "freeze",
    )

    next_calls = {"n": 0}
    closed = {"n": 0}
    real_open = os.open

    class _Entry:
        __slots__ = ("name",)

        def __init__(self, name: str) -> None:
            self.name = name

    class _CountingIt:
        def __iter__(self):  # type: ignore[no-untyped-def]
            return self

        def __next__(self) -> _Entry:
            next_calls["n"] += 1
            if next_calls["n"] > 10_000:
                raise AssertionError("iterator exhausted past dirent cap")
            # Skip . / .. so each next counts as a dirent.
            return _Entry(f"d{next_calls['n']:05d}")

        def close(self) -> None:
            closed["n"] += 1

    def fake_scandir(_dir_fd: int):  # type: ignore[no-untyped-def]
        return _CountingIt()

    def boom_listdir(*_a, **_k):  # type: ignore[no-untyped-def]
        raise AssertionError("os.listdir must not be used on C1 traversal paths")

    def open_synth(path, flags, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        # Materialise synthetic preflight names as empty regular files under dir_fd
        # so fstat passes and the walk can reach the dirent cap.
        name = path if isinstance(path, str) else path
        if (
            isinstance(name, str)
            and len(name) == 6
            and name[0] == "d"
            and name[1:].isdigit()
            and dir_fd is not None
        ):
            try:
                return real_open(name, flags, mode, dir_fd=dir_fd)
            except OSError:
                cfd = real_open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                    dir_fd=dir_fd,
                )
                os.close(cfd)
                return real_open(name, flags, mode, dir_fd=dir_fd)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    try:
        with mock.patch.object(
            store_mod, "_scandir_dir_fd", side_effect=fake_scandir
        ), mock.patch.object(
            store_mod.os, "open", side_effect=open_synth
        ), mock.patch.object(
            fd_dir_mod.os, "listdir", side_effect=boom_listdir
        ), mock.patch.object(
            store_mod.os, "listdir", side_effect=boom_listdir
        ), mock.patch.object(
            types_mod.os, "listdir", side_effect=boom_listdir
        ):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.BUNDLE_TOTAL_DIRENTS
        # max_total_dirents=3 → refuse when counter becomes 4 (cap+1).
        assert next_calls["n"] == 4, f"expected 4 nexts for cap 3, got {next_calls['n']}"
        assert next_calls["n"] < 100
        assert closed["n"] >= 1, "scandir iterator must be closed on cap refuse"
    finally:
        store.close()


def test_list_dir_fd_documented_unused_by_c1_security() -> None:
    """list_dir_fd remains only as non-security compat; C1 walks must not call it."""
    src = Path("src/core/model_activation/store.py").read_text(encoding="utf-8")
    types_src = Path("src/core/model_activation/types.py").read_text(encoding="utf-8")
    assert "list_dir_fd" not in src
    assert "list_dir_fd" not in types_src
    # Compat helper still exists and is documented.
    fd_src = Path("src/core/model_activation/fd_dir.py").read_text(encoding="utf-8")
    assert "unused by C1" in fd_src or "Not used by C1" in fd_src
    assert "def list_dir_fd" in fd_src


def test_stat_rmdir_race_identity_mismatch(tmp_path: Path) -> None:
    """Post-stat/pre-rmdir replacement: empty foreign dir must not be removed.

    Hook fires after cleanup has observed the owned empty directory identity
    and closed its fd, but immediately before the pre-remove identity re-check
    and rmdir. The hook renames the owned shell away and installs an **empty**
    foreign directory at the same name. Foreign (dev, ino) must survive.
    Interior markers are forbidden (would make unsafe rmdir fail by accident).

    Residual note: a non-atomic window remains between the pre-remove re-check
    and the rmdir syscall itself (classic empty-dir TOCTOU); portable POSIX
    cannot close that fully. This test proves the re-check path refuses the
    observed post-stat replacement.
    """
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()
    child = lease.mkdir_owned(root_fd, "sub")
    os.close(child)
    backing = Path(lease.backing_path)
    foreign_id: dict[str, object] = {}
    hook_fired = {"n": 0}

    def pre_remove(
        parent_fd: int, name: str, fid: object, is_dir: bool
    ) -> None:
        if name != "sub" or not is_dir:
            return
        hook_fired["n"] += 1
        owned = backing / "sub"
        if not owned.is_dir():
            return
        owned.rename(tmp_path / "owned-sub-moved")
        foreign = backing / "sub"
        foreign.mkdir()
        # Empty foreign — no marker inside.
        st = os.lstat(foreign)
        foreign_id["dev_ino"] = (st.st_dev, st.st_ino)
        foreign_id["path"] = foreign
        # Outside-only marker.
        (freeze_parent / "_foreign_marker").write_text("outside")

    lease._pre_remove_hook = pre_remove  # noqa: SLF001
    done = lease.release()
    lease._pre_remove_hook = None  # noqa: SLF001

    assert hook_fired["n"] >= 1, "pre-remove hook must run for owned empty sub"
    assert "dev_ino" in foreign_id
    foreign = Path(foreign_id["path"])  # type: ignore[arg-type]
    assert foreign.is_dir()
    assert list(foreign.iterdir()) == []
    st_after = os.lstat(foreign)
    assert (st_after.st_dev, st_after.st_ino) == foreign_id["dev_ino"]
    assert (tmp_path / "owned-sub-moved").exists()
    assert (freeze_parent / "_foreign_marker").exists()
    _ = done


def test_scrub_pending_success_pops_ledger_reaches_complete(
    tmp_path: Path,
) -> None:
    """Pending node: proven remove must pop ledger, close fd, allow complete.

    Old bug: always closed/returned True without popping ledger → forever
    incomplete (pending=0, ledger=1, root closed, FS empty).
    """
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()
    # Simulate O_CREAT then fstat failure: leave as pending (not finalized).
    lease.reserve_identity_slot()
    fd = os.open(
        "m.bin",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
        dir_fd=root_fd,
    )
    os.write(fd, b"pending-bytes")
    node = lease._adopt_pending(fd, root_fd, "m.bin", is_dir=False)  # noqa: SLF001
    assert node in lease._pending  # noqa: SLF001
    assert node.fd == fd
    # Pending owns a stable parent-dir dup (not a borrow of caller's root_fd).
    assert node.parent_fd >= 0
    assert node.parent_fd != root_fd
    owned_parent = int(node.parent_fd)

    assert lease.release() is True
    assert lease.cleanup_complete is True
    assert lease._pending == []  # noqa: SLF001
    assert lease._ledger == {}  # noqa: SLF001
    assert int(lease._root_fd) < 0  # noqa: SLF001
    assert int(lease._parent_fd) < 0  # noqa: SLF001
    # Pending child fd + owned parent dup closed on success.
    with pytest.raises(OSError) as ei:
        os.fstat(fd)
    assert ei.value.errno == errno.EBADF
    with pytest.raises(OSError) as ei_p:
        os.fstat(owned_parent)
    assert ei_p.value.errno == errno.EBADF
    # Zero model-byte residual under freeze parent.
    for p in freeze_parent.rglob("m.bin"):
        pytest.fail(f"model byte residual {p}")
    leftovers = [
        p
        for p in freeze_parent.iterdir()
        if p.is_dir() and p.name.startswith("cadml-freeze-")
    ]
    assert leftovers == []


def test_scrub_pending_mismatch_retains_pending_incomplete(
    tmp_path: Path,
) -> None:
    """Pending remove mismatch: keep same pending fd; foreign name untouched."""
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()
    lease.reserve_identity_slot()
    fd = os.open(
        "m.bin",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
        dir_fd=root_fd,
    )
    os.write(fd, b"owned-pending")
    node = lease._adopt_pending(fd, root_fd, "m.bin", is_dir=False)  # noqa: SLF001
    owned_parent = int(node.parent_fd)

    backing = Path(lease.backing_path)
    # Rename owned file away; plant foreign empty name (no model bytes at name).
    (backing / "m.bin").rename(tmp_path / "owned-moved.bin")
    (backing / "m.bin").write_bytes(b"")  # foreign empty placeholder at name
    foreign_st = os.lstat(backing / "m.bin")
    foreign_id = (foreign_st.st_dev, foreign_st.st_ino)

    assert lease.release() is False
    assert lease.cleanup_complete is False
    assert node in lease._pending  # noqa: SLF001
    assert node.fd == fd  # not closed
    assert node.parent_fd == owned_parent  # owned parent retained on mismatch
    # Ledger still tracks owned pending identity.
    assert any(True for _ in lease._ledger)  # noqa: SLF001
    # Foreign name untouched (same inode).
    st2 = os.lstat(backing / "m.bin")
    assert (st2.st_dev, st2.st_ino) == foreign_id
    # Owned bytes still held via pending fd (not released to FS at name).
    st_fd = os.fstat(fd)
    assert st_fd.st_size == len(b"owned-pending")
    # Honest retry still incomplete while mismatch remains.
    assert lease.release() is False
    assert node.fd == fd
    assert node.parent_fd == owned_parent

    # Cleanup: close pending child + owned parent and drop node so suite doesn't leak.
    try:
        os.close(fd)
    except OSError:
        pass
    node.fd = -1
    try:
        os.close(owned_parent)
    except OSError:
        pass
    node.parent_fd = -1
    try:
        lease._pending.remove(node)  # noqa: SLF001
    except ValueError:
        pass
    # Pop ledger entries so release can finish residual root shell.
    lease._ledger.clear()  # noqa: SLF001
    lease.release()


def test_pending_leases_retained_by_identity_not_value_eq(
    tmp_path: Path,
) -> None:
    """Two distinct equal-looking leases must both be retained (identity)."""
    root = tmp_path / "store"
    root.mkdir()
    store = _store(root, [], freeze_parent=tmp_path / "freeze")
    try:
        a = FreezeResourceLease()
        b = FreezeResourceLease()
        # Field values match (unbound empty leases) but objects differ.
        assert a is not b
        store._retain_pending_lease(a)  # noqa: SLF001
        store._retain_pending_lease(b)  # noqa: SLF001
        store._retain_pending_lease(a)  # noqa: SLF001 — no duplicate identity
        assert len(store._pending_leases) == 2  # noqa: SLF001
        assert store._pending_leases[0] is a  # noqa: SLF001
        assert store._pending_leases[1] is b  # noqa: SLF001
    finally:
        store._pending_leases.clear()  # noqa: SLF001
        store.close()


def test_pending_fstat_failure_retains_pending_no_name_delete(
    tmp_path: Path,
) -> None:
    """Release-time fstat failure must not name-delete or drop the pending fd."""
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()
    lease.reserve_identity_slot()
    os.mkdir("pend", 0o700, dir_fd=root_fd)
    pfd = os.open(
        "pend",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=root_fd,
    )
    node = lease._adopt_pending(pfd, root_fd, "pend", is_dir=True)  # noqa: SLF001
    assert node in lease._pending  # noqa: SLF001
    owned_parent = int(node.parent_fd)

    backing = Path(lease.backing_path)
    (backing / "pend").rename(tmp_path / "owned-pend-moved")
    (backing / "pend").mkdir()
    foreign_st = os.lstat(backing / "pend")
    foreign_id = (foreign_st.st_dev, foreign_st.st_ino)

    real_fstat = os.fstat

    def fail_pending_fstat(fd: int):  # type: ignore[no-untyped-def]
        if fd == pfd:
            raise OSError(errno.EIO, "simulated pending fstat fail")
        return real_fstat(fd)

    with mock.patch.object(store_mod.os, "fstat", side_effect=fail_pending_fstat):
        ok = lease._scrub_pending_node(node)  # noqa: SLF001

    assert ok is False
    assert node.fd == pfd  # not closed/dropped
    assert node.parent_fd == owned_parent  # parent dup retained
    assert node in lease._pending  # noqa: SLF001
    assert (backing / "pend").is_dir()
    st2 = os.lstat(backing / "pend")
    assert (st2.st_dev, st2.st_ino) == foreign_id
    assert list((backing / "pend").iterdir()) == []

    try:
        os.close(pfd)
    except OSError:
        pass
    node.fd = -1
    try:
        os.close(owned_parent)
    except OSError:
        pass
    node.parent_fd = -1
    try:
        lease._pending.remove(node)  # noqa: SLF001
    except ValueError:
        pass
    lease.release()


def test_nested_pending_file_survives_parent_dir_fd_close(
    tmp_path: Path,
) -> None:
    """P1 nested-file: pending under closed sub_fd must still scrub siblings.

    Repro (would fail when pending borrowed the caller's parent fd):
      bind root → finalize already.bin with MODEL-BYTES → mkdir_owned sub →
      O_CREAT sub/m.bin → adopt_pending(fd, sub_fd, ...) → close sub_fd
      (simulate recursive unwind after create_file_owned fstat failure) →
      release() returned False forever and already.bin kept model bytes.

    Fixed: pending owns a stable parent-dir dup; successful retry scrubs
    prior sibling model bytes, empties pending/ledger, closes all owned fds.
    """
    model_bytes = b"MODEL-BYTES"
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()

    already_fd = lease.create_file_owned(root_fd, "already.bin")
    os.write(already_fd, model_bytes)
    os.close(already_fd)

    sub_fd = lease.mkdir_owned(root_fd, "sub")
    lease.reserve_identity_slot()
    mfd = os.open(
        "m.bin",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
        dir_fd=sub_fd,
    )
    os.write(mfd, b"nested-pending")
    node = lease._adopt_pending(mfd, sub_fd, "m.bin", is_dir=False)  # noqa: SLF001
    assert node in lease._pending  # noqa: SLF001
    assert node.fd == mfd
    owned_parent = int(node.parent_fd)
    assert owned_parent >= 0
    assert owned_parent != sub_fd  # stable dup, not borrow

    # Simulate freeze-walk recursive unwind closing the nested dest dir fd.
    os.close(sub_fd)

    assert lease.release() is True
    assert lease.cleanup_complete is True
    assert lease._pending == []  # noqa: SLF001
    assert lease._ledger == {}  # noqa: SLF001
    assert int(lease._root_fd) < 0  # noqa: SLF001
    assert int(lease._parent_fd) < 0  # noqa: SLF001
    with pytest.raises(OSError) as ei:
        os.fstat(mfd)
    assert ei.value.errno == errno.EBADF
    with pytest.raises(OSError) as ei_p:
        os.fstat(owned_parent)
    assert ei_p.value.errno == errno.EBADF
    # Prior sibling model bytes scrubbed — destroy-partial-freeze.
    for p in freeze_parent.rglob("already.bin"):
        pytest.fail(f"model byte residual {p}")
    for p in freeze_parent.rglob("m.bin"):
        pytest.fail(f"pending residual {p}")
    leftovers = [
        p
        for p in freeze_parent.iterdir()
        if p.is_dir() and p.name.startswith("cadml-freeze-")
    ]
    assert leftovers == []


def test_nested_pending_dir_survives_parent_dir_fd_close(
    tmp_path: Path,
) -> None:
    """P1 nested-directory: pending subdir under closed parent still scrubs.

    Same unwind pattern as the nested-file case, but the pending object is a
    directory created under a nested parent that the walk then closes.
    """
    model_bytes = b"MODEL-BYTES"
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()

    already_fd = lease.create_file_owned(root_fd, "already.bin")
    os.write(already_fd, model_bytes)
    os.close(already_fd)

    mid_fd = lease.mkdir_owned(root_fd, "mid")
    lease.reserve_identity_slot()
    os.mkdir("pend", 0o700, dir_fd=mid_fd)
    pfd = os.open(
        "pend",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=mid_fd,
    )
    node = lease._adopt_pending(pfd, mid_fd, "pend", is_dir=True)  # noqa: SLF001
    assert node in lease._pending  # noqa: SLF001
    owned_parent = int(node.parent_fd)
    assert owned_parent >= 0
    assert owned_parent != mid_fd

    os.close(mid_fd)

    assert lease.release() is True
    assert lease.cleanup_complete is True
    assert lease._pending == []  # noqa: SLF001
    assert lease._ledger == {}  # noqa: SLF001
    assert int(lease._root_fd) < 0  # noqa: SLF001
    assert int(lease._parent_fd) < 0  # noqa: SLF001
    with pytest.raises(OSError) as ei:
        os.fstat(pfd)
    assert ei.value.errno == errno.EBADF
    with pytest.raises(OSError) as ei_p:
        os.fstat(owned_parent)
    assert ei_p.value.errno == errno.EBADF
    for p in freeze_parent.rglob("already.bin"):
        pytest.fail(f"model byte residual {p}")
    leftovers = [
        p
        for p in freeze_parent.iterdir()
        if p.is_dir() and p.name.startswith("cadml-freeze-")
    ]
    assert leftovers == []


def test_nested_pending_mismatch_retains_ownership_and_foreign(
    tmp_path: Path,
) -> None:
    """Nested pending name mismatch: retain child+parent dups; foreign survives."""
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()

    already_fd = lease.create_file_owned(root_fd, "already.bin")
    os.write(already_fd, b"MODEL-BYTES")
    os.close(already_fd)

    sub_fd = lease.mkdir_owned(root_fd, "sub")
    lease.reserve_identity_slot()
    mfd = os.open(
        "m.bin",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
        dir_fd=sub_fd,
    )
    os.write(mfd, b"owned-nested")
    node = lease._adopt_pending(mfd, sub_fd, "m.bin", is_dir=False)  # noqa: SLF001
    owned_parent = int(node.parent_fd)
    os.close(sub_fd)

    backing = Path(lease.backing_path)
    (backing / "sub" / "m.bin").rename(tmp_path / "owned-nested-moved.bin")
    (backing / "sub" / "m.bin").write_bytes(b"")  # foreign at name
    foreign_st = os.lstat(backing / "sub" / "m.bin")
    foreign_id = (foreign_st.st_dev, foreign_st.st_ino)

    assert lease.release() is False
    assert lease.cleanup_complete is False
    assert node in lease._pending  # noqa: SLF001
    assert node.fd == mfd
    assert node.parent_fd == owned_parent
    st2 = os.lstat(backing / "sub" / "m.bin")
    assert (st2.st_dev, st2.st_ino) == foreign_id
    # Sibling model bytes retained until pending can complete (honest incomplete).
    assert (backing / "already.bin").read_bytes() == b"MODEL-BYTES"
    st_fd = os.fstat(mfd)
    assert st_fd.st_size == len(b"owned-nested")
    assert lease.release() is False
    assert node.fd == mfd
    assert node.parent_fd == owned_parent

    # Manual teardown for suite hygiene.
    try:
        os.close(mfd)
    except OSError:
        pass
    node.fd = -1
    try:
        os.close(owned_parent)
    except OSError:
        pass
    node.parent_fd = -1
    try:
        lease._pending.remove(node)  # noqa: SLF001
    except ValueError:
        pass
    lease._ledger.clear()  # noqa: SLF001
    lease.release()


def test_adopt_pending_parent_dup_failure_no_orphan_no_claim(
    tmp_path: Path,
) -> None:
    """Parent-fd dup failure: destroy child via live parent or retain; no claim."""
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()
    lease.reserve_identity_slot()
    fd = os.open(
        "m.bin",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
        dir_fd=root_fd,
    )
    os.write(fd, b"dup-fail-bytes")

    with mock.patch.object(
        store_mod, "_dup_dir_fd_cloexec", side_effect=OSError(errno.EMFILE, "dup")
    ):
        with pytest.raises(ActivationRefusal) as ei:
            lease._adopt_pending(fd, root_fd, "m.bin", is_dir=False)  # noqa: SLF001
    assert ei.value.reason is RefusalReason.FREEZE_FAILED
    assert "parent fd dup" in ei.value.detail
    # Must not claim cleanup complete after a failed adopt.
    assert lease.cleanup_complete is False
    # Best-effort destroy succeeded via still-live root_fd → not pending.
    assert lease._pending == []  # noqa: SLF001
    with pytest.raises(OSError) as ei_fd:
        os.fstat(fd)
    assert ei_fd.value.errno == errno.EBADF
    # No model-byte residual at the create name.
    backing = Path(lease.backing_path)
    assert not (backing / "m.bin").exists()

    assert lease.release() is True
    assert lease.cleanup_complete is True


def test_adopt_pending_parent_dup_failure_retains_when_destroy_fails(
    tmp_path: Path,
) -> None:
    """If parent dup fails and immediate destroy also fails: retain child, no claim."""
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()
    lease.reserve_identity_slot()
    fd = os.open(
        "m.bin",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
        dir_fd=root_fd,
    )
    os.write(fd, b"retain-on-dup-fail")

    with mock.patch.object(
        store_mod, "_dup_dir_fd_cloexec", side_effect=OSError(errno.EMFILE, "dup")
    ):
        with mock.patch.object(
            FreezeResourceLease, "_remove_owned_name", return_value=False
        ):
            with pytest.raises(ActivationRefusal) as ei:
                lease._adopt_pending(  # noqa: SLF001
                    fd, root_fd, "m.bin", is_dir=False
                )
    assert ei.value.reason is RefusalReason.FREEZE_FAILED
    assert lease.cleanup_complete is False
    assert len(lease._pending) == 1  # noqa: SLF001
    node = lease._pending[0]  # noqa: SLF001
    assert node.fd == fd  # child not orphaned
    assert node.parent_fd < 0  # no usable parent for name scrub
    st_fd = os.fstat(fd)
    assert st_fd.st_size == len(b"retain-on-dup-fail")
    # Honest incomplete: release cannot finish while pending lacks parent.
    assert lease.release() is False
    assert node.fd == fd

    try:
        os.close(fd)
    except OSError:
        pass
    node.fd = -1
    try:
        lease._pending.remove(node)  # noqa: SLF001
    except ValueError:
        pass
    lease._ledger.clear()  # noqa: SLF001
    lease.release()


def test_store_close_waits_and_no_double_close(tmp_path: Path) -> None:
    root = tmp_path / "store"
    data = b"x"
    _write(root / "m.pt", data)
    pin = _sf_pin("act.s", "art.s", "m.pt", data)
    store = _store(root, [pin])
    out = store.assert_fixed_hash("act.s", "art.s")
    assert out == data
    assert store.close() is True
    assert store.close() is True  # idempotent
    with pytest.raises(ActivationRefusal) as ei:
        store.assert_fixed_hash("act.s", "art.s")
    assert ei.value.reason is RefusalReason.INTERNAL


def test_store_close_retains_incomplete_pending_leases(tmp_path: Path) -> None:
    """close() must not forget a lease whose release() returned False.

    First close/drain returns False and keeps the same lease; second succeeds
    and clears residual ownership. Store-root close remains independent of
    freeze lease fds.
    """
    root = tmp_path / "store"
    root.mkdir()
    store = _store(root, [], freeze_parent=tmp_path / "freeze")
    try:
        lease = FreezeResourceLease()
        lease.bind_root(str(_trusted_freeze_parent(tmp_path / "freeze")))
        # Construction-failure style retain.
        store._retain_pending_lease(lease)  # noqa: SLF001
        assert lease in store._pending_leases  # noqa: SLF001

        real_release = lease.release
        calls: list[int] = []
        lease_id = id(lease)

        def release_false_then_true() -> bool:
            calls.append(lease_id)
            if len(calls) == 1:
                return False
            return bool(real_release())

        with mock.patch.object(lease, "release", side_effect=release_false_then_true):
            assert store.close() is False
            # Same lease retained for retry — not permanently forgotten.
            assert store._pending_leases == [lease]  # noqa: SLF001
            assert id(store._pending_leases[0]) == lease_id  # noqa: SLF001
            # Explicit drain / second close retries the same object.
            assert store.drain_pending_leases() is True
            assert store._pending_leases == []  # noqa: SLF001
        assert calls == [lease_id, lease_id]
        # Fully drained close is complete.
        assert store.close() is True
    finally:
        try:
            store.close()
        except Exception:
            pass


def test_retained_lease_release_serialized_two_threads(tmp_path: Path) -> None:
    """Two concurrent release() calls on one retained lease: max concurrent body=1.

    ControlledStore.drain_pending_leases snapshots then releases outside its
    lock; per-lease serialization must still hold so ledger/fd mutation cannot
    race. Both results honest (True after successful scrub).
    """
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    lease = FreezeResourceLease()
    lease.bind_root(str(freeze_parent))
    root_fd = lease.root_dir_fd()
    # Owned content so scrub does real work.
    ffd = lease.create_file_owned(root_fd, "m.bin")
    os.write(ffd, b"x")
    os.close(ffd)

    real_scrub = lease._scrub_tree
    active = {"n": 0}
    max_active = {"n": 0}
    gate = threading.Lock()

    def slow_scrub(dir_fd: int) -> bool:
        with gate:
            active["n"] += 1
            if active["n"] > max_active["n"]:
                max_active["n"] = active["n"]
        try:
            time.sleep(0.12)
            return bool(real_scrub(dir_fd))
        finally:
            with gate:
                active["n"] -= 1

    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    with mock.patch.object(lease, "_scrub_tree", side_effect=slow_scrub):

        def worker() -> None:
            barrier.wait(timeout=5.0)
            ok = lease.release()
            with results_lock:
                results.append(ok)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        assert not t1.is_alive() and not t2.is_alive()

    assert max_active["n"] == 1, (
        f"release body ran concurrently (max_active={max_active['n']})"
    )
    assert len(results) == 2
    assert all(r is True for r in results), f"honest success expected, got {results}"
    assert lease.cleanup_complete is True
    assert lease._ledger == {}  # noqa: SLF001
    # Idempotent third call.
    assert lease.release() is True


def test_openat2_enosys_runtime_is_internal_no_same_request_fallback(
    tmp_path: Path,
) -> None:
    """Runtime ENOSYS → INTERNAL this request; no same-request component fallback.

    Publishes unprobed (not permanent ``"no"``) so a later request may re-probe.
    Does not require a real kernel openat2 — publishes a synthetic ``ok`` snapshot.
    """
    root = tmp_path / "store"
    data = b"z"
    _write(root / "m.pt", data)
    pin = _sf_pin("act.z", "art.z", "m.pt", data)
    store = _store(root, [pin], impl=ResolverImpl.OPENAT2)

    def enosys_syscall(*_a, **_k):  # type: ignore[no-untyped-def]
        ctypes.set_errno(errno.ENOSYS)
        return -1

    # Synthetic ok snapshot so openat2_available() is True without a real probe.
    resolver_mod._publish_openat2_state("ok", fn=enosys_syscall, nr=437)
    try:
        with mock.patch.object(
            resolver_mod,
            "_component_walk",
            side_effect=AssertionError("must not same-request fallback"),
        ):
            with pytest.raises(ActivationRefusal) as ei:
                store.assert_fixed_hash("act.z", "art.z")
        assert ei.value.reason is RefusalReason.INTERNAL
        # Unprobed + cleared fn/nr — not permanent "no".
        st, fn, n = resolver_mod._snapshot_openat2()
        assert st is None
        assert fn is None
        assert n is None
    finally:
        resolver_mod._publish_openat2_state(None, fn=None, nr=None)
        openat2_available()
        store.close()


def test_openat2_enosys_next_request_reprobes_may_component_fallback(
    tmp_path: Path,
) -> None:
    """After runtime ENOSYS reset, the next request re-probes and may fall back."""
    root = tmp_path / "store"
    data = b"reprobe-bytes"
    _write(root / "m.pt", data)
    pin = _sf_pin("act.r", "art.r", "m.pt", data)

    def enosys_syscall(*_a, **_k):  # type: ignore[no-untyped-def]
        ctypes.set_errno(errno.ENOSYS)
        return -1

    # First request: ENOSYS → INTERNAL, state unprobed.
    resolver_mod._publish_openat2_state("ok", fn=enosys_syscall, nr=437)
    store1 = _store(root, [pin], impl=ResolverImpl.OPENAT2)
    try:
        with mock.patch.object(
            resolver_mod,
            "_component_walk",
            side_effect=AssertionError("must not same-request fallback"),
        ):
            with pytest.raises(ActivationRefusal) as ei:
                store1.assert_fixed_hash("act.r", "art.r")
        assert ei.value.reason is RefusalReason.INTERNAL
        st0, fn0, n0 = resolver_mod._snapshot_openat2()
        assert st0 is None and fn0 is None and n0 is None
    finally:
        store1.close()

    # Next request: openat2_available must enter the probe path (state was None).
    probes = {"n": 0}

    def counting_available() -> bool:
        st, _, _ = resolver_mod._snapshot_openat2()
        if st is None:
            probes["n"] += 1
            # Re-probe outcome for this test: force permanent unavailability so
            # default selection uses component fallback.
            resolver_mod._publish_openat2_state("no")
            return False
        return st == "ok"

    with mock.patch.object(
        resolver_mod, "openat2_available", side_effect=counting_available
    ):
        assert resolver_mod.default_resolver_impl() is ResolverImpl.COMPONENT
        assert probes["n"] >= 1
        store2 = _store(root, [pin], impl=ResolverImpl.COMPONENT)
        try:
            assert store2.assert_fixed_hash("act.r", "art.r") == data
            assert last_open_impl() is ResolverImpl.COMPONENT
        finally:
            store2.close()

    resolver_mod._publish_openat2_state(None, fn=None, nr=None)
    openat2_available()


def test_enotdir_maps_to_not_directory(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    # Create a file where a directory component is expected.
    _write(root / "notdir", b"x")
    fd, _ = open_store_root(str(root))
    try:
        with pytest.raises(ActivationRefusal) as ei:
            open_pinned(
                fd,
                "notdir/leaf.pt",
                TerminalKind.REGULAR_FILE,
                impl=ResolverImpl.COMPONENT,
            )
        assert ei.value.reason is RefusalReason.NOT_DIRECTORY
    finally:
        os.close(fd)


def test_freeze_leaf_flags_include_nonblock() -> None:
    assert os.O_NONBLOCK & store_mod._MEMBER_FLAGS
    assert os.O_NONBLOCK & store_mod._FREEZE_LEAF_FLAGS
    assert os.O_NONBLOCK & store_mod._FREEZE_CREATE_FLAGS
    assert os.O_NONBLOCK & resolver_mod._FILE_FLAGS


def test_constructor_lease_handoff_and_cleanup_retry(tmp_path: Path) -> None:
    """Construction failure retains pending lease when cleanup incomplete."""
    root = tmp_path / "store"
    freeze_parent = _trusted_freeze_parent(tmp_path / "freeze")
    _write(root / "bundle" / "m.bin", b"good")
    pin = _bundle_pin("act.b", "art.b", "bundle", [("m.bin", b"good")])
    store = _store(root, [pin], freeze_parent=freeze_parent)

    # Force digest mismatch after freeze so construction fails post-copy.
    bad = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="ab" * 32,
        store_relpath="bundle",
    )
    store2 = _store(root, [bad], freeze_parent=freeze_parent)
    try:
        with pytest.raises(ActivationRefusal) as ei:
            store2.assert_bundle_digest("act.b", "art.b")
        assert ei.value.reason is RefusalReason.DIGEST_MISMATCH
        # Owned model bytes must be destroyed on construction failure.
        for f in freeze_parent.rglob("m.bin"):
            pytest.fail(f"residual model bytes {f}")
    finally:
        store2.close()
        store.close()


def test_no_commit_reserved_identity_in_production() -> None:
    assert not hasattr(store_mod, "commit_reserved_identity")
    assert not hasattr(FreezeResourceLease, "commit_reserved_identity")


def test_bundle_fifo_source_nonblock_member_flags(tmp_path: Path) -> None:
    """FIFO under bundle root must not block (O_NONBLOCK on member open)."""
    root = tmp_path / "store"
    bdir = root / "bundle"
    bdir.mkdir(parents=True)
    os.mkfifo(bdir / "pipe")
    pin = PinRecord(
        logical_activation_id="act.b",
        artifact_id="art.b",
        kind=ArtifactKind.BUNDLE,
        digest="00" * 32,
        store_relpath="bundle",
    )
    store = _store(root, [pin], freeze_parent=tmp_path / "freeze")
    try:
        t0 = time.monotonic()
        with pytest.raises(ActivationRefusal) as ei:
            store.assert_bundle_digest("act.b", "art.b")
        assert time.monotonic() - t0 < 2.0
        assert ei.value.reason is RefusalReason.SPECIAL_FILE
    finally:
        store.close()
